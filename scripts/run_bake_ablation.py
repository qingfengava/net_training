"""烘焙 v3 增强消融：同一份样式 B 数据按四种增强配置各烘焙一次，每组从 A 权重微调 30 轮，统一评估。

    python scripts/run_bake_ablation.py [--seed 0] [--epochs 30] [--dry-run] [--report-only]

四组：修正核基线 / 仅预测 ROI / 仅摩尔纹 / 两者同时开启。数据源、烘焙种子、初始权重、训练配置和
训练种子完全相同，只有烘焙配置不同。验证分区不叠加两项新增强，脚本会逐字节核对四份验证集一致。
每个模型在三份验证集上评估：新标准验证集（只能看出有没有害处）、合成压力验证集（增强也作用于验证，
与训练增强同源，不代表真实相机）、A→B 实验的旧验证集（与 b_from_a_e30_s0 的成绩衔接）。
串行执行，可续跑；输入变化时拒绝混用旧结果。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUPS = [("base", "default.yaml", "修正核基线"), ("roi", "tracker_roi.yaml", "仅预测 ROI"),
          ("moire", "moire.yaml", "仅摩尔纹"), ("both", "tracker_moire.yaml", "ROI + 摩尔纹")]
SOURCES = ["datasets/sim/style_b_train", "datasets/sim/style_b_val"]
BAKE_SEED = 0  # 与 baked/style_b_v1 相同
OLD_BAKE = Path("baked/style_b_v1")
OLD_RUN = Path("runs/train/b_from_a_e30_s0")
AP50_TOL, ROBOT_TOL = 0.01, 0.01  # 标准验证集上相对基线可接受的损失：AP50 0.01，机器人类别准确率 1 个百分点


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def val_digest(bake: Path) -> str:
    """验证分区的图像、标签和 meta 行的摘要；四组烘焙必须相同，成绩才能直接比较。"""
    h = hashlib.sha256()
    for sub in ("images/val", "labels/val"):
        for p in sorted((bake / sub).iterdir()):
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    rows = sorted(line for line in (bake / "meta.jsonl").read_text().splitlines() if '"split":"val"' in line)
    h.update(hashlib.sha256("\n".join(rows).encode()).digest())
    return h.hexdigest()


def progress(run: Path) -> dict:
    path = run / "results.csv"
    if not path.exists():
        return {}
    try:
        row = list(csv.DictReader(path.open()))[-1]
        return {k: row[k] for k in ("epoch", "time", "metrics/mAP50(B)") if k in row}
    except (IndexError, ValueError, OSError):
        return {}
class Runner:
    """串行执行命令（经 run_capped.sh），写 status.json 和每步日志；任何一步失败就停。"""

    def __init__(self, out: Path, dry_run: bool) -> None:
        self.out, self.dry_run = out, dry_run
        self.status = {"status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
        previous = out / "status.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            self.status["steps"] = old.get("steps", {})
            self.status["started_at"] = old.get("started_at", self.status["started_at"])

    def save(self) -> None:
        write_json(self.out / "status.json", self.status)

    def run(self, key: str, command: list[str], memory_gb: int, run_dir: Path | None = None) -> None:
        cmd = ["scripts/run_capped.sh", str(memory_gb), "--", *command]
        if self.dry_run:
            print(json.dumps({"step": key, "command": cmd}, ensure_ascii=False), flush=True)
            return
        log = self.out / f"{key}.log"
        step = {"status": "running", "command": cmd, "log": str(log)}
        self.status.update(status="running", current_step=key)
        self.status["steps"][key] = step
        self.save()
        start = time.monotonic()
        print(f"开始 {key}，日志：{log}", flush=True)
        with log.open("a") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        step["seconds"] = round(time.monotonic() - start, 1)
                        step["progress"] = progress(run_dir) if run_dir else {}
                        self.save()
                        print(json.dumps({"step": key, "seconds": step["seconds"], **step["progress"]},
                                         ensure_ascii=False), flush=True)
            except BaseException:
                proc.terminate()
                raise
        step.update(status="completed" if code == 0 else "failed", exit_code=code,
                    seconds=round(time.monotonic() - start, 1))
        if code:
            self.status["status"] = "failed"
        self.save()
        if code:
            raise SystemExit(f"{key} 失败，退出码 {code}；请检查 {log}")
def bake_one(r: Runner, name: str, config: str, sources: list[str], workers: int, memory_gb: int) -> Path:
    target = Path("baked") / name
    if target.exists():
        if not (target / "bake_manifest.json").exists():
            raise SystemExit(f"{target} 已存在但没有 bake_manifest.json，像是中断留下的半成品；"
                             "请确认后删除再重跑（脚本不会自行删除数据）。")
        return target
    cmd = [sys.executable, "-m", "nt.bake"]
    for src in sources:
        cmd += ["--src", src]
    cmd += ["--config", f"configs/bake/{config}", "--out", str(target), "--copies", "2",
            "--seed", str(BAKE_SEED), "--workers", str(workers)]
    r.run(f"bake_{name}", cmd, memory_gb)
    return target


def train_one(r: Runner, name: str, init: Path, hyp: Path, config: dict, epochs: int, seed: int, data: Path,
              memory_gb: int) -> Path:
    run = Path("runs/train") / name
    meta_path = run / "nt_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        eff = meta["hyp"]["effective"]
        done = (meta["status"] == "completed" and eff["epochs"] == epochs and eff["seed"] == seed
                and meta["data"]["bake_manifest_sha256"] == sha(data / "bake_manifest.json")
                and all(eff.get(k) == v for k, v in config.items() if k not in {"epochs", "seed"}))
        if done:
            return run
    if run.exists() and not r.dry_run:
        raise SystemExit(f"{run} 已存在且未通过一致性检查，保留原数据并停止。")
    milestones = sorted({e for e in (10, epochs) if e <= epochs})
    r.run(name, [sys.executable, "-m", "nt.train", "--model", str(init), "--data", str(data), "--name", name,
                 "--hyp", str(hyp), "--epochs", str(epochs), "--seed", str(seed), "--device", "0",
                 "--milestones", *map(str, milestones)], memory_gb, run)
    if not r.dry_run:
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "completed" or int(meta["last_metrics"]["epoch"]) != epochs:
            raise SystemExit(f"{name} 没有完成约定轮数，不能作为完整实验结果。")
    return run


def evaluate(r: Runner, key: str, weights: Path, data: Path) -> Path:
    target = r.out / f"{key}.json"
    cmd = [sys.executable, "-m", "nt.eval", "--weights", str(weights), "--data", str(data), "--head", "e2e",
           "--conf", "0.2", "--device", "0", "--out", str(target)]
    if r.dry_run:
        r.run(key, cmd, 5)
        return target
    stamp = {"weights_sha256": sha(weights), "bake_sha256": sha(data / "bake_manifest.json"), "head": "e2e", "conf": 0.2}
    checked = r.out / f"{key}.inputs.json"
    if target.exists() and checked.exists() and json.loads(checked.read_text()) == stamp:
        return target
    target.unlink(missing_ok=True)
    r.run(key, cmd, 5)
    write_json(checked, stamp)
    return target


def check_bakes(bakes: dict[str, Path]) -> dict:
    """四组的图像数和验证分区必须一致，否则成绩不可比，停止并说明。"""
    info = {}
    for g, path in bakes.items():
        m = json.loads((path / "bake_manifest.json").read_text())
        info[g] = {"images": m["images"], "val_digest": val_digest(path), "manifest": m}
    base = info["base"]
    for g, v in info.items():
        if v["images"] != base["images"]:
            raise SystemExit(f"{g} 的图像数 {v['images']} 与基线 {base['images']} 不同，训练集规模不一致，停止。")
        if v["val_digest"] != base["val_digest"]:
            raise SystemExit(f"{g} 的验证分区与基线不同；成绩不能直接比较，先查原因再训练。")
    return info
def metrics(res: dict) -> dict:
    s, w = res["summary"], res["by_width_px"]
    tiny, small = w.get("[0,8)", {}), w.get("[8,16)", {})
    return {"ap50": s["ap50"], "robot_acc": s["robot_numbers"]["class_acc"], "recall": s["recall"],
            "precision": s["precision"], "off_recall": (s["off_class"] or {}).get("recall"),
            "corner_p95": s["corner_p95_orig_px"], "false_positives": res["at_conf"]["false_positives"],
            "tiny_n": tiny.get("n"), "tiny_recall": tiny.get("recall"), "tiny_acc": tiny.get("class_acc"),
            "tiny_correct_per_gt": round(tiny["correct"] / tiny["n"], 4) if tiny.get("n") else None,
            "small_acc": small.get("class_acc")}


def train_widths(bake: Path) -> tuple[list[float], list[int]]:
    """训练分区每个装甲标注在网络输入上的宽度，以及每张图的装甲数。"""
    widths, per_img = [], []
    for line in (bake / "meta.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] != "train":
            continue
        n = 0
        for armor in row["armors"]:
            corners = armor.get("corners_orig")
            if corners:
                xs = [p[0] * row["scale"] for p in corners]
                widths.append(max(xs) - min(xs))
                n += 1
        per_img.append(n)
    return widths, per_img


def fmt(v, digits: int = 4) -> str:
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def table(title: str, rows: list[tuple[str, dict, float | None]]) -> list[str]:
    lines = [f"### {title}", "",
             "| 模型 | 训练分钟 | AP50 | 机器人类别准确率 | 召回率 | 熄灭召回率 | 角点 P95/原图 px | 误检数 | "
             "<8 px 检出且分类正确 | <8 px 类别准确率 | 8–16 px 类别准确率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m, minutes in rows:
        lines.append(f"| {name} | {fmt(minutes, 1)} | {fmt(m['ap50'])} | {fmt(m['robot_acc'])} | {fmt(m['recall'])} | "
                     f"{fmt(m['off_recall'])} | {fmt(m['corner_p95'], 3)} | {m['false_positives']} | "
                     f"{fmt(m['tiny_correct_per_gt'])} | {fmt(m['tiny_acc'])} | {fmt(m['small_acc'])} |")
    return lines + [""]


def verdict(group: str, label: str, m: dict, base: dict) -> str:
    d_ap, d_robot = m["ap50"] - base["ap50"], m["robot_acc"] - base["robot_acc"]
    d_off = None if m["off_recall"] is None or base["off_recall"] is None else m["off_recall"] - base["off_recall"]
    d_tiny = None if m["tiny_correct_per_gt"] is None else m["tiny_correct_per_gt"] - base["tiny_correct_per_gt"]
    text = (f"- {label}：AP50 {d_ap:+.4f}，机器人类别准确率 {100 * d_robot:+.2f} 个百分点，"
            f"熄灭召回 {'-' if d_off is None else f'{100 * d_off:+.2f} 个百分点'}，"
            f"<8 px 检出且分类正确 {'-' if d_tiny is None else f'{100 * d_tiny:+.2f} 个百分点'}，"
            f"角点 P95 {m['corner_p95'] - base['corner_p95']:+.3f} px，误检 {m['false_positives'] - base['false_positives']:+d}。")
    if d_ap < -AP50_TOL or d_robot < -ROBOT_TOL:
        return text + " **超出预设容差，需要查原因。**"
    return text + " 在预设容差内（AP50 ≥ -0.01、机器人类别准确率 ≥ -1 个百分点）。"
def report(out: Path, runs: dict[str, str], bakes: dict[str, Path], fingerprint: dict, seed: int) -> None:
    sets = [("new_val", "新标准验证集（四组验证分区逐字节相同）"),
            ("stress", "合成压力验证集（预测 ROI 和摩尔纹也作用于验证，与训练增强同源）"),
            ("old_val", "A→B 实验的旧验证集（旧模糊核，用来和 b_from_a_e30_s0 的成绩衔接）")]
    old_minutes = json.loads((OLD_RUN / "nt_meta.json").read_text())["wall_time_s"] / 60
    minutes, curves = {}, []
    for g, run in runs.items():
        meta = json.loads((Path("runs/train") / run / "nt_meta.json").read_text())
        minutes[g] = meta["wall_time_s"] / 60
        with (Path("runs/train") / run / "results.csv").open() as stream:
            curves += [{"run": run, **row} for row in csv.DictReader(stream)]
    results = {}
    for key, _ in sets:
        old = json.loads((out / f"old_b_from_a_e30_s0_on_{key}.json").read_text())["heads"]["e2e"]
        results[key] = {"old": metrics(old)}
        for g, run in runs.items():
            results[key][g] = metrics(json.loads((out / f"{run}_on_{key}.json").read_text())["heads"]["e2e"])
    lines = [f"# 烘焙 v3 增强消融（样式 B，A→B 微调 30 轮，种子 {seed}）", "",
             "四组使用同一份样式 B 数据、同一烘焙种子、同一 A 初始权重和同一训练配置，只有烘焙配置不同。",
             "端到端检测头，置信度 0.2。机器人类别准确率要求颜色和编号同时正确，分母是匹配到的装甲；",
             "“<8 px 检出且分类正确”的分母是该宽度档的全部目标，含漏检。", "",
             "## 烘焙统计", "",
             "| 组 | 训练图 | 验证图 | 窗口模式 | 预测 ROI 状态 | 摩尔纹实际生效图数 | 被 ROI 丢掉的装甲 | 训练集装甲标注数 |",
             "|---|---:|---:|---|---|---:|---:|---:|"]
    labels = {}
    for g, cfg, label in GROUPS:
        m = json.loads((bakes[g] / "bake_manifest.json").read_text())
        labels[g] = sum(m["armors"]["train"].values())
        lines.append(f"| {label} | {m['images']['train']} | {m['images']['val']} | {m.get('roi_modes')} | "
                     f"{m.get('tracker_roi_states', '-')} | {m.get('images_with_moire', '-')} | "
                     f"{m.get('armors_dropped_by_roi')} | {labels[g]} |")
    fewer = {g: 100 * (1 - labels[g] / labels["base"]) for g in labels if labels[g] < labels["base"] * 0.99}
    if fewer:
        lines += ["", "预测窗口比随机窗口更紧，会裁掉更多同帧的其他装甲，训练集的装甲标注数因此变少"
                  f"（{', '.join(f'{dict((g, l) for g, _, l in GROUPS)[g]}少 {v:.1f}%' for g, v in fewer.items())}）。"
                  "窗口也更紧，目标在网络输入上更大：训练标签的宽度分布见下表。这两组的成绩差异里混有训练数据分布变化的影响，"
                  "不能全部归因于增强本身。"]
    lines += ["", "训练集装甲标注按网络输入宽度的占比（由各烘焙的 meta.jsonl 统计）：", "",
              "| 组 | <8 px | 8–16 px | 16–32 px | 32–64 px | ≥64 px | 宽度中位数 | 每图装甲数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g, cfg, label in GROUPS:
        widths, per_img = train_widths(bakes[g])
        w = np.array(widths)
        hist = [float(((w >= lo) & (w < hi)).mean()) for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]]
        lines.append(f"| {label} | " + " | ".join(f"{100 * v:.1f}%" for v in hist) +
                     f" | {float(np.median(w)):.1f} px | {sum(per_img) / len(per_img):.2f} |")
    lines += ["", "## 结果", ""]
    for key, title in sets:
        rows = [("b_from_a_e30_s0（旧核、旧增强）", results[key]["old"], old_minutes)]
        rows += [(label, results[key][g], minutes[g]) for g, _, label in GROUPS]
        lines += table(title, rows)
    lines += ["## 相对修正核基线的差值", ""]
    for key, title in sets[:2]:
        lines += [f"**{title}**", ""]
        lines += [verdict(g, label, results[key][g], results[key]["base"]) for g, _, label in GROUPS[1:]] + [""]
    lines += ["## 怎么读这些数字", "",
              "- 新标准验证集里没有摩尔纹，也没有跟踪误差造成的截断窗口。两项增强在这里最好的结果是和基线持平，",
              "  所以它只能回答“有没有害处”，不能回答“对真实相机有没有好处”。",
              "- 压力验证集和训练增强来自同一套模拟，按道理开了增强的模型应该在这里占优，所以它是“增强有没有用”的",
              "  最有利的仿真检验。它的目标数和标准验证集不同（预测窗口把一部分装甲裁出画面），两份表的数字不要互相比较，",
              "  只在各自表内比较。基线在压力验证集上相对标准验证集的降幅有多大，决定了增强最多能带来多少好处。",
              "- 单个训练种子，训练没有开确定性模式。AP50 相差 0.01 以内、机器人类别准确率相差 1 个百分点以内的差别，",
              "  不要当作增强的效果；要下结论就补第二个种子。",
              "- 旧验证集一栏里，新基线与旧模型的差别来自两处：训练用了修正后的模糊核，以及训练数据的随机取窗重新抽样。",
              "- 真实相机上的收益要用代理屏会话或实拍会话验证；ROI 增强还需要 sleepy 实际取窗的记录。", "",
              "本次实验的输入指纹见 inputs.json；各组训练的逐轮指标见 learning_curves.csv。"]
    write_json(out / "comparison.json", {"inputs": fingerprint, "train_minutes": minutes, "results": results})
    with (out / "learning_curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init", type=Path, default=Path("runs/train/y26n_simm7_e100/weights/last.pt"))
    ap.add_argument("--bake-workers", type=int, default=4)
    ap.add_argument("--bake-memory-gb", type=int, default=6)
    ap.add_argument("--train-memory-gb", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印要执行的命令")
    args = ap.parse_args()
    os.chdir(ROOT)
    from nt.train import load_hyp
    hyp = Path("configs/train/pose_transfer_ab.yaml")
    config = load_hyp(hyp)
    out = Path("runs/experiments") / f"bake_ablation_v3_s{args.seed}"
    cfg_files = ["default.yaml", *[g[1] for g in GROUPS], "ablation_stress_val.yaml"]
    fingerprint = {"initial_weights_sha256": sha(args.init), "hyp": config, "seed": args.seed, "epochs": args.epochs,
                   "bake_seed": BAKE_SEED, "sources": {s: sha(Path(s) / "manifest.json") for s in SOURCES},
                   "bake_configs": {c: sha(ROOT / "configs/bake" / c) for c in cfg_files},
                   "code": {f: sha(ROOT / f) for f in ("nt/train.py", "nt/eval.py", "nt/bake.py", "nt/bake_augment.py")}}
    frozen = out / "inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != fingerprint and not args.report_only:
        raise SystemExit("实验输入已变化，拒绝混用旧结果；请换一个种子或独立的实验目录。")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        if not frozen.exists():
            write_json(frozen, fingerprint)
    runs = {g: f"abl_{g}_s{args.seed}" for g, _, _ in GROUPS}
    bakes = {g: Path("baked") / f"style_b_v3_{g}" for g, _, _ in GROUPS}
    stress = Path("baked/style_b_v3_stress_val")
    r = Runner(out, args.dry_run)
    if not args.report_only:
        for g, cfg, _ in GROUPS:
            bake_one(r, bakes[g].name, cfg, SOURCES, args.bake_workers, args.bake_memory_gb)
        bake_one(r, stress.name, "ablation_stress_val.yaml", [SOURCES[1]], args.bake_workers, args.bake_memory_gb)
        if not args.dry_run:
            info = check_bakes(bakes)
            if json.loads((stress / "bake_manifest.json").read_text())["images"].get("val") != info["base"]["images"]["val"]:
                raise SystemExit("压力验证集的图像数与标准验证集不同，停止。")
            write_json(out / "bake_check.json", {g: {"images": v["images"], "val_digest": v["val_digest"]}
                                                 for g, v in info.items()})
        sets = {"new_val": bakes["base"], "stress": stress, "old_val": OLD_BAKE}
        for key, data in sets.items():  # 旧模型先评：训练之前就确认新验证集能被评估
            evaluate(r, f"old_b_from_a_e30_s0_on_{key}", OLD_RUN / "weights/last.pt", data)
        for g, _, _ in GROUPS:
            run = train_one(r, runs[g], args.init, hyp, config, args.epochs, args.seed, bakes[g], args.train_memory_gb)
            for key, data in sets.items():
                evaluate(r, f"{runs[g]}_on_{key}", run / "weights/last.pt", data)
    if args.dry_run:
        return
    report(out, runs, bakes, fingerprint, args.seed)
    r.status.update(status="completed", current_step=None, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    r.save()
    print(f"实验完成：{out / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
