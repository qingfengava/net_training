"""串行运行 A→B 三组迁移实验，记录耗时、校验输入并汇总统一验证结果。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def progress(run: Path) -> dict:
    path = run / "results.csv"
    if not path.exists():
        return {}
    try:
        rows = list(csv.DictReader(path.open()))
        row = rows[-1]
        return {k: row[k] for k in ("epoch", "time", "metrics/mAP50(B)") if k in row}
    except (IndexError, ValueError, OSError):
        return {}


def plot_curves(out: Path, trials: list[tuple[str, Path, int]]) -> None:
    # 只连接同一训练轨迹的真实评估点，横轴使用该轨迹实际记录的累计训练时间。
    os.environ.setdefault("MPLCONFIGDIR", str((out / "matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    direct = json.loads((out / "a_direct_on_b.json").read_text())["heads"]["e2e"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for name, _, epochs in trials:
        run = ROOT / "runs/train" / name
        with (run / "results.csv").open() as stream:
            clock = {int(row["epoch"]): float(row["time"]) / 60 for row in csv.DictReader(stream)}
        points = []
        if "from_a" in name:
            points.append((0, direct))
        for epoch in (10, 30, 60):
            if epoch < epochs:
                s = json.loads((out / f"{name}_e{epoch}.json").read_text())["heads"]["e2e"]["summary"]
                points.append((clock[epoch], s))
        s = json.loads((out / f"{name}_final.json").read_text())["heads"]["e2e"]["summary"]
        points.append((clock[epochs], s))
        for ax, extract in zip(axes, [lambda v: v["ap50"],
                                     lambda v: v["robot_numbers"]["class_acc"],
                                     lambda v: v["corner_p95_orig_px"]]):
            ax.plot([t for t, _ in points], [extract(v) for _, v in points], marker="o", label=name)
    for ax, title in zip(axes, ["Armor AP50 (higher is better)",
                                "Robot class accuracy (higher is better)",
                                "Corner error P95 / original px (lower is better)"]):
        ax.set_title(title)
        ax.set_xlabel("Training time / min")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Style A to B transfer: fixed B validation set, seed 0, e2e head")
    fig.savefig(out / "transfer_curves.png", dpi=160)
    fig.savefig(out / "transfer_curves.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("baked/style_b_v1"))
    parser.add_argument("--init", type=Path, default=Path("runs/train/y26n_simm7_e100/weights/last.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-gb", type=int, default=7)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    from nt.train import load_hyp
    from nt.spec import LabelSpec
    spec = LabelSpec.load()
    hyp = Path("configs/train/pose_transfer_ab.yaml")
    config = load_hyp(hyp)
    source_model = Path("configs/models/yolo26n-pose-armor.yaml")
    out = Path("runs/experiments") / f"style_b_v1_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = args.data / "bake_manifest.json"
    fingerprint = {"bake_sha256": sha(manifest), "initial_weights_sha256": sha(args.init),
                   "model_yaml_sha256": sha(source_model), "hyp": config, "seed": args.seed,
                   "train_code_sha256": sha(ROOT / "nt/train.py"),
                   "eval_code_sha256": sha(ROOT / "nt/eval.py")}
    frozen = out / "inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != fingerprint:
        raise SystemExit("实验输入已变化，拒绝混用旧结果；请使用独立的实验目录。")
    write_json(frozen, fingerprint)
    # 两组微调各自读取相同 A 权重；不把短训练结果作为长训练的起点。
    trials = [(f"b_scratch_e100_s{args.seed}", source_model, 100),
              (f"b_from_a_e30_s{args.seed}", args.init, 30),
              (f"b_from_a_e100_s{args.seed}", args.init, 100)]
    status = {"status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
    if (out / "status.json").exists():
        previous_status = json.loads((out / "status.json").read_text())
        status["steps"] = previous_status.get("steps", {})
        status["started_at"] = previous_status.get("started_at", status["started_at"])

    def execute(key: str, command: list[str], memory: int, run: Path | None = None) -> None:
        cmd = ["scripts/run_capped.sh", str(memory), "--", *command]
        if args.dry_run:
            print(json.dumps({"step": key, "command": cmd}, ensure_ascii=False), flush=True)
            return
        log = out / f"{key}.log"
        status.update(status="running", current_step=key)
        step = {"status": "running", "command": cmd, "log": str(log)}
        status["steps"][key] = step
        write_json(out / "status.json", status)
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
                        step["progress"] = progress(run) if run else {}
                        write_json(out / "status.json", status)
                        print(json.dumps({"step": key, "seconds": step["seconds"], **step["progress"]},
                                         ensure_ascii=False), flush=True)
            except BaseException:
                proc.terminate()
                raise
        step.update(status="completed" if code == 0 else "failed", exit_code=code,
                    seconds=round(time.monotonic() - start, 1))
        write_json(out / "status.json", status)
        if code:
            status["status"] = "failed"
            write_json(out / "status.json", status)
            raise SystemExit(f"{key} 失败，退出码 {code}；请检查 {log}")

    def evaluate(key: str, weights: Path) -> Path:
        target = out / f"{key}.json"
        checked = out / f"{key}.inputs.json"
        stamp = {"weights_sha256": sha(weights), "bake_sha256": fingerprint["bake_sha256"],
                 "head": "e2e", "conf": 0.2}
        if target.exists() and checked.exists() and json.loads(checked.read_text()) == stamp:
            return target
        execute(key, [sys.executable, "-m", "nt.eval", "--weights", str(weights),
                      "--data", str(args.data), "--head", "e2e", "--conf", "0.2",
                      "--device", "0", "--out", str(target)], 5)
        if not args.dry_run:
            write_json(checked, stamp)
        return target

    if not args.report_only:
        evaluate("a_direct_on_b", args.init)
        for name, model, epochs in trials:
            run = Path("runs/train") / name
            metadata = run / "nt_meta.json"
            completed = False
            if metadata.exists():
                meta = json.loads(metadata.read_text())
                effective = meta["hyp"]["effective"]
                completed = (meta["status"] == "completed" and effective["epochs"] == epochs
                             and effective["seed"] == args.seed
                             and meta["model_yaml"]["sha256"] == sha(model)
                             and meta["data"]["bake_manifest_sha256"] == fingerprint["bake_sha256"]
                             and all(effective.get(k) == v for k, v in config.items() if k not in {"epochs", "seed"}))
            if not completed:
                if run.exists() and not args.dry_run:
                    raise SystemExit(f"{run} 已存在且未通过一致性检查，保留原数据并停止。")
                milestones = [str(e) for e in (10, 30, 60) if e <= epochs]
                execute(name, [sys.executable, "-m", "nt.train", "--model", str(model),
                               "--data", str(args.data), "--name", name, "--hyp", str(hyp),
                               "--epochs", str(epochs), "--seed", str(args.seed), "--device", "0",
                               "--milestones", *milestones], args.memory_gb, run)
            if args.dry_run:
                continue
            meta = json.loads(metadata.read_text())
            if meta["status"] != "completed" or int(meta["last_metrics"]["epoch"]) != epochs:
                raise SystemExit(f"{name} 没有完成约定轮数，不能作为完整实验结果。")
            evaluate(f"{name}_final", run / "weights/last.pt")
            for epoch in (10, 30, 60):
                if epoch < epochs:
                    evaluate(f"{name}_e{epoch}", run / "weights" / f"milestone_{epoch:03d}.pt")
    if args.dry_run:
        return

    rows, curves = [], []
    # 只统计真正更换了字形的类别，避免未变化图案掩盖数字迁移的表现。
    digit_ids = [str(c) for c in range(spec.n_classes) if spec.from_class_id(c)[1] in (1, 3, 4)]
    for key, label, epochs in [("a_direct_on_b", "A 直接评估 B", 0),
                               *[(f"{name}_final", name, epochs) for name, _, epochs in trials]]:
        result = json.loads((out / f"{key}.json").read_text())["heads"]["e2e"]
        summary = result["summary"]
        digit_counts = {k: sum(result["by_class"][c][k] for c in digit_ids)
                        for k in ("n", "hits", "correct")}
        changed_digits = {
            **digit_counts,
            "ap50": round(sum(result["ap50_by_class"][c] for c in digit_ids) / len(digit_ids), 4),
            "class_acc": round(digit_counts["correct"] / digit_counts["hits"], 4) if digit_counts["hits"] else None,
            "recall": round(digit_counts["hits"] / digit_counts["n"], 4) if digit_counts["n"] else None,
            "correct_per_gt": round(digit_counts["correct"] / digit_counts["n"], 4) if digit_counts["n"] else None,
        }
        minutes = 0.0
        if epochs:
            run = Path("runs/train") / label
            minutes = json.loads((run / "nt_meta.json").read_text())["wall_time_s"] / 60
            with (run / "results.csv").open() as stream:
                for row in csv.DictReader(stream):
                    curves.append({"run": label, **row})
        rows.append({"run": label, "epochs": epochs, "train_minutes": round(minutes, 2),
                     "ap50": summary["ap50"], "robot_class_accuracy": summary["robot_numbers"]["class_acc"],
                     "recall": summary["recall"], "precision": summary["precision"],
                     "off_recall": (summary["off_class"] or {}).get("recall"),
                     "corner_p50_orig_px": result["corner_error"]["orig_px"]["p50"],
                     "corner_p95_orig_px": summary["corner_p95_orig_px"],
                     "false_positives": result["at_conf"]["false_positives"],
                     "changed_digits": changed_digits, "by_width_px": result["by_width_px"]})
    write_json(out / "comparison.json", {"inputs": fingerprint, "results": rows})
    with (out / "learning_curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    lines = ["# 样式 A→B 迁移实验", "",
             "三组共用独立 B 验证集，固定使用最终检查点、端到端输出和置信度 0.2。",
             "训练分钟只计算本轮 B 适配阶段，不包含已完成的 A 预训练、数据生成和独立评估。", "",
             "| 路线 | 轮数 | 训练分钟 | AP50 | 机器人类别准确率 | 角点 P95/原图 px | 召回率 | 熄灭召回率 | 误检数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['run']} | {row['epochs']} | {row['train_minutes']} | {row['ap50']} | "
                     f"{row['robot_class_accuracy']} | {row['corner_p95_orig_px']} | {row['recall']} | "
                     f"{row['off_recall']} | {row['false_positives']} |")
    lines += ["", "以下单独统计更换字形的红蓝 1/3/4 号；检出且分类正确占比的分母包含漏检目标。", "",
              "| 路线 | 数字 AP50 | 已检出数字类别准确率 | 检出且分类正确 / 全部数字目标 |",
              "|---|---:|---:|---:|"]
    for row in rows:
        digits = row["changed_digits"]
        lines.append(f"| {row['run']} | {digits['ap50']} | {digits['class_acc']} | {digits['correct_per_gt']} |")
    lines += ["", "小目标分组按网络输入上的装甲框宽度统计，包含全部类别；准确率的分母为匹配到的装甲。", "",
              "| 路线 | 宽 <8 px 召回率 | 宽 <8 px 类别准确率 | 宽 8–16 px 召回率 | 宽 8–16 px 类别准确率 |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        tiny, small = row["by_width_px"].get("[0,8)", {}), row["by_width_px"].get("[8,16)", {})
        lines.append(f"| {row['run']} | {tiny.get('recall')} | {tiny.get('class_acc')} | "
                     f"{small.get('recall')} | {small.get('class_acc')} |")
    baseline, short, long = rows[1:]
    saved = 100 * (1 - short["train_minutes"] / baseline["train_minutes"])
    lines += ["",
              f"30 轮微调相对从头训练 100 轮：适配训练时间减少 {saved:.2f}%；"
              f"AP50 变化 {short['ap50'] - baseline['ap50']:+.4f}，"
              f"机器人类别准确率变化 {100 * (short['robot_class_accuracy'] - baseline['robot_class_accuracy']):+.2f} 个百分点，"
              f"角点 P95 变化 {short['corner_p95_orig_px'] - baseline['corner_p95_orig_px']:+.3f} 原图像素。",
              "",
              f"100 轮微调相对从头训练 100 轮：AP50 变化 {long['ap50'] - baseline['ap50']:+.4f}，"
              f"机器人类别准确率变化 {100 * (long['robot_class_accuracy'] - baseline['robot_class_accuracy']):+.2f} 个百分点，"
              f"角点 P95 变化 {long['corner_p95_orig_px'] - baseline['corner_p95_orig_px']:+.3f} 原图像素。",
              "", f"三组训练阶段共 {sum(row['train_minutes'] for row in rows):.2f} 分钟。"]
    lines += ["", "本报告为单个训练种子的仿真结果；不能据此承诺正式样式或真实相机的迁移收益。",
              "机器人类别准确率要求颜色与编号同时正确，范围为哨兵及 1/3/4 号，分母是已检测匹配的装甲；",
              "它不是包含漏检的端到端识别成功率。召回率和误检数按不区分类别的 IoU 匹配统计；",
              "角点误差统计已匹配装甲的可见角点，换算至原图像素。",
              "逐轮 Ultralytics 指标见 learning_curves.csv，独立端到端评估见各路线的 JSON 文件，两者 AP50 不混用。"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    plot_curves(out, trials)
    status.update(status="completed", current_step=None, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(out / "status.json", status)
    print(f"实验完成：{out / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
