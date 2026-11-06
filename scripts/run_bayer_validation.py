"""拜耳取样的验证训练：训练时加拜耳，能不能追回部署时的损失，会不会拉低标准验证集。

    python scripts/run_bayer_validation.py [--seed 0] [--epochs 30] [--dry-run] [--report-only]

只再训一组 abl_bayer_s0（基线 abl_base_s0 是 2026-11-06 增强消融里已经训好的那组，设置完全相同：同一份样式 B 数据、
同一烘焙种子、同一 A 初始权重、pose_transfer_ab.yaml、30 轮、种子 0），训练集烘焙时让每一张仿真图都经过 RGGB 取样和
双线性去马赛克（configs/bake/bayer.yaml，概率 1.0）。两个模型都在两份验证集上评估：
  标准验证集：baked/style_b_v3_base 的验证分区（没有拜耳）。看有没有伤害。
  拜耳验证集：同一批验证图配对地过一遍拜耳（configs/bake/bayer_val.yaml）。看能追回多少。

判定标准在看结果之前定好（下面的常量），报告按它们机械地给出结论。训练集和基线的训练集逐张配对核对：ROI、窗口模式、
其余效果参数和标签必须完全相同，只允许多出拜耳这一步，否则成绩差异不能归因于拜耳。串行、可续跑，输入变化时拒绝混用旧结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_bake_ablation as ab  # noqa: E402

BASE_RUN = "abl_base_s0"
BASE_BAKE = Path("baked/style_b_v3_base")
NAME = "abl_bayer_s0"
TRAIN_BAKE, VAL_BAKE = Path("baked/style_b_v4_bayer"), Path("baked/style_b_v4_bayer_val")
# 预先定好的判定标准。追回比例 = 拜耳训练模型在拜耳验证集上比基线多拿的 AP50 / 基线在拜耳验证集上损失的 AP50。
RECOVER_CLEAR, RECOVER_PARTIAL = 0.5, 0.2
MIN_BASELINE_LOSS = 0.005   # 基线在拜耳验证集上的 AP50 损失小于它，就没有可恢复的东西，不下结论
HARM_AP50, HARM_ROBOT = 0.01, 0.01
# 单个种子的噪声参照（11-06 增强消融里修正核基线对旧模型的差）：AP50 0.003，机器人类别准确率 0.4 点，熄灭召回 1～3 点
NEW_KEYS = ("bayer", "flicker", "moire")


def rows_of(root: Path, split: str) -> dict:
    out = {}
    for line in (root / "meta.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"] == split:
            out[row["image"]] = row
    return out


def check_paired(reference: Path, other: Path, split: str, expect_new: str | None) -> dict:
    """other 与 reference 的同一分区逐张配对：ROI、窗口模式、其余效果参数、标签完全相同。"""
    a, b = rows_of(reference, split), rows_of(other, split)
    if a.keys() != b.keys():
        raise SystemExit(f"{other} 的 {split} 分区图像与基线不同，不能配对")
    with_new = 0
    for name in a:
        fx_a = {k: v for k, v in a[name]["effects"].items() if k not in NEW_KEYS}
        fx_b = {k: v for k, v in b[name]["effects"].items() if k not in NEW_KEYS}
        if (a[name]["roi"], a[name]["roi_mode"], fx_a) != (b[name]["roi"], b[name]["roi_mode"], fx_b):
            raise SystemExit(f"{other}: {name} 的 ROI 或曝光/噪声参数与基线不同，配对失败，停止")
        if [x["corners_net"] for x in a[name]["armors"]] != [x["corners_net"] for x in b[name]["armors"]]:
            raise SystemExit(f"{other}: {name} 的标签与基线不同，停止")
        extra = set(b[name]["effects"]) & set(NEW_KEYS)
        if extra - ({expect_new} if expect_new else set()):
            raise SystemExit(f"{other}: {name} 出现了不该有的效果 {sorted(extra)}，停止")
        with_new += expect_new in b[name]["effects"] if expect_new else 0
    return {"images": len(a), "with_" + (expect_new or "nothing"): with_new}


def judge(base_std: dict, base_bayer: dict, new_std: dict, new_bayer: dict) -> dict:
    loss = base_std["ap50"] - base_bayer["ap50"]
    gain = new_bayer["ap50"] - base_bayer["ap50"]
    out = {"baseline_loss_on_bayer_val": round(loss, 4), "gain_on_bayer_val": round(gain, 4),
           "robot_gain_on_bayer_val_pt": round(100 * (new_bayer["robot_acc"] - base_bayer["robot_acc"]), 2),
           "ap50_change_on_standard_val": round(new_std["ap50"] - base_std["ap50"], 4),
           "robot_change_on_standard_val_pt": round(100 * (new_std["robot_acc"] - base_std["robot_acc"]), 2)}
    if loss < MIN_BASELINE_LOSS:
        out["recovered"] = None
        out["benefit"] = f"基线在拜耳验证集上只损失 {loss:.4f} AP50（< {MIN_BASELINE_LOSS}），没有可恢复的损失，无法判定"
    else:
        out["recovered"] = round(gain / loss, 3)
        out["benefit"] = ("明显追回" if out["recovered"] >= RECOVER_CLEAR else
                          "部分追回，需要第二个种子确认" if out["recovered"] >= RECOVER_PARTIAL else "没有可靠的收益")
    harmed = (out["ap50_change_on_standard_val"] < -HARM_AP50 or out["robot_change_on_standard_val_pt"] < -100 * HARM_ROBOT)
    out["harm"] = "标准验证集上超出容差" if harmed else "标准验证集上在容差内"
    return out


def report(out: Path, fingerprint: dict, pairing: dict) -> None:
    res = {}
    for who, run in (("base", BASE_RUN), ("bayer", NAME)):
        for key, label in (("std", "标准验证集"), ("bayer", "拜耳验证集")):
            res[(who, key)] = ab.metrics(json.loads((out / f"{run}_on_{key}.json").read_text())["heads"]["e2e"])
    meta = json.loads((Path("runs/train") / NAME / "nt_meta.json").read_text())
    base_meta = json.loads((Path("runs/train") / BASE_RUN / "nt_meta.json").read_text())
    verdict = judge(res[("base", "std")], res[("base", "bayer")], res[("bayer", "std")], res[("bayer", "bayer")])
    lines = ["# 拜耳取样的验证训练（样式 B，A→B 微调 30 轮，种子 0）", "",
             "两组唯一的区别是训练集烘焙时是否让每张仿真图都经过 RGGB 取样和双线性去马赛克。基线是 11-06 增强消融里的 abl_base_s0。",
             "端到端检测头，置信度 0.2。机器人类别准确率要求颜色和编号同时正确，分母是匹配到的装甲；",
             "“<8 px 检出且分类正确”的分母是该宽度档的全部目标，含漏检。", "",
             f"训练集配对核对：{pairing['train']}；拜耳验证集配对核对：{pairing['val']}。", ""]
    for key, title in (("std", "标准验证集（没有拜耳）：看有没有伤害"), ("bayer", "拜耳验证集（同一批验证图过一遍拜耳）：看追回多少")):
        lines += ab.table(title, [("基线 abl_base_s0（训练时没有拜耳）", res[("base", key)], base_meta["wall_time_s"] / 60),
                                 ("abl_bayer_s0（训练时每张都过拜耳）", res[("bayer", key)], meta["wall_time_s"] / 60)])
    lines += ["## 判定（标准在看结果前就写进了脚本）", "",
              f"- 基线在拜耳验证集上损失 AP50 {verdict['baseline_loss_on_bayer_val']:+.4f}；拜耳训练模型在拜耳验证集上比基线多 "
              f"{verdict['gain_on_bayer_val']:+.4f}，机器人类别准确率 {verdict['robot_gain_on_bayer_val_pt']:+.2f} 个百分点。",
              f"- 追回比例：{verdict['recovered'] if verdict['recovered'] is not None else '-'}。**{verdict['benefit']}**"
              f"（≥{RECOVER_CLEAR} 明显，{RECOVER_PARTIAL}～{RECOVER_CLEAR} 部分，更低没有可靠收益）。",
              f"- 标准验证集上 AP50 {verdict['ap50_change_on_standard_val']:+.4f}，机器人类别准确率 "
              f"{verdict['robot_change_on_standard_val_pt']:+.2f} 个百分点：**{verdict['harm']}**"
              f"（容差 AP50 -{HARM_AP50}、机器人类别准确率 -{100 * HARM_ROBOT:.0f} 个百分点）。", "",
              "## 怎么读", "",
              "- 单个训练种子。重训噪声参照：AP50 约 0.003，机器人类别准确率约 0.4 个百分点，熄灭召回 1～3 个百分点"
              "（熄灭类只有约 144 个样本）。差在这个范围内不要当作效果。",
              "- 拜耳验证集与训练增强来自同一套模拟，拜耳训练的模型在这里占优是预期的；它回答的是“训练时加拜耳能不能追回"
              "部署时的损失”，不是“真实相机上会好多少”。真实镜头的光学低通可能让混叠比这里弱，也可能更强。",
              "- 这个实验只验证拜耳。频闪的参数还没有实测，没有进入训练。"]
    ab.write_json(out / "comparison.json", {"inputs": fingerprint, "pairing": pairing, "verdict": verdict,
                                            "results": {f"{w}_{k}": v for (w, k), v in res.items()}})
    (out / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init", type=Path, default=Path("runs/train/y26n_simm7_e100/weights/last.pt"))
    ap.add_argument("--bake-workers", type=int, default=3)
    ap.add_argument("--bake-memory-gb", type=int, default=6)
    ap.add_argument("--train-memory-gb", type=int, default=7)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印要执行的命令")
    args = ap.parse_args()
    os.chdir(ROOT)
    from nt.train import load_hyp
    hyp = Path("configs/train/pose_transfer_ab.yaml")
    config = load_hyp(hyp)
    out = Path("runs/experiments/bayer_validation_s0")
    configs = ["default.yaml", "bayer.yaml", "bayer_val.yaml"]
    fingerprint = {"initial_weights_sha256": ab.sha(args.init), "hyp": config, "seed": args.seed, "epochs": args.epochs,
                   "bake_seed": ab.BAKE_SEED, "sources": {s: ab.sha(Path(s) / "manifest.json") for s in ab.SOURCES},
                   "bake_configs": {c: ab.sha(ROOT / "configs/bake" / c) for c in configs},
                   "baseline": {"run": BASE_RUN, "weights_sha256": ab.sha(Path("runs/train") / BASE_RUN / "weights/last.pt"),
                                "bake_manifest_sha256": ab.sha(BASE_BAKE / "bake_manifest.json")},
                   "code": {f: ab.sha(ROOT / f) for f in ("nt/train.py", "nt/eval.py", "nt/bake.py", "nt/bake_augment.py")}}
    frozen = out / "inputs.json"
    if frozen.exists() and json.loads(frozen.read_text()) != fingerprint and not args.report_only:
        raise SystemExit("实验输入已变化，拒绝混用旧结果；请换一个实验目录。")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        if not frozen.exists():
            ab.write_json(frozen, fingerprint)
    r = ab.Runner(out, args.dry_run)
    pairing = {}
    if not args.report_only:
        ab.bake_one(r, TRAIN_BAKE.name, "bayer.yaml", ab.SOURCES, args.bake_workers, args.bake_memory_gb)
        ab.bake_one(r, VAL_BAKE.name, "bayer_val.yaml", [ab.SOURCES[1]], args.bake_workers, args.bake_memory_gb)
        if not args.dry_run:
            m, mb = (json.loads((p / "bake_manifest.json").read_text()) for p in (TRAIN_BAKE, BASE_BAKE))
            if m["images"] != mb["images"]:
                raise SystemExit(f"图像数 {m['images']} 与基线 {mb['images']} 不同，停止")
            if ab.val_digest(TRAIN_BAKE) != ab.val_digest(BASE_BAKE):
                raise SystemExit("拜耳训练集烘焙里的验证分区与基线不同；成绩不能直接比较，停止")
            pairing = {"train": check_paired(BASE_BAKE, TRAIN_BAKE, "train", "bayer"),
                       "val": check_paired(BASE_BAKE, VAL_BAKE, "val", "bayer")}
            ab.write_json(out / "pairing.json", pairing)
            print("配对核对通过：", json.dumps(pairing, ensure_ascii=False), flush=True)
        run = ab.train_one(r, NAME, args.init, hyp, config, args.epochs, args.seed, TRAIN_BAKE, args.train_memory_gb)
        for who, weights in ((BASE_RUN, Path("runs/train") / BASE_RUN / "weights/last.pt"), (NAME, run / "weights/last.pt")):
            ab.evaluate(r, f"{who}_on_std", weights, BASE_BAKE)
            ab.evaluate(r, f"{who}_on_bayer", weights, VAL_BAKE)
    if args.dry_run:
        return
    pairing = pairing or json.loads((out / "pairing.json").read_text())
    report(out, fingerprint, pairing)
    r.status.update(status="completed", current_step=None, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    r.save()
    print(f"实验完成：{out / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
