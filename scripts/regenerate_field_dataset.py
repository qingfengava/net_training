#!/usr/bin/env python3
"""重生成移除悬空组件后的 B 样式数据，并补入原赛场负样本；不会训练或删除旧数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("style_b_field_v1_train", "train", 2700),
    ("style_b_field_v1_val", "val", 600),
    ("field_negative_v1_train", "train", 300),
    ("field_negative_v1_val", "val", 60),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--copies", type=int, default=2)
    args = ap.parse_args()
    os.chdir(ROOT)
    report = ROOT / "runs/verification/field_refresh_20261106"
    report.mkdir(parents=True, exist_ok=True)
    baked = ROOT / "baked/style_b_field_v1"
    for name, _, _ in SOURCES:
        for path in (ROOT / f"runs/sim_{name}", ROOT / f"datasets/sim/{name}"):
            if path.exists():
                raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    if baked.exists() or args.workers < 1 or args.copies < 1:
        raise ValueError("检查输出目录、workers 和 copies")
    if shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError("可用磁盘少于 12 GiB，先检查数据占用")
    steps = []

    def run(name: str, command: list[str]) -> None:
        start = time.monotonic()
        log = report / f"{name}.log"
        print(f"开始 {name}，日志：{log}", flush=True)
        with log.open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        steps.append({"name": name, "seconds": time.monotonic() - start, "command": command})
        (report / "steps.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2) + "\n")
        print(f"完成 {name}，耗时 {steps[-1]['seconds']:.1f} 秒", flush=True)

    for name, role, count in SOURCES:
        run(f"render_{name}", ["scripts/run_sim_dataset.sh", f"configs/sim/{name}.toml", "1800"])
        raw = ROOT / f"runs/sim_{name}"
        done = json.loads((raw / "done.json").read_text())
        if done["samples_written"] != count or done["lost"]:
            raise RuntimeError(f"渲染未完整结束：{name} {done}")
        log = (report / f"render_{name}.log").read_text()
        if 'device_type: Cpu' in log or log.count('removed floating robot accessory:') != 4:
            raise RuntimeError(f"GPU 或悬空组件移除检查失败：{name}")
        if "negative" in name and done["armors_written"]:
            raise RuntimeError(f"负样本出现装甲：{name}")
        # 负样本训练分区也必须使用 train 角色；neg 角色在现有系统中只用于验证。
        run(f"ingest_{name}", ["scripts/run_capped.sh", "3", "--", sys.executable, "-m", "nt.ingest",
            "--src", str(raw), "--format", "canonical-jsonl", "--source", "sim", "--name", name,
            "--dst", f"datasets/sim/{name}", "--role", role, "--link"])
    command = ["scripts/run_capped.sh", "6", "--", sys.executable, "-m", "nt.bake"]
    for name, _, _ in SOURCES:
        command.extend(["--src", f"datasets/sim/{name}"])
    command.extend(["--out", str(baked), "--config", "configs/bake/field_clean.yaml", "--copies", str(args.copies),
        "--seed", "0", "--workers", str(args.workers)])
    run("bake", command)
    run("audit", ["scripts/run_capped.sh", "3", "--", sys.executable, "scripts/audit_field_dataset.py",
        "--baked", str(baked), "--out", str(report)])
    simulator = Path(os.environ.get("SIM_DIR", str(Path.home() / "bevy_robomaster_simulator")))
    provenance = {}
    for relative in ("src/setup.rs", "src/dataset/config.rs", "src/dataset/sampler.rs", "src/dataset/negative.rs"):
        provenance[str(simulator / relative)] = hashlib.sha256((simulator / relative).read_bytes()).hexdigest()
    (report / "simulator_sha256.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"全部完成，训练数据：{baked}，报告：{report / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
