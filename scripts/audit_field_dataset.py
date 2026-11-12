#!/usr/bin/env python3
"""核对原赛场重生成数据：负样本无装甲、分区隔离、空标签、图像与元数据一致。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_guard(sample: dict) -> None:
    scene = sample["scene"]
    guard = scene["negative_guard"]
    assert guard["method"] == "conservative_armor_spheres_v1"
    assert not sample["armors"] and scene["background_only"]
    assert len(guard["armor_centers"]) == 24
    # 用相机坐标下的视锥平面独立复核所有装甲包围球；不依赖检测器或空标签的自我证明。
    x, y, z, w = guard["camera_rotation_xyzw"]
    rotation = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    camera_points = (np.asarray(guard["armor_centers"]) - scene["camera_pos"]) @ rotation
    radius = guard["radius_m"]
    assert radius >= 0.25
    ty = np.tan(np.radians(scene["fov_y_deg"]) * 0.5)
    tx = ty * sample["w"] / sample["h"]
    for px, py, pz in camera_points:
        depth = -pz
        outside = (
            depth + radius <= 1e-5
            or abs(px) - depth*tx > radius*np.sqrt(1 + tx*tx) - 1e-5
            or abs(py) - depth*ty > radius*np.sqrt(1 + ty*ty) - 1e-5
        )
        assert outside, f"装甲可能进入负样本视锥：{sample['id']}"


def preview(root: Path, samples: list[dict], dst: Path) -> None:
    # 等间隔取样，覆盖整个生成过程；保留原场地的颜色，便于人工查看。
    selected = [samples[i] for i in np.linspace(0, len(samples) - 1, min(24, len(samples)), dtype=int)]
    tiles = []
    for sample in selected:
        image = cv2.imread(str(root / sample["image"]))
        assert image is not None
        image = cv2.resize(image, (360, 270))
        cv2.putText(image, sample["id"].split("_")[-1], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(image)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    image = np.concatenate([np.concatenate(tiles[i:i+4], axis=1) for i in range(0, len(tiles), 4)], axis=0)
    assert cv2.imwrite(str(dst), image)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baked", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.baked / "bake_manifest.json").read_text())
    hashes = {"train": set(), "val": set()}
    source_stats = {}
    for source in manifest["sources"]:
        root = Path(source["path"])
        sample_rows = rows(root / "labels.jsonl")
        negative = "field_negative" in source["name"]
        for sample in sample_rows:
            assert sample["scene"]["floating_laser_modules_removed"]
            assert sample["scene"]["background_only"] == negative
            if negative:
                check_guard(sample)
            image = root / sample["image"]
            assert image.exists()
            hashes[source["role"]].add(hashlib.sha256(image.read_bytes()).hexdigest())
        source_stats[source["name"]] = {"role": source["role"], "samples": len(sample_rows),
            "negative_frames": sum(not s["armors"] for s in sample_rows),
            "armors": sum(len(s["armors"]) for s in sample_rows)}
        preview(root, sample_rows, args.out / f"{source['name']}_preview.jpg")
    assert not hashes["train"].intersection(hashes["val"]), "训练与验证出现同一张原图"
    baked_rows = rows(args.baked / "meta.jsonl")
    counts, negatives, generated_negatives = Counter(), Counter(), Counter()
    for item in baked_rows:
        split = item["split"]
        counts[split] += 1
        image = args.baked / item["image"]
        label = args.baked / "labels" / split / (image.stem + ".txt")
        lines = [line for line in label.read_text().splitlines() if line.strip()]
        assert len(lines) == len(item["armors"])
        assert image.exists()
        if not lines:
            negatives[split] += 1
        if "field_negative" in item["source"]:
            assert not lines and item["distractors"] == 0
            generated_negatives[split] += 1
    assert dict(counts) == manifest["images"]
    for split in ("train", "val"):
        assert len(list((args.baked / "images" / split).glob("*.png"))) == counts[split]
        assert len(list((args.baked / "labels" / split).glob("*.txt"))) == counts[split]
        assert generated_negatives[split] > 0
    report = {"sources": source_stats, "baked": str(args.baked), "images": dict(counts),
        "empty_labels": dict(negatives), "explicit_field_negatives": dict(generated_negatives),
        "negative_fraction": {s: negatives[s] / counts[s] for s in counts},
        "train_val_identical_source_images": 0, "negative_guard": "all passed",
        "distractors_enabled": manifest["config"]["distractors"]["prob"] != 0}
    assert not report["distractors_enabled"]
    (args.out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
