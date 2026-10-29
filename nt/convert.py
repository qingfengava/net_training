"""Canonical dataset -> Ultralytics pose dataset (merged color x label classes).

    python -m nt.convert DATASET_DIR OUT_DIR [--copy] [--margin 0.1]

Writes images/ (symlinks unless --copy), labels/, and data.yaml with train and val both pointing at images/.
Real splits are built later by the bake step, by scene or session, never by frame.
Armors whose color has no training class are skipped and counted; nothing is dropped silently.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from nt.canonical import read_samples
from nt.geometry import bbox
from nt.spec import LabelSpec


def pose_line(cid: int, corners, vis, w: int, h: int, margin: float = 0.1) -> str:
    """One Ultralytics pose label line. The box is the corner bbox grown by `margin` and clipped to the image;
    keypoints are clipped to [0, 1], so a corner outside the image keeps its flag (vis 0) at the border."""
    x0, y0, x1, y1 = bbox(corners)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = max(0.0, x0 - mx), min(float(w), x1 + mx)
    y0, y1 = max(0.0, y0 - my), min(float(h), y1 + my)
    row = [
        str(cid),
        f"{(x0 + x1) / 2 / w:.6f}", f"{(y0 + y1) / 2 / h:.6f}",
        f"{(x1 - x0) / w:.6f}", f"{(y1 - y0) / h:.6f}",
    ]
    for (x, y), v in zip(corners, vis):
        row += [f"{min(max(x / w, 0.0), 1.0):.6f}", f"{min(max(y / h, 0.0), 1.0):.6f}", str(v)]
    return " ".join(row)


def canonical_to_yolo(src: Path, dst: Path, spec: LabelSpec, margin: float = 0.1, copy: bool = False) -> dict:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    skipped: Counter = Counter()
    n_samples = n_armors = 0
    for s in read_samples(src / "labels.jsonl"):
        ext = Path(s.image).suffix
        link = dst / "images" / f"{s.id}{ext}"
        if not link.exists():
            if copy:
                shutil.copy2(src / s.image, link)
            else:
                os.symlink((src / s.image).resolve(), link)
        lines = []
        for a in s.armors:
            if not spec.trains(a.color):
                skipped[spec.colors.get(a.color, str(a.color))] += 1
                continue
            row = pose_line(spec.class_id(a.color, a.label), a.corners, a.vis, s.w, s.h, margin)
            lines.append(row)
            n_armors += 1
        (dst / "labels" / f"{s.id}.txt").write_text("".join(ln + "\n" for ln in lines))
        n_samples += 1
    names = "\n".join(f"  {i}: {spec.class_name(i)}" for i in range(spec.n_classes))
    # flip_idx swaps LT<->RT and LB<->RB. Horizontal flips stay off for armor (digits do not mirror).
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: images\nval: images\nkpt_shape: [4, 3]\nflip_idx: [3, 2, 1, 0]\nnames:\n{names}\n"
    )
    return {"samples": n_samples, "armors": n_armors, "skipped_untrained_colors": dict(skipped)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--spec", type=Path, default=None)
    args = ap.parse_args()
    spec = LabelSpec.load(args.spec) if args.spec else LabelSpec.load()
    print(json.dumps(canonical_to_yolo(args.dataset, args.out, spec, args.margin, args.copy)))


if __name__ == "__main__":
    main()
