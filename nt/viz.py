"""Contact sheet of zoomed armor crops with numbered corners, for eyeballing labels after every ingest.
With --baked the argument is a baked dataset and the sheet shows whole network-input images with their labels,
captioned with the ROI mode, the bake scale and the sampled effects.

    python -m nt.viz DATASET_DIR [--n 24] [--seed 0] [--tile 256] [--cols 6] [--out runs/viz/NAME.png]

Each tile is a square crop around one armor, enlarged with nearest-neighbor so individual pixels stay visible.
Corners are drawn 0 LT (green), 1 LB (yellow), 2 RB (blue), 3 RT (magenta); the quad edge runs 0 -> 1 -> 2 -> 3.
Hollow markers are corners with vis=1, crosses are vis=0. Check that 0 sits on the top-left of the screen.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from nt.canonical import Armor, read_samples
from nt.geometry import bbox
from nt.spec import LabelSpec

CORNER_BGR = [(0, 220, 0), (0, 220, 255), (255, 140, 0), (255, 0, 255)]
SHIFT = 4  # sub-pixel precision for drawing


def _p(x: float, y: float) -> tuple[int, int]:
    return int(round(x * (1 << SHIFT))), int(round(y * (1 << SHIFT)))


def render_tile(img: np.ndarray, a: Armor, caption: str, tile: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox(a.corners)
    side = max(x1 - x0, y1 - y0) * 1.8 + 24
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side_i = max(int(round(side)), 8)
    pad = np.zeros((side_i, side_i, 3), np.uint8)
    sx0, sy0 = max(left, 0), max(top, 0)
    sx1, sy1 = min(left + side_i, w), min(top + side_i, h)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = img[sy0:sy1, sx0:sx1]
    scale = tile / side_i
    out = cv2.resize(pad, (tile, tile), interpolation=cv2.INTER_NEAREST)
    pts = [((x - left) * scale, (y - top) * scale) for x, y in a.corners]
    for i in range(4):
        cv2.line(out, _p(*pts[i]), _p(*pts[(i + 1) % 4]), (255, 255, 255), 1, cv2.LINE_AA, SHIFT)
    for i, (x, y) in enumerate(pts):
        color = CORNER_BGR[i]
        v = a.vis[i]
        if v == 0:
            cv2.drawMarker(out, (int(x), int(y)), color, cv2.MARKER_CROSS, 10, 1)
        else:
            cv2.circle(out, _p(x, y), 4 << SHIFT, color, -1 if v == 2 else 1, cv2.LINE_AA, SHIFT)
        cv2.putText(out, str(i), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (tile, 18), (0, 0, 0), -1)
    cv2.putText(out, caption[:42], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_sheet(root: Path, spec: LabelSpec, n: int, seed: int, tile: int, cols: int) -> np.ndarray | None:
    items = []
    for s in read_samples(root / "labels.jsonl"):
        for k, a in enumerate(s.armors):
            if len(a.corners) == 4:
                items.append((s, k))
    random.Random(seed).shuffle(items)
    tiles = []
    for s, k in items[:n]:
        img = cv2.imread(str(root / s.image), cv2.IMREAD_COLOR)
        if img is None:
            continue
        a = s.armors[k]
        name = spec.pair_name(a.color, a.label)
        tiles.append(render_tile(img, a, f"{s.id[-8:]}#{k} {name} vis={''.join(map(str, a.vis))}", tile))
    if not tiles:
        return None
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def make_baked_sheet(root: Path, n: int, seed: int, cols: int, tile_w: int, only_distractors: bool = False) -> np.ndarray | None:
    metas = [json.loads(ln) for ln in (root / "meta.jsonl").read_text().splitlines() if ln]
    if only_distractors:
        metas = [m for m in metas if m["distractors"]]
    random.Random(seed).shuffle(metas)
    tiles = []
    for m in metas[:n]:
        img = cv2.imread(str(root / m["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        k = tile_w / img.shape[1]
        img = cv2.resize(img, (tile_w, round(img.shape[0] * k)), interpolation=cv2.INTER_LINEAR)
        for a in m["armors"]:
            pts = [(x * k, y * k) for x, y in a["corners_net"]]
            for i in range(4):
                cv2.line(img, _p(*pts[i]), _p(*pts[(i + 1) % 4]), (255, 255, 255), 1, cv2.LINE_AA, SHIFT)
            for i, (x, y) in enumerate(pts):
                if a["vis"][i]:
                    cv2.circle(img, _p(x, y), 3 << SHIFT, CORNER_BGR[i], -1, cv2.LINE_AA, SHIFT)
        fx = m["effects"]
        cap = f"{m['roi_mode']} x{m['scale']:.2f} gain{fx['gain']:.2f} n{len(m['armors'])} d{m['distractors']}"
        cv2.rectangle(img, (0, 0), (tile_w, 18), (0, 0, 0), -1)
        cv2.putText(img, cap, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--baked", action="store_true", help="DATASET is a baked dataset")
    ap.add_argument("--distractors-only", action="store_true", help="with --baked: only images that got distractors")
    args = ap.parse_args()
    spec = LabelSpec.load(args.spec) if args.spec else LabelSpec.load()
    if args.baked:
        sheet = make_baked_sheet(args.dataset, args.n, args.seed, args.cols, args.tile, args.distractors_only)
    else:
        sheet = make_sheet(args.dataset, spec, args.n, args.seed, args.tile, args.cols)
    if sheet is None:
        raise SystemExit("no armors to draw")
    out = args.out or Path("runs/viz") / f"{args.dataset.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
