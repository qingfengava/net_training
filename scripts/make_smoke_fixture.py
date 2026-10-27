"""Build the tiny M0 smoke dataset from the sample frames shipped with Daedalus.

Source label line: color typ label x1 y1 x2 y2 x3 y3 x4 y4 (normalized, written by the old Daedalus exporter).
Output label line (Ultralytics pose): cls cx cy w h x1 y1 v1 ... x4 y4 v4, with cls = color * 8 + label.

Usage: python scripts/make_smoke_fixture.py SIM_DATASET_DIR OUT_DIR
The three frames are only good for exercising the pipeline, not for learning anything.
"""
import shutil
import sys
from pathlib import Path

MARGIN = 0.1  # relative padding around the corner box


def convert_line(line: str):
    f = line.split()
    color, _typ, label = int(f[0]), int(f[1]), int(f[2])
    if color not in (0, 1):
        return None  # gray/purple are not part of the 16-class smoke setup
    pts = [(float(f[3 + 2 * i]), float(f[4 + 2 * i])) for i in range(4)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = (x1 - x0) * (1 + 2 * MARGIN), (y1 - y0) * (1 + 2 * MARGIN)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = [color * 8 + label, cx, cy, w, h]
    for x, y in pts:
        out += [x, y, 2]
    return " ".join(f"{v:d}" if isinstance(v, int) else f"{v:.6f}" for v in out)


def main(src: Path, dst: Path) -> None:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n_img = n_obj = 0
    for img in sorted((src / "images").glob("*.jpg")):
        lab = src / "label" / f"{img.stem}.txt"
        rows = [convert_line(ln) for ln in lab.read_text().splitlines() if ln.strip()]
        rows = [r for r in rows if r]
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(rows) + "\n")
        n_img += 1
        n_obj += len(rows)
    names = "\n".join(f"  {i}: {('blue', 'red')[i // 8]}_{i % 8}" for i in range(16))
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: images\nval: images\nkpt_shape: [4, 3]\nflip_idx: [0, 1, 2, 3]\nnames:\n{names}\n"
    )
    print(f"fixture: {n_img} images, {n_obj} objects -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
