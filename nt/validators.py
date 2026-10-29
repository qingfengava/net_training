"""Label validators and the dataset-level check.

Error issues make the ingest gate refuse the whole sample: dropping only the bad armor would leave a real
armor unlabeled, which trains the network to suppress it. Warnings are recorded and never block.

CLI:  python -m nt.validators DATASET_DIR [--no-images] [--spec PATH]   (exit code 1 when any error is found)
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from nt.canonical import Armor, Sample, SOURCES, read_samples
from nt.geometry import (
    aspect_ratio,
    bbox,
    bbox_iou,
    edge_lengths,
    is_convex,
    signed_area,
    top_edge_angle_deg,
)
from nt.spec import LabelSpec

LOSSY_EXTS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warn"
    code: str
    sample_id: str
    armor: int | None
    msg: str

    def to_dict(self) -> dict:
        return asdict(self)


def armor_issues(a: Armor, w: int, h: int, spec: LabelSpec) -> list[tuple[str, str, str]]:
    """(level, code, message) tuples for one armor."""
    out: list[tuple[str, str, str]] = []
    v = spec.validation

    if not spec.valid_pair(a.color, a.label):
        out.append(("error", "E_CLASS_ILLEGAL", f"color={a.color} label={a.label}"))
    elif not spec.trains(a.color):
        out.append(("warn", "W_COLOR_NOT_TRAINED", f"color {spec.colors[a.color]} has no training class"))
    if a.type is not None and a.type not in spec.types:
        out.append(("error", "E_CLASS_ILLEGAL", f"type={a.type!r}"))

    if len(a.corners) != 4 or any(len(p) != 2 for p in a.corners) or len(a.vis) != 4:
        out.append(("error", "E_CORNER_COUNT", "need 4 corners with 2 coordinates each and 4 vis flags"))
        return out
    if any(x not in (0, 1, 2) for x in a.vis):
        out.append(("error", "E_VIS_INVALID", f"vis={a.vis}"))
        return out
    flat = [c for p in a.corners for c in p]
    if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in flat):
        out.append(("error", "E_CORNER_NONFINITE", f"corners={a.corners}"))
        return out

    for name, val, lo, hi in (("occ", a.occ, 0.0, 1.0), ("dist_m", a.dist_m, 1e-3, 1e3)):
        if val is not None and not (math.isfinite(val) and lo <= val <= hi):
            out.append(("error", "E_FIELD_RANGE", f"{name}={val}"))

    m = v["corner_margin_px"]
    for i, (x, y) in enumerate(a.corners):
        if a.vis[i] > 0 and not (-m <= x <= w + m and -m <= y <= h + m):
            out.append(("error", "E_CORNER_OUTSIDE", f"corner {i} at ({x:.1f},{y:.1f}) is outside {w}x{h} but vis={a.vis[i]}"))
            break

    if all(f > 0 for f in a.vis):  # geometry of a quad with absent corners is meaningless
        pts = [tuple(p) for p in a.corners]
        if not is_convex(pts):
            out.append(("error", "E_QUAD_NONCONVEX", "corners do not form a convex quad"))
        else:
            ang = top_edge_angle_deg(pts)
            if signed_area(pts) >= 0:
                out.append(("error", "E_QUAD_ORDER", "winding is not LT, LB, RB, RT"))
            elif abs(ang) > v["max_top_edge_angle_deg"]:
                out.append(("error", "E_QUAD_ORDER", f"LT->RT edge is at {ang:.0f} degrees, corner start is shifted"))
            elif abs(ang) > v["warn_top_edge_angle_deg"]:
                out.append(("warn", "W_QUAD_ROLL", f"LT->RT edge at {ang:.0f} degrees"))
            lengths = edge_lengths(pts)
            if min(lengths) < v["min_edge_px"] or abs(signed_area(pts)) < v["min_area_px"]:
                out.append(("error", "E_QUAD_DEGENERATE", f"edges={[round(e, 1) for e in lengths]}"))
            else:
                lo, hi = v["edge_ratio_range"]
                ratio = aspect_ratio(pts)
                if not lo <= ratio <= hi:
                    out.append(("warn", "W_ASPECT", f"horizontal/vertical edge ratio {ratio:.2f}"))
                if max(lengths) < v["small_armor_warn_px"]:
                    out.append(("warn", "W_SMALL_ARMOR", f"longest edge {max(lengths):.1f}px"))
    return out


def validate_sample(s: Sample, spec: LabelSpec, image_hw: tuple[int, int] | None = None) -> list[Issue]:
    issues: list[Issue] = []

    def add(level: str, code: str, msg: str, armor: int | None = None) -> None:
        issues.append(Issue(level, code, s.id, armor, msg))

    if not s.id:
        add("error", "E_SAMPLE_FIELD", "empty id")
    if s.source not in SOURCES:
        add("error", "E_SAMPLE_FIELD", f"source={s.source!r}")
    if not (isinstance(s.w, int) and isinstance(s.h, int) and s.w > 0 and s.h > 0):
        add("error", "E_SIZE_INVALID", f"w={s.w} h={s.h}")
        return issues
    if image_hw is not None and tuple(image_hw) != (s.h, s.w):
        add("error", "E_SIZE_MISMATCH", f"label says {s.w}x{s.h}, image is {image_hw[1]}x{image_hw[0]}")
    if Path(s.image).suffix.lower() in LOSSY_EXTS:
        add("warn", "W_IMAGE_LOSSY", "JPEG blurs dot-matrix detail; prefer lossless sources")

    for i, a in enumerate(s.armors):
        for level, code, msg in armor_issues(a, s.w, s.h, spec):
            add(level, code, msg, i)

    boxes = {}
    for i, a in enumerate(s.armors):
        if len(a.corners) == 4 and all(len(p) == 2 for p in a.corners):
            try:
                boxes[i] = bbox(a.corners)
            except TypeError:
                continue
    ids = sorted(boxes)
    for ai, i in enumerate(ids):
        for j in ids[ai + 1:]:
            if bbox_iou(boxes[i], boxes[j]) >= spec.v("duplicate_iou"):
                same = (s.armors[i].color, s.armors[i].label) == (s.armors[j].color, s.armors[j].label)
                if same:
                    add("error", "E_ARMOR_DUP", f"armors {i} and {j} are the same label on the same spot", j)
                else:
                    add("warn", "W_LABEL_CONFLICT", f"armors {i} and {j} overlap with different classes", j)
    return issues


def dhash64(img: np.ndarray) -> np.uint64:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.packbits(bits).view(np.uint64)[0]


class NearDuplicateIndex:
    """Finds earlier frames whose 64-bit dHash is within a Hamming distance."""

    def __init__(self, max_hamming: int) -> None:
        self.max_hamming = max_hamming
        self._hashes = np.empty(1024, dtype=np.uint64)
        self._ids: list[str] = []

    def query(self, h: np.uint64) -> str | None:
        n = len(self._ids)
        if n == 0:
            return None
        dist = np.bitwise_count(self._hashes[:n] ^ h)
        hit = np.flatnonzero(dist <= self.max_hamming)
        return self._ids[int(hit[0])] if hit.size else None

    def add(self, sample_id: str, h: np.uint64) -> None:
        n = len(self._ids)
        if n == len(self._hashes):
            self._hashes = np.concatenate([self._hashes, np.empty_like(self._hashes)])
        self._hashes[n] = h
        self._ids.append(sample_id)


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    n_samples: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def by_code(self) -> Counter:
        return Counter(i.code for i in self.issues)


def validate_dataset(root: Path, spec: LabelSpec, decode_images: bool = True) -> Report:
    """Re-validate an existing canonical dataset (labels.jsonl plus images/)."""
    report = Report()
    index = NearDuplicateIndex(spec.v("near_duplicate_hamming"))
    for s in read_samples(root / "labels.jsonl"):
        report.n_samples += 1
        hw = None
        img = None
        path = root / s.image
        if not path.exists():
            report.issues.append(Issue("error", "E_IMAGE_MISSING", s.id, None, str(path)))
        elif decode_images:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                report.issues.append(Issue("error", "E_IMAGE_UNREADABLE", s.id, None, str(path)))
            else:
                hw = img.shape[:2]
        issues = validate_sample(s, spec, hw)
        report.issues.extend(issues)
        if img is not None and not any(i.level == "error" for i in issues):
            h = dhash64(img)
            twin = index.query(h)
            if twin:
                report.issues.append(Issue("warn", "W_NEAR_DUPLICATE_FRAME", s.id, None, f"near duplicate of {twin}"))
            else:
                index.add(s.id, h)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--no-images", action="store_true", help="skip decoding images (no size or duplicate checks)")
    ap.add_argument("--spec", type=Path, default=None)
    args = ap.parse_args()
    spec = LabelSpec.load(args.spec) if args.spec else LabelSpec.load()
    report = validate_dataset(args.dataset, spec, decode_images=not args.no_images)
    print(f"{report.n_samples} samples, {len(report.errors)} errors, {len(report.issues) - len(report.errors)} warnings")
    for code, n in sorted(report.by_code().items()):
        print(f"  {code}: {n}")
    for i in report.errors[:10]:
        print(f"  {i.sample_id} armor={i.armor} {i.code}: {i.msg}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
