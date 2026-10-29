"""Bake canonical datasets into an Ultralytics pose dataset at the network input size.

    python -m nt.bake --src datasets/sim/NAME[:REPEAT] [--src ...] --out baked/NAME2 \\
        [--config configs/bake/default.yaml] [--copies 1] [--seed 0] [--workers 8]

For each sample and copy: choose the whole frame or a tracker-style ROI (4:3, around one armor), draw some
armor-like distractors, apply camera effects at native resolution, resize with bilinear interpolation (what
sleepy does at run time), and write the image plus its label. Labels are mapped with the exact same crop and
scale, in the "edge" pixel convention, so a corner stays on the same physical point. Armors the ROI cuts
ambiguously make the ROI be drawn again.

Lens distortion is not applied yet: the simulator renders a pinhole camera and the real intrinsics are unknown.

Output (never overwrites):
    data.yaml  bake_manifest.json  meta.jsonl  images/{train,val}/*.png  labels/{train,val}/*.txt
meta.jsonl has one line per image with the ROI, scale, sampled effects and per-armor metadata, which
nt.eval uses to map predictions back to original pixels and to bin errors by distance, angle and occlusion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from nt.canonical import Sample, read_manifest, read_samples, sample_to_dict, sample_from_dict
from nt.convert import pose_line
from nt.geometry import bbox
from nt.spec import LabelSpec

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bake" / "default.yaml"
BAKE_VERSION = "2"  # 2: per-dataset role decides the split (nt/canonical.py ROLES)

BGR = {"red": (40, 40, 255), "blue": (255, 90, 20), "white": (255, 255, 255), "green": (60, 255, 60), "amber": (0, 190, 255)}


# ---- config -------------------------------------------------------------------------------------------------

def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise KeyError(f"unknown bake config key {path}{k}")
        out[k] = _merge(base[k], v, f"{path}{k}.") if isinstance(base[k], dict) and isinstance(v, dict) else v
    return out


def load_config(path: Path | str | None = None, override: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if override:
        cfg = _merge(cfg, override)
    w, h = cfg["size"]
    if w * 3 != h * 4:
        raise ValueError(f"size {w}x{h} is not 4:3")
    return cfg


# ---- ROI ----------------------------------------------------------------------------------------------------

def _inside_fraction(box, roi) -> float:
    x0, y0, x1, y1 = box
    rx, ry, rw, rh = roi
    iw = min(x1, rx + rw) - max(x0, rx)
    ih = min(y1, ry + rh) - max(y0, ry)
    area = max((x1 - x0) * (y1 - y0), 1e-9)
    return max(iw, 0.0) * max(ih, 0.0) / area


def choose_roi(rng, img_w: int, img_h: int, boxes: list, cfg: dict) -> tuple[tuple[int, int, int, int], str]:
    """(x, y, w, h) with w:h = 4:3 inside the image, and how it was chosen: full, roi or fallback."""
    full = (0, 0, img_w, img_h)
    four_by_three = abs(img_w * 3 - img_h * 4) <= max(img_w, img_h) * 0.01
    r = cfg["roi"]
    if four_by_three and rng.random() < r["full_frame_prob"]:
        return full, "full"
    lo_amb, hi_amb = r["ambiguous_inside"]
    max_w = min(r["max_width_px"], img_w, img_h * 4 // 3)
    for _ in range(r["tries"]):
        if boxes:
            tx0, ty0, tx1, ty1 = boxes[rng.integers(len(boxes))]
            armor_w = max(tx1 - tx0, 2.0)
            w = armor_w * math.exp(rng.uniform(math.log(r["armor_width_multiple"][0]), math.log(r["armor_width_multiple"][1])))
            cx = (tx0 + tx1) / 2 + rng.uniform(-1, 1) * r["center_jitter"] * w
            cy = (ty0 + ty1) / 2 + rng.uniform(-1, 1) * r["center_jitter"] * w * 0.75
        else:
            w = rng.uniform(r["min_width_px"], max_w)
            cx, cy = rng.uniform(0, img_w), rng.uniform(0, img_h)
        w = int(round(min(max(w, r["min_width_px"]), max_w) / 4) * 4)
        h = w * 3 // 4
        x = int(round(min(max(cx - w / 2, 0), img_w - w)))
        y = int(round(min(max(cy - h / 2, 0), img_h - h)))
        roi = (x, y, w, h)
        if all(not (lo_amb < _inside_fraction(b, roi) < hi_amb) for b in boxes):
            return roi, "roi"
    if four_by_three:
        return full, "fallback"
    w = int(min(max_w, img_w, img_h * 4 // 3) // 4 * 4)
    h = w * 3 // 4
    return (0, 0, w, h), "fallback"


# ---- distractors --------------------------------------------------------------------------------------------

def draw_distractors(img: np.ndarray, rng, cfg: dict, avoid: list, ref_px: float) -> list:
    """Draw armor-like clutter that stays off the labeled armors. Returns the boxes it used."""
    d = cfg["distractors"]
    if rng.random() >= d["prob"]:
        return []
    h, w = img.shape[:2]
    placed = []
    names = list(BGR)
    for _ in range(int(rng.integers(d["count"][0], d["count"][1] + 1))):
        kind = ("bar", "panel", "blob")[int(rng.integers(3))]
        size = ref_px * math.exp(rng.uniform(math.log(d["size_vs_armor"][0]), math.log(d["size_vs_armor"][1])))
        size = max(size, 6.0)
        color = BGR[names[int(rng.integers(len(names)))]]
        for _ in range(10):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            if kind == "bar":
                ang = rng.uniform(0, math.pi)
                dx, dy = math.cos(ang) * size / 2, math.sin(ang) * size / 2
                box = (cx - abs(dx) - 2, cy - abs(dy) - 2, cx + abs(dx) + 2, cy + abs(dy) + 2)
            elif kind == "panel":
                pw, ph = size, size * rng.uniform(0.3, 0.8)
                box = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2)
            else:
                r = size / 3
                box = (cx - r, cy - r, cx + r, cy + r)
            if any(_boxes_touch(box, a, 0.1) for a in avoid + placed):
                continue
            if kind == "bar":
                t = max(2, int(size * rng.uniform(0.05, 0.15)))
                cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), color, t, cv2.LINE_AA)
            elif kind == "panel":
                x0, y0, x1, y1 = (int(v) for v in box)
                cv2.rectangle(img, (x0, y0), (x1, y1), (12, 12, 12), -1)
                pitch = max(2.0, pw / rng.uniform(8, 24))
                for yy in np.arange(y0 + pitch / 2, y1, pitch):
                    for xx in np.arange(x0 + pitch / 2, x1, pitch):
                        if rng.random() < 0.35:
                            cv2.circle(img, (int(xx), int(yy)), max(1, int(pitch * 0.32)), color, -1)
            else:
                cv2.ellipse(img, (int(cx), int(cy)), (int(size / 3), int(size / 4)), rng.uniform(0, 180), 0, 360, color, -1, cv2.LINE_AA)
            placed.append(box)
            break
    return placed


def _boxes_touch(a, b, grow: float) -> bool:
    gx, gy = (b[2] - b[0]) * grow, (b[3] - b[1]) * grow
    return not (a[2] < b[0] - gx or a[0] > b[2] + gx or a[3] < b[1] - gy or a[1] > b[3] + gy)


# ---- camera effects -----------------------------------------------------------------------------------------

def _log_uniform(rng, lo, hi) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def apply_effects(img: np.ndarray, rng, cfg: dict) -> tuple[np.ndarray, dict]:
    """Exposure, white balance, bloom, blur and sensor noise in linear light. Returns the image and what was sampled."""
    e = cfg["effects"]
    used: dict = {}
    lin = (img.astype(np.float32) / 255.0) ** 2.2

    used["gain"] = _log_uniform(rng, *e["gain"])
    used["black_level"] = rng.uniform(*e["black_level"])
    wb = rng.uniform(*e["white_balance"], size=3).astype(np.float32)
    used["white_balance"] = [round(float(v), 3) for v in wb]
    lin = lin * used["gain"] * wb + used["black_level"]

    if rng.random() < e["bloom"]["prob"]:
        sigma = rng.uniform(*e["bloom"]["sigma_px"])
        strength = rng.uniform(*e["bloom"]["strength"])
        halo = cv2.GaussianBlur(np.maximum(lin - e["bloom"]["threshold"], 0.0), (0, 0), sigma)
        lin = lin + strength * halo
        used["bloom"] = [round(float(sigma), 2), round(float(strength), 3)]
    if rng.random() < e["motion_blur"]["prob"]:
        length = int(rng.integers(e["motion_blur"]["length_px"][0], e["motion_blur"]["length_px"][1] + 1))
        ang = rng.uniform(0, 180)
        k = np.zeros((length, length), np.float32)
        c = (length - 1) / 2
        dx, dy = math.cos(math.radians(ang)) * c, math.sin(math.radians(ang)) * c
        cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, 1)
        lin = cv2.filter2D(lin, -1, k / max(float(k.sum()), 1e-6), borderType=cv2.BORDER_REFLECT)
        used["motion_blur"] = [length, round(float(ang), 1)]
    if rng.random() < e["defocus"]["prob"]:
        sigma = rng.uniform(*e["defocus"]["sigma_px"])
        lin = cv2.GaussianBlur(lin, (0, 0), sigma)
        used["defocus"] = round(float(sigma), 2)

    read = rng.uniform(*e["read_noise"])
    shot = rng.uniform(*e["shot_noise"])
    used["noise"] = [round(float(read), 4), round(float(shot), 4)]
    lin = np.clip(lin, 0.0, 1.0)
    noise = rng.standard_normal(lin.shape).astype(np.float32)
    lin = np.clip(lin + noise * (read + shot * np.sqrt(lin)), 0.0, 1.0)
    out = np.clip((lin ** (1 / 2.2)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out, used


# ---- one baked image ----------------------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(spec_path: str, cfg: dict) -> None:
    cv2.setNumThreads(1)
    _WORKER["spec"] = LabelSpec.load(spec_path)
    _WORKER["cfg"] = cfg


def bake_one(task: dict) -> dict:
    spec: LabelSpec = _WORKER["spec"]
    cfg: dict = _WORKER["cfg"]
    sample = sample_from_dict(task["sample"])
    out_w, out_h = cfg["size"]
    rng = np.random.default_rng([task["seed"], zlib.crc32(sample.id.encode()), task["copy"]])

    img = cv2.imread(str(Path(task["root"]) / sample.image), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"cannot read {sample.image}", "sample": sample.id}
    img_h, img_w = img.shape[:2]

    trained = [a for a in sample.armors if spec.trains(a.color) and len(a.corners) == 4]
    skipped_colors = len(sample.armors) - len(trained)
    boxes = [bbox(a.corners) for a in trained]
    roi, how = choose_roi(rng, img_w, img_h, boxes, cfg)
    x0, y0, rw, rh = roi
    scale = out_w / rw
    crop = img[y0:y0 + rh, x0:x0 + rw].copy()

    ref_px = float(np.median([b[2] - b[0] for b in boxes])) if boxes else 0.03 * img_w
    local_boxes = [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in boxes]
    clutter = draw_distractors(crop, rng, cfg, local_boxes, ref_px)
    crop, used = apply_effects(crop, rng, cfg)
    net = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    lines, armors_meta, dropped = [], [], 0
    for a, box in zip(trained, boxes):
        if _inside_fraction(box, roi) < 0.5:
            dropped += 1
            continue
        corners = [[(x - x0) * scale, (y - y0) * scale] for x, y in a.corners]
        vis = [v if (0 <= cx <= out_w and 0 <= cy <= out_h) else 0 for (cx, cy), v in zip(corners, a.vis)]
        cid = spec.class_id(a.color, a.label)
        lines.append(pose_line(cid, corners, vis, out_w, out_h))
        armors_meta.append({
            "cls": cid, "corners_orig": [[round(x, 3), round(y, 3)] for x, y in a.corners],
            "corners_net": [[round(x, 3), round(y, 3)] for x, y in corners], "vis": vis,
            "dist_m": a.dist_m, "yaw_deg": a.yaw_deg, "pitch_deg": a.pitch_deg, "occ": a.occ,
        })

    name = f"{sample.id}_c{task['copy']}"
    split = task["split"]
    out = Path(task["out"])
    cv2.imwrite(str(out / "images" / split / f"{name}.png"), net, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    (out / "labels" / split / f"{name}.txt").write_text("".join(ln + "\n" for ln in lines))
    return {
        "image": f"images/{split}/{name}.png", "sample": sample.id, "source": task["source"], "copy": task["copy"],
        "split": split, "roi": [x0, y0, rw, rh], "roi_mode": how, "scale": scale, "effects": used,
        "distractors": len(clutter), "armors": armors_meta, "dropped_by_roi": dropped,
        "skipped_untrained_colors": skipped_colors, "scene": sample.scene,
    }


# ---- driver -------------------------------------------------------------------------------------------------

def split_of(sample_id: str, val_fraction: float, role: str = "auto") -> str:
    """Train or val. Only "auto" datasets (independent simulator frames) are split per sample; a capture session
    keeps its role as a whole, so frames of a val, hard or neg session can never leak into training."""
    if role == "train":
        return "train"
    if role != "auto":
        return "val"
    h = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) / 2**32
    return "val" if h < val_fraction else "train"


def bake(sources: list[tuple[Path, int]], out: Path, spec: LabelSpec, cfg: dict, copies: int, seed: int, workers: int, spec_path: str, limit: int = 0) -> dict:
    if out.exists():
        raise FileExistsError(f"{out} exists; bakes are never overwritten")
    tasks, src_info, seen = [], [], set()
    for root, repeat in sources:
        man = read_manifest(root / "manifest.json")
        if man.label_spec_sha256 != spec.sha256:
            raise ValueError(f"{root} was ingested with a different label_spec; re-ingest it")
        n = 0
        for s in read_samples(root / "labels.jsonl"):
            if limit and n >= limit:
                break
            if s.id in seen:
                raise ValueError(f"duplicate sample id {s.id}; dataset names must be unique")
            seen.add(s.id)
            split = split_of(s.id, cfg["val_fraction"], man.role)
            # Evaluation sessions are baked once: extra copies would only duplicate the same frame in val.
            n_copies = copies * repeat if split == "train" or man.role == "auto" else 1
            for c in range(n_copies):
                tasks.append({"sample": sample_to_dict(s), "root": str(root), "copy": c, "seed": seed, "split": split, "source": man.name})
            n += 1
        src_info.append({"path": str(root), "name": man.name, "role": man.role, "repeat": repeat, "samples": n,
                         "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()})
    tmp = out.parent / f".{out.name}.tmp"
    if tmp.exists():
        raise FileExistsError(f"{tmp} is left over from a failed bake; remove it")
    for split in ("train", "val"):
        (tmp / "images" / split).mkdir(parents=True)
        (tmp / "labels" / split).mkdir(parents=True)
    for t in tasks:
        t["out"] = str(tmp)

    metas = []
    # spawn, not fork: a parent that has already used OpenCV's thread pool deadlocks forked children.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(spec_path, cfg)) as pool:
        for i, res in enumerate(pool.map(bake_one, tasks, chunksize=8)):
            if "error" in res:
                raise RuntimeError(res["error"])
            metas.append(res)
            if (i + 1) % 500 == 0:
                print(f"  baked {i + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    metas.sort(key=lambda m: m["image"])

    class_counts = {s: Counter() for s in ("train", "val")}
    for m in metas:
        for a in m["armors"]:
            class_counts[m["split"]][spec.class_name(a["cls"])] += 1
    (tmp / "meta.jsonl").write_text("".join(json.dumps(m, separators=(",", ":")) + "\n" for m in metas))
    names = "\n".join(f"  {i}: {spec.class_name(i)}" for i in range(spec.n_classes))
    (tmp / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [4, 3]\nflip_idx: [3, 2, 1, 0]\nnames:\n{names}\n"
    )
    manifest = {
        "bake_version": BAKE_VERSION, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed, "copies": copies, "sources": src_info, "label_spec_sha256": spec.sha256,
        "config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "images": dict(Counter(m["split"] for m in metas)),
        "armors": {s: dict(sorted(c.items())) for s, c in class_counts.items()},
        "roi_modes": dict(Counter(m["roi_mode"] for m in metas)),
        "images_with_distractors": sum(1 for m in metas if m["distractors"]),
        "armors_dropped_by_roi": sum(m["dropped_by_roi"] for m in metas),
        "armors_skipped_untrained_colors": sum(m["skipped_untrained_colors"] for m in metas),
        "meta_sha256": hashlib.sha256((tmp / "meta.jsonl").read_bytes()).hexdigest(),
    }
    (tmp / "bake_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.rename(out)
    return manifest


def _parse_src(text: str) -> tuple[Path, int]:
    path, _, rep = text.partition(":")
    return Path(path), int(rep) if rep else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, help="canonical dataset dir, optionally :REPEAT")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="bake only the first N samples of each source (quality checks)")
    args = ap.parse_args()
    spec_path = str(args.spec) if args.spec else str(Path(__file__).resolve().parents[1] / "configs" / "label_spec.yaml")
    spec = LabelSpec.load(spec_path)
    cfg = load_config(args.config)
    m = bake([_parse_src(s) for s in args.src], args.out, spec, cfg, args.copies, args.seed, args.workers, spec_path, args.limit)
    print(json.dumps({k: m[k] for k in ("images", "armors", "roi_modes", "images_with_distractors", "armors_dropped_by_roi", "armors_skipped_untrained_colors")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
