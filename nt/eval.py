"""Evaluate a pose model on the val (or train) split of a baked dataset, with errors in original pixels.

    python -m nt.eval --weights runs/train/NAME/weights/best.pt --data baked/NAME2 [--split val] \\
        [--conf 0.2] [--head auto|e2e|nms] [--out runs/eval/NAME.json]

Every image is matched to its ground truth by corner-box IoU >= 0.5. Reported:
  * armor AP50: per class, pooled over classes present (a hit needs the right class)
  * at --conf (sleepy uses 0.2): recall and precision ignoring class, class/color/label accuracy among
    matches, and corner error p50/p95/max in network pixels and in ORIGINAL image pixels (the tracker's unit)
  * the same corner error split by camera distance, view angle and occlusion from the simulator's metadata
YOLO26 models are evaluated with both heads: `e2e` is the NMS-free one2one head that gets deployed, `nms` is
the one2many head with NMS. Models without that choice only have `nms`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IOU_MATCH = 0.5
BINS = {
    "dist_m": [0, 2, 4, 6, 9, 1e9],
    "abs_yaw_deg": [0, 20, 40, 60, 91],
    "occ": [0, 0.05, 0.2, 0.4, 1.01],
}


def box_of(pts: np.ndarray) -> np.ndarray:
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], np.float64)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def match(dets: list[dict], gts: list[dict], by_class: bool) -> list[tuple[int, int | None]]:
    """Greedy by confidence. Returns (detection index, matched gt index or None) in confidence order."""
    used: set[int] = set()
    out = []
    for di in sorted(range(len(dets)), key=lambda i: -dets[i]["conf"]):
        best, best_iou = None, IOU_MATCH
        for gi, g in enumerate(gts):
            if gi in used or (by_class and g["cls"] != dets[di]["cls"]):
                continue
            v = iou(dets[di]["box"], g["box"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            used.add(best)
        out.append((di, best))
    return out


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """COCO-style 101-point AP for detections already sorted by confidence."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return float(ap / 101)


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def summarize_errors(errs_net: list[float], errs_orig: list[float]) -> dict:
    return {
        "n_corners": len(errs_orig),
        "net_px": {"p50": pct(errs_net, 50), "p95": pct(errs_net, 95), "max": pct(errs_net, 100)},
        "orig_px": {"p50": pct(errs_orig, 50), "p95": pct(errs_orig, 95), "max": pct(errs_orig, 100)},
    }


def evaluate(per_image: list[tuple[list[dict], list[dict], dict]], conf_thr: float, n_classes: int) -> dict:
    """per_image: (detections, ground truths, image meta). Detections carry box, conf, cls, pts (4x2)."""
    tp_by_cls: dict[int, list[tuple[float, int]]] = {c: [] for c in range(n_classes)}
    n_gt_cls = {c: 0 for c in range(n_classes)}
    n_gt = n_det = n_hit = n_cls_ok = n_color_ok = n_label_ok = 0
    errs_net: list[float] = []
    errs_orig: list[float] = []
    binned: dict[str, dict[str, tuple[list, list]]] = {k: {} for k in BINS}

    for dets, gts, meta in per_image:
        for g in gts:
            n_gt_cls[g["cls"]] += 1
        for di, gi in match(dets, gts, by_class=True):
            tp_by_cls[dets[di]["cls"]].append((dets[di]["conf"], int(gi is not None)))
        confident = [d for d in dets if d["conf"] >= conf_thr]
        n_gt += len(gts)
        n_det += len(confident)
        for di, gi in match(confident, gts, by_class=False):
            if gi is None:
                continue
            d, g = confident[di], gts[gi]
            n_hit += 1
            n_cls_ok += d["cls"] == g["cls"]
            n_color_ok += d["cls"] // 8 == g["cls"] // 8
            n_label_ok += d["cls"] % 8 == g["cls"] % 8
            seen = np.array(g.get("vis", [2, 2, 2, 2])) > 0  # a corner outside the ROI has no meaningful error
            e_net = np.linalg.norm(d["pts"] - g["corners_net"], axis=1)[seen]
            e_orig = e_net / meta["scale"]
            errs_net.extend(e_net.tolist())
            errs_orig.extend(e_orig.tolist())
            for key, value in (("dist_m", g.get("dist_m")), ("abs_yaw_deg", None if g.get("yaw_deg") is None else abs(g["yaw_deg"])), ("occ", g.get("occ"))):
                if value is None:
                    continue
                edges = BINS[key]
                b = next((i for i in range(len(edges) - 1) if edges[i] <= value < edges[i + 1]), None)
                if b is not None:
                    label = f"[{edges[b]:g},{edges[b + 1]:g})"
                    slot = binned[key].setdefault(label, ([], []))
                    slot[0].extend(e_net.tolist())
                    slot[1].extend(e_orig.tolist())

    aps = {}
    for c in range(n_classes):
        if n_gt_cls[c] == 0:
            continue
        rows = sorted(tp_by_cls[c], key=lambda r: -r[0])
        aps[c] = average_precision(np.array([r[1] for r in rows], np.float64), n_gt_cls[c])
    return {
        "images": len(per_image),
        "gt_armors": n_gt,
        "armor_ap50": round(float(np.mean(list(aps.values()))), 4) if aps else None,
        "ap50_by_class": {int(c): round(v, 4) for c, v in aps.items()},
        "at_conf": {
            "conf": conf_thr,
            "recall": round(n_hit / n_gt, 4) if n_gt else None,
            "precision": round(n_hit / n_det, 4) if n_det else None,
            "class_acc": round(n_cls_ok / n_hit, 4) if n_hit else None,
            "color_acc": round(n_color_ok / n_hit, 4) if n_hit else None,
            "label_acc": round(n_label_ok / n_hit, 4) if n_hit else None,
            "false_positives": n_det - n_hit,
        },
        "corner_error": summarize_errors(errs_net, errs_orig),
        "corner_error_by": {k: {lab: summarize_errors(*v) for lab, v in sorted(d.items())} for k, d in binned.items()},
    }


def load_ground_truth(data: Path, split: str) -> list[dict]:
    metas = [json.loads(ln) for ln in (data / "meta.jsonl").read_text().splitlines() if ln]
    return [m for m in metas if m["split"] == split]


def predict(weights: Path, images: list[Path], conf: float, head: str | None, batch: int, device: str) -> list[list[dict]]:
    from nt import offline

    offline.apply()
    from ultralytics import YOLO

    model = YOLO(str(weights), task="pose")
    kwargs = {} if head is None else {"end2end": head == "e2e"}
    out = []
    # Ultralytics treats a Python list as one already-loaded batch and ignores `batch`, so a list of 600 paths
    # became a single (600, 3, H, W) tensor and exhausted the GPU. Slice the list ourselves.
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        for r in model.predict(source=chunk, imgsz=768, conf=conf, max_det=30, device=device, verbose=False, **kwargs):
            dets = []
            if r.boxes is not None and len(r.boxes):
                confs, clss = r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
                kxy = r.keypoints.xy.cpu().numpy().astype(np.float64)
                for c, k, p in zip(confs, clss, kxy):
                    dets.append({"conf": float(c), "cls": int(k), "pts": p, "box": box_of(p)})
            out.append(dets)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="baked dataset directory")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--head", choices=["auto", "e2e", "nms"], default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from nt.spec import LabelSpec

    spec = LabelSpec.load()
    metas = load_ground_truth(args.data, args.split)
    images = [args.data / m["image"] for m in metas]
    gts = [[{"cls": a["cls"], "corners_net": np.array(a["corners_net"], np.float64),
             "box": box_of(np.array(a["corners_net"], np.float64)), "dist_m": a.get("dist_m"),
             "yaw_deg": a.get("yaw_deg"), "occ": a.get("occ"), "vis": a["vis"]} for a in m["armors"]]
            for m in metas]

    import torch

    e2e_capable = False
    if args.head == "auto":
        from ultralytics import YOLO

        e2e_capable = bool(getattr(YOLO(str(args.weights), task="pose").model, "end2end", False))
    heads = {"auto": ["e2e", "nms"] if e2e_capable else [None], "e2e": ["e2e"], "nms": ["nms"]}[args.head]

    report = {"weights": str(args.weights), "data": str(args.data), "split": args.split, "heads": {}}
    for head in heads:
        dets = predict(args.weights, images, 0.001, head, args.batch, args.device)
        res = evaluate([(d, g, m) for d, g, m in zip(dets, gts, metas)], args.conf, spec.n_classes)
        report["heads"][head or "default"] = res
        ce, ac = res["corner_error"]["orig_px"], res["at_conf"]
        print(f"[{head or 'default'}] images={res['images']} armors={res['gt_armors']} AP50={res['armor_ap50']} "
              f"recall={ac['recall']} precision={ac['precision']} class_acc={ac['class_acc']} "
              f"corner_orig_px p50={ce['p50']} p95={ce['p95']}")
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
