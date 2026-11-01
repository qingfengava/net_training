"""Evaluate a pose model on the val (or train) split of a baked dataset, with errors in original pixels.

    python -m nt.eval --weights runs/train/NAME/weights/last.pt --data baked/NAME2 [--split val] \\
        [--conf 0.2] [--head auto|e2e|nms] [--out runs/eval/NAME.json]
    python -m nt.eval --weights exports/NAME/model.onnx --backend onnx|trt-fp16|trt-fp32 --data baked/NAME2 ...

Every image is matched to its ground truth by corner-box IoU >= 0.5. Reported:
  * armor AP50: per class, pooled over classes present (a hit needs the right class)
  * at --conf (sleepy uses 0.2): recall and precision ignoring class, class/color/label accuracy among
    matches, and corner error p50/p95/max in network pixels and in ORIGINAL image pixels (the tracker's unit)
  * the same corner error split by camera distance, view angle and occlusion from the simulator's metadata
  * recall and class accuracy by armor width in network pixels, per-class recall, the confusion matrix, false
    positives split into ones on a real armor (a duplicate with another number, or a loose box) and ones on
    background, and the same recall / precision / class accuracy at the --sweep thresholds
  * `summary`: robot number accuracy (sentry and robot numbers, the hard part), off-class recall and those
    numbers by armor width, so runs compare on one line
YOLO26 models are evaluated with both heads: `e2e` is the NMS-free one2one head that gets deployed, `nms` is
the one2many head with NMS. Models without that choice only have `nms`.

Backends. `pt` runs the checkpoint through Ultralytics. `onnx` runs an nt.export model (the sleepy contract,
decoded the way the C++ decoder reads it) with ONNX Runtime in float32. `trt-fp16` / `trt-fp32` build a
TensorRT engine of that model with trtexec and run it through scripts/trt_runner.py under the system Python
that has the tensorrt bindings (NT_TRT_PYTHON, default /usr/bin/python3; NT_TRTEXEC, default trtexec); the same
images also go through ONNX Runtime, and `vs_onnx` reports the paired difference: how far corners and
confidences move by running the engine, which is what FP16 deployment does to the network's output.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IOU_MATCH = 0.5
BINS = {
    "dist_m": [0, 2, 4, 6, 9, 1e9],
    "abs_yaw_deg": [0, 20, 40, 60, 91],
    "occ": [0, 0.05, 0.2, 0.4, 1.01],
}
WIDTH_BINS = [0, 8, 16, 32, 64, 1e9]  # armor box width in network pixels; below 16 the digits are hard to read
FP_OVERLAP_IOU = 0.1  # a false positive overlapping a labeled armor this much sits on a real armor


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


def _bin_label(edges: list[float], value: float) -> str | None:
    b = next((i for i in range(len(edges) - 1) if edges[i] <= value < edges[i + 1]), None)
    return None if b is None else f"[{edges[b]:g},{edges[b + 1]:g})"


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def at_threshold(per_image, conf_thr: float, parts) -> dict:
    """Class-agnostic matching of the detections at or above conf_thr; everything that depends on it."""
    n_gt = n_det = n_hit = n_cls_ok = n_color_ok = n_label_ok = fp_on_armor = 0
    gt_by_cls: dict[int, int] = {}
    hit_by_cls: dict[int, int] = {}
    ok_by_cls: dict[int, int] = {}
    confusion: dict[int, dict[int, int]] = {}
    width: dict[str, list[int]] = {}
    matches = []  # (det, gt, meta) for corner errors
    for dets, gts, meta in per_image:
        confident = [d for d in dets if d["conf"] >= conf_thr]
        n_gt += len(gts)
        n_det += len(confident)
        hit = {}
        for di, gi in match(confident, gts, by_class=False):
            d = confident[di]
            if gi is None:
                fp_on_armor += any(iou(d["box"], g["box"]) > FP_OVERLAP_IOU for g in gts)
                continue
            hit[gi] = d
        for gi, g in enumerate(gts):
            c = g["cls"]
            gt_by_cls[c] = gt_by_cls.get(c, 0) + 1
            label = _bin_label(WIDTH_BINS, g["box"][2] - g["box"][0])
            slot = width.setdefault(label, [0, 0, 0])
            slot[0] += 1
            d = hit.get(gi)
            if d is None:
                continue
            slot[1] += 1
            slot[2] += d["cls"] == c
            n_hit += 1
            hit_by_cls[c] = hit_by_cls.get(c, 0) + 1
            ok_by_cls[c] = ok_by_cls.get(c, 0) + (d["cls"] == c)
            row = confusion.setdefault(c, {})
            row[d["cls"]] = row.get(d["cls"], 0) + 1
            n_cls_ok += d["cls"] == c
            (dc, dn), (gc, gn) = parts(d["cls"]), parts(c)
            n_color_ok += dc == gc
            n_label_ok += dn == gn
            matches.append((d, g, meta))
    fp = n_det - n_hit
    return {
        "conf": conf_thr,
        "recall": _ratio(n_hit, n_gt),
        "precision": _ratio(n_hit, n_det),
        "class_acc": _ratio(n_cls_ok, n_hit),
        "color_acc": _ratio(n_color_ok, n_hit),
        "label_acc": _ratio(n_label_ok, n_hit),
        "false_positives": fp,
        "false_positives_on_armor": fp_on_armor,
        "false_positives_on_background": fp - fp_on_armor,
        "_by_class": {c: (gt_by_cls[c], hit_by_cls.get(c, 0), ok_by_cls.get(c, 0)) for c in gt_by_cls},
        "_confusion": confusion,
        "_width": width,
        "_matches": matches,
    }


def evaluate(per_image: list[tuple[list[dict], list[dict], dict]], conf_thr: float, n_classes: int,
             parts=None, sweep: tuple[float, ...] = (0.3, 0.5)) -> dict:
    """per_image: (detections, ground truths, image meta). Detections carry box, conf, cls, pts (4x2).

    parts maps a merged class id to (color, number), normally LabelSpec.from_class_id; color and number accuracy
    compare those. Without it two colors times n_classes // 2 labels are assumed.
    """
    if parts is None:
        half = n_classes // 2
        parts = lambda c: (c // half, c % half)  # noqa: E731
    tp_by_cls: dict[int, list[tuple[float, int]]] = {c: [] for c in range(n_classes)}
    n_gt_cls = {c: 0 for c in range(n_classes)}
    for dets, gts, _ in per_image:
        for g in gts:
            n_gt_cls[g["cls"]] += 1
        for di, gi in match(dets, gts, by_class=True):
            tp_by_cls[dets[di]["cls"]].append((dets[di]["conf"], int(gi is not None)))

    main = at_threshold(per_image, conf_thr, parts)
    errs_net: list[float] = []
    errs_orig: list[float] = []
    binned: dict[str, dict[str, tuple[list, list]]] = {k: {} for k in BINS}
    for d, g, meta in main["_matches"]:
        seen = np.array(g.get("vis", [2, 2, 2, 2])) > 0  # a corner outside the ROI has no meaningful error
        e_net = np.linalg.norm(d["pts"] - g["corners_net"], axis=1)[seen]
        e_orig = e_net / meta["scale"]
        errs_net.extend(e_net.tolist())
        errs_orig.extend(e_orig.tolist())
        for key, value in (("dist_m", g.get("dist_m")), ("abs_yaw_deg", None if g.get("yaw_deg") is None else abs(g["yaw_deg"])), ("occ", g.get("occ"))):
            label = None if value is None else _bin_label(BINS[key], value)
            if label is not None:
                slot = binned[key].setdefault(label, ([], []))
                slot[0].extend(e_net.tolist())
                slot[1].extend(e_orig.tolist())

    aps = {}
    for c in range(n_classes):
        if n_gt_cls[c] == 0:
            continue
        rows = sorted(tp_by_cls[c], key=lambda r: -r[0])
        aps[c] = average_precision(np.array([r[1] for r in rows], np.float64), n_gt_cls[c])
    public = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    return {
        "images": len(per_image),
        "gt_armors": sum(n_gt_cls.values()),
        "armor_ap50": round(float(np.mean(list(aps.values()))), 4) if aps else None,
        "ap50_by_class": {int(c): round(v, 4) for c, v in aps.items()},
        "at_conf": public(main),
        "at_conf_sweep": [public(at_threshold(per_image, t, parts)) for t in sweep if t != conf_thr],
        "by_class": {int(c): {"n": n, "hits": h, "correct": ok, "recall": _ratio(h, n), "class_acc": _ratio(ok, h)}
                     for c, (n, h, ok) in sorted(main["_by_class"].items())},
        "by_width_px": {lab: {"n": n, "hits": h, "correct": ok, "recall": _ratio(h, n), "class_acc": _ratio(ok, h)}
                        for lab, (n, h, ok) in sorted(main["_width"].items(), key=lambda kv: float(kv[0][1:].split(",")[0]))},
        "confusion": {int(g): {int(p): n for p, n in sorted(row.items())} for g, row in sorted(main["_confusion"].items())},
        "corner_error": summarize_errors(errs_net, errs_orig),
        "corner_error_by": {k: {lab: summarize_errors(*v) for lab, v in sorted(d.items())} for k, d in binned.items()},
    }


ROBOT_LABELS = {0, 1, 3, 4}  # sentry and the robot numbers; outpost (6) and base (7) are structures


def summarize(res: dict, spec) -> dict:
    """The few numbers that compare runs: robot numbers (sentry and robot ids, the hard classes) and the off
    class, from evaluate()'s per-class counts, plus the confusion matrix with class names."""
    off = spec.n_classes - 1 if spec.off_class else None
    robot = {c for c in range(spec.n_classes) if c != off and spec.from_class_id(c)[1] in ROBOT_LABELS}
    rows = res["by_class"]
    def pooled(classes):
        n, hits, ok = (sum(rows[c][k] for c in classes if c in rows) for k in ("n", "hits", "correct"))
        return {"n": n, "recall": _ratio(hits, n), "class_acc": _ratio(ok, hits)}
    name = spec.class_name
    return {
        "ap50": res["armor_ap50"],
        "recall": res["at_conf"]["recall"], "precision": res["at_conf"]["precision"], "class_acc": res["at_conf"]["class_acc"],
        "robot_numbers": pooled(robot),
        "off_class": pooled({off}) if off is not None else None,
        "corner_p95_orig_px": res["corner_error"]["orig_px"]["p95"],
        "confusion_named": {name(g): {name(p): n for p, n in row.items()} for g, row in res["confusion"].items()},
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


# ---- exported models: the sleepy contract through ONNX Runtime or TensorRT ----------------------------------

def contract_input(path: Path) -> np.ndarray:
    """What sleepy feeds the network for a 768x576 frame: RGB, /255, NCHW float32 (no resize needed)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def decode_contract(out: np.ndarray, spec, conf: float) -> list[dict]:
    """[30, 18] contract rows -> detections with the merged class id, like sleepy's AT decoder reads them."""
    off = spec.off_class
    dets = []
    for row in out:
        if not np.isfinite(row[4]) or row[4] < conf:
            continue
        number, color = int(round(float(row[5]))), int(np.argmax(row[6:10]))
        if off is not None and (color, number) == (off[0], off[1]):
            cls = spec.n_classes - 1
        elif spec.valid_pair(color, number) and color in spec.training_colors:
            cls = spec.class_id(color, number)
        else:
            continue  # an id outside the class table; nt.contract reports those
        pts = row[10:18].reshape(4, 2).astype(np.float64)
        dets.append({"conf": float(row[4]), "cls": cls, "pts": pts, "box": box_of(pts)})
    return dets


def run_onnx(model: Path, images: list[Path]):
    import onnxruntime as ort

    # CPU float32 is the reference: it is what the export is checked against, and the CUDA provider of the venv's
    # onnxruntime-gpu cannot find cuBLAS unless torch happens to be imported first.
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for p in images:
        yield sess.run(None, {name: contract_input(p)})[0][0]


def build_engine(model: Path, precision: str, out: Path) -> Path:
    trtexec = os.environ.get("NT_TRTEXEC") or shutil.which("trtexec")
    if not trtexec:
        raise SystemExit("trtexec not found; set NT_TRTEXEC")
    engine = out / f"{model.parent.name}_{precision}.engine"
    cmd = [trtexec, f"--onnx={model}", f"--saveEngine={engine}", "--skipInference"] + (["--fp16"] if precision == "fp16" else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"trtexec failed:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")
    return engine


def run_trt(engine: Path, images: list[Path]):
    """Frames go to scripts/trt_runner.py on stdin as raw float32, outputs come back the same way."""
    py = os.environ.get("NT_TRT_PYTHON", "/usr/bin/python3")
    proc = subprocess.Popen([py, "-I", str(ROOT / "scripts" / "trt_runner.py"), str(engine)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for p in images:
            proc.stdin.write(contract_input(p).tobytes())
            proc.stdin.flush()
            raw = proc.stdout.read(30 * 18 * 4)
            if len(raw) != 30 * 18 * 4:
                raise SystemExit(f"trt_runner stopped: {proc.stderr.read().decode()[-2000:]}")
            yield np.frombuffer(raw, np.float32).reshape(30, 18)
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)


def paired_difference(a: list[list[dict]], b: list[list[dict]], metas: list[dict], conf: float) -> dict:
    """How far b's detections are from a's on the same images (a = ONNX float32, b = the engine).

    Pairs need the same class and box IoU >= 0.5: one armor often has a few candidates with different numbers,
    and pairing across classes would mix those up. A detection that crosses the threshold, or whose class flips,
    is left unpaired and counted.
    """
    corner_net, corner_orig, conf_diff = [], [], []
    only_a = only_b = 0
    for da, db, meta in zip(a, b, metas):
        da = [d for d in da if d["conf"] >= conf]
        db = [d for d in db if d["conf"] >= conf]
        as_gt = [{"cls": d["cls"], "box": d["box"]} for d in da]
        used = set()
        for di, gi in match(db, as_gt, by_class=True):
            if gi is None:
                only_b += 1
                continue
            used.add(gi)
            e = np.linalg.norm(db[di]["pts"] - da[gi]["pts"], axis=1)
            corner_net.extend(e.tolist())
            corner_orig.extend((e / meta["scale"]).tolist())
            conf_diff.append(abs(db[di]["conf"] - da[gi]["conf"]))
        only_a += len(da) - len(used)
    return {"conf": conf, "matched": len(conf_diff), "only_in_onnx": only_a, "only_in_engine": only_b,
            "corner_diff": summarize_errors(corner_net, corner_orig),
            "conf_diff": {"p50": pct(conf_diff, 50), "p95": pct(conf_diff, 95), "max": pct(conf_diff, 100)}}


def _print(tag: str, res: dict, spec) -> None:
    s = summarize(res, spec)
    ce, ac = res["corner_error"]["orig_px"], res["at_conf"]
    rn, off = s["robot_numbers"], s["off_class"] or {}
    print(f"[{tag}] images={res['images']} armors={res['gt_armors']} AP50={res['armor_ap50']} recall={ac['recall']} "
          f"precision={ac['precision']} class_acc={ac['class_acc']} robot_number_acc={rn['class_acc']} "
          f"off_recall={off.get('recall')} corner_orig_px p50={ce['p50']} p95={ce['p95']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help=".pt checkpoint, or an nt.export model.onnx")
    ap.add_argument("--data", type=Path, required=True, help="baked dataset directory")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--sweep", default="0.3,0.5", help="further confidence thresholds to report")
    ap.add_argument("--head", choices=["auto", "e2e", "nms"], default="auto")
    ap.add_argument("--backend", choices=["auto", "pt", "onnx", "trt-fp16", "trt-fp32"], default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from nt.spec import LabelSpec

    spec = LabelSpec.load()
    sweep = tuple(float(x) for x in args.sweep.split(",") if x)
    backend = args.backend if args.backend != "auto" else ("onnx" if args.weights.suffix == ".onnx" else "pt")
    if backend != "pt" and args.weights.suffix != ".onnx":
        raise SystemExit(f"--backend {backend} needs an exported model.onnx")
    metas = load_ground_truth(args.data, args.split)
    images = [args.data / m["image"] for m in metas]
    gts = [[{"cls": a["cls"], "corners_net": np.array(a["corners_net"], np.float64),
             "box": box_of(np.array(a["corners_net"], np.float64)), "dist_m": a.get("dist_m"),
             "yaw_deg": a.get("yaw_deg"), "occ": a.get("occ"), "vis": a["vis"]} for a in m["armors"]]
            for m in metas]
    report = {"weights": str(args.weights), "data": str(args.data), "split": args.split, "backend": backend, "heads": {}}

    def score(tag: str, dets: list[list[dict]]) -> None:
        res = evaluate([(d, g, m) for d, g, m in zip(dets, gts, metas)], args.conf, spec.n_classes, spec.from_class_id, sweep)
        res["summary"] = summarize(res, spec)
        report["heads"][tag] = res
        _print(tag, res, spec)

    if backend == "pt":
        import torch

        e2e_capable = False
        if args.head == "auto":
            from ultralytics import YOLO

            e2e_capable = bool(getattr(YOLO(str(args.weights), task="pose").model, "end2end", False))
        for head in {"auto": ["e2e", "nms"] if e2e_capable else [None], "e2e": ["e2e"], "nms": ["nms"]}[args.head]:
            score(head or "default", predict(args.weights, images, 0.001, head, args.batch, args.device))
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    else:
        onnx_dets = [decode_contract(o, spec, 0.001) for o in run_onnx(args.weights, images)]
        if backend == "onnx":
            score("onnx", onnx_dets)
        else:
            precision = backend.split("-")[1]
            with tempfile.TemporaryDirectory() as tmp:
                engine = build_engine(args.weights, precision, Path(tmp))
                trt_dets = [decode_contract(o, spec, 0.001) for o in run_trt(engine, images)]
            score("onnx", onnx_dets)
            score(backend, trt_dets)
            report["vs_onnx"] = paired_difference(onnx_dets, trt_dets, metas, args.conf)
            v = report["vs_onnx"]
            print(f"[{backend} vs onnx] matched={v['matched']} only_onnx={v['only_in_onnx']} only_engine={v['only_in_engine']} "
                  f"corner_diff_net_px p50={v['corner_diff']['net_px']['p50']} "
                  f"p95={v['corner_diff']['net_px']['p95']} max={v['corner_diff']['net_px']['max']} orig_px p95={v['corner_diff']['orig_px']['p95']} "
                  f"conf_diff p95={v['conf_diff']['p95']} max={v['conf_diff']['max']}")
            report["engine"] = {"trtexec": os.environ.get("NT_TRTEXEC") or shutil.which("trtexec"), "precision": precision}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
