"""固定权重的极小目标审计：历史输入、真值 ROI、检测驱动 ROI。

python -m nt.tiny --out runs/experiments/tiny_b_v1
只运行推理，不训练。输出逐目标记录、像素对拍、案例图库和三组对比。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from nt import bake
from nt import eval as ev
from nt.canonical import read_samples, sample_to_dict
from nt.spec import LabelSpec

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("full", "oracle_roi", "detector_roi")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=serial, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=serial, allow_nan=False) + "\n")


def ground_truth(meta: dict) -> list[dict]:
    return [{**a, "box": ev.box_of(np.array(a["corners_net"], float))} for a in meta["armors"]]


def outcomes(dets: list[dict], gts: list[dict], conf: float, spec: LabelSpec, scale: float) -> list[dict]:
    """沿用 nt.eval 的一对一匹配；所有目标均参与竞争，避免只匹配小目标造成乐观偏差。"""
    ds = [d for d in dets if d["conf"] >= conf]
    hits = {gi: ds[di] for di, gi in ev.match(ds, gts, by_class=False) if gi is not None}
    result = []
    for gi, gt in enumerate(gts):
        pred = hits.get(gi)
        best = max((ev.iou(d["box"], gt["box"]) for d in ds), default=0.0)
        row = {"state": "unmatched", "prediction": pred, "best_iou": float(best),
               "error_flags": [], "corner_errors_orig_px": []}
        if pred is not None:
            row["state"] = "correct" if pred["cls"] == gt["cls"] else "wrong_class"
            dc, dn = spec.from_class_id(pred["cls"])
            gc, gn = spec.from_class_id(gt["cls"])
            row["error_flags"] = (["wrong_color"] if dc != gc else []) + (["wrong_number"] if dn != gn else [])
            if spec.is_off(dc) and not spec.is_off(gc):
                row["error_flags"].append("false_off")
            visible = np.array(gt["vis"]) > 0
            err = np.linalg.norm(np.array(pred["pts"]) - gt["corners_net"], axis=1)[visible] / scale
            row["corner_errors_orig_px"] = err.tolist()
        elif best >= 0.1:
            row["error_flags"] = ["localization_or_assignment"]
        else:
            row["error_flags"] = ["no_overlap_detection"]
        result.append(row)
    return result


def roi_for(box, width: int, height: int, minimum: int = 384, multiple: float = 16) -> tuple[int, int, int, int]:
    """固定 4:3、保留上下文，中心越界时平移窗口；不修改图像或目标坐标。"""
    x0, y0, x1, y1 = box
    if width < 4 or height < 3 or minimum < 4 or not math.isfinite(multiple) or multiple <= 0:
        raise ValueError("ROI 参数无效")
    maximum = min(width // 4, height // 3) * 4
    wanted = max(minimum, multiple * max(x1 - x0, (y1 - y0) * 4 / 3))
    rw = min(maximum, max(4, int(math.ceil(wanted / 4)) * 4))
    rh = rw * 3 // 4
    x = min(max(round((x0 + x1 - rw) / 2), 0), width - rw)
    y = min(max(round((y0 + y1 - rh) / 2), 0), height - rh)
    return int(x), int(y), rw, rh


def proposal_rois(dets, shape, scale, seed_conf=0.1, max_width=16, max_rois=4, minimum=384, multiple=16):
    """只依赖全图预测生成 ROI；禁止传入真值、失败目标清单或目标类别。"""
    h, w = shape[:2]
    proposals = []
    for d in sorted(dets, key=lambda p: -p["conf"]):
        b = np.asarray(d["box"], float)
        if (d["conf"] < seed_conf or not np.isfinite(b).all()
                or not 0 < b[2] - b[0] < max_width or b[3] <= b[1]):
            continue
        center = (b[:2] + b[2:]) / (2 * scale)
        if not (0 <= center[0] < w and 0 <= center[1] < h):
            continue
        roi = roi_for(b / scale, w, h, minimum, multiple)
        x, y, rw, rh = roi
        rect = [x, y, x + rw, y + rh]
        if any(ev.iou(rect, [q[0], q[1], q[0] + q[2], q[1] + q[3]]) > 0.8 for q in proposals):
            continue
        proposals.append(roi)
        if len(proposals) == max_rois:
            break
    return proposals


def map_crop(dets: list[dict], roi, base_scale: float, out_width: int = 768, crop_id: int = 0) -> list[dict]:
    x, y, w, _ = roi
    rows = []
    for d in dets:
        pts = (np.array(d["pts"], float) / (out_width / w) + [x, y]) * base_scale
        rows.append({**d, "pts": pts, "box": ev.box_of(pts), "origin": "roi", "crop_id": crop_id})
    return rows


def merge(dets: list[dict], conf: float, threshold: float = 0.5) -> list[dict]:
    """与现有分块入口一样按几何和置信度去重，跨类别也去重。"""
    kept = []
    for d in sorted((p for p in dets if p["conf"] >= conf), key=lambda p: -p["conf"]):
        if all(ev.iou(d["box"], k["box"]) <= threshold for k in kept):
            kept.append(d)
    return kept


def choose_controls(rows: list[dict], count: int, seed: int) -> set[str]:
    """按类别和原始输入尺寸分层轮选；先分散图像，再补足样本数。"""
    groups = defaultdict(list)
    for row in rows:
        if row["full"]["state"] == "correct":
            groups[(row["gt_class"], int(row["width_net"] // 2))].append(row)
    for group in groups.values():
        group.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    selected, image_counts = set(), Counter()
    for cap in (1, 2, count):
        changed = True
        while changed and len(selected) < count:
            changed = False
            for key in sorted(groups):
                candidate = next((r for r in groups[key] if r["id"] not in selected and image_counts[r["image"]] < cap), None)
                if candidate is not None:
                    selected.add(candidate["id"])
                    image_counts[candidate["image"]] += 1
                    changed = True
                if len(selected) >= count:
                    break
    return selected


def metrics(rows: list[dict], route: str) -> dict:
    counts = Counter(r[route]["state"] for r in rows)
    n, correct = len(rows), counts["correct"]
    hits = n - counts["unmatched"]
    errors = [e for r in rows for e in r[route]["corner_errors_orig_px"]]
    flags = Counter(f for r in rows for f in r[route]["error_flags"])
    return {"n": n, "hits": hits, "correct": correct, "unmatched": counts["unmatched"],
            "wrong_class": counts["wrong_class"], "recall": ev._ratio(hits, n),
            "class_acc": ev._ratio(correct, hits), "end_to_end_correct": ev._ratio(correct, n),
            "corner_p95_orig_px": ev.pct(errors, 95), "error_flags": dict(flags)}


def add_merge_control(metas, predictions, targets, frames, conf, spec):
    """只用已有预测补做全图去重对照，区分裁剪和后处理对误检的影响。"""
    target_groups = defaultdict(list)
    for row in targets:
        target_groups[row["image"]].append(row)
    frame_map = {frame["image"]: frame for frame in frames}
    for meta, dets in zip(metas, predictions):
        image = meta["image"]
        if image not in target_groups:
            continue
        gts = ground_truth(meta)
        merged = merge(dets, conf)
        states = outcomes(merged, gts, conf, spec, meta["scale"])
        for row in target_groups[image]:
            row["full_merge_only"] = states[row["armor_index"]]
        hits = sum(gi is not None for _, gi in ev.match(merged, gts, False))
        frame_map[image]["full_merge_only_false_positives"] = len(merged) - hits
    return {"metrics": metrics(targets, "full_merge_only"),
            "false_positives_on_audit_frames": sum(f["full_merge_only_false_positives"] for f in frames)}


class Predictor:
    def __init__(self, weights: Path, device: str):
        from nt import offline
        offline.apply()
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.torch, self.device = torch, device
        self.model = YOLO(str(weights), task="pose")
        self.gpu = device != "cpu"

    def sync(self):
        if self.gpu:
            self.torch.cuda.synchronize(self.device if self.device.startswith("cuda") else int(self.device))

    def predict(self, images) -> list[list[dict]]:
        results = self.model.predict(source=images, imgsz=768, conf=0.001, max_det=30, device=self.device,
                                     verbose=False, end2end=True, half=False)
        output = []
        for result in results:
            ds = []
            if result.boxes is not None and len(result.boxes):
                for conf, cls, pts in zip(result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int),
                                          result.keypoints.xy.cpu().numpy().astype(np.float64)):
                    ds.append({"conf": float(conf), "cls": int(cls), "pts": pts, "box": ev.box_of(pts), "origin": "full"})
            output.append(ds)
        return output

    def timed(self, native, roi=None):
        self.sync()
        start = time.perf_counter()
        if roi is not None:
            x, y, w, h = roi
            native = native[y:y + h, x:x + w]
        net = cv2.resize(native, (768, 576), interpolation=cv2.INTER_LINEAR)
        ds = self.predict([net])[0]
        self.sync()
        return ds, net, (time.perf_counter() - start) * 1000


def run(args) -> None:
    spec = LabelSpec.load()
    data = args.data.resolve()
    manifest = json.loads((data / "bake_manifest.json").read_text())
    cfg = manifest["config"]
    if cfg["size"] != [768, 576] or manifest["label_spec_sha256"] != spec.sha256:
        raise ValueError("本实验要求 768×576 输入和当前标签规范")
    if cfg["lens"]["prob"] != 0:
        raise ValueError("当前审计要求无额外镜头形变的烘焙输入，防止原图展示坐标错配")
    if sha(data / "meta.jsonl") != manifest["meta_sha256"]:
        raise ValueError("烘焙元数据哈希不一致")
    reference = json.loads(args.reference.read_text())
    stamp = json.loads(args.reference.with_suffix(".inputs.json").read_text())
    if (stamp["weights_sha256"] != sha(args.weights) or stamp["bake_sha256"] != sha(data / "bake_manifest.json")
            or stamp["head"] != "e2e" or stamp["conf"] != args.conf):
        raise ValueError("权重、数据、输出头或阈值与历史评估不同")
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out
    fingerprint = {"weights": str(args.weights.resolve()), "weights_sha256": stamp["weights_sha256"],
                   "data": str(data), "bake_sha256": stamp["bake_sha256"], "meta_sha256": manifest["meta_sha256"],
                   "reference": str(args.reference.resolve()), "reference_sha256": sha(args.reference),
                   "label_spec_sha256": spec.sha256, "parameters": vars(args),
                   "code_sha256": {name: sha(ROOT / "nt" / name) for name in ("tiny.py", "tiny_report.py", "bake.py", "eval.py")}}
    write_json(out / "inputs.json", fingerprint)
    write_json(out / "status.json", {"stage": "baseline", "complete": False})
    cv2.setNumThreads(1)
    metas = ev.load_ground_truth(data, "val")
    predictor = Predictor(args.weights, args.device)
    predictions = []
    for start in range(0, len(metas), args.batch):
        chunk = [str(data / m["image"]) for m in metas[start:start + args.batch]]
        predictions.extend(predictor.predict(chunk))
        if start % (args.batch * 5) == 0:
            print(f"历史输入推理：{min(start + args.batch, len(metas))}/{len(metas)}", flush=True)
    write_jsonl(out / "baseline_predictions.jsonl", ({"image": m["image"], "detections": d} for m, d in zip(metas, predictions)))
    by_image, targets = {}, []
    for meta, dets in zip(metas, predictions):
        gts = ground_truth(meta)
        states = outcomes(dets, gts, args.conf, spec, meta["scale"])
        small = []
        for gi, (gt, state) in enumerate(zip(gts, states)):
            width = float(gt["box"][2] - gt["box"][0])
            if width >= 8:
                continue
            row = {"id": f"{meta['sample']}_c{meta['copy']}_a{gi:02d}", "image": meta["image"],
                   "sample": meta["sample"], "armor_index": gi, "gt_class": gt["cls"],
                   "gt_name": spec.class_name(gt["cls"]), "width_net": width,
                   "width_orig": width / meta["scale"], "historical_roi_mode": meta["roi_mode"],
                   "historical_roi": meta["roi"], "scale": meta["scale"], "corners_net": gt["corners_net"],
                   "corners_orig": gt["corners_orig"], "vis": gt["vis"],
                   "yaw_deg": gt.get("yaw_deg"), "occ": gt.get("occ"), "effects": meta["effects"], "full": state}
            targets.append(row)
            small.append(row)
        if small:
            by_image[meta["image"]] = (meta, dets, gts, small)
    reproduced = metrics(targets, "full")
    expected = reference["heads"]["e2e"]["by_width_px"]["[0,8)"]
    write_json(out / "baseline_check.json", {"expected": expected, "actual": reproduced})
    if any(reproduced[k] != expected[k] for k in ("n", "hits", "correct")):
        raise ValueError("历史小目标数量未复现，请检查 baseline_check.json，不能沿用原来的失败清单")
    controls = choose_controls(targets, args.controls, args.seed)
    selected = set()
    for row in targets:
        row["group"] = "failure" if row["full"]["state"] != "correct" else ("control" if row["id"] in controls else "not_exported")
        if row["group"] != "not_exported":
            selected.add(row["id"])
    print(f"已复现：{len(targets)} 个小目标，失败 {reproduced['n'] - reproduced['correct']}，正确对照 {len(controls)}", flush=True)

    sources = {}
    needed = {m[0]["source"] for m in by_image.values()}
    fingerprint["sources"] = {}
    for entry in manifest["sources"]:
        if entry["name"] not in needed:
            continue
        root = Path(entry["path"])
        root = root if root.is_absolute() else ROOT / root
        if sha(root / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError(f"源数据清单已变更：{root}")
        samples = {s.id: s for s in read_samples(root / "labels.jsonl")}
        sources[entry["name"]] = (root, samples)
        fingerprint["sources"][entry["name"]] = {"root": str(root), "manifest_sha256": entry["manifest_sha256"],
                                                    "labels_sha256": sha(root / "labels.jsonl")}
    fingerprint["runtime"] = {"torch": predictor.torch.__version__, "opencv": cv2.__version__,
                               "numpy": np.__version__, "precision": "fp32",
                               "gpu": predictor.torch.cuda.get_device_name(int(args.device)) if predictor.gpu else None}
    write_json(out / "inputs.json", fingerprint)
    # 用同一输入尺寸预热单帧路径，后续时间不含载盘、图像重建或报告生成。
    warm = cv2.imread(str(data / next(iter(by_image))), cv2.IMREAD_COLOR)
    for _ in range(10):
        predictor.predict([warm])
    predictor.sync()
    frames, cases, pixel_checks = [], [], []
    from nt.tiny_report import save_case, write_report
    for fi, (image, (meta, base_dets, gts, small)) in enumerate(by_image.items()):
        root, samples = sources[meta["source"]]
        sample = samples[meta["sample"]]
        task = {"sample": sample_to_dict(sample), "root": str(root), "seed": manifest["seed"], "copy": meta["copy"]}
        view = bake.prepare_view(task, spec, cfg)
        if "error" in view:
            raise ValueError(view["error"])
        native = view["native"]
        baked = cv2.imread(str(data / image), cv2.IMREAD_COLOR)
        rebuilt = cv2.resize(native, (768, 576), interpolation=cv2.INTER_LINEAR)
        if (list(view["roi"]) != meta["roi"] or view["effects"] != meta["effects"]
                or not np.array_equal(rebuilt, baked)):
            raise ValueError(f"原始分辨率重建不一致：{image}")
        pixel_checks.append({"image": image, "different_values": 0,
                             "source_png_sha256": sha(root / sample.image),
                             "baked_png_sha256": sha(data / image),
                             "native_pixels_sha256": hashlib.sha256(native.tobytes()).hexdigest()})
        # 单帧结果必须保持历史分类/匹配结论，才可以把单帧时延与配对结果一起使用。
        timed_full, _, full_ms = predictor.timed(native)
        single_states = outcomes(timed_full, gts, args.conf, spec, meta["scale"])
        for row in small:
            s = single_states[row["armor_index"]]
            old = row["full"]
            if (s["state"], s["prediction"]["cls"] if s["prediction"] else None) != (old["state"], old["prediction"]["cls"] if old["prediction"] else None):
                raise ValueError(f"单帧与历史批量输出结论不同：{row['id']}")
        start = time.perf_counter()
        rois = proposal_rois(base_dets, native.shape, meta["scale"], args.seed_conf, args.proposal_width,
                            args.max_rois, args.roi_min, args.roi_multiple)
        deployed = [{**d, "origin": "full"} for d in base_dets]
        roi_images, roi_predictions = [], []
        for ci, roi in enumerate(rois):
            ds, net, _ = predictor.timed(native, roi)
            roi_images.append(net)
            mapped = map_crop(ds, roi, meta["scale"], crop_id=ci)
            roi_predictions.append(mapped)
            deployed.extend(mapped)
        deployed = merge(deployed, args.conf) if rois else [d for d in deployed if d["conf"] >= args.conf]
        deploy_ms = full_ms + (time.perf_counter() - start) * 1000
        deploy_states = outcomes(deployed, gts, args.conf, spec, meta["scale"])
        matched_full = sum(gi is not None for _, gi in ev.match([d for d in base_dets if d["conf"] >= args.conf], gts, False))
        matched_deploy = sum(gi is not None for _, gi in ev.match(deployed, gts, False))
        frame = {"image": image, "roi_mode": meta["roi_mode"], "full_ms": full_ms, "detector_roi_ms": deploy_ms,
                 "inference_calls": 1 + len(rois), "rois_native": rois, "roi_predictions": roi_predictions,
                 "merged_predictions": deployed,
                 "full_false_positives": sum(d["conf"] >= args.conf for d in base_dets) - matched_full,
                 "detector_roi_false_positives": len(deployed) - matched_deploy}
        frames.append(frame)
        for row in small:
            gi = row["armor_index"]
            native_box = gts[gi]["box"] / meta["scale"]
            oracle = roi_for(native_box, native.shape[1], native.shape[0], args.roi_min, args.roi_multiple)
            ds, oracle_net, oracle_ms = predictor.timed(native, oracle)
            mapped = map_crop(ds, oracle, meta["scale"])
            row["oracle_roi"] = outcomes(mapped, gts, args.conf, spec, meta["scale"])[gi]
            row["oracle_roi"].update(roi_native=oracle, latency_ms=oracle_ms)
            row["detector_roi"] = dict(deploy_states[gi])
            covering = [ci for ci, roi in enumerate(rois) if bake._inside_fraction(native_box, roi) >= 0.999]
            row["detector_roi"].update(covered=bool(covering), covering_roi_indices=covering)
            row["review"] = {"status": "pending", "original_readability": "unknown",
                              "causes": [], "notes": "模型改善不自动等同于人工可辨认或缩放损失"}
            if row["id"] in selected:
                ci = covering[0] if covering else None
                cases.append(save_case(out, row, view, baked, oracle_net, oracle,
                                       roi_images[ci] if ci is not None else None, rois[ci] if ci is not None else None))
        if (fi + 1) % 10 == 0 or fi + 1 == len(by_image):
            print(f"配对评估与导图：{fi + 1}/{len(by_image)} 帧，{len(cases)} 个案例", flush=True)
            write_json(out / "status.json", {"stage": "paired", "frames_done": fi + 1, "frames_total": len(by_image), "complete": False})
    merge_control = add_merge_control(metas, predictions, targets, frames, args.conf, spec)
    write_jsonl(out / "targets.jsonl", targets)
    write_jsonl(out / "frames.jsonl", frames)
    write_jsonl(out / "pixel_checks.jsonl", pixel_checks)
    report = {"full_merge_only_control": merge_control,
              "all_tiny": {r: metrics(targets, r) for r in ROUTES},
              "full_frame_only": {r: metrics([t for t in targets if t["historical_roi_mode"] == "full"], r) for r in ROUTES},
              "historical_roi_only": {r: metrics([t for t in targets if t["historical_roi_mode"] != "full"], r) for r in ROUTES},
              "digits_134": {r: metrics([t for t in targets if spec.from_class_id(t["gt_class"])[1] in (1, 3, 4)], r) for r in ROUTES},
              "transitions": {r: dict(Counter(t["full"]["state"] + " -> " + t[r]["state"] for t in targets)) for r in ROUTES[1:]},
              "roi_coverage": {"n": len(targets), "covered": sum(t["detector_roi"]["covered"] for t in targets)},
              "false_positives_on_audit_frames": {r: sum(f[r + "_false_positives"] for f in frames) for r in ("full", "detector_roi")},
              "timing_ms": {r: {"n": len(frames), "p50": ev.pct([f[r + "_ms"] for f in frames], 50),
                                  "p95": ev.pct([f[r + "_ms"] for f in frames], 95),
                                  "p99": ev.pct([f[r + "_ms"] for f in frames], 99)} for r in ("full", "detector_roi")},
              "oracle_per_target_ms": {"p50": ev.pct([t["oracle_roi"]["latency_ms"] for t in targets], 50),
                                         "p95": ev.pct([t["oracle_roi"]["latency_ms"] for t in targets], 95)},
              "inference_calls": dict(Counter(f["inference_calls"] for f in frames)),
              "cases": dict(Counter(c["group"] for c in cases)), "frames_pixel_identical": len(pixel_checks),
              "reacquisition_measured": False, "human_readability_reviewed": False}
    write_json(out / "summary.json", report)
    write_json(out / "cases.json", cases)
    write_report(out, report, cases, fingerprint, targets)
    write_json(out / "status.json", {"stage": "complete", "complete": True, "targets": len(targets), "cases": len(cases)})
    print(json.dumps(report["all_tiny"], ensure_ascii=False), flush=True)


def refresh_report(out: Path) -> None:
    """复用推理结果更新后处理对照和报告，不重跑 GPU、不改变原始实验指纹。"""
    from nt.tiny_report import write_report
    inputs = json.loads((out / "inputs.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    targets = [json.loads(line) for line in (out / "targets.jsonl").read_text().splitlines()]
    frames = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    cases = json.loads((out / "cases.json").read_text())
    if sha(Path(inputs["data"]) / "meta.jsonl") != inputs["meta_sha256"]:
        raise ValueError("原始烘焙元数据已变化，拒绝更新报告")
    metas = ev.load_ground_truth(Path(inputs["data"]), "val")
    prediction_rows = [json.loads(line) for line in (out / "baseline_predictions.jsonl").read_text().splitlines()]
    if [m["image"] for m in metas] != [p["image"] for p in prediction_rows]:
        raise ValueError("预测与元数据顺序不一致")
    summary["full_merge_only_control"] = add_merge_control(metas, [p["detections"] for p in prediction_rows],
                                                         targets, frames, inputs["parameters"]["conf"], LabelSpec.load())
    write_jsonl(out / "targets.jsonl", targets)
    write_jsonl(out / "frames.jsonl", frames)
    write_json(out / "summary.json", summary)
    write_report(out, summary, cases, inputs, targets)
    write_json(out / "report_refresh.json", {"reused_predictions": True,
               "code_sha256": {name: sha(ROOT / "nt" / name) for name in ("tiny.py", "tiny_report.py")}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "baked/style_b_v1")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/train/b_from_a_e100_s0/weights/last.pt")
    ap.add_argument("--reference", type=Path, default=ROOT / "runs/experiments/style_b_v1_s0/b_from_a_e100_s0_final.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report-only", action="store_true", help="复用已保存预测更新报告，不运行模型")
    ap.add_argument("--device", default="0", help="CUDA 设备编号或 cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--controls", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--seed-conf", type=float, default=0.1)
    ap.add_argument("--proposal-width", type=float, default=16)
    ap.add_argument("--max-rois", type=int, default=4)
    ap.add_argument("--roi-min", type=int, default=384)
    ap.add_argument("--roi-multiple", type=float, default=16)
    args = ap.parse_args()
    if (args.batch < 1 or args.controls < 0 or args.max_rois < 1 or args.roi_min < 4
            or not 0 < args.seed_conf <= args.conf <= 1 or not math.isfinite(args.roi_multiple) or args.roi_multiple <= 0
            or not math.isfinite(args.proposal_width) or args.proposal_width <= 0):
        ap.error("采样数、阈值或 ROI 参数无效")
    if args.report_only:
        refresh_report(args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
