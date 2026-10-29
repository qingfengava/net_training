"""Run an exported armor ONNX model on full-resolution frames, for pre-labeling, mining and replay. No training.

Two output layouts are understood, both [1, 30, 18] = box4, conf, class, 4 color scores, 8 keypoint coordinates:
    nt_contract   our own export (nt/export.py): the class column is the label id, colors are one-hot slots.
    praysky       the AT_NN_Detector models in sleepy/model/ (12 classes named s{size}_o{object}, separate
                  4-way color branch). Old light-bar armor only, useful to dry-run the tooling on old footage.
                  Their size-1 classes are RM26 large plates, which RM27 no longer has; they come out with
                  type None.
Keypoints are taken in the network's order (LT, LB, RB, RT); nothing is re-sorted.

Preprocessing is sleepy's CPU path: letterbox with cv2.INTER_LINEAR and 114 padding, RGB, scaled to [0, 1].
That is also what nt/bake.py feeds the network during training.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from nt.geometry import bbox, bbox_iou

PRAYSKY_NAME = re.compile(r"^s(\d)_o(\d)$")


@dataclass
class Detection:
    conf: float
    color: int  # color slot, as in configs/label_spec.yaml
    label: int
    corners: np.ndarray  # (4, 2) LT, LB, RB, RT in original-image pixels, edge convention
    type: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return bbox(self.corners.tolist())


def letterbox(img: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, float, int, int]:
    """sleepy's utils::letterbox: scale to fit, pad with 114, centered. Returns image, scale, pad left, pad top."""
    h, w = img.shape[:2]
    s = min(out_w / w, out_h / h)
    rw, rh = int(w * s + 0.5), int(h * s + 0.5)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pl, pt = (out_w - rw) // 2, (out_h - rh) // 2
    if (rw, rh) == (out_w, out_h):
        return resized, s, 0, 0
    return cv2.copyMakeBorder(resized, pt, out_h - rh - pt, pl, out_w - rw - pl, cv2.BORDER_CONSTANT, value=(114, 114, 114)), s, pl, pt


class ArmorModel:
    def __init__(self, path: Path, device: str = "cpu", threads: int = 4) -> None:
        import onnx
        import onnxruntime as ort

        self.path = Path(path)
        meta = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), so, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        _, c, self.in_h, self.in_w = inp.shape
        if c != 3 or not isinstance(self.in_h, int) or not isinstance(self.in_w, int):
            raise ValueError(f"{path}: expected a static [1,3,H,W] input, got {inp.shape}")
        out = self.sess.get_outputs()[0]
        if list(out.shape[1:]) != [30, 18] and list(out.shape[-1:]) != [18]:
            raise ValueError(f"{path}: expected an [1,N,18] output, got {out.shape}")
        if "nt_contract" in meta:
            self.kind = "nt_contract"
            contract = json.loads(meta["nt_contract"])
            self.contract = contract
            self.label_ids = {int(k) for k in contract["numbers"]}
            self.classes = None
        elif "names" in meta:
            names = _parse_names(meta["names"])
            parsed = {i: PRAYSKY_NAME.match(n) for i, n in names.items()}
            if not names or not all(parsed.values()):
                raise ValueError(f"{path}: class names {names} are neither nt_contract nor praysky s<size>_o<object>")
            self.kind = "praysky"
            self.contract = None
            # size 0 is the small plate, 1 the RM26 large one (no such type in RM27, so no type is claimed);
            # the object id uses the same order as label_spec labels.
            self.classes = {i: ("small" if m.group(1) == "0" else None, int(m.group(2))) for i, m in parsed.items()}
        else:
            raise ValueError(f"{path}: no nt_contract or names metadata, cannot tell how to decode it")
        self.sha256 = _sha256(self.path)

    def describe(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256, "kind": self.kind, "input": [self.in_h, self.in_w]}

    def _run(self, img: np.ndarray, conf: float) -> list[Detection]:
        net, s, pl, pt = letterbox(img, self.in_w, self.in_h)
        x = cv2.cvtColor(net, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        rows = self.sess.run(None, {self.input_name: x})[0].reshape(-1, 18)
        dets = []
        for r in rows:
            c = float(r[4])
            if not np.isfinite(c) or c < conf:
                continue
            cid = int(r[5])
            color = int(np.argmax(r[6:10]))
            if self.kind == "nt_contract":
                if cid not in self.label_ids:
                    continue
                label, typ = cid, None
            else:
                if cid not in self.classes:
                    continue
                typ, label = self.classes[cid]
            k = r[10:18].astype(np.float64).reshape(4, 2)
            corners = np.stack([(k[:, 0] - pl) / s, (k[:, 1] - pt) / s], axis=1)
            if not np.isfinite(corners).all():
                continue
            dets.append(Detection(c, color, label, corners, typ, {"color_scores": [round(float(v), 4) for v in r[6:10]]}))
        return dets

    def detect(self, img: np.ndarray, conf: float = 0.25, tiles: int = 0, merge_iou: float = 0.5) -> list[Detection]:
        """Detections on a full frame. tiles=2 also runs a 2x2 grid of overlapping crops at the model's aspect
        ratio, which roughly doubles the pixels on small, far armors, and merges by corner-box IoU."""
        dets = self._run(img, conf)
        if tiles:
            h, w = img.shape[:2]
            tw = int(round(w / tiles * 1.25))
            th = int(round(tw * self.in_h / self.in_w))
            th = min(th, h)
            for gy in range(tiles):
                for gx in range(tiles):
                    x0 = int(round((w - tw) * gx / max(tiles - 1, 1)))
                    y0 = int(round((h - th) * gy / max(tiles - 1, 1)))
                    for d in self._run(img[y0:y0 + th, x0:x0 + tw], conf):
                        d.corners = d.corners + np.array([x0, y0], np.float64)
                        d.extra["tile"] = [x0, y0, tw, th]
                        dets.append(d)
            dets = nms(dets, merge_iou)
        return dets


def nms(dets: list[Detection], iou_thr: float) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(bbox_iou(d.box, k.box) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def _parse_names(text: str) -> dict[int, str]:
    """Ultralytics writes names as a Python dict literal, e.g. {0: 's0_o0', 1: 's0_o2'}."""
    import ast

    d = ast.literal_eval(text)
    return {int(k): str(v) for k, v in d.items()}


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
