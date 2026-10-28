"""Small synthetic inputs for the M6 tests: frames, a session, and ONNX models with a fixed output.

The fake models ignore the image (the output is a constant plus 0 * sum(input), so the input stays wired) and
carry either praysky-style names metadata or an nt_contract, which is all nt.detector looks at.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NET_W, NET_H = 128, 96  # model input; frames are 256x192, so the letterbox scale is exactly 0.5
FRAME_W, FRAME_H = 256, 192


def write_frames(folder: Path, n: int, seed: int = 0) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        img = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        p = folder / f"img{i:03d}.png"
        cv2.imwrite(str(p), img)
        out.append(p)
    return out


def det_row(conf: float, cls: int, color_slot: int, corners_net) -> np.ndarray:
    r = np.zeros(18, np.float32)
    xs, ys = [c[0] for c in corners_net], [c[1] for c in corners_net]
    r[:4] = [min(xs), min(ys), max(xs), max(ys)]
    r[4], r[5] = conf, cls
    r[6 + color_slot] = 1.0
    r[10:18] = np.array(corners_net, np.float32).ravel()
    return r


# A plate in network pixels (LT, LB, RB, RT) and the same plate in frame pixels.
PLATE_NET = [(40.0, 30.0), (40.0, 42.0), (70.0, 43.0), (70.0, 31.0)]
PLATE_FRAME = [(x * 2, y * 2) for x, y in PLATE_NET]
SECOND_NET = [(90.0, 60.0), (90.0, 70.0), (115.0, 70.0), (115.0, 60.0)]


def fake_model(path: Path, rows: list[np.ndarray], kind: str = "praysky") -> Path:
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh

    out = np.zeros((1, 30, 18), np.float32)
    for i, r in enumerate(rows):
        out[0, i] = r
    nodes = [
        h.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        h.make_node("Mul", ["s", "zero"], ["z"]),
        h.make_node("Add", ["const", "z"], ["output0"]),
    ]
    graph = h.make_graph(
        nodes, "fake",
        [h.make_tensor_value_info("images", T.FLOAT, [1, 3, NET_H, NET_W])],
        [h.make_tensor_value_info("output0", T.FLOAT, [1, 30, 18])],
        [nh.from_array(out, "const"), nh.from_array(np.array(0.0, np.float32), "zero")],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 8
    meta = model.metadata_props.add()
    if kind == "praysky":
        meta.key, meta.value = "names", "{0: 's0_o0', 1: 's0_o3', 2: 's1_o1'}"
    else:
        from nt.export import contract_for
        from nt.spec import LabelSpec

        meta.key, meta.value = "nt_contract", json.dumps(contract_for(LabelSpec.load()))
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path
