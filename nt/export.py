"""Export a trained pose model to the ONNX contract sleepy's AT decoder reads.

    python -m nt.export --weights runs/train/NAME/weights/best.pt --out exports/NAME [--opset 17] [--heads]
    python -m nt.export --split exports/NAME            (add the head split to an existing export)

Output tensor [1, 30, 18], one row per detection:
    0..3   box x1, y1, x2, y2 in network-input pixels
    4      confidence in [0, 1]
    5      armor number id, float, from label_spec: 0 sentry, 1, 3, 4, 6 outpost, 7 base (sleepy's ArmorClass ids;
           RM27 has no robot 2 or 5), or 8 (UNKNOWN) for a dark screen
    6..9   color scores, one-hot over [blue, red, gray, purple]; a dark screen is gray (sleepy: NONE)
    10..17 keypoints x, y for LT, LB, RB, RT in network-input pixels

Training uses merged classes (label_spec class table: training colors x labels, then the off class). Ultralytics
exports the stock end-to-end graph (or NMS inside the graph for non-end-to-end models) as box4, conf, cls, then
(x, y, visibility) per keypoint; this module appends a few operators that look the merged class up in the class
table (number and color slot) and drop the visibility columns, so the
decoder needs no change and the model keeps one export path. A contract.json is written next to the model
and embedded in the ONNX metadata under `nt_contract`; nt.contract checks both.

--heads (or --split on an existing export) also cuts model.onnx at the 9 detection-head convolutions, for NPUs
that quantize every tensor (RKNN): model_heads.onnx runs from the image to the 9 head outputs (box, class and
keypoint maps per stride, each its own tensor, so each gets its own quantization scale), and decode.onnx runs
from those 9 tensors to the same [1, 30, 18] output (grid decode, sigmoid, top-k, class remap) on the CPU in
float. In a quantized model whose head outputs carry a Q/DQ pair (nt.quant rknn, nt.train --qat rknn) the cut
is after that pair, so the 8-bit scale stays with the NPU part and decode.onnx has no Q/DQ.
decode(heads(x)) equals model(x); the split is checked on export and described in split.json (head
tensor names, kind, stride, shape) and in each file's `nt_split` metadata. docs/rknn_int8_qat_notes.md explains why the cut belongs there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from nt.spec import LabelSpec

CONTRACT_VERSION = 2  # 2: off class (number 8, gray) and the class table in the contract
MAX_DET = 30
IMG_H, IMG_W = 576, 768
N_KPT = 4
STOCK_COLS = 6 + N_KPT * 3
OUT_COLS = 6 + 4 + N_KPT * 2


def contract_for(spec: LabelSpec, source: dict | None = None) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "task": "armor",
        "input": {
            "name": "images", "shape": [1, 3, IMG_H, IMG_W], "dtype": "float32", "layout": "NCHW",
            "color_order": "RGB", "range": [0.0, 1.0],
            "resize": "bilinear, no antialias; 4:3 inputs need no padding",
            "pixel_convention": spec.pixel_convention,
        },
        "output": {
            "name": "output0", "shape": [1, MAX_DET, OUT_COLS], "dtype": "float32",
            "columns": {"box": [0, 4], "conf": 4, "number": 5, "color_scores": [6, 10], "keypoints_xy": [10, 18]},
            "coordinates": "network-input pixels",
            "keypoint_order": list(spec.corner_order),
        },
        "numbers": {str(k): spec.number_name(k) for k in spec.number_ids},
        "color_slots": {str(k): v for k, v in sorted(spec.colors.items())},
        "training_colors": list(spec.training_colors),
        "off_class": None if spec.off_class is None else {"color": spec.off_class[0], "number": spec.off_class[1]},
        "classes": [list(c) for c in spec.classes],
        "label_spec_sha256": spec.sha256,
        "corner_definition": spec.corner_definition,
        "source": source or {},
    }


def append_remap(model, spec: LabelSpec):
    """Append the column remap to a stock [1, N, 18] graph and return the new ModelProto."""
    import onnx
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh

    g = model.graph
    # The stock output is called output0 as well; give it another name so the remapped tensor can take it.
    stock = "stock_output"
    old_name = g.output[0].name
    for node in g.node:
        for i, name in enumerate(node.output):
            if name == old_name:
                node.output[i] = stock
    class_numbers = np.array([float(n) for _, n in spec.classes], np.float32)
    class_slots = np.array([float(c) for c, _ in spec.classes], np.float32)
    nodes, inits = [], []

    def const(name, arr):
        inits.append(nh.from_array(np.asarray(arr), name))
        return name

    def slice_cols(src, lo, hi, out):
        nodes.append(h.make_node("Slice", [src, const(f"{out}_lo", np.array([lo], np.int64)),
                                           const(f"{out}_hi", np.array([hi], np.int64)),
                                           const(f"{out}_ax", np.array([2], np.int64))], [out]))
        return out

    slice_cols(stock, 0, 5, "p_box_conf")                     # box4 + conf
    slice_cols(stock, 5, 6, "p_cls")                          # merged class id as float
    # merged class -> (number, color slot) through the class table; the off class has no number of its own
    nodes.append(h.make_node("Cast", ["p_cls"], ["p_cls_i"], to=T.INT64))
    nodes.append(h.make_node("Squeeze", ["p_cls_i", const("c_ax2", np.array([2], np.int64))], ["p_cls_v"]))
    nodes.append(h.make_node("Gather", [const("c_numbers", class_numbers), "p_cls_v"], ["p_num_v"], axis=0))
    nodes.append(h.make_node("Unsqueeze", ["p_num_v", "c_ax2"], ["p_number"]))
    nodes.append(h.make_node("Gather", [const("c_slots", class_slots), "p_cls_v"], ["p_slot"], axis=0))
    nodes.append(h.make_node("Unsqueeze", ["p_slot", "c_ax2"], ["p_slot_u"]))
    nodes.append(h.make_node("Equal", ["p_slot_u", const("c_range", np.arange(len(spec.colors), dtype=np.float32))], ["p_eq"]))
    nodes.append(h.make_node("Cast", ["p_eq"], ["p_color"], to=T.FLOAT))
    kp = []
    for i in range(N_KPT):                                    # x, y of each keypoint; the visibility column is dropped
        kp.append(slice_cols(stock, 6 + 3 * i, 8 + 3 * i, f"p_kp{i}"))
    nodes.append(h.make_node("Concat", ["p_box_conf", "p_number", "p_color", *kp], ["output0"], axis=2))

    out_info = h.make_tensor_value_info("output0", T.FLOAT, [1, MAX_DET, OUT_COLS])
    del g.output[:]
    g.output.append(out_info)
    g.node.extend(nodes)
    g.initializer.extend(inits)
    onnx.checker.check_model(model)
    return model


def _is_quantized_conv_output(n, producer: dict) -> bool:
    """DequantizeLinear(QuantizeLinear(Conv output)): a head output that the model rounds to 8 bits."""
    q = producer.get(n.input[0]) if n.op_type == "DequantizeLinear" else None
    return q is not None and q.op_type == "QuantizeLinear" and getattr(producer.get(q.input[0]), "op_type", None) == "Conv"


def head_tensors(model) -> list[str]:
    """The detection-head outputs: the last Conv outputs before the graph output (no Conv after them). When the
    model quantizes such an output (rknn PTQ or QAT), the cut goes after its Q/DQ pair, so the NPU part keeps the
    output's 8-bit scale."""
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    heads, stack, seen = [], [g.output[0].name], set()
    while stack:
        t = stack.pop()
        if t in seen or t not in producer:
            continue
        seen.add(t)
        n = producer[t]
        if n.op_type == "Conv" or _is_quantized_conv_output(n, producer):
            heads.append(t)
        else:
            stack.extend(n.input)
    return sorted(heads)


def _head_info(name: str, shape: list[int]) -> dict:
    kind = "box" if "cv2" in name else "cls" if "cv3" in name else "kpt" if "cv4" in name else "?"
    return {"name": name, "kind": kind, "stride": IMG_H // shape[2] if len(shape) == 4 and shape[2] else None, "shape": shape}


def split_at_heads(model_path: Path) -> dict:
    """Write model_heads.onnx and decode.onnx next to model_path, check them, and record the split."""
    import onnx
    import onnxruntime as ort

    out_dir = model_path.parent
    model = onnx.load(str(model_path))
    heads = head_tensors(model)
    if len(heads) != 9:
        raise RuntimeError(f"expected 9 head convolutions, found {len(heads)}: {heads}")
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in inferred.graph.value_info}
    in_name, out_name = model.graph.input[0].name, model.graph.output[0].name
    heads_path, decode_path = out_dir / "model_heads.onnx", out_dir / "decode.onnx"
    onnx.utils.extract_model(str(model_path), str(heads_path), [in_name], heads)
    onnx.utils.extract_model(str(model_path), str(decode_path), heads, [out_name])

    rng = np.random.default_rng(0)
    full = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    npu = ort.InferenceSession(str(heads_path), providers=["CPUExecutionProvider"])
    cpu = ort.InferenceSession(str(decode_path), providers=["CPUExecutionProvider"])
    for _ in range(2):
        x = rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32)
        h = npu.run(None, {in_name: x})
        y = cpu.run(None, dict(zip(heads, h)))[0]
        ref = full.run(None, {in_name: x})[0]
        if not np.allclose(y, ref, atol=1e-4):
            raise RuntimeError(f"decode(heads(x)) differs from the model by {float(np.abs(y - ref).max())}")
    split = {"npu": heads_path.name, "cpu": decode_path.name, "heads": [_head_info(n, shapes.get(n, [])) for n in heads],
             "note": "run npu on the image, feed its 9 outputs by name to cpu; cpu output == model.onnx output"}
    for path in (heads_path, decode_path):
        m = onnx.load(str(path))
        entry = m.metadata_props.add()
        entry.key, entry.value = "nt_split", json.dumps({"part": path.stem, **split}, sort_keys=True)
        onnx.save(m, str(path))
    # beside contract.json, not inside it: the contract describes model.onnx and must match its embedded copy
    (out_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    return split


def export(weights: Path, out_dir: Path, spec: LabelSpec, opset: int = 17, device: str = "cpu", heads: bool = False,
           simplify: bool | None = None) -> Path:
    """simplify None: onnxslim on, unless the model carries fake-quant modules (QAT), whose weight Q/DQ it would fold."""
    import onnx
    from nt import offline

    offline.apply()
    from ultralytics import YOLO

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; exports are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = _export(weights, out_dir, spec, opset, device, simplify)
        if heads:
            split_at_heads(path)
        return path
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)  # a failed export leaves nothing half-written
        raise


def _export(weights: Path, out_dir: Path, spec: LabelSpec, opset: int, device: str, simplify: bool | None = None) -> Path:
    import onnx
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as tmp:
        # Ultralytics writes the ONNX next to a .pt checkpoint, or to the working directory for yaml-built
        # models. A private copy keeps both cases in tmp, also for a relative --weights path after the chdir.
        local = Path(tmp) / Path(weights).name
        shutil.copy2(weights, local)
        model = YOLO(str(local), task="pose")
        e2e = bool(getattr(model.model, "end2end", False))
        from nt import qat

        quantized = qat.is_quantized(model.model)
        if simplify is None:
            simplify = not quantized
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            stock = Path(model.export(format="onnx", imgsz=[IMG_H, IMG_W], batch=1, half=False, dynamic=False,
                                      simplify=simplify, opset=opset, max_det=MAX_DET, nms=not e2e, device=device, verbose=False))
        finally:
            os.chdir(cwd)
        stock = Path(tmp) / stock.name if not stock.is_absolute() else stock
        proto = onnx.load(str(stock))
    shape = [d.dim_value for d in proto.graph.output[0].type.tensor_type.shape.dim]
    if shape != [1, MAX_DET, STOCK_COLS]:
        raise RuntimeError(f"stock export has shape {shape}, expected [1, {MAX_DET}, {STOCK_COLS}]")
    proto = append_remap(proto, spec)

    source = {
        "weights": str(weights),
        "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest() if Path(weights).exists() else None,
        "end_to_end": e2e,
        "opset": opset,
        **({"quantization": qat.describe(model.model)} if quantized else {}),
    }
    contract = contract_for(spec, source)
    entry = proto.metadata_props.add()
    entry.key, entry.value = "nt_contract", json.dumps(contract, sort_keys=True)
    path = out_dir / "model.onnx"
    onnx.save(proto, str(path))
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heads", action="store_true", help="also write model_heads.onnx + decode.onnx (RKNN-style split)")
    ap.add_argument("--split", type=Path, help="an existing export directory: only add the head split")
    args = ap.parse_args()
    if args.split:
        split = split_at_heads(args.split / "model.onnx")
        print(f"wrote {args.split / split['npu']} and {args.split / split['cpu']}; decode(heads(x)) == model(x)")
        return
    if not args.weights or not args.out:
        ap.error("--weights and --out are required unless --split is given")
    path = export(args.weights, args.out, LabelSpec.load(), args.opset, args.device, args.heads)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
