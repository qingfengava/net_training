"""Post-training quantization (PTQ) of an nt.export model into a QDQ ONNX, the format TensorRT and RKNN both load.

    python -m nt.quant ptq --model exports/NAME/model.onnx --calib-data baked/X --scheme tensorrt|rknn \\
        --out exports/NAME_ptq_tensorrt [--calib-split train] [--calib-n 256] [--method minmax|entropy|percentile]
    python -m nt.eval --weights exports/NAME_ptq_tensorrt/model.onnx --backend trt-int8 \\
        --reference exports/NAME/model.onnx --data baked/VAL

ONNX Runtime's quantize_static calibrates on --calib-n frames of the calibration split (train by default, so the
evaluation images stay unseen) preprocessed exactly as sleepy does, and inserts QuantizeLinear/DequantizeLinear
around Conv and MatMul only. Everything after the detection heads (grid decode, sigmoid, top-k, the class remap)
has neither and stays float, which is where docs/rknn_int8_qat_notes.md says the cut belongs. Weights are int8
per output channel; activations per tensor:
    tensorrt  int8 symmetric (TensorRT's explicit quantization needs zero point 0) on Conv/MatMul inputs only;
              bias and the head outputs stay float
    rknn      uint8 with a zero point (RKNN SDK recommendation; symmetric parameters also load), on inputs and
              outputs, so the 9 head outputs are 8-bit tensors as an NPU hands them to the CPU decode
The result is the PTQ baseline a QAT model (nt.train --qat) has to beat; both are evaluated the same way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMES = ("tensorrt", "rknn")


STRIDE = 8  # frames per calibration pass: ORT merges the ranges of the passes, so memory stays bounded


class _Frames:
    """onnxruntime CalibrationDataReader over baked images, with the len/set_range that strided calibration uses."""

    def __init__(self, paths: list[Path], input_name: str):
        from nt.eval import contract_input

        self._paths = list(paths)
        self._name = input_name
        self._load = contract_input
        self.set_range(0, len(self._paths))

    def __len__(self) -> int:
        return len(self._paths)

    def set_range(self, start_index: int, end_index: int) -> None:
        self._it = iter(self._paths[start_index:end_index])

    def get_next(self):
        p = next(self._it, None)
        return None if p is None else {self._name: self._load(p)}


def calibration_frames(data: Path, split: str, n: int) -> list[Path]:
    from nt.eval import load_ground_truth

    metas = load_ground_truth(data, split)
    if not metas:
        raise SystemExit(f"{data} has no {split} images")
    paths = sorted(data / m["image"] for m in metas)
    step = max(1, len(paths) // n)
    return paths[::step][:n]


def ptq(model: Path, out_dir: Path, calib: list[Path], scheme: str, method: str = "minmax") -> Path:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    if scheme not in SCHEMES:
        raise ValueError(f"scheme {scheme!r}: one of {SCHEMES}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = onnx.load(str(model))
    input_name = src.graph.input[0].name
    # CalibStridedMinMax: calibrate STRIDE frames at a time and merge, instead of holding every intermediate tensor
    # of every frame (about 0.2 GB per frame). (CalibMaxIntermediateOutputs drops the data in onnxruntime 1.23.)
    calib = calib[: len(calib) // STRIDE * STRIDE]
    if not calib:
        raise ValueError(f"need at least {STRIDE} calibration frames")
    extra = {"WeightSymmetric": True, "ActivationSymmetric": scheme == "tensorrt", "CalibStridedMinMax": STRIDE}
    if scheme == "tensorrt":
        # Q/DQ only on the inputs of Conv/MatMul (TensorRT's recommended placement): the head outputs, i.e. the box,
        # class and keypoint maps the decode reads, stay in float/FP16 instead of being rounded to 8 bits
        extra["QuantizeBias"] = False
        extra["OpTypesToExcludeOutputQuantization"] = ["Conv", "MatMul"]
    out = out_dir / "model.onnx"
    quantize_static(
        str(model), str(out), _Frames(calib, input_name),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8 if scheme == "tensorrt" else QuantType.QUInt8, weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul"],
        calibrate_method={"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                          "percentile": CalibrationMethod.Percentile}[method],
        extra_options=extra,
    )
    q = onnx.load(str(out))
    contract = None
    for p in src.metadata_props:
        if p.key == "nt_contract":
            contract = json.loads(p.value)
    if contract is not None:
        contract.setdefault("source", {})["quantization"] = {
            "kind": "ptq", "tool": "onnxruntime quantize_static", "scheme": scheme, "method": method,
            "calibration_frames": len(calib), "op_types": ["Conv", "MatMul"],
            "float_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
        del q.metadata_props[:]
        entry = q.metadata_props.add()
        entry.key, entry.value = "nt_contract", json.dumps(contract, sort_keys=True)
        onnx.save(q, str(out))
        (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return out


def count_qdq(path: Path) -> dict:
    import onnx

    ops = [n.op_type for n in onnx.load(str(path)).graph.node]
    return {"QuantizeLinear": ops.count("QuantizeLinear"), "DequantizeLinear": ops.count("DequantizeLinear")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ptq")
    p.add_argument("--model", type=Path, required=True, help="an nt.export model.onnx (float)")
    p.add_argument("--calib-data", type=Path, required=True, help="baked dataset with the calibration frames")
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-n", type=int, default=256)
    p.add_argument("--scheme", choices=SCHEMES, required=True)
    p.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = calibration_frames(args.calib_data, args.calib_split, args.calib_n)
    out = ptq(args.model, args.out, frames, args.scheme, args.method)
    print(f"wrote {out} ({count_qdq(out)}, {len(frames)} calibration frames)")


if __name__ == "__main__":
    main()
