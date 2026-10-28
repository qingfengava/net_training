"""Check an exported armor model against the output contract, and decode it the way sleepy's AT mode does.

    python -m nt.contract exports/NAME/model.onnx [--cpp]

check_model() runs the ONNX through onnxruntime and returns a list of problems (empty means it conforms):
interface (names, shape, dtype), embedded contract metadata against the current label_spec, and value
semantics on random and structured inputs: confidence in [0,1], integer number ids in range, color scores one-hot
on a trained color, finite keypoints. Value semantics matter because the stock export has the same 18 columns
with a different meaning, so a shape check alone would pass a wrongly exported model.

reference_decode() is a Python port of ArmorInfer::post_processAT_impl in sleepy; with --cpp the real C++
decoder is run on the same tensor (scripts/run_cpp_parity.sh) and the two must agree.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from nt.export import CONTRACT_VERSION, IMG_H, IMG_W, MAX_DET, OUT_COLS, contract_for
from nt.spec import LabelSpec

# Mirrors ModelTraits_AT in sleepy/src/tasks/auto_aim/armor_detect/armor_infer.cpp.
SLEEPY_CLASSES = ["sentry", "one", "two", "three", "four", "five", "outpost", "base", "unknown"]
SLEEPY_COLORS = ["blue", "red", "none", "purple"]
ROOT = Path(__file__).resolve().parents[1]


def reference_decode(out: np.ndarray, conf_threshold: float) -> list[dict]:
    """What sleepy's AT decoder makes of an [N, 18] tensor. Keypoints 0..3 are LT, LB, RB, RT and are kept as is.

    Until 2026-10-28 sleepy re-derived the roles by sorting y then x, which rotated the roles by 90 degrees once
    the top edge was tilted past atan(h/w) (about 22 degrees for a small plate). It now trusts the network order.
    """
    armors = []
    for row in out:
        conf = float(row[4])
        if not np.isfinite(conf) or conf < conf_threshold:
            continue
        cls_id = int(row[5])
        if cls_id < 0 or cls_id >= len(SLEEPY_CLASSES):
            continue
        color_id = int(np.argmax(row[6:10]))  # first maximum wins, as in the C++ loop
        pts = [(float(row[10 + 2 * k]), float(row[11 + 2 * k])) for k in range(4)]
        armors.append({
            "conf": conf, "number": SLEEPY_CLASSES[cls_id], "color": SLEEPY_COLORS[color_id],
            "lt": pts[0], "lb": pts[1], "rb": pts[2], "rt": pts[3],
        })
    return armors


def run_cpp(out: np.ndarray, conf_threshold: float) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        f.write(struct.pack(f"{out.size}f", *out.astype(np.float32).ravel()))
        path = f.name
    try:
        res = subprocess.run([str(ROOT / "scripts" / "run_cpp_parity.sh"), "run", path, str(len(out)), str(conf_threshold)],
                             capture_output=True, text=True, check=True)
    finally:
        Path(path).unlink(missing_ok=True)
    return json.loads(res.stdout.strip().splitlines()[-1])["armors"]


def check_values(out: np.ndarray, spec: LabelSpec, where: str) -> list[str]:
    problems = []
    if out.shape != (1, MAX_DET, OUT_COLS) or out.dtype != np.float32:
        return [f"{where}: output is {out.dtype}{out.shape}, expected float32{(1, MAX_DET, OUT_COLS)}"]
    rows = out[0]
    if not np.isfinite(rows).all():
        problems.append(f"{where}: non-finite values")
    if ((rows[:, 4] < 0) | (rows[:, 4] > 1)).any():
        problems.append(f"{where}: confidence outside [0,1]")
    number = rows[:, 5]
    if (number != np.round(number)).any() or number.min() < 0 or number.max() >= len(spec.labels):
        problems.append(f"{where}: number column is not an integer id in [0,{len(spec.labels) - 1}]")
    colors = rows[:, 6:10]
    onehot = (colors.sum(axis=1) == 1) & ((colors == 0) | (colors == 1)).all(axis=1)
    if not onehot.all():
        problems.append(f"{where}: color scores are not one-hot")
    else:
        trained = set(spec.training_colors)
        if not set(int(i) for i in colors.argmax(axis=1)) <= trained:
            problems.append(f"{where}: a color outside training_colors {sorted(trained)} was produced")
    return problems


def check_model(path: Path, spec: LabelSpec | None = None, cpp: bool = False) -> list[str]:
    import onnx
    import onnxruntime as ort

    spec = spec or LabelSpec.load()
    problems: list[str] = []
    proto = onnx.load(str(path))
    meta = {p.key: p.value for p in proto.metadata_props}
    if "nt_contract" not in meta:
        problems.append("no nt_contract metadata embedded")
    else:
        embedded = json.loads(meta["nt_contract"])
        expected = contract_for(spec, embedded.get("source"))
        if embedded.get("contract_version") != CONTRACT_VERSION:
            problems.append(f"contract_version {embedded.get('contract_version')} != {CONTRACT_VERSION}")
        if embedded != expected:
            diff = sorted(k for k in expected if embedded.get(k) != expected[k])
            problems.append(f"embedded contract differs from the current label_spec in: {diff}")
    sibling = path.with_name("contract.json")
    if sibling.exists() and "nt_contract" in meta and json.loads(sibling.read_text()) != json.loads(meta["nt_contract"]):
        problems.append("contract.json next to the model differs from the embedded one")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    if inp.name != "images" or list(inp.shape) != [1, 3, IMG_H, IMG_W]:
        problems.append(f"input is {inp.name}{inp.shape}, expected images[1,3,{IMG_H},{IMG_W}]")
    if outp.name != "output0" or list(outp.shape) != [1, MAX_DET, OUT_COLS]:
        problems.append(f"output is {outp.name}{outp.shape}, expected output0[1,{MAX_DET},{OUT_COLS}]")
    if problems:
        return problems

    rng = np.random.default_rng(0)
    inputs = {
        "random": rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32),
        "black": np.zeros((1, 3, IMG_H, IMG_W), np.float32),
        "white": np.ones((1, 3, IMG_H, IMG_W), np.float32),
    }
    for name, x in inputs.items():
        out = sess.run(None, {inp.name: x})[0]
        problems += check_values(out, spec, name)
        if problems:
            continue
        ref = reference_decode(out[0], 0.0)
        if cpp:
            real = run_cpp(out[0], 0.0)
            problems += compare_decoders(ref, real, name)
    return problems


def compare_decoders(ref: list[dict], real: list[dict], where: str, tol: float = 1e-4) -> list[str]:
    if len(ref) != len(real):
        return [f"{where}: python reference decoded {len(ref)} armors, sleepy's decoder {len(real)}"]
    problems = []
    for i, (a, b) in enumerate(zip(ref, real)):
        for key in ("number", "color"):
            if a[key] != b[key]:
                problems.append(f"{where}: armor {i} {key}: reference {a[key]} vs sleepy {b[key]}")
        if abs(a["conf"] - b["conf"]) > tol:
            problems.append(f"{where}: armor {i} confidence {a['conf']} vs {b['conf']}")
        for k in ("lt", "lb", "rb", "rt"):
            if max(abs(a[k][0] - b[k][0]), abs(a[k][1] - b[k][1])) > 1e-3:
                problems.append(f"{where}: armor {i} {k}: reference {a[k]} vs sleepy {b[k]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--cpp", action="store_true", help="also run sleepy's real decoder (needs a built sleepy)")
    args = ap.parse_args()
    problems = check_model(args.model, cpp=args.cpp)
    if problems:
        print("\n".join(f"FAIL {p}" for p in problems))
        sys.exit(1)
    print(f"{args.model} conforms to contract v{CONTRACT_VERSION}" + (" and agrees with sleepy's decoder" if args.cpp else ""))


if __name__ == "__main__":
    main()
