"""M0 smoke test: GPU, model build, ONNX export contract, AMP training, non-square training behavior.

Run:  . scripts/env.sh && python -m nt.smoke_m0 [--skip-train]
Prints PASS/FAIL per check and exits non-zero if any fails. All outputs go to a temp dir that is removed at the end.
"""
import argparse
import csv
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/models/yolo26n-pose-armor.yaml"
FIXTURE = ROOT / "tests/fixtures/m0_smoke/data.yaml"
IMG_H, IMG_W = 576, 768
MAX_DET = 30
NC, NKPT, KDIM = 16, 4, 3
OUT_COLS = 4 + 1 + 1 + NKPT * KDIM  # box, conf, cls, kpts with visibility

results: list[tuple[str, bool, str]] = []
ctx: dict = {}


def check(name, fn, *args, **kwargs):
    t0 = time.time()
    try:
        info = fn(*args, **kwargs)
        results.append((name, True, str(info)))
        print(f"PASS {name}: {info} ({time.time() - t0:.1f}s)", flush=True)
        return info
    except Exception as e:  # noqa: BLE001 - a smoke test reports every failure
        results.append((name, False, repr(e)))
        print(f"FAIL {name}: {e!r}", flush=True)
        return None


def gpu_check():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    arch = torch.cuda.get_arch_list()
    assert "sm_120" in arch, f"sm_120 missing from {arch}"
    a = torch.randn(512, 512, device="cuda")
    ref = (a.cpu() @ a.cpu().T)
    got = (a @ a.T).cpu()
    assert torch.allclose(ref, got, rtol=1e-3, atol=1e-2), "GPU matmul mismatch"
    with torch.autocast("cuda", dtype=torch.float16):
        h = a @ a.T
    assert torch.isfinite(h).all(), "fp16 autocast produced non-finite values"
    return f"torch {torch.__version__}, cuda {torch.version.cuda}, {torch.cuda.get_device_name(0)}, arch {arch}"


def build_model(workdir: Path):
    from ultralytics import YOLO

    local_yaml = workdir / MODEL_YAML.name  # export writes next to the yaml, keep the repo clean
    shutil.copy2(MODEL_YAML, local_yaml)
    model = YOLO(str(local_yaml), task="pose")
    head = model.model.model[-1]
    assert type(head).__name__ == "Pose26", type(head).__name__
    assert head.nc == NC, head.nc
    assert list(head.kpt_shape) == [NKPT, KDIM], head.kpt_shape
    assert head.end2end, "end2end head expected"
    n_params = sum(p.numel() for p in model.model.parameters())
    ctx["model"] = model
    return f"Pose26 nc={head.nc} kpt_shape={list(head.kpt_shape)} params={n_params / 1e6:.2f}M"


def export_check(workdir: Path):
    import os

    import numpy as np
    import onnxruntime as ort

    model = ctx["model"]
    cwd = os.getcwd()
    os.chdir(workdir)  # a yaml-built model exports into the current directory, not next to the yaml
    try:
        path = model.export(
            format="onnx", imgsz=[IMG_H, IMG_W], batch=1, half=False, dynamic=False, simplify=True,
            max_det=MAX_DET, device="cpu", verbose=False,
        )
    finally:
        os.chdir(cwd)
    path = (workdir / path).resolve()
    assert path.parent == workdir.resolve(), f"export landed in {path.parent}"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    assert inp.name == "images" and list(inp.shape) == [1, 3, IMG_H, IMG_W], (inp.name, inp.shape)
    assert list(out.shape) == [1, MAX_DET, OUT_COLS], f"output shape {out.shape}"
    x = np.random.default_rng(0).random((1, 3, IMG_H, IMG_W), dtype=np.float32)
    y = sess.run(None, {inp.name: x})[0]
    assert y.shape == (1, MAX_DET, OUT_COLS) and np.isfinite(y).all()
    assert (y[0, :, 4] >= 0).all() and (y[0, :, 4] <= 1).all(), "conf column not in [0,1]"
    cls = y[0, :, 5]
    assert (cls == np.round(cls)).all() and cls.min() >= 0 and cls.max() < NC, "cls column not integral ids"
    meta = sess.get_modelmeta().custom_metadata_map
    return (
        f"in {inp.name}{list(inp.shape)} out {out.name}{list(out.shape)}; "
        f"meta end2end={meta.get('end2end')} kpt_shape={meta.get('kpt_shape')} imgsz={meta.get('imgsz')}"
    )


def train_run(workdir: Path, tag: str, **overrides):
    from ultralytics import YOLO

    from nt import offline

    offline.apply()
    local_yaml = workdir / MODEL_YAML.name
    model = YOLO(str(local_yaml), task="pose")
    from ultralytics.models.yolo.detect import DetectionTrainer

    # The trainer keeps no reference to the current batch, so record shapes where the batch is prepared.
    shapes: set[tuple[int, ...]] = set()
    orig_preprocess = DetectionTrainer.preprocess_batch

    def recording_preprocess(self, batch):
        out = orig_preprocess(self, batch)
        shapes.add(tuple(out["img"].shape))
        return out

    DetectionTrainer.preprocess_batch = recording_preprocess
    args = dict(
        data=str(FIXTURE), imgsz=IMG_W, epochs=2, batch=3, workers=0, device=0, amp=True, plots=False,
        project=str(workdir / "runs"), name=tag, exist_ok=True, fliplr=0.0, mosaic=0.0, close_mosaic=0,
        val=True, verbose=False, seed=0, pretrained=False,
    )
    args.update(overrides)
    try:
        model.train(**args)
    finally:
        DetectionTrainer.preprocess_batch = orig_preprocess
    save_dir = Path(model.trainer.save_dir)
    assert (save_dir / "weights/last.pt").exists(), "last.pt missing"
    with open(save_dir / "results.csv") as f:
        rows = list(csv.DictReader(f))
    last = {k.strip(): float(v) for k, v in rows[-1].items() if v.strip()}
    losses = {k: v for k, v in last.items() if k.startswith("train/")}
    assert losses and all(math.isfinite(v) for v in losses.values()), f"non-finite loss {losses}"
    return f"train batch shapes {sorted(shapes)}; final losses " + ", ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in losses.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="nt_m0_"))
    try:
        check("gpu", gpu_check)
        if check("build", build_model, workdir):
            check("export", export_check, workdir)
        if not args.skip_train:
            check("train amp+MuSGD, square imgsz=768", train_run, workdir, "sq", optimizer="MuSGD")
            check("train rect=True imgsz=768", train_run, workdir, "rect", rect=True)
            check("train imgsz=[576,768] (list)", train_run, workdir, "list", imgsz=[IMG_H, IMG_W])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
