"""Train a pose model on a baked dataset.

    python -m nt.train --model configs/models/yolo26n-pose-armor.yaml --data baked/NAME --name y26n_a \\
        [--hyp configs/train/pose_default.yaml] [--epochs N] [--batch B] [--workers W] [--seed S] [--fraction F] [--device 0]

The baked data.yaml records the absolute path of the machine that baked it. A copy with this machine's path is
written next to the run (the dataset itself is never edited), so a baked directory can be copied to a cloud
machine and trained there unchanged.

Runs under nt.offline (no downloads, no telemetry). Writes runs/train/<name>/ with Ultralytics' weights and
results.csv plus nt_meta.json: package versions, the exact arguments, hashes of the model yaml, hyper-parameter
file and baked-dataset manifest, wall time and the last metrics row. That file is what a registry card is made of.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Pixel-level transforms that keep geometry and hue, so the keypoints and the color labels stay valid. Ultralytics
# applies its own default list whenever albumentations is installed and none is given, and that list has ToGray,
# which turns a red or blue armor gray; so nt.train always passes a list, empty when the hyp file has none.
ALBUMENTATIONS_ALLOWED = frozenset({
    "MotionBlur", "GaussianBlur", "Defocus", "GaussNoise", "ImageCompression", "RandomBrightnessContrast", "RandomGamma",
    "Sharpen",
})
# Blur, MedianBlur and Downscale are not on the list: their even kernels or resampling move the content by up to
# about 1-2 px against the labeled corners. MotionBlur only with allow_shifted false and direction_range [0, 0].


def build_albumentations(specs: list | None) -> list:
    """hyp `albumentations: [{name: MotionBlur, p: 0.15, blur_limit: [3, 7]}, ...]` -> transform objects."""
    if not specs:
        return []
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # albumentations checks PyPI on import otherwise
    import albumentations as A

    out = []
    for spec in specs:
        kwargs = dict(spec)
        name = kwargs.pop("name")
        out.append(getattr(A, name)(**{k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}))
    return out


def load_hyp(path: Path) -> dict:
    """A hyp file, or a small one with `base: other.yaml` (relative to it) whose keys override the base's."""
    hyp = yaml.safe_load(Path(path).read_text()) or {}
    base = hyp.pop("base", None)
    return {**load_hyp(Path(path).parent / base), **hyp} if base else hyp


def check_hyp(hyp: dict) -> None:
    """Refuse settings that make runs silently incomparable or contradict the labels."""
    for spec in hyp.get("albumentations") or []:
        if not isinstance(spec, dict) or spec.get("name") not in ALBUMENTATIONS_ALLOWED:
            raise SystemExit(f"albumentations entry {spec!r}: allowed are {sorted(ALBUMENTATIONS_ALLOWED)} "
                             "(pixel-level, keep geometry and hue)")
        if spec["name"] == "MotionBlur" and (spec.get("allow_shifted", True) or list(spec.get("direction_range", [-1, 1])) != [0, 0]):
            raise SystemExit("MotionBlur needs allow_shifted: false and direction_range: [0.0, 0.0], or it moves the content")
    opt = str(hyp.get("optimizer", "auto"))
    if opt.lower() == "auto":
        raise SystemExit("optimizer is 'auto': Ultralytics then picks MuSGD or AdamW by the iteration count, so a bigger "
                         "dataset or more epochs changes the optimizer; set optimizer, lr0 and momentum in the hyp file")
    if opt.lower() in {"adam", "adamax", "adamw", "nadam", "radam"} and hyp.get("warmup_bias_lr", 0.1) > 0.01:
        raise SystemExit(f"warmup_bias_lr {hyp.get('warmup_bias_lr', 0.1)} is meant for SGD; Ultralytics keeps it at 0 for Adam")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="baked dataset directory")
    ap.add_argument("--name", required=True)
    ap.add_argument("--hyp", type=Path, default=ROOT / "configs" / "train" / "pose_default.yaml")
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int, help="weight init and data order; repeat a run with another seed to measure noise")
    ap.add_argument("--fraction", type=float, help="train on only this share of the training images (pipeline checks)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    # Ultralytics puts a relative project under runs/<task>/, so the run would land elsewhere than nt_meta.json
    args.project = args.project.resolve()

    from nt import offline

    offline.apply()
    import torch
    import ultralytics
    from ultralytics import YOLO

    from nt import modules

    modules.register()  # yaml names such as CoordAtt

    hyp = load_hyp(args.hyp)
    for key in ("epochs", "batch", "workers", "seed", "fraction"):
        if getattr(args, key) is not None:
            hyp[key] = getattr(args, key)
    check_hyp(hyp)
    data_yaml = args.data / "data.yaml"
    manifest = args.data / "bake_manifest.json"
    if not data_yaml.exists() or not manifest.exists():
        raise SystemExit(f"{args.data} is not a baked dataset (needs data.yaml and bake_manifest.json)")

    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(args.data.resolve())
    args.project.mkdir(parents=True, exist_ok=True)
    local_data = args.project / f".{args.name}.data.yaml"
    local_data.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    train_args = {k: v for k, v in hyp.items() if k != "albumentations"}
    train_args["augmentations"] = build_albumentations(hyp.get("albumentations"))
    model = YOLO(str(args.model), task="pose")
    t0 = time.time()
    status = "completed"
    try:
        model.train(data=str(local_data), project=str(args.project), name=args.name, device=args.device, exist_ok=False, verbose=False, **train_args)
    except BaseException as e:  # keep the metadata of an interrupted run too
        status = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        trainer = getattr(model, "trainer", None)
        save_dir = Path(trainer.save_dir) if trainer is not None else args.project / args.name
        last = {}
        results = save_dir / "results.csv"
        if results.exists():
            rows = list(csv.DictReader(results.open()))
            last = {k.strip(): v.strip() for k, v in rows[-1].items()} if rows else {}
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "nt_meta.json").write_text(json.dumps({
            "status": status,
            "wall_time_s": round(time.time() - t0, 1),
            "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "model_yaml": {"path": str(args.model), "sha256": _sha(args.model)},
            "hyp": {"path": str(args.hyp), "sha256": _sha(args.hyp), "effective": hyp},
            "data": {"path": str(args.data), "bake_manifest_sha256": _sha(manifest),
                     "images": json.loads(manifest.read_text()).get("images")},
            "last_metrics": last,
            "optimizer": _optimizer_of(trainer),
        }, indent=2, sort_keys=True) + "\n")


def _optimizer_of(trainer) -> dict | None:
    opt = getattr(trainer, "optimizer", None)
    if opt is None:
        return None
    g = opt.param_groups[0]
    return {"name": type(opt).__name__, "lr0": g.get("initial_lr", g.get("lr")), "betas_or_momentum": g.get("betas", g.get("momentum"))}


if __name__ == "__main__":
    main()
