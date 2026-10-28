"""Canonical dataset format shared by sim export, real capture, and exchanged data.

datasets/<source>/<name>/
    manifest.json    counts, class histogram, issue summary, hashes, label_spec hash (the only file git tracks)
    labels.jsonl     one Sample per line
    images/          files named <sample id>.<ext>
    issues.jsonl     warnings on accepted samples
    rejected.jsonl   samples the ingest gate refused, with the reasons

Colors and labels are the integer slots from configs/label_spec.yaml. Corners are [LT, LB, RB, RT] in pixel
coordinates with the "edge" convention. vis follows Ultralytics: 0 absent, 1 present but occluded, 2 visible.
Unknown keys are an error on load, so schema drift from the Rust exporter shows up immediately.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterator

SOURCES = ("sim", "real", "exchange")
# What a dataset may be used for. "auto" splits train/val by a hash of the sample id (simulator data, where
# frames are independent). Real data comes in capture sessions whose frames are correlated, so the whole
# session has one role: train, val (real_val), hard (real_hard) or neg (frames that must stay empty).
# Only "train" and "auto" ever reach a training split; see nt/bake.py.
ROLES = ("auto", "train", "val", "hard", "neg")


class SchemaError(ValueError):
    pass


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)  # k1, k2, p1, p2, k3


@dataclass
class Armor:
    color: int
    label: int
    corners: list[list[float]]
    vis: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    type: str | None = None
    occ: float | None = None  # occluded fraction of the face, 0..1
    dist_m: float | None = None
    yaw_deg: float | None = None  # armor yaw relative to the camera
    pitch_deg: float | None = None
    screen_state: str | None = None  # e.g. on, off, flashing; free text until the official style is known
    pose_cam: list[float] | None = None  # rvec + tvec in the camera frame


@dataclass
class Sample:
    id: str
    image: str  # path relative to the dataset root
    w: int
    h: int
    source: str
    armors: list[Armor] = field(default_factory=list)  # empty means a negative frame
    camera: Camera | None = None
    origin: str | None = None  # where the sample came from before ingest
    scene: dict = field(default_factory=dict)  # seeds, lighting, background and other generator parameters


def _build(cls, d: dict, what: str):
    if not isinstance(d, dict):
        raise SchemaError(f"{what} must be an object, got {type(d).__name__}")
    extra = set(d) - {f.name for f in fields(cls)}
    if extra:
        raise SchemaError(f"unknown {what} keys: {sorted(extra)}")
    try:
        return cls(**d)
    except TypeError as e:
        raise SchemaError(f"{what}: {e}") from e


def sample_from_dict(d: dict) -> Sample:
    d = dict(d)
    d["armors"] = [_build(Armor, a, "armor") for a in d.get("armors", [])]
    if d.get("camera") is not None:
        d["camera"] = _build(Camera, d["camera"], "camera")
    return _build(Sample, d, "sample")


def sample_to_dict(s: Sample) -> dict:
    d = asdict(s)
    for a in d["armors"]:
        a["corners"] = [[round(float(x), 4), round(float(y), 4)] for x, y in a["corners"]]
    return d


def read_samples(path: Path) -> Iterator[Sample]:
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield sample_from_dict(json.loads(line))
            except (json.JSONDecodeError, SchemaError) as e:
                raise SchemaError(f"{path}:{n}: {e}") from e


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Manifest:
    name: str
    source: str
    created_utc: str
    importer: str
    src_path: str
    label_spec_sha256: str
    corner_definition: str
    pixel_convention: str
    counts: dict = field(default_factory=dict)
    class_counts: dict = field(default_factory=dict)
    image_formats: dict = field(default_factory=dict)
    issues: dict = field(default_factory=dict)  # code -> count, accepted samples only
    rejected: dict = field(default_factory=dict)  # code -> count of rejected samples carrying it
    labels_sha256: str = ""
    images_sha256: str = ""  # sha256 over sorted "<id> <image sha256>" lines
    notes: list[str] = field(default_factory=list)
    role: str = "auto"  # one of ROLES; manifests written before roles existed read back as "auto"


def write_manifest(path: Path, m: Manifest) -> None:
    path.write_text(json.dumps(asdict(m), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    return _build(Manifest, json.loads(path.read_text(encoding="utf-8")), "manifest")
