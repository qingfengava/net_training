"""Ingest gate: turn an external or generated dataset into an immutable canonical dataset.

    python -m nt.ingest --src DIR --format ultralytics-pose|daedalus-old|canonical-jsonl|labelme \\
        --source sim|real|exchange --name NAME --dst datasets/<source>/<name> \\
        [--role auto|train|val|hard|neg] [--drop-near-duplicates] [--link] [--dry-run]

Labelme annotation projects are normally imported with `python -m nt.annotate import`, which fills in the
source and role from the capture session.

Every sample is validated. A sample with any error is refused as a whole and listed in rejected.jsonl;
warnings are kept in issues.jsonl and counted in the manifest. The destination must not exist (datasets are
immutable) and is built in a hidden sibling directory first, so a failed run leaves nothing behind.
--dry-run runs the full gate and prints the summary without writing anything.
--link hard-links images instead of copying them (same filesystem), for large simulator exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

from nt.canonical import ROLES, SOURCES, Armor, Camera, Manifest, Sample, dumps, read_samples, sample_to_dict, write_manifest
from nt.geometry import canonical_permutation
from nt.spec import LabelSpec
from nt.validators import Issue, NearDuplicateIndex, dhash64, validate_sample

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
IMPORTER_VERSION = "1"


@dataclass
class RawSample:
    image_path: Path
    origin: str
    armors: list[Armor]  # corners normalized to 0..1 unless `normalized` is False
    normalized: bool = True  # False: corners are already pixels and declared_hw is the size the source claims
    declared_hw: tuple[int, int] | None = None
    camera: Camera | None = None
    scene: dict | None = None
    # (code, message) errors the reader found before the armors could be built, e.g. an unknown label in a
    # labelme file; the gate refuses the sample with them like any other error.
    problems: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Importer:
    name: str
    read: Callable[[Path, LabelSpec], tuple[list[RawSample], list[str]]]
    reorder_corners: bool  # the source does not promise the LT, LB, RB, RT start corner


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_ultralytics_pose(src: Path, spec: LabelSpec) -> tuple[list[RawSample], list[str]]:
    """src/images/*.ext + src/labels/*.txt, lines `cls cx cy w h x y v x y v x y v x y v` (v optional).

    The bbox column is ignored: boxes are rebuilt from the corners on export. cls is the merged id.
    """
    raws = []
    for img in _images(src / "images"):
        armors = []
        lab = src / "labels" / f"{img.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            f = line.split()
            if not f:
                continue
            nums = [float(x) for x in f]
            kp = nums[5:]
            step = 3 if len(kp) == 12 else 2
            pair = spec.from_class_id(int(nums[0]))
            color, label = pair if pair else (-1, -1)
            pts = [[kp[i * step], kp[i * step + 1]] for i in range(4)] if len(kp) in (8, 12) else []
            vis = [int(kp[i * 3 + 2]) for i in range(4)] if step == 3 else [2] * 4
            armors.append(Armor(color=color, label=label, corners=pts, vis=vis if pts else []))
        raws.append(RawSample(img, img.name, armors))
    return raws, ["bbox column ignored; corners are the label"]


def read_daedalus_old(src: Path, spec: LabelSpec) -> tuple[list[RawSample], list[str]]:
    """src/images/*.jpg + src/label/*.txt from the removed Daedalus exporter (commit 6e4b85a).

    Line: `color typ label x1 y1 .. x4 y4`, normalized. Known quirks, handled or recorded here:
    corners were sorted by angle so the start corner varies from frame to frame (fixed by ingest reordering),
    coordinates were truncated to whole pixels, and armors that were not fully visible were dropped.
    """
    raws = []
    for img in _images(src / "images"):
        armors = []
        lab = src / "label" / f"{img.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            f = line.split()
            if not f:
                continue
            color, typ, label = int(f[0]), int(f[1]), int(f[2])
            pts = [[float(f[3 + 2 * i]), float(f[4 + 2 * i])] for i in range(4)]
            # typ 1 was the RM26 large plate; RM27 has small armor only, so it is kept without a type
            armors.append(Armor(color=color, label=label, corners=pts, type={0: "small", 1: None}.get(typ, "?")))
        raws.append(RawSample(img, img.name, armors))
    return raws, [
        "daedalus-old: corner start was unstable and has been reordered to LT, LB, RB, RT",
        "daedalus-old: RM26 large plates are kept with type null; RM27 has no large armor",
        "daedalus-old: coordinates were truncated to whole pixels",
        "daedalus-old: partially visible armors were never labeled",
    ]


def read_canonical_jsonl(src: Path, spec: LabelSpec) -> tuple[list[RawSample], list[str]]:
    """src/labels.jsonl + src/images/ written by the Daedalus exporter (src/dataset/ in the simulator).

    Corners are pixels already in LT, LB, RB, RT order with per-corner visibility, so nothing is reordered
    or rescaled; the size the exporter claims is checked against the decoded image by the gate.
    """
    raws = []
    for smp in read_samples(src / "labels.jsonl"):
        raws.append(
            RawSample(
                image_path=src / smp.image,
                origin=smp.origin or smp.id,
                armors=smp.armors,
                normalized=False,
                declared_hw=(smp.h, smp.w),
                camera=smp.camera,
                scene=smp.scene,
            )
        )
    return raws, ["canonical-jsonl: labels come straight from the simulator, which also records scene parameters"]


def read_labelme_project(src: Path, spec: LabelSpec) -> tuple[list[RawSample], list[str]]:
    """An annotation project from `nt.annotate export`: batches of labelme JSON files over a capture session.

    Only frames flagged `reviewed` are read; `bad_frame` ones are left out. Corners stay in the order the
    annotator left them (the export writes LT, LB, RB, RT), so the gate's order check is the safety net.
    """
    from nt.annotate import read_project

    return read_project(src, spec)


IMPORTERS = {
    "ultralytics-pose": Importer("ultralytics-pose", read_ultralytics_pose, reorder_corners=False),
    "daedalus-old": Importer("daedalus-old", read_daedalus_old, reorder_corners=True),
    "canonical-jsonl": Importer("canonical-jsonl", read_canonical_jsonl, reorder_corners=False),
    "labelme": Importer("labelme", read_labelme_project, reorder_corners=False),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _place_image(src: Path, dst: Path, link: bool) -> None:
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystem or no permission: fall back to a copy
    shutil.copy2(src, dst)


def ingest(
    src: Path,
    dst: Path,
    fmt: str,
    source: str,
    name: str,
    spec: LabelSpec,
    drop_near_duplicates: bool = False,
    dry_run: bool = False,
    link: bool = False,
    role: str = "auto",
) -> dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if fmt not in IMPORTERS:
        raise ValueError(f"format must be one of {sorted(IMPORTERS)}")
    if dst.exists():
        raise FileExistsError(f"{dst} exists; datasets are immutable, use a new directory")
    importer = IMPORTERS[fmt]
    raws, notes = importer.read(src, spec)

    tmp = dst.parent / f".{dst.name}.tmp"
    if not dry_run:
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / "images").mkdir(parents=True)

    index = NearDuplicateIndex(spec.v("near_duplicate_hamming"))
    accepted: list[Sample] = []
    warn_issues: list[Issue] = []
    rejected: list[dict] = []
    image_sha: dict[str, str] = {}
    reordered = 0

    for i, raw in enumerate(raws):
        sid = f"{name}_{i:06d}"
        ext = raw.image_path.suffix.lower()
        img = cv2.imread(str(raw.image_path), cv2.IMREAD_COLOR)
        if img is None:
            rejected.append({"id": sid, "origin": raw.origin, "issues": [Issue("error", "E_IMAGE_UNREADABLE", sid, None, raw.origin).to_dict()]})
            continue
        h, w = img.shape[:2]
        armors = []
        for a in raw.armors:
            if len(a.corners) == 4:
                px = [[x * w, y * h] for x, y in a.corners] if raw.normalized else [list(c) for c in a.corners]
                vis = list(a.vis)
                if importer.reorder_corners:
                    perm, changed = canonical_permutation(px)
                    if changed:
                        px, vis = [px[k] for k in perm], [vis[k] for k in perm]
                        reordered += 1
                a = Armor(**{**a.__dict__, "corners": px, "vis": vis})
            armors.append(a)
        dh, dw = raw.declared_hw if raw.declared_hw else (h, w)
        sample = Sample(
            id=sid, image=f"images/{sid}{ext}", w=dw, h=dh, source=source, armors=armors,
            camera=raw.camera, origin=raw.origin, scene=raw.scene or {},
        )
        issues = [Issue("error", code, sid, None, msg) for code, msg in raw.problems]
        issues += validate_sample(sample, spec, (h, w))
        if role == "neg" and armors:
            issues.append(Issue("error", "E_NEG_HAS_ARMOR", sid, None, f"{len(armors)} armors in a dataset that must stay empty"))

        if not any(x.level == "error" for x in issues):
            twin = index.query(dhash64(img))
            if twin:
                level = "error" if drop_near_duplicates else "warn"
                issues.append(Issue(level, "W_NEAR_DUPLICATE_FRAME" if level == "warn" else "E_NEAR_DUPLICATE", sid, None, f"near duplicate of {twin}"))
        if any(x.level == "error" for x in issues):
            rejected.append({"id": sid, "origin": raw.origin, "issues": [x.to_dict() for x in issues]})
            continue
        index.add(sid, dhash64(img))
        accepted.append(sample)
        warn_issues.extend(issues)
        if not dry_run:
            _place_image(raw.image_path, tmp / sample.image, link)
            image_sha[sid] = _sha256(tmp / sample.image)

    if reordered:
        notes.append(f"reordered corners on {reordered} armors")
    class_counts = Counter(spec.pair_name(a.color, a.label) for s in accepted for a in s.armors)
    labels_text = "".join(dumps(sample_to_dict(s)) + "\n" for s in accepted)
    manifest = Manifest(
        name=name,
        source=source,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        importer=f"{importer.name}@{IMPORTER_VERSION}",
        src_path=str(src.resolve()),
        label_spec_sha256=spec.sha256,
        corner_definition=spec.corner_definition,
        pixel_convention=spec.pixel_convention,
        counts={
            "input_samples": len(raws),
            "accepted_samples": len(accepted),
            "rejected_samples": len(rejected),
            "armors": sum(len(s.armors) for s in accepted),
            "negative_frames": sum(1 for s in accepted if not s.armors),
        },
        class_counts=dict(sorted(class_counts.items())),
        image_formats=dict(Counter(Path(s.image).suffix for s in accepted)),
        issues=dict(sorted(Counter(i.code for i in warn_issues).items())),
        rejected=dict(sorted(Counter(i["code"] for r in rejected for i in r["issues"] if i["level"] == "error").items())),
        labels_sha256=hashlib.sha256(labels_text.encode()).hexdigest(),
        images_sha256=hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(image_sha.items())).encode()).hexdigest() if image_sha else "",
        notes=notes,
        role=role,
    )
    if not dry_run:
        (tmp / "labels.jsonl").write_text(labels_text, encoding="utf-8")
        (tmp / "issues.jsonl").write_text("".join(dumps(i.to_dict()) + "\n" for i in warn_issues), encoding="utf-8")
        (tmp / "rejected.jsonl").write_text("".join(dumps(r) + "\n" for r in rejected), encoding="utf-8")
        write_manifest(tmp / "manifest.json", manifest)
        tmp.rename(dst)
    return json.loads(json.dumps(manifest.__dict__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--format", dest="fmt", choices=sorted(IMPORTERS), required=True)
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--name", required=True, help="prefix of the sample ids, e.g. hebust_2026")
    ap.add_argument("--dst", type=Path, required=True, help="datasets/<source>/<name>")
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--drop-near-duplicates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--link", action="store_true", help="hard-link images instead of copying")
    ap.add_argument("--role", choices=ROLES, default="auto",
                    help="what the dataset may be used for; real capture sessions set it (see nt/canonical.py)")
    args = ap.parse_args()
    spec = LabelSpec.load(args.spec) if args.spec else LabelSpec.load()
    m = ingest(args.src, args.dst, args.fmt, args.source, args.name, spec, args.drop_near_duplicates, args.dry_run, args.link,
               args.role)
    print(json.dumps({k: m[k] for k in ("counts", "class_counts", "issues", "rejected", "notes")}, indent=2, ensure_ascii=False))
    if not args.dry_run:
        print(f"wrote {args.dst}")
    sys.exit(0)


if __name__ == "__main__":
    main()
