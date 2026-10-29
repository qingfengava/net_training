"""Annotation projects in labelme format: export with pre-labels, track progress, self-check, import.

    python -m nt.annotate export --session sessions/S --out annot/P [--model M.onnx] [--conf 0.5] [--review-conf 0.1]
                                 [--tiles 0|2] [--frames LIST.txt] [--batch-size 200] [--copy]
    python -m nt.annotate export --dataset datasets/real/D --frames LIST.txt --out annot/P [--model M.onnx]   (re-review)
    python -m nt.annotate status annot/P
    python -m nt.annotate check  annot/P [--all]
    python -m nt.annotate import annot/P --dst datasets/real/NAME [--allow-partial] [--dry-run]

annot/<project>/
    project.json          where the frames came from (session or dataset), its role, the pre-label model and thresholds
    labelme_config.yaml   label list and flags; pass it to labelme with --config
    README.md             how to open a batch, and the conventions (see docs/labeling_guide.md)
    batch_000/            one folder per batch, self-contained (images + JSON), so it can be handed to one person
        batch.json        frame ids, image and JSON file names, optional assignee
        <frame>.png       hard link to the session frame (--copy for a folder that leaves this machine)
        <frame>.json      labelme document; the conventions are in nt/labelme_io.py

Pre-labels: detections at or above --conf become ordinary labels the annotator corrects; those between
--review-conf and --conf, those whose geometry fails or looks suspicious to the label checks, and those cut by
the image border are written as "?<label>" and must be resolved by hand. Nothing from the model reaches a dataset unless a person ticked "reviewed" on the frame.
A re-review export (--dataset) writes the existing labels, plus as "?" shapes the model detections that do
not match any of them, so label errors found by mining can be fixed and imported as a new dataset version.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from nt import labelme_io as lm
from nt import session as sess
from nt.canonical import Armor, Camera, Sample, read_manifest, read_samples
from nt.geometry import aspect_ratio, bbox, bbox_iou
from nt.ingest import RawSample
from nt.spec import LabelSpec
from nt.validators import armor_issues, validate_sample

PROJECT_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]

README = """# 标注批次 {name}

来源：{source}（用途 {role}），共 {n_frames} 帧，分成 {n_batches} 批。预标注模型：{model}。

打开一批（每批是一个独立文件夹，可以整个交给一个人）：

    labelme {name}/batch_000 --config {name}/labelme_config.yaml --nodata --autosave

规则见仓库里的 docs/labeling_guide.md。要点：
- 装甲是 4 点多边形，点的顺序固定为 左上、左下、右下、右上。
- 被挡住的角放一个 occluded 点，出画的角放一个 outside 点。
- 以 ? 开头的形状是没确认的预标注，必须改成正式标签或删掉。
- 一帧检查完勾选 reviewed；废帧勾选 bad_frame。只有 reviewed 的帧会入库。

自检：python -m nt.annotate check {name}
"""


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _read_frame_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


# Pre-labels that become "?" even at high confidence. On old footage the model's wrong guesses were mostly
# quads squeezed onto a single light bar (horizontal/vertical edge ratio far below a plate's) and half plates
# cut by the image border. MIN_ASPECT suits the light-bar plates (about 2.4 seen head-on, near 0.8 at 70 deg
# of yaw); set --min-aspect again once the RM27 screen's shape is known.
REVIEW_WARNINGS = {"W_QUAD_ROLL"}
MIN_ASPECT = 0.6
BORDER_PX = 2.0


def _det_shapes(spec: LabelSpec, d, w: int, h: int, conf: float, unresolved_reason: str | None = None,
                min_aspect: float = MIN_ASPECT, group_id: int | None = None) -> list[dict]:
    a = Armor(color=d.color, label=d.label, corners=d.corners.tolist(), vis=[2, 2, 2, 2], type=d.type)
    flagged = [code for level, code, _ in armor_issues(a, w, h, spec)
               if (level == "error" and code != "E_CORNER_OUTSIDE") or code in REVIEW_WARNINGS]
    if not flagged and aspect_ratio(a.corners) < min_aspect:
        flagged.append(f"aspect {aspect_ratio(a.corners):.2f} < {min_aspect}")
    x0, y0, x1, y1 = d.box
    if x0 < BORDER_PX or y0 < BORDER_PX or x1 > w - BORDER_PX or y1 > h - BORDER_PX:
        flagged.append("touches image border")
    reason = unresolved_reason or (f"conf {d.conf:.2f} < {conf:.2f}" if d.conf < conf else None)
    if flagged:
        reason = (reason + "; " if reason else "") + ",".join(flagged)
    desc = f"auto conf={d.conf:.3f}" + (f" review: {reason}" if reason else "")
    if not spec.valid_pair(d.color, d.label):
        return []
    return lm.armor_shapes(spec, d.color, d.label, d.corners.tolist(), w=w, h=h, description=desc,
                           unresolved=reason is not None, group_id=group_id)


def export(
    out: Path,
    spec: LabelSpec,
    session_dir: Path | None = None,
    dataset_dir: Path | None = None,
    model_path: Path | None = None,
    conf: float = 0.5,
    review_conf: float = 0.1,
    tiles: int = 0,
    frames_file: Path | None = None,
    batch_size: int = 200,
    copy: bool = False,
    device: str = "cpu",
    min_aspect: float = MIN_ASPECT,
) -> dict:
    if (session_dir is None) == (dataset_dir is None):
        raise ValueError("give exactly one of --session or --dataset")
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new project name")
    if not 0 <= review_conf <= conf:
        raise ValueError("need 0 <= review_conf <= conf")
    wanted = _read_frame_list(frames_file)

    # items: (frame id, image path, width, height, existing armors or None, scene extras)
    items = []
    if session_dir is not None:
        meta = sess.load(session_dir)
        source = {"kind": "session", "name": meta["name"], "path": str(session_dir.resolve()), "role": meta["role"],
                  "camera": meta["camera"], "real_source": "real"}
        frames = meta["frames"]
        if wanted is not None:
            by_id = {f["id"]: f for f in frames}
            missing = [i for i in wanted if i not in by_id]
            if missing:
                raise ValueError(f"{len(missing)} frame ids are not in the session, e.g. {missing[:3]}")
            frames = [by_id[i] for i in dict.fromkeys(wanted)]
        for f in frames:
            items.append((f["id"], session_dir / f["image"], meta["width"], meta["height"], None,
                          {"session": meta["name"], "frame_index": f["index"], "t_s": f["t_s"]}))
    else:
        man = read_manifest(dataset_dir / "manifest.json")
        if man.label_spec_sha256 != spec.sha256:
            raise ValueError(f"{dataset_dir} was ingested with a different label_spec")
        if wanted is None:
            raise ValueError("a re-review export needs --frames (sample ids or origins, e.g. from nt.mine)")
        samples = list(read_samples(dataset_dir / "labels.jsonl"))
        cameras = {json.dumps(s.camera.__dict__, sort_keys=True) if s.camera else None for s in samples}
        if len(cameras) > 1:
            raise ValueError(f"{dataset_dir} mixes {len(cameras)} cameras; a re-review project keeps one")
        camera = json.loads(next(iter(cameras))) if cameras and next(iter(cameras)) else None
        source = {"kind": "dataset", "name": man.name, "path": str(dataset_dir.resolve()), "role": man.role,
                  "camera": camera, "real_source": man.source}
        index = {s.id: s for s in samples} | {s.origin: s for s in samples if s.origin}
        missing = [i for i in wanted if i not in index]
        if missing:
            raise ValueError(f"{len(missing)} ids are not in the dataset, e.g. {missing[:3]}")
        for i in dict.fromkeys(wanted):
            s = index[i]
            fid = s.origin or s.id
            items.append((fid, dataset_dir / s.image, s.w, s.h, s.armors, {**s.scene, "re_review_of": s.id}))
    if not items:
        raise ValueError("no frames to export")

    model = None
    if model_path is not None:
        from nt.detector import ArmorModel

        model = ArmorModel(model_path, device=device)

    tmp = out.parent / f".{out.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    counts = Counter()
    try:
        n_batches = (len(items) + batch_size - 1) // batch_size
        for b in range(n_batches):
            bname = f"batch_{b:03d}"
            bdir = tmp / bname
            bdir.mkdir()
            entries = []
            for fid, img_path, w, h, existing, scene in items[b * batch_size:(b + 1) * batch_size]:
                ext = img_path.suffix.lower()
                image_name, json_name = f"{fid}{ext}", f"{fid}.json"
                _link_or_copy(img_path, bdir / image_name, copy)
                shapes = []
                if existing is not None:
                    for a in existing:
                        shapes += lm.armor_shapes(spec, a.color, a.label, a.corners, a.vis, w, h,
                                                  description="existing label", group_id=len(shapes) + 1)
                        counts["existing_armors"] += 1
                if model is not None:
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None or img.shape[:2] != (h, w):
                        raise ValueError(f"{img_path}: unreadable or not {w}x{h}")
                    # an old model can still report robots 2 and 5, which RM27 does not have
                    dets = [d for d in model.detect(img, review_conf, tiles) if spec.valid_pair(d.color, d.label)]
                    for d in dets:
                        if existing is not None:
                            # re-review: only what the labels do not already cover is worth a look
                            if any(bbox_iou(d.box, bbox(a.corners)) > 0.5 for a in existing):
                                continue
                            shapes += _det_shapes(spec, d, w, h, conf, "model sees an unlabeled armor", min_aspect, len(shapes) + 1)
                        else:
                            shapes += _det_shapes(spec, d, w, h, conf, None, min_aspect, len(shapes) + 1)
                        counts["pre_labels"] += 1
                    counts["unresolved"] += sum(1 for s in shapes if s["label"].startswith(lm.UNRESOLVED_PREFIX))
                doc = lm.document(image_name, w, h, shapes, nt_meta={"frame_id": fid, **scene})
                (bdir / json_name).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                entries.append({"id": fid, "image": image_name, "json": json_name, "scene": scene})
            (bdir / "batch.json").write_text(json.dumps({"batch": bname, "assignee": None, "frames": entries}, indent=1,
                                                        ensure_ascii=False) + "\n", encoding="utf-8")
        project = {
            "project_version": PROJECT_VERSION, "name": out.name,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source, "label_spec_sha256": spec.sha256,
            "model": None if model is None else model.describe(),
            "conf": conf, "review_conf": review_conf, "tiles": tiles, "min_aspect": min_aspect, "batch_size": batch_size,
            "frames": len(items), "batches": n_batches, "counts": dict(counts),
            "frames_file": None if frames_file is None else str(frames_file),
        }
        (tmp / "project.json").write_text(json.dumps(project, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (tmp / "labelme_config.yaml").write_text(yaml.safe_dump(lm.labelme_config(spec), allow_unicode=True, sort_keys=False),
                                                 encoding="utf-8")
        (tmp / "README.md").write_text(README.format(
            name=out.name, source=f"{source['kind']} {source['name']}", role=source["role"], n_frames=len(items),
            n_batches=n_batches, model="无" if model is None else f"{model.path.name} ({model.kind})"), encoding="utf-8")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return project


def load_project(path: Path) -> dict:
    p = json.loads((path / "project.json").read_text(encoding="utf-8"))
    if p.get("project_version") != PROJECT_VERSION:
        raise ValueError(f"{path}: project_version {p.get('project_version')} is not {PROJECT_VERSION}")
    return p


def iter_frames(path: Path):
    """(batch dir, batch.json dict, frame entry) for every frame of a project, in batch order."""
    for bdir in sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("batch_")):
        batch = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
        for entry in batch["frames"]:
            yield bdir, batch, entry


def _load_doc(bdir: Path, entry: dict) -> tuple[dict | None, list[tuple[str, str]]]:
    jp = bdir / entry["json"]
    if not jp.exists():
        return None, [("E_LM_MISSING", f"{jp.name} is missing")]
    try:
        doc = json.loads(jp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, [("E_LM_JSON", f"{jp.name}: {e}")]
    problems = []
    if doc.get("imagePath") != entry["image"]:
        problems.append(("E_LM_IMAGE", f"{jp.name} points at {doc.get('imagePath')!r}, expected {entry['image']!r}"))
    return doc, problems


def read_project(path: Path, spec: LabelSpec) -> tuple[list[RawSample], list[str]]:
    """RawSamples of the reviewed frames, for nt.ingest's labelme importer."""
    project = load_project(path)
    if project["label_spec_sha256"] != spec.sha256:
        raise ValueError(f"{path} was exported under a different label_spec; re-export it")
    cam = project["source"].get("camera")
    camera = Camera(**cam) if cam else None
    raws, skipped = [], Counter()
    for bdir, batch, entry in iter_frames(path):
        doc, problems = _load_doc(bdir, entry)
        if doc is None:
            skipped["unreadable"] += 1
            continue
        armors, more, flags, _ = lm.from_document(doc, spec)
        if not flags["reviewed"]:
            skipped["unreviewed"] += 1
            continue
        if flags["bad_frame"]:
            skipped["bad_frame"] += 1
            continue
        scene = {**entry.get("scene", {}), "batch": batch["batch"], "assignee": batch.get("assignee"),
                 "annotation_project": project["name"]}
        if project.get("model"):
            scene["prelabel_model_sha256"] = project["model"]["sha256"]
        raws.append(RawSample(
            image_path=bdir / entry["image"], origin=entry["id"], armors=armors, normalized=False,
            declared_hw=(int(doc["imageHeight"]), int(doc["imageWidth"])), camera=camera, scene=scene,
            problems=problems + more,
        ))
    notes = [f"labelme project {project['name']} from {project['source']['kind']} {project['source']['name']}"]
    notes += [f"skipped {n} {k} frames" for k, n in sorted(skipped.items())]
    return raws, notes


def frame_problems(path: Path, spec: LabelSpec, include_unreviewed: bool = False) -> tuple[list[tuple[str, str, str]], Counter]:
    """(file, code, message) for every problem the import would refuse, plus progress counts."""
    out, counts = [], Counter()
    for bdir, _, entry in iter_frames(path):
        counts["frames"] += 1
        doc, problems = _load_doc(bdir, entry)
        name = f"{bdir.name}/{entry['json']}"
        if doc is None:
            out += [(name, c, m) for c, m in problems]
            continue
        armors, more, flags, _ = lm.from_document(doc, spec)
        counts["reviewed"] += flags["reviewed"]
        counts["bad_frame"] += flags["bad_frame"]
        counts["done"] += flags["reviewed"] or flags["bad_frame"]
        counts["unresolved"] += sum(1 for c, _ in more if c == "E_LM_UNRESOLVED")
        counts["armors"] += len(armors)
        if not (flags["reviewed"] or include_unreviewed) or flags["bad_frame"]:
            continue
        sample = Sample(id=entry["id"], image=entry["image"], w=int(doc["imageWidth"]), h=int(doc["imageHeight"]),
                        source="real", armors=armors)
        out += [(name, c, m) for c, m in problems + more]
        out += [(name, i.code, f"armor {i.armor}: {i.msg}" if i.armor is not None else i.msg)
                for i in validate_sample(sample, spec) if i.level == "error"]
    return out, counts


def status(path: Path, spec: LabelSpec) -> list[dict]:
    rows = []
    for bdir in sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("batch_")):
        batch = json.loads((bdir / "batch.json").read_text(encoding="utf-8"))
        c = Counter(frames=len(batch["frames"]))
        for entry in batch["frames"]:
            doc, _ = _load_doc(bdir, entry)
            if doc is None:
                c["missing"] += 1
                continue
            armors, problems, flags, _ = lm.from_document(doc, spec)
            c["reviewed"] += flags["reviewed"]
            c["bad_frame"] += flags["bad_frame"]
            c["armors"] += len(armors)
            c["unresolved"] += sum(1 for code, _ in problems if code == "E_LM_UNRESOLVED")
        rows.append({"batch": bdir.name, "assignee": batch.get("assignee"), **c})
    return rows


def import_project(path: Path, dst: Path, spec: LabelSpec, allow_partial: bool = False, dry_run: bool = False, link: bool = True) -> dict:
    from nt.ingest import ingest

    project = load_project(path)
    _, counts = frame_problems(path, spec)
    open_frames = counts["frames"] - counts["done"]
    if open_frames and not allow_partial:
        raise ValueError(f"{open_frames} of {counts['frames']} frames are neither reviewed nor bad_frame; "
                         "finish them or pass --allow-partial")
    src = project["source"]
    return ingest(path, dst, "labelme", src.get("real_source", "real"), dst.name, spec, dry_run=dry_run, link=link, role=src["role"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--session", type=Path)
    e.add_argument("--dataset", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--model", type=Path)
    e.add_argument("--conf", type=float, default=0.5, help="plain labels at or above this; below it down to --review-conf they are \"?\"")
    e.add_argument("--review-conf", type=float, default=0.1)
    e.add_argument("--tiles", type=int, default=0, choices=[0, 2])
    e.add_argument("--frames", type=Path, help="text file with one frame id (or dataset sample id) per line")
    e.add_argument("--batch-size", type=int, default=200)
    e.add_argument("--copy", action="store_true", help="copy images instead of hard-linking them")
    e.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    e.add_argument("--min-aspect", type=float, default=MIN_ASPECT,
                   help="pre-labels flatter than this horizontal/vertical edge ratio become \"?\"")
    for name in ("status", "check"):
        p = sub.add_parser(name)
        p.add_argument("project", type=Path)
        if name == "check":
            p.add_argument("--all", action="store_true", help="also check frames nobody marked reviewed")
    i = sub.add_parser("import")
    i.add_argument("project", type=Path)
    i.add_argument("--dst", type=Path, required=True)
    i.add_argument("--allow-partial", action="store_true")
    i.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    spec = LabelSpec.load()

    if args.cmd == "export":
        p = export(args.out, spec, args.session, args.dataset, args.model, args.conf, args.review_conf, args.tiles,
                   args.frames, args.batch_size, args.copy, args.device, args.min_aspect)
        print(f"wrote {args.out}: {p['frames']} frames in {p['batches']} batches, {json.dumps(p['counts'])}")
        print(f"open with: labelme {args.out}/batch_000 --config {args.out}/labelme_config.yaml --nodata --autosave")
    elif args.cmd == "status":
        rows = status(args.project, spec)
        keys = ["batch", "assignee", "frames", "reviewed", "bad_frame", "armors", "unresolved", "missing"]
        print("  ".join(f"{k:>10s}" for k in keys))
        for r in rows:
            print("  ".join(f"{str(r.get(k, 0) if r.get(k) is not None else '-'):>10s}" for k in keys))
        tot = Counter()
        for r in rows:
            tot.update({k: r.get(k, 0) for k in keys[2:]})
        print(f"total: {tot['reviewed']}/{tot['frames']} reviewed, {tot['bad_frame']} bad, {tot['armors']} armors, "
              f"{tot['unresolved']} unresolved pre-labels")
    elif args.cmd == "check":
        problems, counts = frame_problems(args.project, spec, args.all)
        for name, code, msg in problems:
            print(f"{name}: {code} {msg}")
        print(f"{len(problems)} problems in {counts['reviewed']} reviewed of {counts['frames']} frames")
        sys.exit(1 if problems else 0)
    else:
        m = import_project(args.project, args.dst, spec, args.allow_partial, args.dry_run)
        print(json.dumps({k: m[k] for k in ("counts", "class_counts", "issues", "rejected", "notes", "role")}, indent=2, ensure_ascii=False))
        if not args.dry_run:
            print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
