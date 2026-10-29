"""labelme JSON conventions for armor annotation, and conversion to and from canonical armors.

One JSON per image, next to it, readable by labelme (and X-AnyLabeling, which uses the same format):
    armor       polygon, exactly 4 points in the order LT, LB, RB, RT, label "<color>_<label>" from label_spec,
                e.g. red_three. RM27 has small armor only, so there is no size to mark.
    occluded    point placed on a corner that is hidden but whose position can be estimated (vis 1)
    outside     point placed on a corner that lies outside the image; the corner is clamped to the border (vis 0)
                Markers written by the export share the group_id of their polygon, so deleting a polygon leaves
                markers whose group no longer exists; those are dropped. A marker without a group_id (added by
                hand) goes to the nearest corner and is an error when it sits on no corner.
    ?<anything> a pre-label the annotator still has to resolve: rename it to a real label or delete it.
                Low-confidence model output and suspicious mined cases are written this way, and a frame that
                still holds one is refused on import.
Image flags:
    reviewed    a person checked every armor in the frame; only reviewed frames are imported
    bad_frame   unusable (corrupt, ambiguous); left out of the dataset and counted

Coordinates: labelme maps a click to image coordinates as widget position / zoom with pixel i covering
[i, i+1), the same "edge" convention as label_spec, so no half-pixel shift is applied in either direction.
Our own metadata goes in a top-level "nt" key, which labelme keeps when it saves the file.
"""
from __future__ import annotations

import math
from typing import Iterable

from nt.canonical import Armor
from nt.geometry import edge_lengths
from nt.spec import LabelSpec

MARKERS = {"occluded": 1, "outside": 0}
FLAGS = ("reviewed", "bad_frame")
UNRESOLVED_PREFIX = "?"
LABELME_VERSION = "5.4.1"  # format version written into the file; labelme 5.x and 6.x read it


def armor_label(spec: LabelSpec, color: int, label: int) -> str:
    return spec.pair_name(color, label)


def armor_labels(spec: LabelSpec) -> dict[str, tuple[int, int]]:
    """label text -> (color slot, label id), for every color, trained or not: canonical data keeps them all. The
    off color also gets <off color>_<off name> (gray_unknown) for a dark screen whose number nobody can see."""
    names = {armor_label(spec, c, l): (c, l) for c in sorted(spec.colors) for l in sorted(spec.labels)}
    if spec.off_class is not None:
        c, n, _ = spec.off_class
        names[armor_label(spec, c, n)] = (c, n)
    return names


def labelme_config(spec: LabelSpec) -> dict:
    """A labelme --config file: fixed label list and the image flags."""
    labels = list(armor_labels(spec)) + list(MARKERS)
    cfg = {
        "auto_save": True,
        "with_image_data": False,
        "keep_prev": False,
        "sort_labels": False,
        "validate_label": "exact",
        "labels": labels,
        "flags": list(FLAGS),
        "label_flags": {},
        "shape_color": "auto",
    }
    return cfg


def polygon(points: Iterable[Iterable[float]], label: str, description: str = "", group_id: int | None = None) -> dict:
    return {
        "label": label,
        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def point(xy: Iterable[float], label: str, group_id: int | None = None) -> dict:
    x, y = xy
    return {"label": label, "points": [[round(float(x), 3), round(float(y), 3)]], "group_id": group_id, "description": "",
            "shape_type": "point", "flags": {}, "mask": None}


def document(image_name: str, w: int, h: int, shapes: list[dict], flags: dict | None = None, nt_meta: dict | None = None) -> dict:
    doc = {
        "version": LABELME_VERSION,
        "flags": {f: False for f in FLAGS} | (flags or {}),
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    if nt_meta is not None:
        doc["nt"] = nt_meta
    return doc


def armor_shapes(spec: LabelSpec, color: int, label: int, corners, vis=(2, 2, 2, 2), w: int | None = None, h: int | None = None,
                 description: str = "", unresolved: bool = False, group_id: int | None = None) -> list[dict]:
    """The polygon for one armor plus a marker per corner that is not plainly visible.

    Corners outside the image are clamped to its border (labelme cannot place points outside) and marked
    "outside", whatever their vis said, because a clamped coordinate is not the corner's position.
    """
    pts, marks = [], []
    for (x, y), v in zip(corners, vis):
        cx, cy = float(x), float(y)
        out = w is not None and h is not None and not (0 <= cx <= w and 0 <= cy <= h)
        if out:
            cx, cy = min(max(cx, 0.0), float(w)), min(max(cy, 0.0), float(h))
        pts.append((cx, cy))
        if out or v == 0:
            marks.append(point((cx, cy), "outside", group_id))
        elif v == 1:
            marks.append(point((cx, cy), "occluded", group_id))
    text = armor_label(spec, color, label)
    return [polygon(pts, UNRESOLVED_PREFIX + text if unresolved else text, description, group_id)] + marks


def from_document(doc: dict, spec: LabelSpec) -> tuple[list[Armor], list[tuple[str, str]], dict, list[str]]:
    """Armors, (code, message) problems, image flags and informational notes of one labelme document.

    Problems are errors for the ingest gate: a frame with any of them is refused as a whole.
    """
    names = armor_labels(spec)
    problems: list[tuple[str, str]] = []
    polys, marks = [], []
    for i, s in enumerate(doc.get("shapes") or []):
        label = s.get("label", "")
        kind = s.get("shape_type")
        pts = s.get("points") or []
        if label.startswith(UNRESOLVED_PREFIX):
            problems.append(("E_LM_UNRESOLVED", f"shape {i} {label!r} is an unresolved pre-label: rename or delete it"))
        elif label in MARKERS:
            if kind != "point" or len(pts) != 1:
                problems.append(("E_LM_SHAPE", f"shape {i} {label!r} must be a single point"))
            else:
                marks.append((i, label, pts[0], s.get("group_id")))
        elif label in names:
            if kind != "polygon":
                problems.append(("E_LM_SHAPE", f"shape {i} {label!r} must be a polygon, is {kind!r}"))
            elif len(pts) != 4:
                problems.append(("E_LM_POINTS", f"shape {i} {label!r} has {len(pts)} points, needs 4 (LT, LB, RB, RT)"))
            else:
                polys.append((i, label, s))
        else:
            problems.append(("E_LM_LABEL", f"shape {i}: unknown label {label!r}"))

    armors, groups, notes = [], {}, []
    for i, label, s in polys:
        color, lab = names[label]
        gid = s.get("group_id")
        if gid is not None:
            if gid in groups:
                problems.append(("E_LM_GROUP", f"shapes {groups[gid]} and {i} are both polygons of group {gid}; give one a new group id"))
            groups[gid] = len(armors)
        armors.append(Armor(color=color, label=lab, corners=[[float(x), float(y)] for x, y in s["points"]], vis=[2, 2, 2, 2],
                            type="small"))
    # A group id of a polygon that is no longer there means the annotator deleted it; its markers go with it.
    # Ids of unresolved "?" polygons count as present, so their markers do not vanish before they are resolved.
    pending = {s.get("group_id") for s in (doc.get("shapes") or [])
               if s.get("label", "").startswith(UNRESOLVED_PREFIX) and s.get("group_id") is not None}

    claimed: dict[tuple[int, int], str] = {}
    for i, label, (mx, my), gid in marks:
        if gid is not None and gid not in groups:
            if gid not in pending:
                notes.append(f"dropped {label!r} marker of deleted armor group {gid}")
            continue
        hits = []
        for a_idx, a in enumerate(armors):
            if gid is not None and a_idx != groups[gid]:
                continue
            radius = max(6.0, 0.35 * min(edge_lengths(a.corners)))
            for c_idx, (x, y) in enumerate(a.corners):
                d = math.dist((x, y), (mx, my))
                if d <= radius:
                    hits.append((d, a_idx, c_idx))
        if not hits:
            problems.append(("E_LM_MARKER", f"shape {i} {label!r} at ({mx:.1f},{my:.1f}) is not on any armor corner"))
            continue
        hits.sort()
        if len(hits) > 1 and hits[1][0] - hits[0][0] < 1.0 and hits[1][1:] != hits[0][1:]:
            problems.append(("E_LM_MARKER", f"shape {i} {label!r} at ({mx:.1f},{my:.1f}) is as close to two corners; move it onto one"))
            continue
        _, a_idx, c_idx = hits[0]
        if (a_idx, c_idx) in claimed:
            problems.append(("E_LM_MARKER", f"shape {i}: corner {c_idx} of armor {a_idx} already has a {claimed[(a_idx, c_idx)]!r} marker"))
            continue
        claimed[(a_idx, c_idx)] = label
        armors[a_idx].vis[c_idx] = MARKERS[label]

    flags = {f: bool((doc.get("flags") or {}).get(f)) for f in FLAGS}
    return armors, problems, flags, notes
