"""Replay evaluation on a recorded session: stability of what the tracker would be fed, no labels needed.

    python -m nt.replay sessions/S --model M.onnx --out runs/replay/NAME [--conf 0.5] [--tiles 0|2]
                        [--geometry configs/geometry.yaml] [--plot]

The model runs on every session frame in order. Detections are chained into tracks by box overlap with the
previous frame (same color), and each detection is solved with IPPE PnP the way sleepy's armor_pnp does
(object points LT, LB, RB, RT; the solution whose front faces the camera), using the session's intrinsics
and configs/geometry.yaml. Reported, per session and per track:
    continuity   track length, frames missing inside a track (dropouts), number/color changes (flips)
    jitter       frame-to-frame change of distance (% of distance) and of plate yaw and pitch (deg), as
                 median and p95; "yaw jumps" count changes above --yaw-jump, the typical PnP ambiguity flip
    negatives    for a session with role neg every detection is a false positive: count and rate per minute
Jitter mixes target motion with estimation noise, so compare models on the same session, not across sessions.
Use dense sessions (`nt.session create --every 1`); with a stride the frame-to-frame numbers mostly measure
motion. Without intrinsics only continuity and negatives are computed.

Outputs: summary.json, frames.csv (one row per detection), and with --plot tracks.png (longest tracks).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from nt import session as sess
from nt.detector import ArmorModel, Detection
from nt.geometry import bbox_iou
from nt.spec import LabelSpec

ROOT = Path(__file__).resolve().parents[1]


def load_geometry(path: Path, spec: LabelSpec) -> dict:
    g = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "size" not in g:
        raise ValueError(f"{path}: needs size: {{width_m, height_m}} (RM27 has one armor size)")
    return g


def object_points(width: float, height: float) -> np.ndarray:
    # sleepy ArmorKeyPoint3D::build(): x forward (0), y left, z up; order LT, LB, RB, RT
    return np.array([[0, width / 2, height / 2], [0, width / 2, -height / 2], [0, -width / 2, -height / 2],
                     [0, -width / 2, height / 2]], np.float64)


def solve(d: Detection, K: np.ndarray, dist: np.ndarray, geometry: dict) -> dict | None:
    size = geometry["size"]
    obj = object_points(size["width_m"], size["height_m"])
    # labels use the edge convention, OpenCV puts pixel centers on integers
    img = (d.corners - 0.5).astype(np.float64)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    for r, t, e in zip(rvecs, tvecs, errs):
        R, _ = cv2.Rodrigues(r)
        front = -R[:, 0]
        t = t.ravel()
        if front @ (-t) > 0:
            # yaw: plate normal around the camera's vertical axis, 0 when it faces the camera; pitch likewise
            yaw = math.degrees(math.atan2(-front[0], -front[2]))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, front[1]))))
            return {"dist_m": float(np.linalg.norm(t)), "yaw_deg": yaw, "pitch_deg": pitch,
                    "reproj_px": float(np.ravel(e)[0])}
    return None


def link_tracks(dets: list[list[Detection]], max_gap: int = 2, iou_thr: float = 0.3) -> list[list[tuple[int, int]]]:
    """Tracks as lists of (frame position, detection index). A track survives up to max_gap missing frames."""
    tracks: list[list[tuple[int, int]]] = []
    open_tracks: list[int] = []
    for k, ds in enumerate(dets):
        taken = set()
        still_open = []
        cands = []
        for t_idx in open_tracks:
            fk, di = tracks[t_idx][-1]
            if k - fk > max_gap + 1:
                continue
            prev = dets[fk][di]
            for j, d in enumerate(ds):
                if d.color == prev.color:
                    cands.append((bbox_iou(prev.box, d.box), t_idx, j))
            still_open.append(t_idx)
        assigned = set()
        for iou, t_idx, j in sorted(cands, reverse=True):
            if iou < iou_thr or t_idx in assigned or j in taken:
                continue
            tracks[t_idx].append((k, j))
            assigned.add(t_idx)
            taken.add(j)
        for j in range(len(ds)):
            if j not in taken:
                tracks.append([(k, j)])
                still_open.append(len(tracks) - 1)
        open_tracks = still_open
    return tracks


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.abs(np.array(values))
    return {"n": int(a.size), "median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4)}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def replay(session_dir: Path, model: ArmorModel, spec: LabelSpec, geometry: dict, conf: float, tiles: int, yaw_jump: float,
           min_track: int = 5) -> tuple[dict, list[dict], list[list[tuple[int, int]]], list[list[Detection]], list[list[dict | None]]]:
    meta = sess.load(session_dir)
    frames = sorted(meta["frames"], key=lambda f: f["index"])
    camera = sess.camera_of(meta)
    K = dist = None
    if camera is not None:
        K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], np.float64)
        dist = np.array(camera.dist, np.float64)
    dets, poses = [], []
    for i, f in enumerate(frames):
        img = cv2.imread(str(session_dir / f["image"]), cv2.IMREAD_COLOR)
        ds = model.detect(img, conf, tiles)
        dets.append(ds)
        poses.append([solve(d, K, dist, geometry) if K is not None else None for d in ds])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(frames)} frames", file=sys.stderr, flush=True)

    steps = np.diff([f["index"] for f in frames])
    stride = int(np.median(steps)) if len(steps) else 1
    tracks = link_tracks(dets)
    long_tracks = [t for t in tracks if len(t) >= min_track]
    d_dist, d_yaw, d_pitch, yaw_jumps, flips, dropouts = [], [], [], 0, 0, 0
    per_track = []
    for t in long_tracks:
        t_flips = t_drop = t_jumps = 0
        for (k0, j0), (k1, j1) in zip(t, t[1:]):
            a, b = dets[k0][j0], dets[k1][j1]
            gap = (frames[k1]["index"] - frames[k0]["index"]) // max(stride, 1) - 1
            t_drop += max(gap, 0)
            if (a.color, a.label) != (b.color, b.label):
                t_flips += 1
            pa, pb = poses[k0][j0], poses[k1][j1]
            if pa and pb and gap <= 0:
                d_dist.append(100 * (pb["dist_m"] - pa["dist_m"]) / pa["dist_m"])
                dy = _wrap(pb["yaw_deg"] - pa["yaw_deg"])
                d_yaw.append(dy)
                d_pitch.append(pb["pitch_deg"] - pa["pitch_deg"])
                if abs(dy) > yaw_jump:
                    t_jumps += 1
        flips += t_flips
        dropouts += t_drop
        yaw_jumps += t_jumps
        first = dets[t[0][0]][t[0][1]]
        per_track.append({"start_frame": frames[t[0][0]]["id"], "frames": len(t), "dropouts": t_drop, "flips": t_flips,
                          "yaw_jumps": t_jumps, "class": f"{spec.colors[first.color]}_{spec.labels[first.label]}"})
    n_det = sum(len(ds) for ds in dets)
    duration = None
    if frames and frames[-1].get("t_s") is not None and frames[0].get("t_s") is not None:
        duration = frames[-1]["t_s"] - frames[0]["t_s"]
    summary = {
        "session": meta["name"], "role": meta["role"], "model": model.describe(), "conf": conf, "tiles": tiles,
        "geometry": geometry.get("corner_definition"), "frames": len(frames), "frame_stride": stride,
        "duration_s": duration, "detections": n_det,
        "frames_with_detections": sum(1 for ds in dets if ds),
        "classes": dict(Counter(f"{spec.colors[d.color]}_{spec.labels[d.label]}" for ds in dets for d in ds)),
        "tracks": {"all": len(tracks), f"at_least_{min_track}_frames": len(long_tracks),
                   "dropouts": dropouts, "flips": flips, "yaw_jumps": yaw_jumps},
        "jitter": None if K is None else {"distance_pct": _stats(d_dist), "yaw_deg": _stats(d_yaw), "pitch_deg": _stats(d_pitch),
                                          "yaw_jump_threshold_deg": yaw_jump},
        "pnp_failures": sum(1 for ps in poses for p in ps if p is None) if K is not None else None,
        "per_track": sorted(per_track, key=lambda r: -r["frames"])[:50],
    }
    if meta["role"] == "neg":
        summary["negatives"] = {"false_positives": n_det,
                                "per_minute": None if not duration else round(60 * n_det / duration, 2)}
    rows = []
    for k, (f, ds, ps) in enumerate(zip(frames, dets, poses)):
        for j, (d, p) in enumerate(zip(ds, ps)):
            rows.append({"frame": f["id"], "index": f["index"], "t_s": f["t_s"], "det": j, "conf": round(d.conf, 4),
                         "class": f"{spec.colors[d.color]}_{spec.labels[d.label]}",
                         **({} if p is None else {k2: round(v, 4) if isinstance(v, float) else v for k2, v in p.items()})})
    return summary, rows, long_tracks, dets, poses


def plot(path: Path, frames: list[dict], tracks, poses, n: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for t in sorted(tracks, key=len, reverse=True)[:n]:
        xs = [frames[k]["index"] for k, j in t if poses[k][j]]
        axes[0].plot(xs, [poses[k][j]["dist_m"] for k, j in t if poses[k][j]], ".-", ms=2, lw=0.8)
        axes[1].plot(xs, [poses[k][j]["yaw_deg"] for k, j in t if poses[k][j]], ".-", ms=2, lw=0.8)
    axes[0].set_ylabel("distance (m)")
    axes[1].set_ylabel("plate yaw (deg)")
    axes[1].set_xlabel("frame index")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--tiles", type=int, default=0, choices=[0, 2])
    ap.add_argument("--geometry", type=Path, default=ROOT / "configs" / "geometry.yaml")
    ap.add_argument("--yaw-jump", type=float, default=20.0)
    ap.add_argument("--min-track", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    spec = LabelSpec.load()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    model = ArmorModel(args.model, device=args.device)
    summary, rows, tracks, dets, poses = replay(args.session, model, spec, load_geometry(args.geometry, spec), args.conf,
                                                args.tiles, args.yaw_jump, args.min_track)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    keys = list(dict.fromkeys(k for r in rows for k in r))  # every column, in first-seen order
    with open(args.out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if args.plot and summary["jitter"] is not None:
        plot(args.out / "tracks.png", sorted(sess.load(args.session)["frames"], key=lambda f: f["index"]), tracks, poses)
    print(json.dumps({k: summary[k] for k in ("frames", "detections", "tracks", "jitter") if k in summary}, indent=1))
    if "negatives" in summary:
        print("negatives:", summary["negatives"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
