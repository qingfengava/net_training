"""Capture sessions: the unit real data is collected, labeled and split by.

    python -m nt.session create --src VIDEO_OR_IMAGE_DIR --name NAME --role train|val|hard|neg \\
        [--camera sleepy.yaml|camera.json] [--camera-key KEY] [--every N] [--start S] [--end S] [--max N] \\
        [--dedupe-hamming H] [--notes TEXT] [--root sessions]
    python -m nt.session show sessions/NAME

A session is one recording (or one folder of stills) from one camera setup. Frames of a session are strongly
correlated, so the whole session gets one role and is never split between training and evaluation:
    train   may be used for training
    val     real_val: the held-out set the model decisions are made on
    hard    real_hard: blur, occlusion, dark scenes, reported separately
    neg     frames that must stay empty (field LED screens, monitors, light-bar modules); every detection is an error

sessions/<name>/
    session.json   source file and its hash, role, camera intrinsics, frame list with time stamps and image stats
    frames/        <name>_<index>.png for video sources; stills keep their own format so a JPEG stays visibly lossy

Sessions are immutable like datasets: the directory is built next to its destination and renamed at the end.
Videos are decoded with OpenCV. A lossy source codec (H.264, MJPEG, ...) is recorded as a warning, because
compression blurs the dot-matrix detail the network has to learn; record losslessly when you can.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from nt.canonical import ROLES, Camera
from nt.ingest import IMAGE_EXTS
from nt.spec import LabelSpec
from nt.validators import dhash64

ROOT = Path(__file__).resolve().parents[1]
SESSION_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
LOSSLESS_CODECS = {"ffv1", "png", "rawvideo", "huffyuv", "utvideo", "ffvhuff", "qtrle"}
SESSION_ROLES = tuple(r for r in ROLES if r != "auto")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_codec(path: Path) -> str | None:
    """Codec name from ffprobe when it is installed; OpenCV's FOURCC is the fallback."""
    if shutil.which("ffprobe"):
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        name = res.stdout.strip().splitlines()[0] if res.returncode == 0 and res.stdout.strip() else ""
        if name:
            return name.lower()
    cap = cv2.VideoCapture(str(path))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip().lower() or None


def _camera_nodes(node, path: str = "") -> list[tuple[str, dict]]:
    """Every mapping that holds camera_matrix and distortion_coefficients, with its dotted key path."""
    found = []
    if isinstance(node, dict):
        if "camera_matrix" in node and "distortion_coefficients" in node:
            found.append((path, node))
        for k, v in node.items():
            found += _camera_nodes(v, f"{path}.{k}" if path else str(k))
    return found


def load_camera(path: Path, key: str | None = None) -> tuple[Camera, tuple[int, int] | None, str]:
    """Intrinsics from a sleepy config (ROS camera_info layout) or a JSON {fx, fy, cx, cy, dist}.

    Returns the camera, the (width, height) it was calibrated at when the file says so, and a description.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        d = json.loads(text)
        dist = [float(x) for x in d.get("dist", [0.0] * 5)]
        size = (int(d["width"]), int(d["height"])) if "width" in d and "height" in d else None
        return Camera(float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), (dist + [0.0] * 5)[:5]), size, str(path)
    nodes = _camera_nodes(yaml.safe_load(text))
    if key is not None:
        nodes = [(p, n) for p, n in nodes if p == key]
    if not nodes:
        raise ValueError(f"{path}: no camera_info block{f' at {key}' if key else ''}")
    if len(nodes) > 1:
        raise ValueError(f"{path}: several camera_info blocks, pick one with --camera-key: {[p for p, _ in nodes]}")
    where, n = nodes[0]
    k = [float(x) for x in n["camera_matrix"]["data"]]
    dist = [float(x) for x in n["distortion_coefficients"]["data"]]
    if len(k) != 9:
        raise ValueError(f"{path}:{where}: camera_matrix needs 9 values")
    if n.get("distortion_model", "plumb_bob") not in ("plumb_bob", "rational_polynomial"):
        raise ValueError(f"{path}:{where}: distortion model {n['distortion_model']!r} is not supported")
    size = (int(n["image_width"]), int(n["image_height"])) if "image_width" in n and "image_height" in n else None
    return Camera(k[0], k[4], k[2], k[5], (dist + [0.0] * 5)[:5]), size, f"{path}:{where}"


def _frame_stats(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return {"sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2), "brightness": round(float(gray.mean()), 2)}


def _iter_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield idx, img
            idx += 1
    finally:
        cap.release()


def create(
    src: Path,
    root: Path,
    name: str,
    role: str,
    camera_path: Path | None = None,
    camera_key: str | None = None,
    every: int = 1,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int = 0,
    dedupe_hamming: int | None = None,
    notes: str = "",
) -> Path:
    if not NAME_RE.match(name):
        raise ValueError(f"session name {name!r}: use letters, digits, '_' and '-'")
    if role not in SESSION_ROLES:
        raise ValueError(f"role must be one of {SESSION_ROLES}")
    if every < 1:
        raise ValueError("--every must be at least 1")
    out = root / name
    if out.exists():
        raise FileExistsError(f"{out} exists; sessions are immutable, use a new name")
    if dedupe_hamming is None:
        dedupe_hamming = LabelSpec.load().v("near_duplicate_hamming")

    warnings: list[str] = []
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"no images in {src}")
        source = {"kind": "images", "path": str(src.resolve()), "files": len(files), "fps": None, "codec": None}
        lossy = sorted({p.suffix.lower() for p in files} & {".jpg", ".jpeg"})
        if lossy:
            warnings.append(f"stills are {', '.join(lossy)} (lossy)")
        frames_in = ((i, p) for i, p in enumerate(files))
        fps = None
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError(f"cannot open {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        codec = _probe_codec(src)
        source = {"kind": "video", "path": str(src.resolve()), "sha256": _sha256(src), "bytes": src.stat().st_size,
                  "codec": codec, "fps": fps, "frames_reported": n_frames}
        if codec not in LOSSLESS_CODECS:
            warnings.append(f"source codec {codec!r} is lossy: compression blurs dot-matrix detail")
        frames_in = _iter_video(src)

    camera, calib_size, camera_desc = (None, None, None)
    if camera_path is not None:
        camera, calib_size, camera_desc = load_camera(camera_path, camera_key)
    else:
        warnings.append("no camera intrinsics: PnP-based replay metrics are unavailable for this session")

    tmp = root / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "frames").mkdir(parents=True)
    kept, dropped_dupes, size, last_hash = [], 0, None, None
    try:
        for idx, item in frames_in:
            t = idx / fps if fps else None
            if t is not None and (t < start_s or (end_s is not None and t > end_s)):
                if end_s is not None and t > end_s:
                    break
                continue
            if idx % every:
                continue
            img = cv2.imread(str(item), cv2.IMREAD_COLOR) if isinstance(item, Path) else item
            if img is None:
                warnings.append(f"unreadable frame {item}")
                continue
            hw = img.shape[:2]
            if size is None:
                size = hw
            elif hw != size:
                raise ValueError(f"frame {idx} is {hw[1]}x{hw[0]}, the session started at {size[1]}x{size[0]}")
            h = dhash64(img)
            if dedupe_hamming >= 0 and last_hash is not None and int(np.bitwise_count(h ^ last_hash)) <= dedupe_hamming:
                dropped_dupes += 1
                continue
            last_hash = h
            fid = f"{name}_{idx:06d}"
            if isinstance(item, Path):
                image = f"frames/{fid}{item.suffix.lower()}"
                shutil.copy2(item, tmp / image)
            else:
                image = f"frames/{fid}.png"
                cv2.imwrite(str(tmp / image), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            kept.append({"id": fid, "index": idx, "t_s": None if t is None else round(t, 4), "image": image,
                         "dhash": f"{int(h):016x}", **_frame_stats(img)})
            if max_frames and len(kept) >= max_frames:
                break
        if not kept:
            raise ValueError("no frames selected; check --start/--end/--every")
        h_px, w_px = size
        if calib_size is not None and calib_size != (w_px, h_px):
            raise ValueError(f"camera was calibrated at {calib_size[0]}x{calib_size[1]} but frames are {w_px}x{h_px}")
        meta = {
            "session_version": SESSION_VERSION, "name": name, "role": role,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source, "width": w_px, "height": h_px,
            "camera": None if camera is None else camera.__dict__, "camera_source": camera_desc,
            "selection": {"every": every, "start_s": start_s, "end_s": end_s, "max_frames": max_frames,
                          "dedupe_hamming": dedupe_hamming, "dropped_near_duplicates": dropped_dupes, "kept": len(kept)},
            "warnings": warnings, "notes": notes, "frames": kept,
        }
        (tmp / "session.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def load(path: Path) -> dict:
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    if meta.get("session_version") != SESSION_VERSION:
        raise ValueError(f"{path}: session_version {meta.get('session_version')} is not {SESSION_VERSION}")
    return meta


def camera_of(meta: dict) -> Camera | None:
    return None if meta.get("camera") is None else Camera(**meta["camera"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="build a session from a video or an image folder")
    c.add_argument("--src", type=Path, required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--role", choices=SESSION_ROLES, required=True)
    c.add_argument("--camera", type=Path, default=None, help="sleepy config with a camera_info block, or camera JSON")
    c.add_argument("--camera-key", default=None, help="dotted path of the camera_info block when the file has several")
    c.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    c.add_argument("--start", type=float, default=0.0, help="seconds")
    c.add_argument("--end", type=float, default=None, help="seconds")
    c.add_argument("--max", type=int, default=0, help="stop after N kept frames")
    c.add_argument("--dedupe-hamming", type=int, default=None,
                   help="drop a frame whose dHash is within H bits of the previous kept one (-1 keeps all; default from label_spec)")
    c.add_argument("--notes", default="")
    c.add_argument("--root", type=Path, default=ROOT / "sessions")
    s = sub.add_parser("show", help="summarize a session")
    s.add_argument("session", type=Path)
    args = ap.parse_args()

    if args.cmd == "create":
        args.root.mkdir(parents=True, exist_ok=True)
        out = create(args.src, args.root, args.name, args.role, args.camera, args.camera_key, args.every, args.start,
                     args.end, args.max, args.dedupe_hamming, args.notes)
        meta = load(out)
        print(f"wrote {out}: {len(meta['frames'])} frames, role {meta['role']}")
        for w in meta["warnings"]:
            print(f"  warning: {w}")
        return
    meta = load(args.session)
    sharp = [f["sharpness"] for f in meta["frames"]]
    print(json.dumps({k: meta[k] for k in ("name", "role", "width", "height", "camera_source", "selection", "warnings", "notes")},
                     indent=2, ensure_ascii=False))
    print(f"frames: {len(meta['frames'])}, sharpness p10/p50/p90 {np.percentile(sharp, 10):.0f}/{np.percentile(sharp, 50):.0f}/{np.percentile(sharp, 90):.0f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
