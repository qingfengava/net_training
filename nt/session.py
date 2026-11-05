"""Capture sessions: the unit real data is collected, labeled and split by.

    python -m nt.session create --src VIDEO_OR_IMAGE_DIR_OR_RECORDER_PNG --name NAME --role train|val|hard|neg \\
        [--camera sleepy.yaml|camera.json] [--camera-key KEY] [--every N] [--start S] [--end S] [--max N] \\
        [--dedupe-hamming H] [--poses FILE.txt | --no-poses] [--notes TEXT] [--root sessions]
    python -m nt.session show sessions/NAME

A session is one recording (or one folder of stills) from one camera setup. Frames of a session are strongly
correlated, so the whole session gets one role and is never split between training and evaluation:
    train   may be used for training
    val     real_val: the held-out set the model decisions are made on
    hard    real_hard: blur, occlusion, dark scenes, reported separately
    neg     frames that must stay empty (field LED screens, monitors, light-bar modules); every detection is an error

sessions/<name>/
    session.json   source file and its hash, role, camera intrinsics, frame list with time stamps, gimbal
                   orientation (sleepy recordings) and image stats
    frames/        <name>_<index>.png for video sources; stills keep their own format so a JPEG stays visibly lossy

Sessions are immutable like datasets: the directory is built next to its destination and renamed at the end.
A sleepy video recording is <time>.avi plus <time>.txt with `t qw qx qy qz` per frame.
The .txt next to a video is read by default: each frame's t_s and q_wxyz (gimbal orientation in gimbal_odom) come
from it, and --start/--end refer to that time; without it t_s is index / container fps.
Recorder PNG 会话使用 frames.jsonl 配对图像、时间、姿态和 camera_frame_id；默认不去重。
可传入会话根目录、frames/ 子目录或 frames.jsonl。PNG 直接复制，不重新编码或交换颜色通道。

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

from nt import recorder
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


def load_poses(path: Path) -> tuple[list[tuple[float, list[float]]], list[str]]:
    """sleepy's recorder file: line i belongs to video frame i and reads `t qw qx qy qz`, with t in seconds since
    the recording started and q the gimbal orientation in gimbal_odom at that frame (src/utils/io/recorder.cpp).

    sleepy's own reader skips a line it cannot parse, which would pair every later frame with the wrong line;
    here any bad line is an error, except a broken last line, which a killed recorder leaves behind.
    Quaternions are normalized as sleepy does on load. Returns the (t, [w, x, y, z]) rows and warnings.
    """
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    rows, warnings = [], []
    for n, ln in enumerate(lines, 1):
        try:
            vals = [float(v) for v in ln.split()]
            if len(vals) != 5 or not all(np.isfinite(vals)):
                raise ValueError
            q = np.array(vals[1:], np.float64)
            norm = float(np.linalg.norm(q))
            if norm < 1e-6:
                raise ValueError
        except ValueError:
            if n == len(lines):
                warnings.append(f"{path.name}: dropped a broken last line {n}")
                break
            raise ValueError(f"{path}: line {n} is not 't qw qx qy qz': {ln!r}") from None
        if rows and vals[0] < rows[-1][0]:
            raise ValueError(f"{path}: line {n} goes back in time ({vals[0]} after {rows[-1][0]})")
        rows.append((vals[0], [round(float(v), 7) for v in q / norm]))
    if not rows:
        raise ValueError(f"{path}: no pose lines")
    return rows, warnings


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
    poses_path: Path | None | bool = None,
) -> Path:
    """poses_path: sleepy's recorder file for a video; None looks for <video>.txt next to it, False ignores it."""
    if not NAME_RE.match(name):
        raise ValueError(f"session name {name!r}: use letters, digits, '_' and '-'")
    if role not in SESSION_ROLES:
        raise ValueError(f"role must be one of {SESSION_ROLES}")
    if every < 1:
        raise ValueError("--every must be at least 1")
    out = root / name
    if out.exists():
        raise FileExistsError(f"{out} exists; sessions are immutable, use a new name")
    warnings: list[str] = []
    recorder_rows = {}
    recorder_root = recorder.png_root(src)
    if recorder_root is not None:
        if poses_path:
            raise ValueError("Recorder PNG 使用 frames.jsonl 中的姿态，不支持 --poses 覆盖")
        rows, source, warnings = recorder.load_png(recorder_root)
        recorder_rows = {row["index"]: row for row in rows}
        frames_in = ((row["index"], row["path"]) for row in rows)
        fps, poses = None, None
        source["poses_ignored"] = poses_path is False
    elif src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"no images in {src}")
        source = {"kind": "images", "path": str(src.resolve()), "files": len(files), "fps": None, "codec": None}
        lossy = sorted({p.suffix.lower() for p in files} & {".jpg", ".jpeg"})
        if lossy:
            warnings.append(f"stills are {', '.join(lossy)} (lossy)")
        frames_in = ((i, p) for i, p in enumerate(files))
        fps = None
        if poses_path:
            raise ValueError("--poses goes with a video source")
        poses = None
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
        if poses_path is None and src.with_suffix(".txt").exists():
            poses_path = src.with_suffix(".txt")
        poses = None
        if poses_path:
            poses, pose_warnings = load_poses(Path(poses_path))
            warnings += pose_warnings
            source["poses"] = {"path": str(Path(poses_path).resolve()), "sha256": _sha256(Path(poses_path)), "lines": len(poses),
                               "format": "sleepy recorder: t qw qx qy qz, gimbal orientation in gimbal_odom"}
    source.setdefault("time_base", "poses" if poses else ("fps" if fps else None))
    if dedupe_hamming is None:
        # Recorder 会话默认保留连续帧；抽帧和去重必须由用户显式选择。
        dedupe_hamming = -1 if recorder_root is not None else LabelSpec.load().v("near_duplicate_hamming")

    camera, calib_size, camera_desc = (None, None, None)
    if camera_path is not None:
        camera, calib_size, camera_desc = load_camera(camera_path, camera_key)
    else:
        warnings.append("no camera intrinsics: PnP-based replay metrics are unavailable for this session")

    tmp = root / f".{name}.tmp"
    if tmp.exists():
        raise FileExistsError(f"{tmp} 已存在，保留未完成数据；请换一个会话名")
    (tmp / "frames").mkdir(parents=True)
    kept, dropped_dupes, size, last_hash = [], 0, None, None
    decoded, stopped_early, no_pose = 0, False, 0
    try:
        for idx, item in frames_in:
            decoded = idx + 1
            q = None
            record = recorder_rows.get(idx)
            if record is not None:
                t = record["t_s"]
                q = record["q_wxyz"] if poses_path is not False else None
            elif poses is not None:
                # frame times come from the recorder: the container frame rate is nominal and the recorder drops
                # frames when its writer is busy, so index / fps drifts from the real time
                t, q = poses[idx] if idx < len(poses) else (None, None)
                no_pose += idx >= len(poses)
            else:
                t = idx / fps if fps else None
            if t is not None and (t < start_s or (end_s is not None and t > end_s)):
                if end_s is not None and t > end_s:
                    stopped_early = True
                    break
                continue
            if idx % every:
                continue
            flags = cv2.IMREAD_UNCHANGED if record is not None else cv2.IMREAD_COLOR
            img = cv2.imread(str(item), flags) if isinstance(item, Path) else item
            if img is None:
                if record is not None:
                    raise ValueError(f"Recorder PNG 无法解码：{item}")
                warnings.append(f"unreadable frame {item}")
                continue
            if record is not None:
                channels = img.shape[2] if img.ndim == 3 else 1
                if (img.shape[:2] != (record["height"], record["width"])
                        or channels != record["channels"] or img.dtype != np.uint8):
                    raise ValueError(f"Recorder PNG 的实际尺寸、通道或位深与元数据不符：{item}")
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
            extras = ({k: record[k] for k in ("write_index", "camera_frame_id", "source_image",
                                              "source_pixel_format", "pixel_format")} if record is not None else {})
            kept.append({"id": fid, "index": idx,
                         "t_s": t if record is not None or t is None else round(t, 4), "image": image,
                         "dhash": f"{int(h):016x}", **({"q_wxyz": q} if q is not None else {}),
                         **extras, **_frame_stats(img)})
            if max_frames and len(kept) >= max_frames:
                stopped_early = True
                break
        if not kept:
            raise ValueError("no frames selected; check --start/--end/--every")
        if poses is not None:
            if no_pose:
                warnings.append(f"the pose file has {len(poses)} lines but the video goes on: {no_pose} frames have no time or pose")
            elif not stopped_early and decoded < len(poses):
                warnings.append(f"the pose file has {len(poses)} lines but the video only {decoded} frames (truncated video?)")
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


def frame_scene(meta: dict, frame: dict) -> dict:
    """标注和代理屏入库时保留采集元数据，方便按原始帧追溯。"""
    keys = ("camera_frame_id", "write_index", "q_wxyz", "source_image", "source_pixel_format", "pixel_format")
    return {"session": meta["name"], "frame_index": frame["index"], "t_s": frame["t_s"],
            **{k: frame[k] for k in keys if k in frame}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="从视频、图片目录或 Recorder PNG 会话创建采集会话")
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
                   help="相邻保留帧的 dHash 去重阈值；-1 不去重，Recorder PNG 默认 -1，其他来源默认从 label_spec 读取")
    c.add_argument("--poses", type=Path, default=None,
                   help="sleepy recorder file (t qw qx qy qz per frame); default: <video>.txt next to the video if it exists")
    c.add_argument("--no-poses", action="store_true", help="忽略姿态；视频回退到容器帧率，Recorder PNG 仍保留 JSONL 的真实时间")
    c.add_argument("--notes", default="")
    c.add_argument("--root", type=Path, default=ROOT / "sessions")
    s = sub.add_parser("show", help="summarize a session")
    s.add_argument("session", type=Path)
    args = ap.parse_args()

    if args.cmd == "create":
        args.root.mkdir(parents=True, exist_ok=True)
        out = create(args.src, args.root, args.name, args.role, args.camera, args.camera_key, args.every, args.start,
                     args.end, args.max, args.dedupe_hamming, args.notes, False if args.no_poses else args.poses)
        meta = load(out)
        print(f"wrote {out}: {len(meta['frames'])} frames, role {meta['role']}, time base {meta['source'].get('time_base')}")
        for w in meta["warnings"]:
            print(f"  warning: {w}")
        return
    meta = load(args.session)
    sharp = [f["sharpness"] for f in meta["frames"]]
    print(json.dumps({k: meta[k] for k in ("name", "role", "width", "height", "camera_source", "selection", "warnings", "notes")}
                     | {"time_base": meta["source"].get("time_base"), "poses": meta["source"].get("poses")},
                     indent=2, ensure_ascii=False))
    print(f"frames: {len(meta['frames'])}, sharpness p10/p50/p90 {np.percentile(sharp, 10):.0f}/{np.percentile(sharp, 50):.0f}/{np.percentile(sharp, 90):.0f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
