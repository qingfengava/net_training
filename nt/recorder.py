"""读取 sleepy Recorder 的 PNG 会话，保持图像与 JSONL 元数据逐帧对应。"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


def png_root(src: Path) -> Path | None:
    """同时识别会话根目录、frames 子目录和显式指定的清单。"""
    if src.is_dir() and (src / "frames.jsonl").is_file():
        return src.resolve()
    if src.is_dir() and src.name == "frames" and (src.parent / "frames.jsonl").is_file():
        return src.parent.resolve()
    if src.is_file() and src.name == "frames.jsonl":
        return src.parent.resolve()
    return None


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    return float(value)


def _incomplete_record(line: str, error: json.JSONDecodeError) -> bool:
    """仅接受仍可能补全为 JSON 对象的常见尾部截断，不忽略任意坏行。"""
    if not line.lstrip().startswith("{") or error.msg == "Extra data":
        return False
    if error.pos == len(line) or error.msg == "Unterminated string starting at":
        return True
    tail = line[error.pos:]
    if error.msg == "Expecting value":
        return tail in ("-", "t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul")
    # 数字的小数点或指数可能在文件结尾被截断。
    return error.msg == "Expecting ',' delimiter" and re.fullmatch(r"\.|[eE][+-]?", tail) is not None


def load_png(root: Path) -> tuple[list[dict], dict, list[str]]:
    """JSONL 是唯一配对依据；poses.txt 仅供旧工具使用，不参与导入。"""
    root = root.resolve()
    manifest = root / "frames.jsonl"
    raw = manifest.read_bytes()
    lines = [(n, line) for n, line in enumerate(raw.decode("utf-8").splitlines(), 1) if line.strip()]
    rows, warnings, paths = [], [], set()
    last_camera_id = None
    missing_ids = gap_events = unknown_ids = 0
    shape = None
    for n, line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            # 只恢复断电留下的未结束尾行；完整坏行或中间坏行必须报错。
            if n == lines[-1][0] and not raw.endswith(b"\n") and _incomplete_record(line, exc):
                warnings.append(f"{manifest.name}: 忽略截断的最后一行 {n}")
                break
            raise ValueError(f"{manifest}: 第 {n} 行 JSON 损坏") from exc
        try:
            if not isinstance(record, dict) or record.get("mode") != "png_sequence":
                raise ValueError("必须是 mode=png_sequence 的 Recorder 记录")
            index = _integer(record["write_index"], "write_index")
            frame_id = _integer(record["camera_frame_id"], "camera_frame_id", -1)
            t = _number(record["time_offset_s"], "time_offset_s")
            width = _integer(record["width"], "width", 1)
            height = _integer(record["height"], "height", 1)
            channels = _integer(record["channels"], "channels", 1)
            fmt = record["pixel_format"]
            if fmt not in ("BGR", "RGB", "GRAY") or channels != (1 if fmt == "GRAY" else 3):
                raise ValueError("pixel_format 与 channels 不匹配或不受支持")
            current_shape = (height, width, channels)
            if shape is not None and current_shape != shape:
                raise ValueError("同一会话的图像尺寸或通道数发生变化")
            shape = current_shape
            if rows and (index <= rows[-1]["write_index"] or t <= rows[-1]["t_s"]):
                raise ValueError("write_index 和 time_offset_s 必须严格递增，不能重复或倒序")
            if frame_id >= 0:
                if last_camera_id is not None:
                    if frame_id <= last_camera_id:
                        raise ValueError("已知 camera_frame_id 必须严格递增")
                    gap = frame_id - last_camera_id - 1
                    missing_ids += gap
                    gap_events += gap > 0
                last_camera_id = frame_id
            else:
                unknown_ids += 1
            image = record["image"]
            if not isinstance(image, str) or not image:
                raise ValueError("image 必须是相对于会话目录的 PNG 路径")
            rel = Path(image)
            path = (root / rel).resolve()
            if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root):
                raise ValueError("image 不能越出会话目录")
            if rel.suffix.lower() != ".png" or not path.is_file():
                raise ValueError(f"PNG 图像不存在：{image}")
            if path in paths:
                raise ValueError(f"image 重复引用：{image}")
            paths.add(path)
            # 所有记录先检查 PNG 头，抽帧时也不能漏掉尺寸不一致的源图。
            with Image.open(path) as png:
                if (png.format != "PNG" or png.size != (width, height)
                        or png.mode != ("L" if channels == 1 else "RGB")):
                    raise ValueError(f"PNG 头的尺寸、通道或位深与元数据不符：{image}")
            pose = record["pose"]
            q = np.array([_number(pose[k], f"pose.{k}") for k in ("qw", "qx", "qy", "qz")])
            norm = float(np.linalg.norm(q))
            if not math.isfinite(norm) or norm < 1e-6:
                raise ValueError("姿态四元数的模长无效")
            rows.append({"index": index, "write_index": index, "camera_frame_id": frame_id,
                         "t_s": t, "q_wxyz": (q / norm).tolist(), "source_image": rel.as_posix(),
                         "source_pixel_format": fmt, "pixel_format": "GRAY" if channels == 1 else "BGR",
                         "width": width, "height": height, "channels": channels, "path": path})
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{manifest}: 第 {n} 行无效：{exc}") from exc
    if not rows:
        raise ValueError(f"{manifest}: 没有完整有效的记录")
    unlisted = [p for p in (root / "frames").glob("*.png") if p.resolve() not in paths]
    if unlisted:
        warnings.append(f"清单外有 {len(unlisted)} 张 PNG（可能是中断时留下的文件），未导入")
    if missing_ids:
        warnings.append(f"camera_frame_id 中有 {missing_ids} 个未记录编号；可能包含限频、写盘或上游丢帧，不能单独认定为录像丢帧")
    source = {"kind": "recorder_png", "path": str(root), "codec": "png", "fps": None,
              "frames_reported": len(rows), "time_base": "recorder_jsonl",
              "manifest": {"path": str(manifest), "sha256": hashlib.sha256(raw).hexdigest()},
              "unlisted_pngs": len(unlisted),
              "camera_frame_ids": {"missing_between_known_ids": missing_ids, "gap_events": gap_events,
                                   "unknown_records": unknown_ids}}
    return rows, source, warnings
