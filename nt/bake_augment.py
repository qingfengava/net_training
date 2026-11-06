"""烘焙中的对齐模糊核、预测 ROI 扰动、点阵空间采样、拜耳取样和全局快门频闪增强。

新增随机流由调用方独立建立，关闭开关时不消耗旧烘焙的随机数。
ROI 是静态误差模拟；摩尔纹是有限谐波的亮度残差模型；拜耳取样走 RGGB 马赛克、8 位量化和双线性去马赛克，
产生滤色阵列的混叠和伪色；频闪按刷新频率、占空比、曝光时间和扫描方式算每一行的积分亮度。参数都仍需实拍校准。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from nt.canonical import Sample
from nt.geometry import bbox


def validate_config(cfg: dict) -> None:
    """在创建输出目录前检查新配置，拒绝无效概率和无法解释的参数。"""
    def scalar(value, name, lo=0.0, hi=float("inf")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{name} 必须在 [{lo}, {hi}] 内")

    def pair(value, name, lo=0.0, hi=float("inf")):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} 必须是两个数的范围")
        scalar(value[0], name, lo, hi)
        scalar(value[1], name, lo, hi)
        if value[0] > value[1]:
            raise ValueError(f"{name} 下限大于上限")

    motion = cfg["effects"]["motion_blur"]
    if motion.get("kernel", "legacy_v2") not in ("centered_v1", "legacy_v2"):
        raise ValueError("未知 motion_blur.kernel")
    pair(motion["length_px"], "motion_blur.length_px", 1, 255)
    if any(int(v) != v for v in motion["length_px"]):
        raise ValueError("motion_blur.length_px 必须为整数")
    t = cfg["roi"].get("tracker")
    if t is not None:
        for key in ("prob", "search_prob", "patrol_prob", "recovery_prob"):
            scalar(t[key], f"roi.tracker.{key}", 0, 1)
        if t["search_prob"] + t["patrol_prob"] + t["recovery_prob"] > 1:
            raise ValueError("ROI 各状态概率之和不能大于 1")
        if not isinstance(t["train_only"], bool):
            raise ValueError("roi.tracker.train_only 必须为布尔值")
        pair(t["envelope_multiple"], "envelope_multiple", 1, 100)
        for key in ("age_ms", "recovery_age_ms", "velocity_error_frame_s"):
            pair(t[key], key)
        for key in ("expand_ratio", "base_expand_ratio", "lost_time_ms"):
            scalar(t[key], key, 1e-6)
        for key in ("center_sigma", "recovery_sigma_per_s", "log_scale_sigma"):
            scalar(t[key], key, 0, 10)
    m = cfg["effects"].get("moire")
    if m is not None:
        scalar(m["prob"], "moire.prob", 0, 1)
        if not isinstance(m["train_only"], bool):
            raise ValueError("moire.train_only 必须为布尔值")
        if m["grid_source"] not in ("scene", "config") or m["dot"] not in ("square", "circle"):
            raise ValueError("moire 的点阵来源或点形状无效")
        if len(m["grid"]) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or not 2 <= v <= 512 for v in m["grid"]):
            raise ValueError("moire.grid 必须为 2～512 的两个整数")
        pair(m["fill"], "moire.fill", 0.1, 1)
        pair(m["strength"], "moire.strength", 0, 0.5)
        pair(m["optical_sigma_px"], "moire.optical_sigma_px", 0, 5)
        pair(m["pixel_aperture"], "moire.pixel_aperture", 0.05, 1)
        if not isinstance(m["harmonics"], int) or not 1 <= m["harmonics"] <= 5:
            raise ValueError("moire.harmonics 必须为 1～5 的整数")
    b = cfg["effects"].get("bayer")
    if b is not None:
        scalar(b["prob"], "bayer.prob", 0, 1)
        if not isinstance(b["train_only"], bool):
            raise ValueError("bayer.train_only 必须为布尔值")
        if b["pattern"] != "rggb":
            raise ValueError("bayer.pattern 目前只支持 rggb（海康 BayerRG8，与 sleepy 的 nppiCFAToRGB 网格一致）")
        if b["demosaic"] != "bilinear":
            raise ValueError("bayer.demosaic 目前只支持 bilinear")
    f = cfg["effects"].get("flicker")
    if f is not None:
        scalar(f["prob"], "flicker.prob", 0, 1)
        if not isinstance(f["train_only"], bool):
            raise ValueError("flicker.train_only 必须为布尔值")
        pair(f["refresh_hz"], "flicker.refresh_hz", 10, 100000)
        pair(f["duty"], "flicker.duty", 0.02, 1)
        pair(f["exposure_ms"], "flicker.exposure_ms", 0.01, 100)
        pair(f["scan_span"], "flicker.scan_span", 0, 1)
        scalar(f["frame_sigma"], "flicker.frame_sigma", 0, 2)
        if f["grid_source"] not in ("scene", "config"):
            raise ValueError("flicker.grid_source 必须是 scene 或 config")
        if isinstance(f["rows"], bool) or not isinstance(f["rows"], int) or not 2 <= f["rows"] <= 512:
            raise ValueError("flicker.rows 必须为 2～512 的整数")
        if f["scan_direction"] not in ("down", "up", "random"):
            raise ValueError("flicker.scan_direction 必须是 down、up 或 random")
        if not isinstance(f["synchronized"], bool):
            raise ValueError("flicker.synchronized 必须为布尔值")


def centered_motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """在奇数支撑域积分线段，双线性分配权重并强制中心对称。

    length 保留原配置的线段跨度语义；偶数长度也可使用，不移动图像。
    显式对称消除线段离散化和端点舍入引起的质心偏移。
    """
    if not math.isfinite(length) or not 1 <= length <= 255 or not math.isfinite(angle_deg):
        raise ValueError("运动模糊长度或角度无效")
    radius = int(math.ceil((length - 1) / 2))
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float64)
    ts = np.linspace(-(length - 1) / 2, (length - 1) / 2, max(3, int(math.ceil(length * 16)) | 1))
    weights = np.ones(len(ts)); weights[[0, -1]] = 0.5
    a = math.radians(angle_deg)
    xs, ys = radius + ts * math.cos(a), radius + ts * math.sin(a)
    ix, iy = np.floor(xs).astype(int), np.floor(ys).astype(int)
    for dx, wx in ((0, 1 - (xs - ix)), (1, xs - ix)):
        for dy, wy in ((0, 1 - (ys - iy)), (1, ys - iy)):
            np.add.at(kernel, (np.clip(iy + dy, 0, 2 * radius), np.clip(ix + dx, 0, 2 * radius)), weights * wx * wy)
    kernel = (kernel + kernel[::-1, ::-1]) * 0.5
    return (kernel / kernel.sum()).astype(np.float32)


def enabled(options: dict, sample: Sample, split: str) -> bool:
    """新增强默认仅施加于仿真训练样本，不给真实照片重复造退化。"""
    return bool(options.get("prob", 0)) and sample.source == "sim" and (not options.get("train_only", True) or split == "train")


def fit_roi(cx: float, cy: float, wanted_w: float, width: int, height: int) -> tuple[int, int, int, int]:
    """保持 4:3；边界处平移窗口，不截掉窗口一侧后再拉伸。"""
    maximum = min(width // 4, height // 3) * 4
    if maximum < 4:
        raise ValueError("图像过小，无法构建 4:3 ROI")
    w = min(maximum, max(4, math.ceil(wanted_w / 4) * 4))
    h = w * 3 // 4
    return int(np.clip(round(cx - w / 2), 0, width - w)), int(np.clip(round(cy - h / 2), 0, height - h)), w, h


def tracker_roi(rng, sample: Sample, boxes: list, cfg: dict, split: str, *, lens_applied=False, base_indices=()):
    """采样单帧预测误差；也可读取 scene.roi_prediction 的真实预测包络。

    不借助当前真值拒绝漏裁窗口。所有与窗口相交的目标由烘焙保留标注。
    没有预测记录时，以单板为参照采样整车包络，绝不按编号猜测车体归属。
    """
    options = cfg["roi"].get("tracker", {})
    if not enabled(options, sample, split) or rng.random() >= options["prob"]:
        return None
    width, height = sample.w, sample.h
    full = fit_roi(width / 2, height / 2, width, width, height)
    meta = {"model": "tracker_error_v1", "source": "synthetic"}
    prediction = sample.scene.get("roi_prediction")
    if prediction is not None:
        if lens_applied:
            raise ValueError("真实预测包络不能与新增镜头畸变同时使用；请先统一坐标系")
        if prediction.get("image_size") != [width, height]:
            raise ValueError("scene.roi_prediction.image_size 与原图不一致")
        box = np.asarray(prediction.get("bbox"), float)
        age = float(prediction.get("age_ms", -1))
        state = prediction.get("state")
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]) or not math.isfinite(age) or age < 0 or state not in ("tracking", "recovering", "search", "patrol"):
            raise ValueError("scene.roi_prediction 的 bbox、age_ms 或 state 无效")
        cx, cy = (box[:2] + box[2:]) / 2
        envelope = max(box[2] - box[0], (box[3] - box[1]) * 4 / 3)
        expand = options["base_expand_ratio"] if prediction.get("is_base", False) else options["expand_ratio"]
        base_w = max(cfg["roi"]["min_width_px"], envelope * expand)
        meta.update(source="recorded_prediction", state=state, age_ms=age, predicted_bbox=box.tolist())
    else:
        u = rng.random()
        search, patrol, recovery = (options[k] for k in ("search_prob", "patrol_prob", "recovery_prob"))
        state = "search" if u < search else "patrol" if u < search + patrol else "recovering" if u < search + patrol + recovery else "tracking"
        meta["state"] = state
        if not boxes or state in ("search", "patrol"):
            meta["reason"] = "no_target" if not boxes else state
            return full, "tracker_full", meta
        idx = int(rng.integers(len(boxes)))
        x0, y0, x1, y1 = boxes[idx]
        multiple = math.exp(rng.uniform(*np.log(options["envelope_multiple"])))
        envelope = max(x1 - x0, (y1 - y0) * 4 / 3) * multiple
        age = float(rng.uniform(*options["recovery_age_ms" if state == "recovering" else "age_ms"]))
        # 速度误差用画幅/秒表示，延迟使用秒；不把随机速度当成真实跟踪速度。
        speed = rng.uniform(*options["velocity_error_frame_s"])
        angle = rng.uniform(0, 2 * math.pi)
        velocity = np.array([math.cos(angle) * width, math.sin(angle) * height]) * speed
        sigma = options["center_sigma"] + options["recovery_sigma_per_s"] * age / 1000
        error = rng.normal(0, sigma, 2) * [envelope, envelope * 0.75] - velocity * age / 1000
        cx, cy = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + error
        scale_error = math.exp(float(np.clip(rng.normal(0, options["log_scale_sigma"]), -2, 2)))
        # boxes 与 trained 顺序一致；基地标签由调用方放在独立索引列表中。
        is_base = idx in base_indices
        expand = options["base_expand_ratio"] if is_base else options["expand_ratio"]
        base_w = max(cfg["roi"]["min_width_px"], envelope * expand * scale_error)
        meta.update(target_index=idx, age_ms=age, envelope_multiple=multiple, center_error_px=error.tolist(),
                    velocity_error_px_s=velocity.tolist(), scale_error=scale_error)
    meta["base_width_px"] = float(base_w)
    if state in ("search", "patrol") or age >= options["lost_time_ms"] or not (0 <= cx < width and 0 <= cy < height):
        meta["reason"] = state if state in ("search", "patrol") else "expired_or_outside"
        return full, "tracker_full", meta
    maximum = min(cfg["roi"]["max_width_px"], width, height * 4 / 3)
    base_w = min(base_w, maximum)
    wanted = base_w + (maximum - base_w) * min(age / options["lost_time_ms"], 1)
    roi = fit_roi(cx, cy, wanted, width, height)
    meta["predicted_center_px"] = [float(cx), float(cy)]
    meta["boundary_shift_px"] = [roi[0] + roi[2] / 2 - float(cx), roi[1] + roi[3] / 2 - float(cy)]
    return roi, "tracker_roi", meta


def _grid(sample: Sample, options: dict):
    if options["grid_source"] == "config":
        return (*options["grid"], options["dot"])
    screen = sample.scene.get("screen", {})
    looks = [r.get("look", {}) for group in ("robots", "structures") for r in screen.get(group, [])]
    grids = {(r.get("cols"), r.get("rows"), r.get("dot")) for r in looks}
    # 旧 canonical 没有每块板到车体的对应关系；不猜测混合点阵的归属。
    if len(grids) != 1:
        return None
    cols, rows, dot = next(iter(grids))
    return (cols, rows, dot) if isinstance(cols, int) and isinstance(rows, int) and 2 <= cols <= 512 and 2 <= rows <= 512 and dot in ("square", "circle") else None


def quad_coordinates(quad, shape):
    """四边形内每个像素中心的面内坐标 (u, v) 及其对像素坐标的偏导。

    quad 的顺序是 LT、LB、RB、RT，u 沿 LT→RT，v 沿 LT→LB；坐标采用项目的像素边缘约定，
    像素 (i, j) 的中心在 (j + 0.5, i + 0.5)。返回 u、v、inside 掩码，以及偏导 (du/dx, du/dy, dv/dx, dv/dy)，
    偏导不含点阵行列数，调用方乘以 cols 或 rows 即得每像素跨过的点数。
    """
    h, w = shape
    transform = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32))
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    xx += 0.5; yy += 0.5
    den = transform[2, 0] * xx + transform[2, 1] * yy + transform[2, 2]
    safe_den = np.where(np.abs(den) > 1e-9, den, 1.0)
    u = (transform[0, 0] * xx + transform[0, 1] * yy + transform[0, 2]) / safe_den
    v = (transform[1, 0] * xx + transform[1, 1] * yy + transform[1, 2]) / safe_den
    inside = (np.abs(den) > 1e-9) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    slopes = ((transform[0, 0] - u * transform[2, 0]) / safe_den, (transform[0, 1] - u * transform[2, 1]) / safe_den,
              (transform[1, 0] - v * transform[2, 0]) / safe_den, (transform[1, 1] - v * transform[2, 1]) / safe_den)
    return u, v, inside, slopes


def sampling_pattern(quad, shape, grid, fill, phase, optical_sigma, aperture, harmonics):
    """在原图像素中心采样点阵的有限傅里叶展开。

    单应变换提供空间相位及局部频率；高斯光学传递函数先衰减高频，
    方形像素孔径的 sinc 响应再积分，最后离散采样产生混叠。
    返回单位均值模型和面内掩码；坐标采用项目的像素边缘约定。
    """
    h, w = shape
    u, v, inside, (dudx, dudy, dvdx, dvdy) = quad_coordinates(quad, shape)
    cols, rows, dot = grid
    ux, uy, vx, vy = dudx * cols, dudy * cols, dvdx * rows, dvdy * rows
    pattern = np.ones((h, w), np.float64)
    if dot == "circle":
        # 圆点的系数用对称网格积分；方点采用解析系数。
        axis = (np.arange(96) + 0.5) / 96 - 0.5
        gy, gx = np.meshgrid(axis, axis, indexing="ij")
        mask = gx * gx + gy * gy <= (fill / 2) ** 2
    for n in range(-harmonics, harmonics + 1):
        for m in range(-harmonics, harmonics + 1):
            if n < 0 or (n == 0 and m <= 0):
                continue
            coefficient = np.sinc(n * fill) * np.sinc(m * fill) if dot == "square" else float(np.cos(2 * math.pi * (n * gx[mask] + m * gy[mask])).mean())
            fx, fy = n * ux + m * vx, n * uy + m * vy
            transfer = np.exp(-2 * math.pi ** 2 * optical_sigma ** 2 * (fx * fx + fy * fy))
            transfer *= np.sinc(aperture * fx) * np.sinc(aperture * fy)
            theta = 2 * math.pi * (n * (u * cols + phase[0]) + m * (v * rows + phase[1]))
            pattern += 2 * coefficient * transfer * np.cos(theta)
    # 一个点间距内渐入，避免四边形边缘产生人为轮廓。
    edge = np.clip(np.minimum.reduce([u * cols, (1 - u) * cols, v * rows, (1 - v) * rows]), 0, 1)
    return np.clip(pattern, 0, 4), inside * edge


def _pwm_integral(t, duty: float):
    """周期为 1、占空比 duty、每周期前 duty 为亮的方波，从 0 到 t 的积分（t 可以是数组）。"""
    whole = np.floor(t)
    return whole * duty + np.minimum(t - whole, duty)


def row_exposure_gain(rows: int, refresh_hz: float, duty: float, exposure_s: float, start_s: float,
                      scan_phase: float, scan_span: float, descending: bool = True) -> np.ndarray:
    """全局快门下每一行在曝光窗口里收到的光，相对"不调光、一直亮"的比值，期望值为 1。

    行 r 的 PWM 相位是 scan_phase + scan_span * (r / rows)（以刷新周期为单位）。曝光窗口 [start, start + T]
    内的亮时间用方波原函数闭式求出，除以 duty * T 归一化：曝光里恰好含整数个刷新周期时所有行都是 1，
    此时没有频闪；不足一个周期时，各行的相位不同，亮度在 0 到 1/duty 之间跳变。
    """
    if rows < 1 or refresh_hz <= 0 or not 0 < duty <= 1 or exposure_s <= 0:
        raise ValueError("频闪参数无效")
    r = np.arange(rows, dtype=np.float64) / rows
    if not descending:
        r = r[::-1]
    shift = scan_phase + scan_span * r
    a = start_s * refresh_hz - shift
    b = (start_s + exposure_s) * refresh_hz - shift
    lit = _pwm_integral(b, duty) - _pwm_integral(a, duty)
    return lit / (duty * exposure_s * refresh_hz)


def apply_flicker(img, rng, sample: Sample, armors, roi, options: dict, split: str):
    """亮屏区域按行乘以 PWM 积分比，再乘整屏亮度因子；不移动几何、不更改颜色或熄灭标签。

    刷新频率、占空比、曝光时间都按范围采样，换成官方模块的数值只需改配置。一帧里所有屏共用同一个曝光起点，
    因为全局快门同时开始曝光；每块屏有各自的扫描相位，除非 synchronized 为真（所有屏同步扫描）。
    仿真器已经画了暗带的屏（screen_state == flicker）不再叠加，避免重复变暗。
    """
    if not enabled(options, sample, split) or rng.random() >= options["prob"]:
        return img, None
    report = {"model": "global_shutter_pwm_v1", "applied": [], "skipped": {}}
    refresh = float(math.exp(rng.uniform(math.log(options["refresh_hz"][0]), math.log(options["refresh_hz"][1]))))
    duty = float(rng.uniform(*options["duty"]))
    exposure_s = float(rng.uniform(*options["exposure_ms"])) / 1000
    start_s = float(rng.random() / refresh)
    shared_phase = float(rng.random())
    report.update(refresh_hz=refresh, duty=duty, exposure_ms=exposure_s * 1000, periods_in_exposure=exposure_s * refresh,
                  exposure_start_ms=start_s * 1000)
    out = img.copy()
    rx, ry, rw, rh = roi
    for idx, armor in enumerate(armors):
        reason = None
        if armor.color not in (0, 1) or armor.screen_state not in ("on",):
            reason = "not_plain_lit_screen" if armor.screen_state != "flicker" else "already_flicker"
        elif any(v != 2 for v in armor.vis) or (armor.occ or 0) > 0:
            reason = "occluded"
        if reason:
            report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
            continue
        quad = np.array(armor.corners, np.float64) - [rx, ry]
        x0, y0, x1, y1 = bbox(quad)
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(rw, math.ceil(x1)), min(rh, math.ceil(y1))
        if x1 <= x0 or y1 <= y0 or abs(cv2.contourArea(quad.astype(np.float32))) < 1:
            continue
        rows = _screen_rows(sample, options)
        phase = shared_phase if options["synchronized"] else float(rng.random())
        down = options["scan_direction"] == "down" or (options["scan_direction"] == "random" and rng.random() < 0.5)
        span = float(rng.uniform(*options["scan_span"]))
        sigma = options["frame_sigma"]
        frame = float(math.exp(np.clip(rng.normal(0, sigma), -1.5, 1.5))) if sigma else 1.0
        gain_rows = row_exposure_gain(rows, refresh, duty, exposure_s, start_s, phase, span, down)
        # 面内行号 v*rows 定位到屏的哪一行；只在四边形内部施加，边缘一个点距内渐入
        u, v, inside, _ = quad_coordinates(quad - [x0, y0], (y1 - y0, x1 - x0))
        row_index = np.clip(np.floor(v * rows).astype(int), 0, rows - 1)
        edge = np.clip(np.minimum.reduce([u * 8, (1 - u) * 8, v * rows, (1 - v) * rows]), 0, 1)
        gain = 1 + (gain_rows[row_index] * frame - 1) * inside * edge
        patch = out[y0:y1, x0:x1]
        linear = (patch.astype(np.float64) / 255) ** 2.2
        changed = np.clip(np.clip(linear * gain[..., None], 0, 1) ** (1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
        changed_values = int(np.count_nonzero(changed != patch))
        patch[:] = changed
        report["applied"].append({"armor_index": idx, "rows": rows, "scan_phase": phase, "scan_span": span,
                                  "descending": bool(down), "frame_gain": frame,
                                  "row_gain_min": float(gain_rows.min()), "row_gain_max": float(gain_rows.max()),
                                  "changed_values": changed_values})
    return out, report


def _screen_rows(sample: Sample, options: dict) -> int:
    """这块屏有几行：grid_source 为 scene 时取场景里的点阵行数（缺失或混合布局退回 rows），为 config 时取 rows。"""
    if options["grid_source"] == "scene":
        screen = sample.scene.get("screen", {})
        rows = {r.get("look", {}).get("rows") for group in ("robots", "structures") for r in screen.get(group, [])}
        if len(rows) == 1 and isinstance(next(iter(rows)), int) and 2 <= next(iter(rows)) <= 512:
            return int(next(iter(rows)))
    return int(options["rows"])


_BILINEAR_CHROMA = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64) / 4
_BILINEAR_GREEN = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float64) / 4


def bayer_mosaic(rgb: np.ndarray, origin=(0, 0)) -> np.ndarray:
    """RGB（红、绿、蓝顺序，任意数值类型）按 RGGB 排布取样为单通道。origin 是这块图像左上角在整帧里的 (x, y)，
    奇偶决定哪些像素是红色位点：整帧的 (偶行, 偶列) 是红，(奇行, 奇列) 是蓝，其余是绿。"""
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[:h, :w]
    row, col = (yy + origin[1]) & 1, (xx + origin[0]) & 1
    channel = np.where((row == 0) & (col == 0), 0, np.where((row == 1) & (col == 1), 2, 1))
    return np.take_along_axis(rgb, channel[..., None], axis=2)[..., 0]


def bayer_demosaic(mosaic: np.ndarray, origin=(0, 0)) -> np.ndarray:
    """RGGB 马赛克的双线性去马赛克，输出红、绿、蓝顺序的浮点图。

    和 sleepy 用的 nppiCFAToRGB（RGGB 网格）对照过：红、蓝、绿位点的脉冲响应与这里的核完全一致；
    点距 3 px 的红色点阵上平均差 1.1/255，对角条纹 0.02/255，白噪声上 5/255（NPP 在强边缘处按方向选插值）。
    边界用反射 101，和 NPP 在边缘位点的脉冲响应一致。
    """
    f = mosaic.astype(np.float64)
    h, w = f.shape
    yy, xx = np.mgrid[:h, :w]
    row, col = (yy + origin[1]) & 1, (xx + origin[0]) & 1
    planes = []
    for site, kernel in (((row == 0) & (col == 0), _BILINEAR_CHROMA), (((row ^ col) == 1), _BILINEAR_GREEN),
                         ((row == 1) & (col == 1), _BILINEAR_CHROMA)):
        planes.append(cv2.filter2D(np.where(site, f, 0.0), -1, kernel, borderType=cv2.BORDER_REFLECT_101))
    return np.stack(planes, axis=-1)


def apply_bayer(img, rng, sample: Sample, roi, options: dict, split: str):
    """让整帧经过 RGGB 取样和双线性去马赛克，产生点阵与滤色阵列的混叠和伪色。

    img 是烘焙流程里的 8 位 BGR 图，放在 apply_effects 之后（曝光、模糊、噪声之后）。取样本身就是 8 位 Bayer 数据，
    不再单独量化。噪声放在去马赛克之前还是之后是个容易想错的点：在每个传感器像素上独立加噪声，和先在 RGB 三个
    通道里独立加噪声再取样，数学上等价（取样只保留每个位点的一个通道），去马赛克后的噪声标准差都是原来的 0.75～0.79
    倍（红、蓝约 0.75，绿约 0.79，与双线性核的理论值一致），所以这里不需要把噪声搬进马赛克。
    只改变像素值，不移动任何几何，所以标签不受影响。ROI 原点的奇偶决定红色位点的相位。
    """
    if not enabled(options, sample, split) or rng.random() >= options["prob"]:
        return img, None
    rx, ry = roi[0], roi[1]
    mosaic = bayer_mosaic(img[..., ::-1], (rx, ry))
    out = np.clip(bayer_demosaic(mosaic, (rx, ry)) + 0.5, 0, 255).astype(np.uint8)[..., ::-1].copy()
    report = {"model": "bayer_rggb_bilinear_v1", "origin_parity": [rx & 1, ry & 1], "pattern": options["pattern"],
              "demosaic": options["demosaic"], "changed_values": int(np.count_nonzero(out != img))}
    return out, report


def apply_moire(img, rng, sample: Sample, armors, roi, options: dict, split: str):
    """仅改变完整可见亮屏区域的亮度，不移动几何、不更改颜色或熄灭标签。"""
    if not enabled(options, sample, split) or rng.random() >= options["prob"]:
        return img, None
    grid = _grid(sample, options)
    report = {"model": "dot_sampling_residual_v1", "applied": [], "skipped": {}}
    if grid is None:
        report["reason"] = "missing_or_ambiguous_grid"
        return img, report
    report["grid"] = list(grid)
    out = img.copy()
    rx, ry, rw, rh = roi
    for idx, armor in enumerate(armors):
        reason = None
        if armor.color not in (0, 1) or armor.screen_state not in ("on", "flicker"):
            reason = "not_lit_screen"
        elif any(v != 2 for v in armor.vis) or (armor.occ or 0) > 0:
            reason = "occluded"
        if reason:
            report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
            continue
        quad = np.array(armor.corners, np.float64) - [rx, ry]
        x0, y0, x1, y1 = bbox(quad)
        x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        x1, y1 = min(rw, math.ceil(x1)), min(rh, math.ceil(y1))
        if x1 <= x0 or y1 <= y0 or abs(cv2.contourArea(quad.astype(np.float32))) < 1:
            continue
        fill = float(rng.uniform(*options["fill"]))
        strength = float(rng.uniform(*options["strength"]))
        sigma = float(rng.uniform(*options["optical_sigma_px"]))
        aperture = float(rng.uniform(*options["pixel_aperture"]))
        phase = rng.random(2).tolist()
        pattern, mask = sampling_pattern(quad - [x0, y0], (y1 - y0, x1 - x0), grid, fill, phase, sigma, aperture, options["harmonics"])
        gain = 1 + strength * np.clip(pattern - 1, -1, 1) * mask
        patch = out[y0:y1, x0:x1]
        linear = (patch.astype(np.float64) / 255) ** 2.2
        changed = np.clip(np.clip(linear * gain[..., None], 0, 1) ** (1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
        changed_values = int(np.count_nonzero(changed != patch))
        patch[:] = changed
        report["applied"].append({"armor_index": idx, "fill": fill, "strength": strength,
                                  "optical_sigma_px": sigma, "pixel_aperture": aperture, "phase": phase,
                                  "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                                  "changed_values": changed_values})
    return out, report
