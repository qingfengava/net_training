#!/usr/bin/env python3
"""核对 nt.bake_augment.bayer_demosaic 与 sleepy 实际使用的 NVIDIA NPP 去马赛克是否一致。

    python -I scripts/check_demosaic_vs_npp.py

sleepy 的 src/utils/cuda/cvtcolor.cu 用 nppiCFAToRGB_8u_C1C3R_Ctx（RGGB 网格，NPPI_INTER_UNDEFINED）。本脚本通过 ctypes
直接调用同一个函数，用到 libcudart 和 libnppc/libnppicc，不需要编译 sleepy。换 GPU 驱动、CUDA 版本或相机的 Bayer 网格后
重新跑一遍：

  1. 脉冲响应：红、蓝、绿位点上的一个脉冲，输出的响应必须和双线性核完全相同（确定性检验，失败则退出码 1）。
  2. 点阵、条纹、边缘图像上的平均差必须小于阈值（失败则退出码 1）。
  3. 白噪声上的差只报告不断言：NPP 在强边缘处按方向选插值，和双线性不同是预期的。

没有 CUDA 或 NPP 时退出码 2，不报告任何通过。2026-11-06 在 RTX 5060、CUDA 13.2 上的结果见 docs/bake_augment_v3.md。
"""
import ctypes as C
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nt.bake_augment import bayer_demosaic, bayer_mosaic  # noqa: E402

RGGB = 1  # NppiBayerGridPosition
LIBS = {"cudart": ("libcudart.so.13", "libcudart.so.12"), "nppc": ("libnppc.so.13", "libnppc.so.12"),
        "nppicc": ("libnppicc.so.13", "libnppicc.so.12")}
# 平均绝对差阈值（灰度级，0～255）。实测：红色点阵 1.1，对角条纹 0.02，竖直红黑边 0.04；阈值留出余量。
MEAN_TOLERANCE = {"red dots, pitch 3 px": 3.0, "red diagonal stripes": 0.5, "red/black vertical edge": 0.5}


class Size(C.Structure):
    _fields_ = [("w", C.c_int), ("h", C.c_int)]


class Rect(C.Structure):
    _fields_ = [("x", C.c_int), ("y", C.c_int), ("w", C.c_int), ("h", C.c_int)]


class Ctx(C.Structure):  # NppStreamContext
    _fields_ = [("stream", C.c_void_p), ("device", C.c_int), ("sm", C.c_int), ("thr_sm", C.c_int), ("thr_blk", C.c_int),
                ("shared", C.c_size_t), ("major", C.c_int), ("minor", C.c_int), ("flags", C.c_uint), ("reserved", C.c_int)]


def load(names):
    for name in names:
        try:
            return C.CDLL(name)
        except OSError:
            continue
    return None


def open_npp():
    libs = {key: load(names) for key, names in LIBS.items()}
    missing = [key for key, lib in libs.items() if lib is None]
    if missing:
        raise SystemExit(f"找不到 {', '.join(missing)}；没有 CUDA/NPP，无法核对（退出码 2）") from None
    fn = getattr(libs["nppicc"], "nppiCFAToRGB_8u_C1C3R_Ctx", None)
    if fn is None:
        raise SystemExit("libnppicc 里没有 nppiCFAToRGB_8u_C1C3R_Ctx（退出码 2）")
    fn.argtypes = [C.c_void_p, C.c_int, Size, Rect, C.c_void_p, C.c_int, C.c_int, C.c_int, Ctx]
    fn.restype = C.c_int
    return libs["cudart"], libs["nppc"], fn


def npp_demosaic(libs, bayer: np.ndarray) -> np.ndarray:
    cudart, nppc, fn = libs
    h, w = bayer.shape
    bayer = np.ascontiguousarray(bayer, np.uint8)
    ctx = Ctx()
    if nppc.nppGetStreamContext(C.byref(ctx)) != 0:
        raise SystemExit("nppGetStreamContext 失败：没有可用的 GPU？（退出码 2）")
    src, dst = C.c_void_p(), C.c_void_p()
    if cudart.cudaMalloc(C.byref(src), w * h) or cudart.cudaMalloc(C.byref(dst), w * h * 3):
        raise SystemExit("cudaMalloc 失败（退出码 2）")
    try:
        cudart.cudaMemcpy(src, bayer.ctypes.data_as(C.c_void_p), C.c_size_t(w * h), 1)
        status = fn(src, w, Size(w, h), Rect(0, 0, w, h), dst, w * 3, RGGB, 0, ctx)
        out = np.empty((h, w, 3), np.uint8)
        cudart.cudaMemcpy(out.ctypes.data_as(C.c_void_p), dst, C.c_size_t(w * h * 3), 2)
    finally:
        cudart.cudaFree(src)
        cudart.cudaFree(dst)
    if status < 0:
        raise SystemExit(f"NPP 返回错误 {status}（退出码 2）")
    return out  # R、G、B 顺序


def main() -> int:
    libs = open_npp()
    failures = []
    expected = {"chroma": np.array([[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]]),
                "green": np.array([[0, .25, 0], [.25, 1, .25], [0, .25, 0]])}
    print("脉冲响应（脉冲 200，位点奇偶以 RGGB 整帧为准）：")
    for (py, px), channel, kernel, name in (((0, 0), 0, "chroma", "红色位点"), ((1, 1), 2, "chroma", "蓝色位点"),
                                           ((0, 1), 1, "green", "绿色位点"), ((1, 0), 1, "green", "绿色位点")):
        bayer = np.zeros((16, 16), np.uint8)
        y, x = 8 + py, 8 + px
        bayer[y, x] = 200
        out = npp_demosaic(libs, bayer).astype(float) / 200
        ok = np.allclose(out[y - 1:y + 2, x - 1:x + 2, channel], expected[kernel], atol=1 / 200)
        others = [c for c in range(3) if c != channel]
        ok &= not out[..., others].any()
        mine = bayer_demosaic(bayer.astype(float))[y - 1:y + 2, x - 1:x + 2, channel] / 200
        ok &= bool(np.allclose(mine, expected[kernel], atol=1e-12))
        print(f"  {name} ({py},{px}): NPP 与双线性核{'一致' if ok else '不一致'}")
        if not ok:
            failures.append(f"脉冲响应 {name} ({py},{px})")
    rng = np.random.default_rng(0)
    h, w = 64, 96
    cases = {"random (只报告)": rng.integers(0, 256, (h, w, 3), dtype=np.uint8)}
    step = np.zeros((h, w, 3), np.uint8)
    step[:, w // 2:] = (230, 40, 20)
    cases["red/black vertical edge"] = step
    yy, xx = np.mgrid[:h, :w]
    stripes = np.zeros((h, w, 3), np.uint8)
    stripes[(xx + yy) % 20 < 10] = (240, 30, 30)
    cases["red diagonal stripes"] = stripes
    dots = np.zeros((h, w, 3), np.uint8)
    for y in range(3, h, 3):
        for x in range(3, w, 3):
            dots[y:y + 2, x:x + 2] = (240, 30, 20)
    cases["red dots, pitch 3 px"] = dots
    print("图像对照（去掉最外 2 圈像素，红、绿、蓝顺序）：")
    for name, rgb in cases.items():
        mosaic = bayer_mosaic(rgb)
        npp = npp_demosaic(libs, mosaic).astype(float)
        mine = np.clip(np.round(bayer_demosaic(mosaic)), 0, 255)
        diff = np.abs(npp - mine)[2:-2, 2:-2]
        limit = MEAN_TOLERANCE.get(name)
        verdict = "" if limit is None else (" 通过" if diff.mean() < limit else f" 超出阈值 {limit}")
        print(f"  {name:28s} 平均差 {diff.mean():6.3f}  最大差 {diff.max():6.1f}{verdict}")
        if limit is not None and diff.mean() >= limit:
            failures.append(f"{name} 平均差 {diff.mean():.3f} ≥ {limit}")
    if failures:
        print("失败：" + "；".join(failures))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
