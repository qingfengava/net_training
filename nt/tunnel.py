"""隧道关键点检测器：导出 ONNX、像素误差评估、序列推理转储。

三个子命令共用同一套预处理与解码，保证“评估的就是部署的”：
  export  best.pt → ONNX（imgsz 736×960，Ultralytics 原生 end2end 输出）+ contract.json
  eval    ONNX 在 baked 数据集（val / val_stress）上按原图像素误差评估
  dump    ONNX 在 Daedalus 序列图像上推理，写出原始输出张量（sleepy tunnel_replay 读取）

ONNX 输出布局（[1, MAX_DET, 30]，坐标为网络输入像素）：
  [0:4]   x1 y1 x2 y2
  [4]     conf
  [5]     cls（恒 0）
  [6:30]  8 × (x, y, vis)，K1..K8；vis 取值范围以 export 后的实测核对为准
sleepy 的 decode_best 以同一布局解码。
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 960, 736   # 原图 4:3 缩到 960×720，再居中填充到 stride32 倍数 736（与 Ultralytics rect 训练一致）
N_KPT = 8
MAX_DET = 10
STRIDE = 6 + N_KPT * 3
FLIP_IDX = [7, 6, 5, 4, 3, 2, 1, 0]


# ----------------------------------------------------------------------------------------- 导出
def export(weights: Path, out_dir: Path) -> Path:
    from nt import offline

    offline.apply()
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pt = Path(tmp) / "model.pt"
        shutil.copy(weights, tmp_pt)
        model = YOLO(str(tmp_pt))
        produced = Path(
            model.export(
                format="onnx", imgsz=[IMG_H, IMG_W], batch=1, half=False, dynamic=False,
                simplify=True, opset=17, max_det=MAX_DET, device="cpu", verbose=False,
            )
        )
        dst = out_dir / "tunnel.onnx"
        shutil.copy(produced, dst)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    assert list(out.shape)[1:] == [MAX_DET, STRIDE], f"输出形状 {out.shape} 与约定不符"
    contract = {
        "onnx": dst.name,
        "weights": str(weights),
        "input": {"name": inp.name, "shape": list(inp.shape), "color": "RGB", "range": "0..1",
                  "resize": "cv2.INTER_AREA 缩放到 960x720 后上下各填充 8 行灰(114)至 736x960；scale=2/3, pad_x=0, pad_y=8"},
        "output": {"name": out.name, "shape": list(out.shape),
                   "columns": "x1 y1 x2 y2 conf cls (x y vis)*8，网络输入像素",
                   "stride": STRIDE, "max_det": MAX_DET},
        "keypoints": ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"],
        "flip_idx": FLIP_IDX,
    }
    (out_dir / "contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False))
    print(f"[export] {dst}  in={inp.shape}  out={out.shape}")
    return dst


# ----------------------------------------------------------------------------------------- 推理
def make_session(onnx_path: Path, force_cpu: bool = False):
    """创建 ONNX Runtime 会话，优先 CUDA，失败时回退 CPU 并明确提示。

    onnxruntime-gpu 1.23 需要 CUDA 12 的 cuBLAS/cuDNN/cudart/cuFFT；本机系统只装了 CUDA 13，
    这些 .so.12 来自 venv 里的 nvidia-* pip 包，需要 preload_dlls() 预加载才找得到。
    """
    import logging

    import onnxruntime as ort

    log = logging.getLogger("nt.tunnel")
    if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            log.warning("ort.preload_dlls() 失败: %s", e)
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if sess.get_providers()[0] == "CUDAExecutionProvider":
            return sess
        log.warning("CUDA provider 未能加载，回退 CPU（耗时不具参考性）")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """BGR 原图 → [1,3,736,960] float32，返回 (tensor, scale, pad_x, pad_y)。与训练一致用 INTER_AREA。"""
    import cv2

    h, w = img_bgr.shape[:2]
    scale = min(IMG_W / w, IMG_H / h)
    nw, nh = round(w * scale), round(h * scale)
    if (nw, nh) != (w, h):
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (IMG_W - nw) // 2, (IMG_H - nh) // 2
    canvas = np.full((IMG_H, IMG_W, 3), 114, np.uint8)
    canvas[y0:y0 + nh, x0:x0 + nw] = img_bgr
    t = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(t), float(scale), float(x0), float(y0)


def infer(sess, tensor: np.ndarray) -> np.ndarray:
    return sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]  # [MAX_DET, STRIDE]


def decode_best(out: np.ndarray, scale: float, pad_x: float, pad_y: float, conf_thr: float):
    """与 sleepy decode_best 同逻辑：取最高置信度行，坐标还原到原图像素。无检测返回 None。"""
    i = int(np.argmax(out[:, 4]))
    if out[i, 4] < conf_thr:
        return None
    k = out[i, 6:].reshape(N_KPT, 3).copy()
    k[:, 0] = (k[:, 0] - pad_x) / scale
    k[:, 1] = (k[:, 1] - pad_y) / scale
    return float(out[i, 4]), k  # k: [8,3] = 原图 x,y,vis


# ----------------------------------------------------------------------------------------- 评估
def _stats(e: np.ndarray) -> str:
    if len(e) == 0:
        return "n=0"
    return (f"n={len(e):5d} p50={np.percentile(e, 50):5.2f} p90={np.percentile(e, 90):5.2f} "
            f"p95={np.percentile(e, 95):5.2f} p99={np.percentile(e, 99):6.2f} max={e.max():7.2f}")


def evaluate(onnx_path: Path, baked: Path, split: str, conf_thr: float, vis_thr: float, out_json: Path | None):
    import cv2

    sess = make_session(onnx_path)
    rows = [json.loads(l) for l in (baked / "meta.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    print(f"[eval] {split}: {len(rows)} 张  conf>={conf_thr}  vis>={vis_thr}")

    err_all, err_kp = [], [[] for _ in range(N_KPT)]
    err_by_band: dict[str, list] = {}
    bands = [(0.0, 0.8), (0.8, 1.5), (1.5, 2.5), (2.5, 99.0)]
    n_pos = n_pos_det = n_neg = n_neg_fp = 0
    vis_tp = vis_fp = vis_fn = vis_tn = 0
    det_by_band = {f"{a}-{b}": [0, 0] for a, b in bands}
    worst = []
    vis_vals = []

    for r in rows:
        stem = r["image"].rsplit(".", 1)[0]
        lab = baked / "labels" / f"{stem}.txt"
        is_pos = lab.exists() and lab.stat().st_size > 0   # 与 bake 规则一致：label 非空即正样本
        img = cv2.imread(str(baked / "images" / r["image"]))   # 960×720 的训练同款 jpg
        t, s, px, py = preprocess(img)
        det = decode_best(infer(sess, t), s, px, py, conf_thr)
        if not is_pos:
            n_neg += 1
            n_neg_fp += det is not None
            continue
        # 预测在 960×720 图上还原到“该 jpg 坐标”，需再除以 bake 缩放到原图 1440×1080
        sc = r["scale"][0]
        gt = np.array(r["kp_px"], np.float64).reshape(N_KPT, 2)
        vis = np.array(r["vis"])
        d = r["ref_dist"]
        band = next(f"{a}-{b}" for a, b in bands if a <= d < b)
        n_pos += 1
        det_by_band[band][1] += 1
        if det is None:
            continue
        n_pos_det += 1
        det_by_band[band][0] += 1
        _, k = det
        k[:, :2] /= sc
        for j in range(N_KPT):
            pred_vis = k[j, 2] >= vis_thr
            vis_vals.append((int(vis[j]), float(k[j, 2])))
            if vis[j] == 2:
                e = float(np.hypot(*(k[j, :2] - gt[j])))
                err_all.append(e); err_kp[j].append(e)
                err_by_band.setdefault(band, []).append(e)
                worst.append((e, r["image"], j))
                vis_tp += pred_vis; vis_fn += (not pred_vis)
            else:
                vis_fp += pred_vis; vis_tn += (not pred_vis)

    err_all = np.array(err_all)
    print(f"  正样本检出 {n_pos_det}/{n_pos} = {n_pos_det / max(n_pos, 1):.3f}    负样本误检 {n_neg_fp}/{n_neg}")
    for b, (hit, tot) in det_by_band.items():
        print(f"    距离 {b:>8} m 检出 {hit}/{tot}")
    print(f"  关键点像素误差（原图 1440×1080，仅 GT 可见点，仅检出帧）\n    全部   {_stats(err_all)}")
    for j in range(N_KPT):
        print(f"    K{j + 1}     {_stats(np.array(err_kp[j]))}")
    for b in sorted(err_by_band):
        print(f"    {b:>8} m {_stats(np.array(err_by_band[b]))}")
    print(f"  vis 分类：可见点召回 {vis_tp / max(vis_tp + vis_fn, 1):.3f}  非可见点误报 {vis_fp / max(vis_fp + vis_tn, 1):.3f}")
    vv = np.array(vis_vals)
    if len(vv):
        print(f"  vis 输出范围 [{vv[:, 1].min():.3f}, {vv[:, 1].max():.3f}]")
    worst.sort(reverse=True)
    print("  最差 5 点:", [(round(e, 1), im, f"K{j + 1}") for e, im, j in worst[:5]])

    if out_json:
        out_json.write_text(json.dumps({
            "split": split, "pos": n_pos, "pos_det": n_pos_det, "neg": n_neg, "neg_fp": n_neg_fp,
            "err_p50": float(np.percentile(err_all, 50)), "err_p95": float(np.percentile(err_all, 95)),
            "err_rms": float(np.sqrt((err_all ** 2).mean())),
        }, indent=2))


# ----------------------------------------------------------------------------------------- 转储
def dump(onnx_path: Path, seq_dir: Path):
    """对 seq_dir 下每个 runNNN.images.txt 推理，写 runNNN.det.bin：
    int32 frames,N,stride + float32 scale,pad_x,pad_y + float32[frames*N*stride]。"""
    import cv2

    sess = make_session(onnx_path)
    for lst in sorted(seq_dir.glob("run*.images.txt")):
        paths = [p for p in lst.read_text().splitlines() if p.strip()]
        outs = []
        for p in paths:
            t, s, px, py = preprocess(cv2.imread(p))
            outs.append(infer(sess, t))
        arr = np.stack(outs).astype(np.float32)
        dst = lst.with_name(lst.name.replace(".images.txt", ".det.bin"))
        with open(dst, "wb") as f:
            f.write(struct.pack("<3i3f", arr.shape[0], arr.shape[1], arr.shape[2], s, px, py))
            f.write(arr.tobytes())
        print(f"[dump] {dst.name}: {arr.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("weights", type=Path); e.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("eval"); v.add_argument("onnx", type=Path); v.add_argument("--baked", type=Path, required=True)
    v.add_argument("--split", default="val"); v.add_argument("--conf", type=float, default=0.4)
    v.add_argument("--vis", type=float, default=0.5); v.add_argument("--json", type=Path)
    d = sub.add_parser("dump"); d.add_argument("onnx", type=Path); d.add_argument("seq_dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.weights, a.out)
    elif a.cmd == "eval":
        evaluate(a.onnx, a.baked, a.split, a.conf, a.vis, a.json)
    else:
        dump(a.onnx, a.seq_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
