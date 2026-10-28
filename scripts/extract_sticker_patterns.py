"""Rasterize the sentry, outpost and base stickers of Daedalus' armor models into pattern masks for the screen.

    python -I scripts/extract_sticker_patterns.py [--glb ~/bevy_robomaster_simulator/assets/vehicle.glb] \\
        [--out configs/screens/patterns] [--px-per-mm 4]

Writes <label>.png (white = lit, black = dark, tight to the artwork) for sentry, outpost and base. A screen
spec with `content.bitmap_dir` pointing at the folder draws these labels with the pattern instead of the font,
resampled to the dot grid. The stickers are the ones the light-bar armors carry in the simulator (node names
3G, 3O and 3B under 3_ARMOR_C); the outpost and base ones are the same artwork as the 2027 build-spec
manual's appendix figures 1 and 3.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np

STICKERS = {"sentry": "3G", "outpost": "3O", "base": "3B"}
ROOT = "3_ARMOR_ROOT"


def load_glb(path: Path) -> tuple[dict, bytes]:
    b = path.read_bytes()
    clen = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + clen]), b[20 + clen + 8:]


def accessor(j: dict, blob: bytes, idx: int) -> np.ndarray:
    a = j["accessors"][idx]
    bv = j["bufferViews"][a["bufferView"]]
    if bv.get("byteStride"):
        raise ValueError("interleaved buffers are not supported")
    comp = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[a["componentType"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    return np.frombuffer(blob, comp, count=a["count"] * n, offset=bv.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], n)


def node_matrix(n: dict) -> np.ndarray:
    m = np.eye(4)
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m[:3, :3] = r * np.array(n.get("scale", [1, 1, 1]))
    m[:3, 3] = n.get("translation", [0, 0, 0])
    return m


def sticker_triangles(j: dict, blob: bytes, name: str) -> np.ndarray:
    """Triangles of a sticker in the armor-root frame, meters, shape (N, 3, 3)."""
    nodes = j["nodes"]
    by_name = {n.get("name"): i for i, n in enumerate(nodes)}
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    i, root = by_name[name], by_name[ROOT]
    m = np.eye(4)
    k = i
    while k != root:
        m = node_matrix(nodes[k]) @ m
        k = parent[k]
    tris = []
    for p in j["meshes"][nodes[i]["mesh"]]["primitives"]:
        pts = accessor(j, blob, p["attributes"]["POSITION"]).astype(np.float64)
        pts = (m[:3, :3] @ pts.T).T + m[:3, 3]
        idx = accessor(j, blob, p["indices"]).ravel() if "indices" in p else np.arange(len(pts))
        tris.append(pts[idx].reshape(-1, 3, 3))
    return np.concatenate(tris)


def rasterize(tris: np.ndarray, px_per_mm: float) -> np.ndarray:
    # In the vehicle.glb armor-root frame the plate's width runs along +x and its height along +z (up), so the
    # image is (x, -z) with y pointing down.
    xy = np.stack([tris[..., 0], -tris[..., 2]], -1) * 1000.0
    lo = xy.reshape(-1, 2).min(0)
    hi = xy.reshape(-1, 2).max(0)
    w, h = (np.ceil((hi - lo) * px_per_mm) + 1).astype(int)
    img = np.zeros((h, w), np.uint8)
    for t in xy:
        cv2.fillPoly(img, [np.round((t - lo) * px_per_mm * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", type=Path, default=Path.home() / "bevy_robomaster_simulator" / "assets" / "vehicle.glb")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "screens" / "patterns")
    ap.add_argument("--px-per-mm", type=float, default=4.0)
    args = ap.parse_args()
    j, blob = load_glb(args.glb)
    args.out.mkdir(parents=True, exist_ok=True)
    for label, node in STICKERS.items():
        img = rasterize(sticker_triangles(j, blob, node), args.px_per_mm)
        cv2.imwrite(str(args.out / f"{label}.png"), img)
        print(f"{label}: {node} -> {args.out / f'{label}.png'} ({img.shape[1]}x{img.shape[0]} px, "
              f"{img.shape[1] / args.px_per_mm:.1f} x {img.shape[0] / args.px_per_mm:.1f} mm)")


if __name__ == "__main__":
    main()
