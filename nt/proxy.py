"""Proxy dot-matrix screen: real-camera images with exact corner labels before the official armor exists.

    python -m nt.proxy render --out proxies/NAME --label three --color red [--text 3] [--display 1920x1080]
                              [--cols 32 --rows 16] [--dot round|square] [--fill 0.6] [--panel 0.62]
    python -m nt.proxy label --session sessions/S --proxy proxies/NAME/proxy.json --dst datasets/real/NAME
                             [--min-markers 2] [--max-rms 1.0]

render writes proxy.png, to be shown full screen on a monitor or tablet (or printed flat), and proxy.json with
the geometry in display pixels: a dot-matrix panel showing the text, and four ArUco markers (DICT_4X4_50,
ids 0-3) in the display corners. Film it with the robot camera as a capture session, then `label` finds the
markers in every frame, fits the display-to-image homography, and maps the panel outline into the image.
That gives the four corners (LT, LB, RB, RT of the dot grid's outer edge) to sub-pixel accuracy with no
manual work, through the real lens, sensor, exposure and compression.

What this is for: measuring what the real camera does to a dot-matrix pattern (moire, blur, bloom, aliasing at
distance), checking a corner definition, and testing a sim-trained model on real optics. It is not the
official screen: an LCD's own pixel grid, brightness and viewing-angle falloff differ from an LED matrix, so
these datasets get their own name and are kept out of training unless you decide otherwise.

Conventions: display coordinates use the edge convention (pixel i covers [i, i+1)), like the labels.
OpenCV returns marker corners with pixel centers at integers, so 0.5 is added after all camera math. With
the session's intrinsics the homography is fitted on undistorted points and the panel corners are distorted
back, so labels are in raw image pixels as sleepy sees them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from nt import session as sess
from nt.canonical import Armor, Sample, dumps, sample_to_dict
from nt.spec import LabelSpec

ARUCO_DICT = "DICT_4X4_50"
PROXY_VERSION = 1
DEFAULT_TEXT = {"sentry": "S", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "outpost": "O", "base": "B"}
LIT_BGR = {"red": (40, 40, 255), "blue": (255, 90, 20), "gray": (200, 200, 200), "purple": (220, 40, 200)}
UNLIT_BGR = (38, 38, 38)


# 5x7 dot font, the common LED-matrix one; scaled by whole dots so every stroke stays a clean run of dots.
FONT_5X7 = {
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    "S": "01111 10000 10000 01110 00001 00001 11110", "O": "01110 10001 10001 10001 10001 10001 01110",
    "B": "11110 10001 10001 11110 10001 10001 11110",
}


def _glyph_mask(text: str, cols: int, rows: int) -> np.ndarray:
    """Which dots are lit. Text made of FONT_5X7 characters is drawn with whole-dot scaling and one font
    column between characters; anything else falls back to an anti-aliased font averaged per dot."""
    if text and all(ch in FONT_5X7 for ch in text):
        glyphs = [np.array([[c == "1" for c in row] for row in FONT_5X7[ch].split()]) for ch in text]
        gap = np.zeros((7, 1), bool)
        line = np.hstack([g if i == 0 else np.hstack([gap, g]) for i, g in enumerate(glyphs)])
        k = min((rows - 2) // 7, (cols - 2) // line.shape[1])
        if k >= 1:
            big = np.kron(line, np.ones((k, k), bool))
            mask = np.zeros((rows, cols), bool)
            y0, x0 = (rows - big.shape[0]) // 2, (cols - big.shape[1]) // 2
            mask[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
            return mask
    cell = 16
    canvas = np.zeros((rows * cell, cols * cell), np.uint8)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(2, rows * cell // 9)
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, thick)
    scale = min(0.86 * cols * cell / tw, 0.78 * rows * cell / th)
    thick = max(2, int(thick * scale / 2.2))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    org = ((cols * cell - tw) // 2, (rows * cell + th) // 2)
    cv2.putText(canvas, text, org, font, scale, 255, thick, cv2.LINE_AA)
    return canvas.reshape(rows, cell, cols, cell).mean(axis=(1, 3)) > 96


def render(out: Path, spec: LabelSpec, label: str, color: str, text: str | None = None, display=(1920, 1080), cols: int = 32,
           rows: int = 16, dot: str = "round", fill: float = 0.6, panel: float = 0.62) -> dict:
    labels = {v: k for k, v in spec.labels.items()}
    colors = {v: k for k, v in spec.colors.items()}
    if label not in labels or color not in colors:
        raise ValueError(f"label must be one of {sorted(labels)}, color one of {sorted(colors)}")
    if out.exists():
        raise FileExistsError(f"{out} exists")
    W, H = display
    text = text if text is not None else DEFAULT_TEXT[label]
    img = np.zeros((H, W, 3), np.uint8)

    # Panel: an integer pitch, centered, `panel` of the display width at most.
    pitch = int(min(panel * W / cols, 0.7 * H / rows))
    if pitch < 4:
        raise ValueError("dots would be smaller than 4 display pixels; use fewer cols/rows or a larger display")
    pw, ph = cols * pitch, rows * pitch
    x0, y0 = (W - pw) // 2, (H - ph) // 2
    lit = _glyph_mask(text, cols, rows)
    r = max(1, int(round(pitch * fill / 2)))
    for gy in range(rows):
        for gx in range(cols):
            c = LIT_BGR[color] if lit[gy, gx] else UNLIT_BGR
            cx, cy = x0 + gx * pitch + pitch // 2, y0 + gy * pitch + pitch // 2
            if dot == "round":
                cv2.circle(img, (cx, cy), r, c, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (cx - r, cy - r), (cx + r - 1, cy + r - 1), c, -1)

    # Markers in the four display corners, each on a white quiet zone of one module.
    side = int(min(0.17 * H, (W - pw) / 2 - 8, (H - ph) / 2 + 0.25 * H))
    side = max(side, 48)
    quiet = max(side // 6, 8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    markers = {}
    spots = {0: (quiet, quiet), 1: (W - side - quiet, quiet), 2: (W - side - quiet, H - side - quiet), 3: (quiet, H - side - quiet)}
    for mid, (mx, my) in spots.items():
        img[my - quiet:my + side + quiet, mx - quiet:mx + side + quiet] = 255
        img[my:my + side, mx:mx + side] = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, mid, side), cv2.COLOR_GRAY2BGR)
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left (edge convention)
        markers[str(mid)] = [[mx, my], [mx + side, my], [mx + side, my + side], [mx, my + side]]
    if not (x0 >= spots[0][0] + side + quiet or y0 >= spots[0][1] + side + quiet):
        raise ValueError("markers overlap the panel; lower --panel")

    corners = [[x0, y0], [x0, y0 + ph], [x0 + pw, y0 + ph], [x0 + pw, y0]]  # LT, LB, RB, RT, outer edge of the dot grid
    meta = {
        "proxy_version": PROXY_VERSION, "name": out.name, "display": [W, H], "aruco_dict": ARUCO_DICT, "markers": markers,
        "panel_corners": corners, "color": colors[color], "label": labels[label], "text": text,
        "grid": {"cols": cols, "rows": rows, "pitch_px": pitch, "dot": dot, "fill": fill},
        "corner_definition": "outer edge of the dot grid (proxy)",
        "panel_size_mm": None,
        "notes": "Show proxy.png full screen at 100% scaling. Measure the panel on the physical display and fill "
                 "panel_size_mm [w, h] if PnP numbers are needed.",
    }
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "proxy.png"), img)
    (out / "proxy.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def _distort(pts_undist: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Pixel points without distortion -> raw pixels, plumb_bob model (OpenCV center convention)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    x, y = (pts_undist[:, 0] - cx) / fx, (pts_undist[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd * fx + cx, yd * fy + cy], axis=1)


class ProxyLabeler:
    def __init__(self, proxy: dict, camera=None, min_markers: int = 2, max_rms: float = 1.0) -> None:
        self.proxy = proxy
        self.min_markers, self.max_rms = min_markers, max_rms
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, proxy["aruco_dict"]))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.K = self.dist = None
        if camera is not None:
            self.K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], np.float64)
            self.dist = np.array(camera.dist, np.float64)

    def label(self, img: np.ndarray) -> tuple[np.ndarray | None, dict]:
        """Panel corners (4, 2) in raw image pixels, edge convention, or None with a skip code in info
        (no_markers, few_markers, homography_failed, homography_rms)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        found, ids, _ = self.detector.detectMarkers(gray)
        info = {"markers": 0}
        if ids is None:
            return None, {**info, "skip": "no_markers"}
        src, dst = [], []
        for c, i in zip(found, ids.ravel()):
            key = str(int(i))
            if key in self.proxy["markers"]:
                # display corners: edge convention -> centers at integers, to match OpenCV's image coordinates
                src += [[x - 0.5, y - 0.5] for x, y in self.proxy["markers"][key]]
                dst += c.reshape(4, 2).tolist()
                info["markers"] += 1
        if info["markers"] < self.min_markers:
            return None, {**info, "skip": "few_markers"}
        src, dst = np.array(src, np.float64), np.array(dst, np.float64)
        if self.K is not None:
            dst = cv2.undistortPoints(dst.reshape(-1, 1, 2), self.K, self.dist, P=self.K).reshape(-1, 2)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            return None, {**info, "skip": "homography_failed"}
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
        info["h_rms_px"] = round(rms, 3)
        if rms > self.max_rms:
            return None, {**info, "skip": "homography_rms"}
        panel = np.array(self.proxy["panel_corners"], np.float64) - 0.5
        corners = cv2.perspectiveTransform(panel.reshape(-1, 1, 2), H).reshape(-1, 2)
        if self.K is not None:
            corners = _distort(corners, self.K, self.dist)
        return corners + 0.5, info


def label_session(session_dir: Path, proxy_path: Path, dst: Path, spec: LabelSpec, min_markers: int = 2, max_rms: float = 1.0) -> dict:
    from nt.ingest import ingest

    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    if proxy.get("proxy_version") != PROXY_VERSION:
        raise ValueError(f"{proxy_path}: proxy_version {proxy.get('proxy_version')} is not {PROXY_VERSION}")
    meta = sess.load(session_dir)
    camera = sess.camera_of(meta)
    labeler = ProxyLabeler(proxy, camera, min_markers, max_rms)
    staging = dst.parent / f".{dst.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    rows, skipped = [], {}
    w, h = meta["width"], meta["height"]
    try:
        for f in meta["frames"]:
            img = cv2.imread(str(session_dir / f["image"]), cv2.IMREAD_COLOR)
            corners, info = labeler.label(img)
            if corners is None:
                skipped[info["skip"]] = skipped.get(info["skip"], 0) + 1
                continue
            vis = [2 if 0 <= x <= w and 0 <= y <= h else 0 for x, y in corners]
            image = f"images/{f['id']}{Path(f['image']).suffix}"
            try:
                (staging / image).hardlink_to(session_dir / f["image"])
            except OSError:
                shutil.copy2(session_dir / f["image"], staging / image)
            s = Sample(id=f["id"], image=image, w=w, h=h, source="real",
                       armors=[Armor(color=proxy["color"], label=proxy["label"], corners=corners.tolist(), vis=vis)],
                       camera=camera, origin=f["id"],
                       scene={"session": meta["name"], "frame_index": f["index"], "t_s": f["t_s"], "proxy": proxy["name"],
                              "proxy_text": proxy["text"], **info})
            rows.append(dumps(sample_to_dict(s)))
        (staging / "labels.jsonl").write_text("".join(r + "\n" for r in rows), encoding="utf-8")
        if not rows:
            raise ValueError(f"no frame of {session_dir} could be labeled: {skipped}")
        m = ingest(staging, dst, "canonical-jsonl", "real", dst.name, spec, link=True, role=meta["role"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    m["proxy_skipped"] = skipped
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--label", required=True, help="label_spec label name, e.g. three")
    r.add_argument("--color", required=True, help="label_spec color name, e.g. red")
    r.add_argument("--text", default=None, help="what the panel shows; default is the label's digit or letter")
    r.add_argument("--display", default="1920x1080", help="display resolution WxH")
    r.add_argument("--cols", type=int, default=32)
    r.add_argument("--rows", type=int, default=16)
    r.add_argument("--dot", choices=["round", "square"], default="round")
    r.add_argument("--fill", type=float, default=0.6, help="dot diameter / pitch")
    r.add_argument("--panel", type=float, default=0.62, help="panel width as a fraction of the display width")
    lb = sub.add_parser("label")
    lb.add_argument("--session", type=Path, required=True)
    lb.add_argument("--proxy", type=Path, required=True)
    lb.add_argument("--dst", type=Path, required=True)
    lb.add_argument("--min-markers", type=int, default=2)
    lb.add_argument("--max-rms", type=float, default=1.0, help="homography fit RMS on marker corners, pixels")
    args = ap.parse_args()
    spec = LabelSpec.load()
    if args.cmd == "render":
        W, H = (int(v) for v in args.display.lower().split("x"))
        meta = render(args.out, spec, args.label, args.color, args.text, (W, H), args.cols, args.rows, args.dot, args.fill, args.panel)
        print(f"wrote {args.out}/proxy.png ({W}x{H}), panel {meta['grid']['cols']}x{meta['grid']['rows']} dots at "
              f"{meta['grid']['pitch_px']} px pitch, text {meta['text']!r}")
        return
    m = label_session(args.session, args.proxy, args.dst, spec, args.min_markers, args.max_rms)
    print(json.dumps({k: m[k] for k in ("counts", "issues", "rejected", "role", "proxy_skipped")}, indent=2))
    print(f"wrote {args.dst}")
    sys.exit(0)


if __name__ == "__main__":
    main()
