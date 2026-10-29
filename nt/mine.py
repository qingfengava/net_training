"""Bad-case mining: rank frames worth a person's time, and write them as a frame list for nt.annotate.

    python -m nt.mine session  sessions/S --model M.onnx --out runs/mine/S.txt [--top 200] [--min-gap 5] [--tiles 0|2]
    python -m nt.mine dataset  datasets/real/D --model M.onnx --out runs/mine/D.txt [--top 200]
then
    python -m nt.annotate export --session sessions/S --frames runs/mine/S.txt --model M.onnx --out annot/P
    python -m nt.annotate export --dataset datasets/real/D --frames runs/mine/D.txt --model M.onnx --out annot/P

session mode needs no labels. A frame scores for
    uncertain    detections between --review-conf and --conf
    bad_geometry a detection the label checks would refuse (non-convex, wrong order, degenerate)
    flicker      an armor seen in the frames before and after but not in this one, or the other way round
    flip         the same armor (matched by box overlap) changes color or number from the previous frame
    jump         the corners of a matched armor move more than --jump times its size between frames
Temporal checks compare consecutive session frames, so they mean something only when the session keeps
neighboring frames (a small --every); with a large stride they mostly report motion.
dataset mode compares the model with the labels: missed armors, extra detections, wrong class, and corner
errors above --corner-px; large disagreements are often label mistakes, so they go to re-review.

Selected frames are at least --min-gap frame indices apart, so one bad second does not fill the list.
The output text file has one id per line with the score and reasons after a '#'; a JSON with all scores is
written next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from nt import session as sess
from nt.canonical import Armor, read_manifest, read_samples
from nt.detector import ArmorModel, Detection
from nt.geometry import bbox, bbox_iou
from nt.spec import LabelSpec
from nt.validators import armor_issues

WEIGHTS = {"uncertain": 1.0, "bad_geometry": 2.0, "flicker": 2.0, "flip": 3.0, "jump": 1.5,
           "missed": 3.0, "extra": 2.0, "wrong_class": 3.0, "corner_error": 2.0}


def _size(d: Detection) -> float:
    x0, y0, x1, y1 = d.box
    return max(x1 - x0, y1 - y0, 1.0)


def _match(prev: list[Detection], cur: list[Detection], iou_thr: float = 0.3) -> list[tuple[int, int]]:
    pairs = sorted(((bbox_iou(p.box, c.box), i, j) for i, p in enumerate(prev) for j, c in enumerate(cur)), reverse=True)
    used_p, used_c, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_thr:
            break
        if i in used_p or j in used_c:
            continue
        used_p.add(i)
        used_c.add(j)
        out.append((i, j))
    return out


def score_session(frames: list[dict], dets: list[list[Detection]], spec: LabelSpec, w: int, h: int, conf: float,
                  jump: float) -> list[dict]:
    """Per-frame reasons from detections at review_conf and above. frames are in index order."""
    rows = []
    confident = [[d for d in ds if d.conf >= conf] for ds in dets]
    # Frames count as neighbors when they are at most two typical strides apart; a gap left by the session's
    # duplicate filter or a cut in the recording must not be read as a flicker.
    steps = np.diff([f["index"] for f in frames])
    stride = float(np.median(steps)) if len(steps) else 1.0

    def adjacent(a: int, b: int) -> bool:
        return abs(frames[a]["index"] - frames[b]["index"]) <= 2 * stride

    for k, f in enumerate(frames):
        reasons = Counter()
        for d in dets[k]:
            if d.conf < conf:
                reasons["uncertain"] += 1
            a = Armor(color=d.color, label=d.label, corners=d.corners.tolist())
            if any(level == "error" and code != "E_CORNER_OUTSIDE" for level, code, _ in armor_issues(a, w, h, spec)):
                reasons["bad_geometry"] += 1
        if 0 < k < len(frames) - 1 and adjacent(k - 1, k) and adjacent(k, k + 1):
            before, now, after = confident[k - 1], confident[k], confident[k + 1]
            # in both neighbors but missing here, or here but in neither neighbor
            both = [i for i, _ in _match(before, after)]
            seen_now = {i for i, _ in _match(before, now)}
            reasons["flicker"] += sum(1 for i in both if i not in seen_now)
            lonely = set(range(len(now))) - {j for _, j in _match(before, now)} - {j for j, _ in _match(now, after)}
            reasons["flicker"] += len(lonely)
        if k > 0 and adjacent(k - 1, k):
            for i, j in _match(confident[k - 1], confident[k]):
                p, c = confident[k - 1][i], confident[k][j]
                if (p.color, p.label) != (c.color, c.label):
                    reasons["flip"] += 1
                if float(np.abs(p.corners - c.corners).max()) > jump * _size(c):
                    reasons["jump"] += 1
        reasons = +reasons  # drop the zero counts
        score = sum(WEIGHTS[r] * n for r, n in reasons.items())
        rows.append({"id": f["id"], "index": f["index"], "score": score, "reasons": dict(reasons)})
    return rows


def score_labeled(samples, dets: list[list[Detection]], spec: LabelSpec, corner_px: float) -> list[dict]:
    rows = []
    for idx, (s, ds) in enumerate(zip(samples, dets)):
        reasons, worst = Counter(), 0.0
        gts = [a for a in s.armors if len(a.corners) == 4]
        pairs = sorted(((bbox_iou(d.box, bbox(a.corners)), i, j) for i, d in enumerate(ds) for j, a in enumerate(gts)), reverse=True)
        used_d, used_g = set(), set()
        for iou, i, j in pairs:
            if iou < 0.3:
                break
            if i in used_d or j in used_g:
                continue
            used_d.add(i)
            used_g.add(j)
            d, a = ds[i], gts[j]
            if (d.color, d.label) != (a.color, a.label):
                reasons["wrong_class"] += 1
            vis = np.array(a.vis) > 0
            if vis.any():
                err = float(np.linalg.norm(d.corners - np.array(a.corners), axis=1)[vis].max())
                worst = max(worst, err)
                if err > corner_px:
                    reasons["corner_error"] += 1
        reasons["missed"] += len(gts) - len(used_g)
        reasons["extra"] += len(ds) - len(used_d)
        reasons = +reasons
        score = sum(WEIGHTS[r] * n for r, n in reasons.items())
        index = s.scene.get("frame_index", idx) if isinstance(s.scene, dict) else idx
        rows.append({"id": s.id, "origin": s.origin, "index": index, "score": score, "reasons": dict(reasons),
                     "max_corner_error_px": round(worst, 2)})
    return rows


def select(rows: list[dict], top: int, min_gap: int) -> list[dict]:
    chosen = []
    for r in sorted((r for r in rows if r["score"] > 0), key=lambda r: (-r["score"], r["index"])):
        if all(abs(r["index"] - c["index"]) >= min_gap for c in chosen):
            chosen.append(r)
        if len(chosen) >= top:
            break
    return sorted(chosen, key=lambda r: r["index"])


def write(rows: list[dict], chosen: list[dict], out: Path, header: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {json.dumps(header, ensure_ascii=False)}"]
    for r in chosen:
        why = ",".join(f"{k}={v}" for k, v in sorted(r["reasons"].items()))
        lines.append(f"{r['id']}  # score {r['score']:.1f} {why}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    totals = Counter()
    for r in rows:
        totals.update(r["reasons"])
    out.with_suffix(".json").write_text(json.dumps({**header, "totals": dict(totals), "frames": rows}, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("session", "dataset"):
        p = sub.add_parser(name)
        p.add_argument("path", type=Path)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--top", type=int, default=200)
        p.add_argument("--min-gap", type=int, default=5, help="minimum frame-index distance between selected frames")
        p.add_argument("--conf", type=float, default=0.5)
        p.add_argument("--review-conf", type=float, default=0.1)
        p.add_argument("--tiles", type=int, default=0, choices=[0, 2])
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        if name == "session":
            p.add_argument("--jump", type=float, default=0.5, help="corner move between frames, in armor sizes")
        else:
            p.add_argument("--corner-px", type=float, default=4.0)
    args = ap.parse_args()
    spec = LabelSpec.load()
    model = ArmorModel(args.model, device=args.device)

    if args.cmd == "session":
        meta = sess.load(args.path)
        frames = sorted(meta["frames"], key=lambda f: f["index"])
        dets = []
        for i, f in enumerate(frames):
            img = cv2.imread(str(args.path / f["image"]), cv2.IMREAD_COLOR)
            # an old model can still report robots 2 and 5, which RM27 does not have
            dets.append([d for d in model.detect(img, args.review_conf, args.tiles) if spec.valid_pair(d.color, d.label)])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(frames)} frames", file=sys.stderr, flush=True)
        rows = score_session(frames, dets, spec, meta["width"], meta["height"], args.conf, args.jump)
        header = {"mode": "session", "session": meta["name"], "role": meta["role"], "model": model.describe(),
                  "conf": args.conf, "review_conf": args.review_conf, "tiles": args.tiles, "jump": args.jump}
    else:
        man = read_manifest(args.path / "manifest.json")
        samples = list(read_samples(args.path / "labels.jsonl"))
        dets = []
        for i, s in enumerate(samples):
            img = cv2.imread(str(args.path / s.image), cv2.IMREAD_COLOR)
            dets.append([d for d in model.detect(img, args.conf, args.tiles) if spec.valid_pair(d.color, d.label)])
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(samples)} samples", file=sys.stderr, flush=True)
        rows = score_labeled(samples, dets, spec, args.corner_px)
        header = {"mode": "dataset", "dataset": man.name, "role": man.role, "model": model.describe(), "conf": args.conf,
                  "tiles": args.tiles, "corner_px": args.corner_px}
    chosen = select(rows, args.top, args.min_gap)
    write(rows, chosen, args.out, header)
    totals = Counter()
    for r in rows:
        totals.update(r["reasons"])
    print(f"{len(rows)} frames scored, {sum(1 for r in rows if r['score'] > 0)} with reasons, {len(chosen)} selected -> {args.out}")
    print("reasons over all frames:", dict(totals))


if __name__ == "__main__":
    main()
