"""Pick a checkpoint: evaluate every snapshot of a training run and rank them by one combined score.

    python -m nt.select runs/train/NAME --data baked/VAL [--split val] [--conf 0.2] [--head e2e]
        [--score ap50=1,robot_numbers=1,off_recall=0.5,corner_p95=-0.02] [--out runs/select/NAME.json]

The snapshots are weights/epochN.pt (written every save_period epochs) plus last.pt and best.pt. Each one goes
through nt.eval on the same images. The score is a weighted sum of these `nt.eval` summary numbers:
    ap50           armor AP50 (needs the right class)
    recall, precision, class_acc     at --conf
    robot_numbers  number accuracy over sentry, 1, 3, 4: the hard classes
    off_recall     recall of the off (dark screen) class
    corner_p95     corner error p95 in original pixels; give it a negative weight
The default weighs AP50 and robot numbers equally, the off class half, and 1 px of corner p95 like 2 points of
number accuracy. Change it with --score; the weights are written into the report.

Select on one set and report on another: the score of the winner on the set it was picked on is biased upwards.
Once a real val session exists, pick on real data, not on simulator val. Ultralytics' best.pt is chosen by its
own fitness on the training val split, which is why it is ranked here like any other snapshot.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE = "ap50=1,robot_numbers=1,off_recall=0.5,corner_p95=-0.02"
METRICS = ("ap50", "recall", "precision", "class_acc", "robot_numbers", "off_recall", "corner_p95")


def parse_score(text: str) -> dict[str, float]:
    weights = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        key, _, value = part.partition("=")
        if key not in METRICS:
            raise SystemExit(f"--score: unknown metric {key!r}; use {', '.join(METRICS)}")
        weights[key] = float(value)
    if not weights:
        raise SystemExit("--score is empty")
    return weights


def metrics_of(summary: dict) -> dict[str, float | None]:
    """The rankable numbers from nt.eval's summary."""
    return {
        "ap50": summary["ap50"], "recall": summary["recall"], "precision": summary["precision"],
        "class_acc": summary["class_acc"], "robot_numbers": (summary["robot_numbers"] or {}).get("class_acc"),
        "off_recall": (summary["off_class"] or {}).get("recall"), "corner_p95": summary["corner_p95_orig_px"],
    }


def score(metrics: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted sum; None when a weighted metric is missing (for example no off-class armor in the set)."""
    if any(metrics.get(k) is None for k in weights):
        return None
    return round(sum(w * float(metrics[k]) for k, w in weights.items()), 5)


def snapshots(run: Path) -> list[tuple[str, Path]]:
    """(name, path) for every checkpoint of a run, epochs in order, then last and best."""
    w = run / "weights"
    if not w.is_dir():
        raise SystemExit(f"{run} has no weights/ directory")
    epochs = sorted(w.glob("epoch*.pt"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    out = [(p.stem, p) for p in epochs]
    out += [(n, w / f"{n}.pt") for n in ("last", "best") if (w / f"{n}.pt").exists()]
    if not out:
        raise SystemExit(f"no checkpoints in {w}")
    return out


def rank(rows: list[dict]) -> list[dict]:
    """Highest score first; rows without a score go last."""
    return sorted(rows, key=lambda r: (r["score"] is None, -(r["score"] or 0.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="training run directory, runs/train/NAME")
    ap.add_argument("--data", type=Path, required=True, help="baked dataset to evaluate on")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--head", choices=["e2e", "nms"], default="e2e", help="e2e is the head that gets deployed")
    ap.add_argument("--score", default=DEFAULT_SCORE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from nt.eval import box_of, evaluate, load_ground_truth, predict, summarize
    from nt.spec import LabelSpec

    weights = parse_score(args.score)
    spec = LabelSpec.load()
    metas = load_ground_truth(args.data, args.split)
    if not metas:
        raise SystemExit(f"{args.data} has no {args.split} images")
    images = [args.data / m["image"] for m in metas]
    gts = [[{"cls": a["cls"], "corners_net": np.array(a["corners_net"], np.float64),
             "box": box_of(np.array(a["corners_net"], np.float64)), "dist_m": a.get("dist_m"),
             "yaw_deg": a.get("yaw_deg"), "occ": a.get("occ"), "vis": a["vis"]} for a in m["armors"]] for m in metas]
    rows = []
    for name, path in snapshots(args.run):
        dets = predict(path, images, 0.001, args.head, args.batch, args.device)
        res = evaluate([(d, g, m) for d, g, m in zip(dets, gts, metas)], args.conf, spec.n_classes, spec.from_class_id)
        m = metrics_of(summarize(res, spec))
        rows.append({"checkpoint": name, "path": str(path), "score": score(m, weights), **m})
        shown = " ".join(f"{k}={m[k]}" for k in METRICS)
        print(f"{name:10s} score={rows[-1]['score']} {shown}", flush=True)
    ranked = rank(rows)
    best = ranked[0]
    print(f"\nbest: {best['checkpoint']} ({best['path']}), score {best['score']}")
    report = {"run": str(args.run), "data": str(args.data), "split": args.split, "conf": args.conf, "head": args.head,
              "score_weights": weights, "ranked": ranked, "selected": best["path"]}
    out = args.out or ROOT / "runs" / "select" / f"{args.run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
