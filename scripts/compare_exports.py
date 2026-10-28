"""Compare two Daedalus exports of the same seed armor by armor, to check a change in the corner ordering.

    python3 -I scripts/compare_exports.py OLD_RAW_DIR NEW_RAW_DIR [OLD_REJECTED.jsonl]

Attempt k draws the same scene in both runs (the seed is mixed with the attempt number), and `origin` is
"seq<k>". For armors present in both, matched by class and position, it reports:
  same     identical corners in the same order
  mirrored the new corners are the old ones in a different order (the old ordering had flipped)
  moved    corners differ by more than 0.05 px in any case (something else changed)
If a rejected.jsonl of the old export is given, the armors it refused are listed separately, so you can see
whether they are now recovered.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    out = {}
    for line in (path / "labels.jsonl").read_text().splitlines():
        d = json.loads(line)
        out[d["origin"]] = d
    return out


def center(a) -> tuple:
    c = a["corners"]
    return sum(p[0] for p in c) / 4, sum(p[1] for p in c) / 4


def main() -> None:
    old, new = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    refused = set()
    if len(sys.argv) > 3:
        for line in Path(sys.argv[3]).read_text().splitlines():
            r = json.loads(line)
            for i in r["issues"]:
                if i["code"] == "E_QUAD_ORDER":
                    refused.add((r["origin"], i["armor"]))
    tally = {"accepted": Counter(), "refused": Counter()}
    unmatched_old = 0
    for origin, d_old in old.items():
        d_new = new.get(origin)
        if d_new is None:
            continue
        for k, a in enumerate(d_old["armors"]):
            ca = center(a)
            match = min(
                (b for b in d_new["armors"] if (b["color"], b["label"]) == (a["color"], a["label"])),
                key=lambda b: math.dist(center(b), ca),
                default=None,
            )
            if match is None or math.dist(center(match), ca) > 2.0:
                unmatched_old += 1
                continue
            same = all(math.dist(p, q) < 0.05 for p, q in zip(a["corners"], match["corners"]))
            reordered = not same and all(
                min(math.dist(p, q) for q in match["corners"]) < 0.05 for p in a["corners"]
            )
            kind = "same" if same else "mirrored" if reordered else "moved"
            tally["refused" if (origin, k) in refused else "accepted"][kind] += 1
    for group, counts in tally.items():
        print(f"{group:9s} {dict(counts)}")
    print(f"old armors with no counterpart in the new export (view-angle limit changed): {unmatched_old}")


if __name__ == "__main__":
    main()
