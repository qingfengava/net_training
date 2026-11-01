"""nt.select: the score, the ranking and which files count as checkpoints."""
import tempfile
import unittest
from pathlib import Path

from nt.select import DEFAULT_SCORE, metrics_of, parse_score, rank, score, snapshots


def summary(ap50, robot, off, p95):
    return {"ap50": ap50, "recall": 0.9, "precision": 0.75, "class_acc": 0.84,
            "robot_numbers": {"n": 10, "recall": 0.9, "class_acc": robot},
            "off_class": None if off is None else {"n": 5, "recall": off, "class_acc": 1.0},
            "corner_p95_orig_px": p95}


class SelectTest(unittest.TestCase):
    def test_default_score_and_ranking(self):
        w = parse_score(DEFAULT_SCORE)
        self.assertEqual(w, {"ap50": 1.0, "robot_numbers": 1.0, "off_recall": 0.5, "corner_p95": -0.02})
        a = metrics_of(summary(0.82, 0.78, 0.75, 6.2))
        b = metrics_of(summary(0.83, 0.76, 0.70, 6.0))
        self.assertAlmostEqual(score(a, w), 0.82 + 0.78 + 0.375 - 0.124, places=5)
        rows = [{"checkpoint": "b", "score": score(b, w)}, {"checkpoint": "a", "score": score(a, w)},
                {"checkpoint": "none", "score": None}]
        self.assertEqual([r["checkpoint"] for r in rank(rows)], ["a", "b", "none"])

    def test_missing_metric_gives_no_score_and_bad_keys_are_refused(self):
        w = parse_score(DEFAULT_SCORE)
        self.assertIsNone(score(metrics_of(summary(0.8, 0.7, None, 6.0)), w))  # no off-class armor in the set
        self.assertIsNotNone(score(metrics_of(summary(0.8, 0.7, None, 6.0)), parse_score("ap50=1")))
        for bad in ("mAP=1", ""):
            with self.assertRaises(SystemExit):
                parse_score(bad)

    def test_snapshots_are_ordered_by_epoch_then_last_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "weights"
            w.mkdir()
            for n in ("epoch10", "epoch5", "epoch100", "last", "best"):
                (w / f"{n}.pt").write_bytes(b"x")
            self.assertEqual([n for n, _ in snapshots(Path(tmp))], ["epoch5", "epoch10", "epoch100", "last", "best"])
            with self.assertRaises(SystemExit):
                snapshots(Path(tmp) / "missing")


if __name__ == "__main__":
    unittest.main()
