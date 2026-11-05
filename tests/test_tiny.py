"""极小目标实验的坐标、匹配、无真值候选选择及样本分母约束。"""
import unittest

import numpy as np

from nt import tiny
from nt.spec import LabelSpec

SPEC = LabelSpec.load()


def detection(box, cls=0, conf=0.8):
    x0, y0, x1, y1 = box
    return {"cls": cls, "conf": conf, "box": np.array(box, float),
            "pts": np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]], float)}


def target(box, cls=0):
    d = detection(box, cls)
    return {"cls": cls, "box": d["box"], "corners_net": d["pts"], "vis": [2] * 4}


class TinyTest(unittest.TestCase):
    def test_roi_bounds_aspect_and_coverage(self):
        for box in ([0, 0, 12, 14], [1428, 1068, 1440, 1080], [700, 500, 712, 514]):
            x, y, w, h = tiny.roi_for(box, 1440, 1080)
            self.assertEqual(w * 3, h * 4)
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 1440)
            self.assertLessEqual(y + h, 1080)
            self.assertEqual(tiny.bake._inside_fraction(box, (x, y, w, h)), 1)
        self.assertEqual(tiny.roi_for([0, 0, 3, 2], 96, 72), (0, 0, 96, 72))

    def test_proposals_use_only_predictions_and_respect_budget(self):
        self.assertEqual(tiny.proposal_rois([], (1080, 1440, 3), 0.5), [])
        ds = [detection([80, 60, 86, 67], conf=0.9), detection([81, 61, 87, 68], conf=0.8),
              detection([650, 450, 656, 457], conf=0.7), detection([330, 250, 336, 257], conf=0.01),
              detection([300, 200, 400, 300], conf=0.99)]
        rois = tiny.proposal_rois(ds, (1080, 1440, 3), 0.5, max_rois=2)
        self.assertEqual(len(rois), 2)
        self.assertLess(rois[0][0], rois[1][0])
        self.assertEqual(tiny.proposal_rois(ds, (1080, 1440, 3), 0.5, max_rois=1), rois[:1])

    def test_roi_coordinates_map_back_to_historical_input(self):
        mapped = tiny.map_crop([detection([20, 30, 40, 50])], (100, 200, 384, 288), 0.5)[0]
        np.testing.assert_array_equal(mapped["pts"], [[55, 107.5], [55, 112.5], [60, 112.5], [60, 107.5]])
        np.testing.assert_array_equal(mapped["box"], [55, 107.5, 60, 112.5])

    def test_matching_does_not_drop_misses_or_share_detections(self):
        gts = [target([0, 0, 8, 8]), target([0, 0, 8, 8]), target([20, 20, 28, 28])]
        rows = tiny.outcomes([detection([0, 0, 8, 8]), detection([20, 20, 28, 28], cls=1)], gts, 0.2, SPEC, 0.5)
        counts = tiny.metrics([{"full": row} for row in rows], "full")
        self.assertEqual((counts["n"], counts["hits"], counts["correct"]), (3, 2, 1))
        self.assertEqual(counts["end_to_end_correct"], 0.3333)
        self.assertEqual(counts["unmatched"], 1)
        self.assertEqual(rows[-1]["error_flags"], ["wrong_number"])

    def test_localization_candidate_is_distinct_from_no_detection(self):
        row = tiny.outcomes([detection([2, 2, 10, 10])], [target([0, 0, 8, 8])], 0.2, SPEC, 1)[0]
        self.assertEqual(row["state"], "unmatched")
        self.assertEqual(row["error_flags"], ["localization_or_assignment"])
        self.assertLess(row["best_iou"], 0.5)

    def test_merge_does_not_choose_class_using_truth(self):
        ds = [detection([0, 0, 8, 8], cls=0, conf=0.6), detection([0, 0, 8, 8], cls=1, conf=0.8)]
        merged = tiny.merge(ds, 0.2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["cls"], 1)

    def test_controls_are_reproducible_balanced_and_exclude_errors(self):
        rows = [{"id": str(i), "image": f"im{i // 2}", "gt_class": i % 3, "width_net": 3 + i % 4,
                 "full": {"state": "correct" if i else "wrong_class"}} for i in range(31)]
        picked = tiny.choose_controls(rows, 9, 0)
        self.assertEqual(len(picked), 9)
        self.assertNotIn("0", picked)
        self.assertEqual(picked, tiny.choose_controls(list(reversed(rows)), 9, 0))
        self.assertEqual(len({r["image"] for r in rows if r["id"] in picked}), 9)
        self.assertEqual(len(tiny.choose_controls(rows, 100, 0)), 30)
        self.assertEqual(tiny.choose_controls(rows, 0, 0), set())


if __name__ == "__main__":
    unittest.main()
