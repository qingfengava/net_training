import unittest

import numpy as np

from nt.eval import average_precision, box_of, decode_contract, evaluate, iou, match, paired_difference, summarize
from nt.spec import LabelSpec

SPEC = LabelSpec.load()


def quad(cx, cy, w=40.0, h=16.0):
    return np.array([[cx - w / 2, cy - h / 2], [cx - w / 2, cy + h / 2], [cx + w / 2, cy + h / 2], [cx + w / 2, cy - h / 2]])


def det(cx, cy, cls, conf=0.9, shift=(0.0, 0.0)):
    pts = quad(cx, cy) + np.array(shift)
    return {"conf": conf, "cls": cls, "pts": pts, "box": box_of(pts)}


def gt(cx, cy, cls, **extra):
    pts = quad(cx, cy)
    return {"cls": cls, "corners_net": pts, "box": box_of(pts), **extra}


META = {"scale": 2.0}


class EvalTest(unittest.TestCase):
    def test_perfect_predictions(self):
        r = evaluate([([det(100, 100, 3), det(300, 200, 9)], [gt(100, 100, 3), gt(300, 200, 9)], META)], 0.2, 16)
        self.assertEqual(r["armor_ap50"], 1.0)
        self.assertEqual((r["at_conf"]["recall"], r["at_conf"]["precision"], r["at_conf"]["class_acc"]), (1.0, 1.0, 1.0))
        self.assertEqual(r["corner_error"]["net_px"]["max"], 0.0)

    def test_missed_armor_false_positive_and_wrong_class(self):
        dets = [det(100, 100, 4), det(500, 400, 3)]  # right place wrong label, plus an invented armor
        gts = [gt(100, 100, 3), gt(300, 200, 9)]
        r = evaluate([(dets, gts, META)], 0.2, 16)["at_conf"]
        self.assertEqual((r["recall"], r["precision"], r["false_positives"]), (0.5, 0.5, 1))
        self.assertEqual((r["class_acc"], r["color_acc"], r["label_acc"]), (0.0, 1.0, 0.0))

    def test_corner_error_is_reported_in_original_pixels(self):
        r = evaluate([([det(100, 100, 3, shift=(3.0, 4.0))], [gt(100, 100, 3)], META)], 0.2, 16)["corner_error"]
        self.assertAlmostEqual(r["net_px"]["p50"], 5.0)
        self.assertAlmostEqual(r["orig_px"]["p50"], 2.5)

    def test_only_visible_corners_count(self):
        g = gt(100, 100, 3, vis=[2, 2, 0, 2])
        d = det(100, 100, 3)
        d["pts"] = d["pts"].copy()
        d["pts"][2] += 50.0  # wrong, but that corner is outside the crop
        r = evaluate([([d], [g], META)], 0.2, 16)["corner_error"]
        self.assertEqual((r["n_corners"], r["net_px"]["max"]), (3, 0.0))

    def test_bins_follow_metadata(self):
        g = gt(100, 100, 3, dist_m=5.0, yaw_deg=-30.0, occ=0.0)
        r = evaluate([([det(100, 100, 3, shift=(1, 0))], [g], META)], 0.2, 16)["corner_error_by"]
        self.assertEqual(list(r["dist_m"]), ["[4,6)"])
        self.assertEqual(list(r["abs_yaw_deg"]), ["[20,40)"])
        self.assertEqual(list(r["occ"]), ["[0,0.05)"])

    def test_low_confidence_is_ignored_at_threshold_but_counts_for_ap(self):
        r = evaluate([([det(100, 100, 3, conf=0.1)], [gt(100, 100, 3)], META)], 0.2, 16)
        self.assertEqual(r["at_conf"]["recall"], 0.0)
        self.assertEqual(r["armor_ap50"], 1.0)

    def test_greedy_matching_uses_each_gt_once(self):
        m = match([det(100, 100, 3, 0.9), det(101, 100, 3, 0.8)], [gt(100, 100, 3)], by_class=True)
        self.assertEqual(sorted(g is None for _, g in m), [False, True])

    def test_average_precision_basics(self):
        self.assertEqual(average_precision(np.array([1.0, 1.0]), 2), 1.0)
        self.assertAlmostEqual(average_precision(np.array([0.0, 1.0]), 1), 0.5, delta=0.01)
        self.assertTrue(np.isnan(average_precision(np.array([]), 0)))
        self.assertEqual(average_precision(np.array([]), 3), 0.0)
        self.assertEqual(iou(box_of(quad(0, 0)), box_of(quad(0, 0))), 1.0)


class GroundTruthAsPredictionTest(unittest.TestCase):
    """Feeding labels back as predictions must score perfectly; a regression guard for the whole path."""

    def scene(self):
        rng = np.random.default_rng(0)
        images = []
        for i in range(30):
            gts = [gt(100 + 90 * k, 80 + 40 * k, (3 * i + k) % 16, dist_m=1 + (i + k) % 10, yaw_deg=float((i * 7 + k * 13) % 80),
                      occ=((i + k) % 5) / 10, vis=[2, 2, 2, 2]) for k in range(1 + i % 3)]
            images.append((gts, {"scale": 1.0 + (i % 4)}))
        return rng, images

    def test_perfect_noisy_and_empty(self):
        rng, images = self.scene()
        def preds(noise):
            out = []
            for gts, _ in images:
                dets = []
                for g in gts:
                    pts = g["corners_net"] + rng.normal(0, noise, (4, 2))
                    dets.append({"conf": 0.9, "cls": g["cls"], "pts": pts, "box": box_of(pts)})
                out.append(dets)
            return out
        perfect = evaluate([(d, g, m) for d, (g, m) in zip(preds(0.0), images)], 0.2, 16)
        self.assertEqual((perfect["armor_ap50"], perfect["at_conf"]["recall"], perfect["at_conf"]["precision"]), (1.0, 1.0, 1.0))
        self.assertEqual(perfect["corner_error"]["net_px"]["max"], 0.0)
        noisy = evaluate([(d, g, m) for d, (g, m) in zip(preds(1.0), images)], 0.2, 16)
        self.assertTrue(0.9 < noisy["corner_error"]["net_px"]["p50"] / 1.177 < 1.15)  # Rayleigh median of sigma = 1
        self.assertGreater(noisy["corner_error"]["net_px"]["p95"], noisy["corner_error"]["net_px"]["p50"])
        none = evaluate([([], g, m) for g, m in images], 0.2, 16)
        self.assertEqual((none["armor_ap50"], none["at_conf"]["recall"]), (0.0, 0.0))


class ColorAndLabelAccuracyTest(unittest.TestCase):
    def test_uses_the_spec_label_count(self):
        # class 3 is blue four, class 9 is red four: same number, other color
        r = evaluate([([det(100, 100, 3)], [gt(100, 100, 9)], META)], 0.2, SPEC.n_classes, SPEC.from_class_id)["at_conf"]
        self.assertEqual((r["color_acc"], r["label_acc"], r["class_acc"]), (0.0, 1.0, 0.0))



class BreakdownTest(unittest.TestCase):
    """The analyses that used to live in throwaway scripts: confusion, size bins, sweep, false-positive kinds."""

    def scene(self):
        sentry, one, off = SPEC.class_id(0, 0), SPEC.class_id(0, 1), SPEC.n_classes - 1
        gts = [gt(100, 100, sentry), gt(300, 100, one), gt(500, 100, off)]
        small = {"cls": one, "corners_net": quad(100, 300, w=6.0, h=3.0), "box": box_of(quad(100, 300, w=6.0, h=3.0))}
        dets = [det(100, 100, sentry, 0.9),         # right
                det(300, 100, sentry, 0.6),         # right place, wrong number
                det(302, 100, one, 0.4),            # a second candidate on the same armor: a false positive on an armor
                det(700, 300, one, 0.8)]            # nothing there: a false positive on background
        return [(dets, gts + [small], META)], sentry, one, off

    def test_confusion_size_bins_and_false_positive_kinds(self):
        per_image, sentry, one, off = self.scene()
        r = evaluate(per_image, 0.2, SPEC.n_classes, SPEC.from_class_id)
        self.assertEqual(r["confusion"], {sentry: {sentry: 1}, one: {sentry: 1}})
        self.assertEqual((r["at_conf"]["false_positives_on_armor"], r["at_conf"]["false_positives_on_background"]), (1, 1))
        self.assertEqual(r["by_width_px"]["[0,8)"], {"n": 1, "hits": 0, "correct": 0, "recall": 0.0, "class_acc": None})
        self.assertEqual(r["by_width_px"]["[32,64)"]["n"], 3)
        self.assertEqual(r["by_class"][off]["recall"], 0.0)
        sweep = {s["conf"]: s for s in r["at_conf_sweep"]}
        self.assertEqual(sorted(sweep), [0.3, 0.5])
        self.assertEqual(sweep[0.5]["false_positives"], 1)  # the 0.4 duplicate is gone at 0.5
        s = summarize(r, SPEC)
        self.assertEqual(s["robot_numbers"], {"n": 3, "recall": 0.6667, "class_acc": 0.5})
        self.assertEqual(s["off_class"], {"n": 1, "recall": 0.0, "class_acc": None})
        self.assertEqual(s["confusion_named"][SPEC.class_name(one)], {SPEC.class_name(sentry): 1})


class ContractDecodeTest(unittest.TestCase):
    def row(self, conf, number, slot, cx=100.0):
        r = np.zeros(18, np.float32)
        r[4], r[5], r[6 + slot] = conf, number, 1.0
        r[10:18] = quad(cx, 100).ravel()
        return r

    def test_rows_map_back_to_merged_classes(self):
        off_color, off_number = SPEC.off_class[:2]
        rows = np.stack([self.row(0.9, 4, 1), self.row(0.8, off_number, off_color, 300), self.row(0.7, 2, 0, 500),
                         self.row(0.1, 1, 0, 700)])
        dets = decode_contract(rows, SPEC, 0.2)
        # red four, the off class; number 2 does not exist in RM27 and is dropped; 0.1 is under the threshold
        self.assertEqual([d["cls"] for d in dets], [SPEC.class_id(1, 4), SPEC.n_classes - 1])
        np.testing.assert_allclose(dets[0]["pts"], quad(100, 100))

    def test_paired_difference_measures_engine_drift(self):
        a = [[det(100, 100, 3, 0.9), det(300, 100, 4, 0.5)]]
        b = [[det(100, 100, 3, 0.88, shift=(0.3, 0.4)), det(300, 100, 5, 0.5)]]  # second one changed class
        v = paired_difference(a, b, [{"scale": 2.0}], 0.2)
        self.assertEqual((v["matched"], v["only_in_onnx"], v["only_in_engine"]), (1, 1, 1))
        self.assertAlmostEqual(v["corner_diff"]["net_px"]["max"], 0.5)
        self.assertAlmostEqual(v["corner_diff"]["orig_px"]["max"], 0.25)
        self.assertAlmostEqual(v["conf_diff"]["max"], 0.02)


if __name__ == "__main__":
    unittest.main()
