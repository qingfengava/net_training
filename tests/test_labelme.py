"""labelme conventions: corner order, visibility markers, unresolved pre-labels, and the config file."""
import unittest

from nt import labelme_io as lm
from nt.spec import LabelSpec
from test_geometry import rect

SPEC = LabelSpec.load()


def doc_with(shapes, reviewed=True):
    return lm.document("f.png", 300, 200, shapes, flags={"reviewed": reviewed})


class FromDocumentTest(unittest.TestCase):
    def test_polygon_keeps_its_order_and_label_and_is_small(self):
        pts = rect(150, 100, 60, 25, 4)
        shapes = lm.armor_shapes(SPEC, 1, 3, pts)
        armors, problems, flags, _ = lm.from_document(doc_with(shapes), SPEC)
        self.assertEqual(problems, [])
        self.assertTrue(flags["reviewed"])
        (a,) = armors
        self.assertEqual((a.color, a.label, a.type, a.vis), (1, 3, "small", [2, 2, 2, 2]))
        for got, want in zip(a.corners, pts):
            self.assertAlmostEqual(got[0], want[0], places=2)
            self.assertAlmostEqual(got[1], want[1], places=2)

    def test_markers_set_visibility_and_outside_corners_are_clamped(self):
        pts = [(-10.0, 50.0), (-10.0, 80.0), (60.0, 80.0), (60.0, 50.0)]  # left edge outside the image
        shapes = lm.armor_shapes(SPEC, 0, 1, pts, vis=[2, 2, 1, 2], w=300, h=200, group_id=1)
        armors, problems, _, _ = lm.from_document(doc_with(shapes), SPEC)
        self.assertEqual(problems, [])
        self.assertEqual(armors[0].vis, [0, 0, 1, 2])
        self.assertEqual(armors[0].corners[0], [0.0, 50.0])

    def test_hand_placed_marker_must_sit_on_a_corner(self):
        pts = rect(150, 100, 60, 25, 0)
        shapes = lm.armor_shapes(SPEC, 0, 1, pts) + [lm.point((10, 10), "occluded")]
        _, problems, _, _ = lm.from_document(doc_with(shapes), SPEC)
        self.assertEqual([c for c, _ in problems], ["E_LM_MARKER"])
        near = lm.armor_shapes(SPEC, 0, 1, pts) + [lm.point((pts[2][0] + 1.5, pts[2][1] - 1), "occluded")]
        armors, problems, _, _ = lm.from_document(doc_with(near), SPEC)
        self.assertEqual(problems, [])
        self.assertEqual(armors[0].vis, [2, 2, 1, 2])

    def test_deleting_a_prelabel_drops_its_markers(self):
        pts = [(-10.0, 50.0), (-10.0, 80.0), (60.0, 80.0), (60.0, 50.0)]
        shapes = lm.armor_shapes(SPEC, 0, 1, pts, w=300, h=200, group_id=7)
        polygon_removed = [s for s in shapes if s["shape_type"] != "polygon"]
        armors, problems, _, notes = lm.from_document(doc_with(polygon_removed), SPEC)
        self.assertEqual((armors, problems), ([], []))
        self.assertTrue(notes)

    def test_unresolved_unknown_and_malformed_shapes_are_errors(self):
        pts = rect(150, 100, 60, 25, 0)
        shapes = (lm.armor_shapes(SPEC, 0, 1, pts, unresolved=True)
                  + [lm.polygon(pts, "green_three")]
                  + [lm.polygon(pts[:3], "red_three")]
                  + [{**lm.polygon(pts, "red_three"), "shape_type": "rectangle"}])
        _, problems, _, _ = lm.from_document(doc_with(shapes), SPEC)
        self.assertEqual(sorted(c for c, _ in problems), ["E_LM_LABEL", "E_LM_POINTS", "E_LM_SHAPE", "E_LM_UNRESOLVED"])

    def test_dark_screen_with_an_unseen_number(self):
        pts = rect(150, 100, 60, 25, 0)
        armors, problems, _, _ = lm.from_document(doc_with([lm.polygon(pts, "gray_unknown")]), SPEC)
        self.assertEqual(problems, [])
        self.assertEqual((armors[0].color, armors[0].label), SPEC.off_class[:2])
        _, problems, _, _ = lm.from_document(doc_with([lm.polygon(pts, "red_unknown")]), SPEC)
        self.assertEqual([c for c, _ in problems], ["E_LM_LABEL"])

    def test_labelme_config_lists_every_label_once(self):
        cfg = lm.labelme_config(SPEC)
        self.assertEqual(len(cfg["labels"]), len(set(cfg["labels"])))
        self.assertIn("red_three", cfg["labels"])
        self.assertIn("occluded", cfg["labels"])
        self.assertEqual(cfg["validate_label"], "exact")
        self.assertEqual(list(cfg["flags"]), ["reviewed", "bad_frame"])
        self.assertEqual(cfg["label_flags"], {})  # no size flag: RM27 has small armor only


if __name__ == "__main__":
    unittest.main()
