import math
import unittest

from nt.geometry import (
    aspect_ratio,
    bbox_iou,
    canonical_permutation,
    is_convex,
    signed_area,
    top_edge_angle_deg,
)


def rect(cx, cy, w, h, roll_deg=0.0):
    """Corners in spec order LT, LB, RB, RT (image space, y down), rotated clockwise by roll_deg."""
    t = math.radians(roll_deg)
    c, s = math.cos(t), math.sin(t)
    base = [(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)]
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in base]


class GeometryTest(unittest.TestCase):
    def test_spec_order_properties(self):
        pts = rect(100, 50, 60, 20)
        self.assertLess(signed_area(pts), 0)
        self.assertTrue(is_convex(pts))
        self.assertAlmostEqual(top_edge_angle_deg(pts), 0.0)
        self.assertAlmostEqual(aspect_ratio(pts), 3.0)

    def test_roll_sign(self):
        self.assertAlmostEqual(top_edge_angle_deg(rect(0, 0, 60, 20, 15)), 15.0)

    def test_bowtie_is_not_convex(self):
        self.assertFalse(is_convex([(0, 0), (10, 10), (0, 10), (10, 0)]))

    def test_canonical_permutation_recovers_spec_order(self):
        for roll in (-30, 0, 25):
            base = rect(300, 200, 60, 25, roll)
            for k in range(4):
                for reverse in (False, True):
                    cand = base[k:] + base[:k]
                    if reverse:
                        cand = cand[::-1]
                    perm, changed = canonical_permutation(cand)
                    self.assertEqual([cand[i] for i in perm], base, f"roll={roll} k={k} reverse={reverse}")
                    self.assertEqual(changed, cand != base)

    def test_nonconvex_is_left_alone(self):
        perm, changed = canonical_permutation([(0, 0), (10, 10), (0, 10), (10, 0)])
        self.assertEqual((perm, changed), ([0, 1, 2, 3], False))

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


if __name__ == "__main__":
    unittest.main()
