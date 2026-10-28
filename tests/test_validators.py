import unittest

import numpy as np

from nt.canonical import Armor, Sample
from nt.spec import LabelSpec
from nt.validators import NearDuplicateIndex, armor_issues, dhash64, validate_sample
from test_geometry import rect

SPEC = LabelSpec.load()
W, H = 640, 480


def armor(pts=None, **kw):
    pts = pts if pts is not None else rect(320, 240, 60, 25, 5)
    return Armor(**{"color": 1, "label": 3, "corners": [list(p) for p in pts], **kw})


def codes(a, w=W, h=H):
    return {c for _, c, _ in armor_issues(a, w, h, SPEC)}


def sample(armors, image="images/s.png", w=W, h=H):
    return Sample(id="s", image=image, w=w, h=h, source="sim", armors=armors)


class ArmorValidationTest(unittest.TestCase):
    def test_clean_armor(self):
        self.assertEqual(codes(armor()), set())

    def test_reversed_winding(self):
        self.assertIn("E_QUAD_ORDER", codes(armor(rect(320, 240, 60, 25, 5)[::-1])))

    def test_start_corner_shifted_by_one(self):
        base = rect(320, 240, 60, 25, 5)
        self.assertIn("E_QUAD_ORDER", codes(armor(base[1:] + base[:1])))

    def test_bowtie(self):
        base = rect(320, 240, 60, 25)
        self.assertIn("E_QUAD_NONCONVEX", codes(armor([base[0], base[2], base[1], base[3]])))

    def test_corner_outside_only_matters_when_visible(self):
        pts = rect(2, 240, 60, 25)  # left half sticks out of the image
        self.assertIn("E_CORNER_OUTSIDE", codes(armor(pts)))
        self.assertNotIn("E_CORNER_OUTSIDE", codes(armor(pts, vis=[0, 0, 2, 2])))

    def test_illegal_class_and_untrained_color(self):
        self.assertIn("E_CLASS_ILLEGAL", codes(armor(color=9)))
        self.assertIn("E_CLASS_ILLEGAL", codes(armor(label=8)))
        self.assertIn("E_CLASS_ILLEGAL", codes(armor(type="huge")))
        self.assertEqual(codes(armor(color=2)), {"W_COLOR_NOT_TRAINED"})

    def test_degenerate_and_nonfinite(self):
        self.assertIn("E_QUAD_DEGENERATE", codes(armor(rect(320, 240, 2, 1))))
        bad = rect(320, 240, 60, 25)
        bad[1] = (float("nan"), 3.0)
        self.assertIn("E_CORNER_NONFINITE", codes(armor(bad)))

    def test_bad_structure_and_ranges(self):
        self.assertIn("E_CORNER_COUNT", codes(armor(rect(320, 240, 60, 25)[:3])))
        self.assertIn("E_VIS_INVALID", codes(armor(vis=[2, 2, 2, 3])))
        self.assertIn("E_FIELD_RANGE", codes(armor(occ=1.5)))

    def test_warnings(self):
        self.assertIn("W_QUAD_ROLL", codes(armor(rect(320, 240, 60, 25, 50))))
        self.assertIn("W_SMALL_ARMOR", codes(armor(rect(320, 240, 7, 4))))
        self.assertIn("W_ASPECT", codes(armor(rect(320, 240, 120, 5))))


class SampleValidationTest(unittest.TestCase):
    def levels(self, s, hw=None):
        return {(i.level, i.code) for i in validate_sample(s, SPEC, hw)}

    def test_duplicate_label_is_an_error(self):
        self.assertIn(("error", "E_ARMOR_DUP"), self.levels(sample([armor(), armor()])))

    def test_overlap_with_other_class_is_a_warning(self):
        self.assertEqual(self.levels(sample([armor(), armor(label=2)])), {("warn", "W_LABEL_CONFLICT")})

    def test_size_mismatch(self):
        self.assertIn(("error", "E_SIZE_MISMATCH"), self.levels(sample([armor()]), hw=(480, 641)))

    def test_jpeg_warns_and_negative_frame_is_fine(self):
        self.assertEqual(self.levels(sample([], image="images/s.jpg")), {("warn", "W_IMAGE_LOSSY")})

    def test_bad_sample_fields(self):
        s = sample([armor()], w=0)
        self.assertIn(("error", "E_SIZE_INVALID"), self.levels(s))


class DuplicateIndexTest(unittest.TestCase):
    def test_identical_frames_match_and_different_ones_do_not(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        index = NearDuplicateIndex(max_hamming=2)
        index.add("a", dhash64(a))
        self.assertEqual(index.query(dhash64(a.copy())), "a")
        self.assertIsNone(index.query(dhash64(b)))

    def test_index_grows_past_initial_capacity(self):
        index = NearDuplicateIndex(max_hamming=0)
        for i in range(1100):
            index.add(f"id{i}", np.uint64(i * 2654435761 % (1 << 63)))
        self.assertEqual(index.query(np.uint64(1099 * 2654435761 % (1 << 63))), "id1099")


if __name__ == "__main__":
    unittest.main()
