"""Class table of label_spec: training colors x labels with gaps, then one off class for dark screens."""
import unittest

from nt.spec import LabelSpec

SPEC = LabelSpec.load()


class ClassTableTest(unittest.TestCase):
    def test_classes_are_contiguous_with_the_off_class_last(self):
        self.assertEqual(SPEC.label_ids, (0, 1, 3, 4, 6, 7))
        self.assertEqual(SPEC.n_classes, 13)
        self.assertEqual(SPEC.class_name(12), "gray_unknown")
        for cid in range(SPEC.n_classes):
            color, number = SPEC.from_class_id(cid)
            label = number if not SPEC.is_off(color) else 3  # any label folds into the off class
            self.assertEqual(SPEC.class_id(color, label), cid)

    def test_any_label_on_a_dark_screen_is_the_off_class(self):
        self.assertEqual({SPEC.class_id(2, l) for l in (*SPEC.label_ids, 8)}, {12})
        self.assertTrue(SPEC.valid_pair(2, 8) and SPEC.valid_pair(2, 3))
        self.assertFalse(SPEC.valid_pair(1, 8) or SPEC.valid_pair(1, 2) or SPEC.trains(3))
        with self.assertRaises(ValueError):
            SPEC.class_id(3, 1)  # purple trains nothing


if __name__ == "__main__":
    unittest.main()
