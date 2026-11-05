"""验证增强的像素对齐、物理采样行为、裁剪标注及禁用路径。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt import bake
from nt.bake_augment import apply_moire, centered_motion_kernel, sampling_pattern, tracker_roi
from nt.canonical import Armor, Sample, sample_to_dict
from nt.spec import LabelSpec
from test_bake import NO_EFFECTS


def plate(corners=None, **kwargs):
    return Armor(1, 3, corners or [[20, 20], [20, 100], [140, 100], [140, 20]], screen_state="on", **kwargs)


def sample(armors=None, **kwargs):
    return Sample("example", "input.png", 480, 360, "sim", armors=armors or [], **kwargs)


def configuration(override=None):
    return bake.load_config(override=override or {})


class MotionKernelTest(unittest.TestCase):
    def test_impulse_stays_at_same_location_for_even_and_odd_lengths(self):
        image = np.zeros((41, 41), np.float32); image[20, 20] = 1
        yy, xx = np.mgrid[:41, :41]
        for length in range(1, 13):
            for angle in np.linspace(0, 180, 73):
                k = centered_motion_kernel(length, angle)
                self.assertEqual(k.shape[0] % 2, 1)
                self.assertTrue(np.all(k >= 0))
                self.assertAlmostEqual(float(k.sum()), 1, places=6)
                out = cv2.filter2D(image, -1, k)
                mass = out.astype(np.float64).sum()
                self.assertAlmostEqual(float((out * xx).sum() / mass), 20, places=6)
                self.assertAlmostEqual(float((out * yy).sum() / mass), 20, places=6)

    def test_direction_and_extent_are_not_replaced_by_noop_or_isotropic_blur(self):
        yy, xx = np.mgrid[:41, :41]
        image = np.zeros((41, 41), np.float32); image[20, 20] = 1
        for angle in (0, 30, 90, 135):
            out = cv2.filter2D(image, -1, centered_motion_kernel(9, angle))
            cov = np.array([[(out * (xx - 20) ** 2).sum(), (out * (xx - 20) * (yy - 20)).sum()],
                            [(out * (xx - 20) * (yy - 20)).sum(), (out * (yy - 20) ** 2).sum()]])
            values, vectors = np.linalg.eigh(cov)
            self.assertGreater(values[-1], 3)
            self.assertGreater(values[-1], 8 * values[0])
            expected = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
            self.assertGreater(abs(np.dot(vectors[:, -1], expected)), 0.999)

    def test_complete_effect_path_preserves_center_and_legacy_is_explicit(self):
        cfg = configuration(NO_EFFECTS)
        cfg["effects"]["motion_blur"].update(prob=1, length_px=[4, 4])
        image = np.zeros((41, 41, 3), np.uint8); image[20, 20] = 230
        yy, xx = np.mgrid[:41, :41]
        for seed in range(30):
            out, used = bake.apply_effects(image, np.random.default_rng(seed), cfg)
            weight = out[..., 0].astype(float)
            self.assertAlmostEqual((weight * xx).sum() / weight.sum(), 20, places=6)
            self.assertAlmostEqual((weight * yy).sum() / weight.sum(), 20, places=6)
            self.assertEqual(used["motion_blur_kernel"], "centered_v1")
        cfg["effects"]["motion_blur"].pop("kernel")
        old, used = bake.apply_effects(image, np.random.default_rng(1), cfg)
        cfg["effects"]["motion_blur"]["kernel"] = "legacy_v2"
        explicit, _ = bake.apply_effects(image, np.random.default_rng(1), cfg)
        np.testing.assert_array_equal(old, explicit)
        self.assertNotIn("motion_blur_kernel", used)


class TrackerRoiTest(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration({"roi": {"tracker": {"prob": 1}}})
        self.s = sample([plate([[160, 150], [160, 190], [240, 190], [240, 150]])])
        self.boxes = [(160, 150, 240, 190)]

    def test_states_geometry_and_reproducibility(self):
        states = set()
        for seed in range(200):
            value = tracker_roi(np.random.default_rng(seed), self.s, self.boxes, self.cfg, "train")
            self.assertEqual(value, tracker_roi(np.random.default_rng(seed), self.s, self.boxes, self.cfg, "train"))
            (x, y, w, h), _, meta = value
            states.add(meta["state"])
            self.assertEqual(w * 3, h * 4)
            self.assertTrue(0 <= x < x + w <= 480 and 0 <= y < y + h <= 360)
        self.assertEqual(states, {"search", "tracking", "recovering", "patrol"})

    def test_recorded_prediction_expands_with_age_and_expires(self):
        widths = []
        for age in (0, 200, 800, 1500):
            self.s.scene["roi_prediction"] = {"bbox": [290, 160, 310, 180], "image_size": [480, 360],
                                                "state": "recovering", "age_ms": age}
            roi, how, meta = tracker_roi(np.random.default_rng(2), self.s, self.boxes, self.cfg, "train")
            widths.append(roi[2])
            self.assertEqual(meta["source"], "recorded_prediction")
        self.assertEqual(widths, sorted(widths))
        self.assertEqual((roi, how), ((0, 0, 480, 360), "tracker_full"))

    def test_disabled_real_and_validation_do_not_consume_random_stream(self):
        for source, split, prob in (("real", "train", 1), ("sim", "val", 1), ("sim", "train", 0)):
            self.s.source = source; self.cfg["roi"]["tracker"]["prob"] = prob
            rng = np.random.default_rng(7); before = copy.deepcopy(rng.bit_generator.state)
            self.assertIsNone(tracker_roi(rng, self.s, self.boxes, self.cfg, split))
            self.assertEqual(before, rng.bit_generator.state)

    def test_recorded_prediction_is_not_recentred_using_truth(self):
        self.s.scene["roi_prediction"] = {"bbox": [390, 270, 410, 290], "image_size": [480, 360],
                                            "state": "tracking", "age_ms": 0}
        roi, _, _ = tracker_roi(np.random.default_rng(2), self.s, self.boxes, self.cfg, "train")
        self.assertEqual(bake._inside_fraction(self.boxes[0], roi), 0)
        with self.assertRaises(ValueError):
            tracker_roi(np.random.default_rng(2), self.s, self.boxes, self.cfg, "train", lens_applied=True)
        self.s.scene["roi_prediction"]["image_size"] = [1920, 1080]
        with self.assertRaises(ValueError):
            tracker_roi(np.random.default_rng(2), self.s, self.boxes, self.cfg, "train")

    def test_partial_armor_retains_label_and_outside_corners_are_invisible(self):
        self.s.scene["roi_prediction"] = {"bbox": [290, 160, 310, 180], "image_size": [480, 360],
                                            "state": "tracking", "age_ms": 0}
        cfg = configuration({**NO_EFFECTS, "roi": {"tracker": {"prob": 1}}})
        spec = LabelSpec.load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((360, 480, 3), np.uint8)
            image[150:190, 160:240] = 180
            cv2.imwrite(str(root / "input.png"), image)
            for kind in ("images", "labels"):
                (root / kind / "train").mkdir(parents=True)
            bake._WORKER.update(spec=spec, cfg=cfg)
            task = {"sample": sample_to_dict(self.s), "root": str(root), "out": str(root), "copy": 0,
                    "seed": 0, "split": "train", "source": "example"}
            result = bake.bake_one(task)
            self.assertLess(bake._inside_fraction(self.boxes[0], result["roi"]), 0.5)
            self.assertEqual(len(result["armors"]), 1)
            self.assertEqual(result["armors"][0]["vis"], [0, 0, 2, 2])
            fields = (root / "labels/train/example_c0.txt").read_text().split()
            self.assertEqual(len(fields), 17)
            self.assertTrue(all(0 <= float(fields[i]) <= 1 for i in (1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15)))
            for (ox, oy), (nx, ny) in zip(self.s.armors[0].corners, result["armors"][0]["corners_net"]):
                self.assertAlmostEqual((ox - result["roi"][0]) * result["scale"], nx, places=3)
                self.assertAlmostEqual((oy - result["roi"][1]) * result["scale"], ny, places=3)


class MoireTest(unittest.TestCase):
    def setUp(self):
        self.options = configuration({"effects": {"moire": {"prob": 1, "grid_source": "config"}}})["effects"]["moire"]

    def test_sampling_aliases_to_expected_beat_and_optics_attenuate_it(self):
        quad = [[0, 0], [0, 64], [128, 64], [128, 0]]
        def profile(sigma):
            pattern, _ = sampling_pattern(quad, (64, 128), (120, 4, "square"), .5, [.17, .31], sigma, .5, 1)
            row = pattern.mean(axis=0); row -= row.mean()
            return np.abs(np.fft.rfft(row))
        sharp, soft = profile(.08), profile(2)
        self.assertEqual(int(np.argmax(sharp[1:]) + 1), 8)
        self.assertGreater(sharp[8], 1)
        self.assertLess(soft[8], sharp[8] * 1e-4)

    def test_whole_period_pixel_aperture_removes_grid_and_phase_changes_pattern(self):
        quad = [[0, 0], [0, 64], [128, 64], [128, 0]]
        pattern, _ = sampling_pattern(quad, (64, 128), (128, 64, "square"), .5, [.2, .3], 0, 1, 3)
        np.testing.assert_allclose(pattern, 1, atol=1e-12)
        a, _ = sampling_pattern(quad, (64, 128), (120, 48, "square"), .5, [.2, .3], .15, .6, 3)
        b, _ = sampling_pattern(quad, (64, 128), (120, 48, "square"), .5, [.7, .8], .15, .6, 3)
        self.assertGreater(np.abs(a - b).mean(), .05)

    def test_locality_off_and_occluded_screens_and_repeatability(self):
        armor = plate(); s = sample([armor]); image = np.full((120, 160, 3), (30, 60, 180), np.uint8)
        before = sample_to_dict(s)
        out, report = apply_moire(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        again, report2 = apply_moire(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        np.testing.assert_array_equal(out, again); self.assertEqual(report, report2)
        self.assertEqual(before, sample_to_dict(s))
        np.testing.assert_array_equal(out[:20], image[:20])
        np.testing.assert_array_equal(out[:, :20], image[:, :20])
        self.assertGreater(np.count_nonzero(out != image), 0)
        self.assertTrue(np.all(out[..., 2] > out[..., 1]))
        for changed in (plate(occ=.1), plate(vis=[2, 1, 2, 2])):
            same, meta = apply_moire(image, np.random.default_rng(3), s, [changed], (0, 0, 160, 120), self.options, "train")
            np.testing.assert_array_equal(same, image); self.assertEqual(meta["applied"], [])
        armor.color = 2; armor.screen_state = "off"
        same, meta = apply_moire(image, np.random.default_rng(3), s, [armor], (0, 0, 160, 120), self.options, "train")
        np.testing.assert_array_equal(same, image); self.assertEqual(meta["applied"], [])

    def test_scene_grid_and_unknown_grid_do_not_silently_use_b_layout(self):
        s = sample([plate()]); self.options["grid_source"] = "scene"
        image = np.full((120, 160, 3), 120, np.uint8)
        same, meta = apply_moire(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        np.testing.assert_array_equal(same, image)
        self.assertEqual(meta["reason"], "missing_or_ambiguous_grid")
        s.scene["screen"] = {"robots": [{"look": {"cols": 34, "rows": 28, "dot": "circle"}}]}
        out, meta = apply_moire(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        self.assertEqual(meta["grid"], [34, 28, "circle"])
        self.assertGreater(np.count_nonzero(out != image), 0)

    def test_projected_grid_is_consistent_between_full_image_and_native_crop(self):
        quad = np.array([[31.25, 21.5], [25.5, 108.25], [141.75, 98.5], [130.25, 30.75]])
        args = ((64, 48, "square"), .6, [.2, .4], .2, .8, 3)
        whole, mask = sampling_pattern(quad, (128, 160), *args)
        crop, crop_mask = sampling_pattern(quad - [20, 16], (96, 128), *args)
        np.testing.assert_allclose(whole[16:112, 20:148], crop, atol=1e-10)
        np.testing.assert_allclose(mask[16:112, 20:148], crop_mask, atol=1e-10)

    def test_off_real_and_validation_paths_are_exact_noops(self):
        image = np.full((120, 160, 3), 120, np.uint8); s = sample([plate()])
        for source, split, prob in (("real", "train", 1), ("sim", "val", 1), ("sim", "train", 0)):
            s.source = source; self.options["prob"] = prob
            rng = np.random.default_rng(2); state = copy.deepcopy(rng.bit_generator.state)
            out, meta = apply_moire(image, rng, s, s.armors, (0, 0, 160, 120), self.options, split)
            np.testing.assert_array_equal(out, image); self.assertIsNone(meta)
            self.assertEqual(state, rng.bit_generator.state)


class ConfigTest(unittest.TestCase):
    def test_invalid_augmentation_settings_fail_early(self):
        for override in ({"effects": {"motion_blur": {"kernel": "wrong"}}},
                         {"roi": {"tracker": {"search_prob": 1, "patrol_prob": 1}}},
                         {"roi": {"tracker": {"lost_time_ms": 0}}},
                         {"effects": {"moire": {"pixel_aperture": [0, 1]}}},
                         {"effects": {"moire": {"strength": [0, 2]}}},
                         {"effects": {"moire": {"harmonics": 100}}}):
            with self.assertRaises(ValueError):
                configuration(override)


if __name__ == "__main__":
    unittest.main()
