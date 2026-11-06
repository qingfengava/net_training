"""验证增强的像素对齐、物理采样行为、裁剪标注及禁用路径。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt import bake
from nt.bake_augment import (_pwm_integral, apply_bayer, apply_flicker, apply_moire, bayer_demosaic, bayer_mosaic,
                             centered_motion_kernel, row_exposure_gain, sampling_pattern, tracker_roi)
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


class FlickerTest(unittest.TestCase):
    def setUp(self):
        self.options = configuration({"effects": {"flicker": {"prob": 1, "grid_source": "config", "rows": 48}}})["effects"]["flicker"]

    def test_whole_number_of_refresh_periods_in_exposure_shows_no_flicker(self):
        for hz, exposure_s in ((500.0, 0.004), (1000.0, 0.002), (240.0, 1 / 240), (2000.0, 0.0015)):
            for start in (0.0, 0.000123, 0.0007):
                gain = row_exposure_gain(48, hz, 0.4, exposure_s, start, 0.3, 1.0)
                np.testing.assert_allclose(gain, 1.0, atol=1e-12)

    def test_short_exposure_gives_rows_different_brightness_with_unit_mean(self):
        gain = row_exposure_gain(48, 240.0, 0.5, 0.0015, 0.0, 0.0, 1.0)  # 0.36 个周期
        self.assertGreater(gain.max() - gain.min(), 1.0)
        self.assertLessEqual(gain.max(), 1 / 0.5 + 1e-9)
        self.assertGreaterEqual(gain.min(), 0)
        starts = np.random.default_rng(0).random(4000) / 240
        mean = np.mean([row_exposure_gain(48, 240.0, 0.5, 0.0015, s, 0.0, 1.0).mean() for s in starts])
        self.assertAlmostEqual(mean, 1.0, delta=0.02)

    def test_synchronized_scan_makes_all_rows_equal_and_direction_reverses_the_stripes(self):
        flat = row_exposure_gain(48, 240.0, 0.5, 0.0015, 0.001, 0.2, 0.0)
        np.testing.assert_allclose(flat, flat[0])
        down = row_exposure_gain(48, 240.0, 0.5, 0.0015, 0.001, 0.2, 0.7, True)
        up = row_exposure_gain(48, 240.0, 0.5, 0.0015, 0.001, 0.2, 0.7, False)
        np.testing.assert_allclose(down, up[::-1])
        self.assertGreater(np.abs(down - up).max(), 0.1)

    def test_pwm_integral_matches_numeric_integration(self):
        t = np.linspace(0, 3.7, 20001)
        wave = ((t % 1) < 0.35).astype(float)
        numeric = np.concatenate([[0], np.cumsum((wave[1:] + wave[:-1]) / 2 * np.diff(t))])
        self.assertLess(np.abs(numeric - _pwm_integral(t, 0.35)).max(), 2e-4)

    def test_stripes_follow_screen_rows_and_only_lit_visible_screens_change(self):
        armor = plate([[20, 20], [20, 100], [140, 100], [140, 20]])
        s = sample([armor]); image = np.full((120, 160, 3), (40, 80, 200), np.uint8)
        self.options.update(refresh_hz=[240, 240], exposure_ms=[1.5, 1.5], duty=[0.5, 0.5], scan_span=[1, 1], scan_direction="down")
        out, report = apply_flicker(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        self.assertEqual(report["model"], "global_shutter_pwm_v1")
        np.testing.assert_array_equal(out[:15], image[:15]); np.testing.assert_array_equal(out[:, :15], image[:, :15])
        column = out[25:95, 80, 2].astype(float)
        self.assertGreater(column.max() - column.min(), 30)  # 条纹沿屏的行方向变化
        row = out[60, 30:130, 2].astype(float)
        self.assertLess(row.max() - row.min(), 6)             # 同一行内亮度一致（行间才不同）
        self.assertTrue(np.all(out[25:95, 30:130, 2] >= out[25:95, 30:130, 1]))  # 不改颜色顺序
        for changed in (plate(occ=.1), plate(vis=[2, 1, 2, 2])):
            same, meta = apply_flicker(image, np.random.default_rng(3), s, [changed], (0, 0, 160, 120), self.options, "train")
            np.testing.assert_array_equal(same, image); self.assertEqual(meta["applied"], [])
        for state in ("off", "flicker"):
            other = plate(); other.screen_state = state
            same, meta = apply_flicker(image, np.random.default_rng(3), s, [other], (0, 0, 160, 120), self.options, "train")
            np.testing.assert_array_equal(same, image); self.assertEqual(meta["applied"], [])
        again, report2 = apply_flicker(image, np.random.default_rng(3), s, s.armors, (0, 0, 160, 120), self.options, "train")
        np.testing.assert_array_equal(out, again); self.assertEqual(report, report2)

    def test_rendered_brightness_ratio_equals_the_recorded_row_gain_and_clips_at_full_scale(self):
        # 用未饱和的暗红屏逐行核对：渲染的线性光比值就是模型的行增益；再用接近满刻度的屏幕核对限幅。
        armor = plate([[20, 10], [20, 110], [140, 110], [140, 10]])
        s = sample([armor]); rows = 40
        self.options.update(refresh_hz=[300, 300], exposure_ms=[1.2, 1.2], duty=[0.4, 0.4], scan_span=[1, 1],
                            scan_direction="down", rows=rows)
        dim = np.full((120, 160, 3), (4, 10, 40), np.uint8)
        out, report = apply_flicker(dim, np.random.default_rng(4), s, s.armors, (0, 0, 160, 120), self.options, "train")
        applied = report["applied"][0]
        gains = row_exposure_gain(rows, report["refresh_hz"], report["duty"], report["exposure_ms"] / 1000,
                                  report["exposure_start_ms"] / 1000, applied["scan_phase"], applied["scan_span"], applied["descending"])
        linear = lambda x: (np.asarray(x, float) / 255) ** 2.2
        encode = lambda lin: np.clip(np.floor(np.clip(lin, 0, 1) ** (1 / 2.2) * 255 + 0.5), 0, 255)
        checked = 0
        for r in range(1, rows - 1):  # 首尾两行在屏的边缘渐入区里，增益被有意稀释
            y = 10 + int((r + 0.5) * 100 / rows)
            expected = encode(linear(dim[y, 80, 2]) * gains[r])  # 渲染 = 模型行增益，再取整到 8 位
            self.assertEqual(int(out[y, 80, 2]), int(expected), f"row {r}")
            checked += 1
        self.assertEqual(checked, rows - 2)
        edge_row = int(out[10 + int(0.5 * 100 / rows), 80, 2])  # 边缘行与未渐入的期望不同：渐入把增益向 1 拉
        self.assertNotEqual(edge_row, int(encode(linear(dim[11, 80, 2]) * gains[0])))
        bright = np.full((120, 160, 3), (4, 20, 250), np.uint8)
        clipped, _ = apply_flicker(bright, np.random.default_rng(4), s, s.armors, (0, 0, 160, 120), self.options, "train")
        boosted, saturated = 0, 0
        for r in range(1, rows - 1):
            y = 10 + int((r + 0.5) * 100 / rows)
            value = int(clipped[y, 80, 2])
            if gains[r] > 1.05:   # 增益大于 1 的行不能变暗：限幅在满刻度，而不是溢出回绕成小数
                boosted += 1
                self.assertGreaterEqual(value, 250, f"row {r}")
                saturated += value == 255
            elif gains[r] < 0.01:  # LED 在整个曝光里都是灭的：该行熄灭
                self.assertLess(value, 5, f"row {r}")
        self.assertGreater(boosted, 5)
        self.assertGreater(saturated, 0)                              # 确实有行被推到满刻度

    def test_off_real_and_validation_paths_are_exact_noops(self):
        image = np.full((120, 160, 3), 120, np.uint8); s = sample([plate()])
        for source, split, prob in (("real", "train", 1), ("sim", "val", 1), ("sim", "train", 0)):
            s.source = source; self.options["prob"] = prob
            rng = np.random.default_rng(2); state = copy.deepcopy(rng.bit_generator.state)
            out, meta = apply_flicker(image, rng, s, s.armors, (0, 0, 160, 120), self.options, split)
            np.testing.assert_array_equal(out, image); self.assertIsNone(meta)
            self.assertEqual(state, rng.bit_generator.state)


class BayerTest(unittest.TestCase):
    def setUp(self):
        self.options = configuration({"effects": {"bayer": {"prob": 1}}})["effects"]["bayer"]

    def test_neutral_flat_and_smooth_images_do_not_get_false_color(self):
        flat = np.full((32, 40, 3), (90, 120, 200), np.uint8)
        out, _ = apply_bayer(flat, np.random.default_rng(0), sample(), (0, 0, 40, 32), self.options, "train")
        np.testing.assert_array_equal(out[2:-2, 2:-2], flat[2:-2, 2:-2])
        ramp = np.repeat(np.linspace(30, 220, 64)[None, :, None], 3, axis=2).repeat(40, axis=0).astype(np.uint8)
        out, _ = apply_bayer(ramp, np.random.default_rng(0), sample(), (0, 0, 64, 40), self.options, "train")
        self.assertLessEqual(int(np.abs(out[4:-4, 4:-4].astype(int) - ramp[4:-4, 4:-4]).max()), 2)

    def test_impulse_response_matches_nvidia_npp_bilinear_kernels(self):
        for (py, px), channel, expected in (
                ((0, 0), 0, [[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]]),   # 红色位点只在红通道响应
                ((1, 1), 2, [[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]]),   # 蓝色位点
                ((0, 1), 1, [[0, .25, 0], [.25, 1, .25], [0, .25, 0]]),       # 绿色位点
                ((1, 0), 1, [[0, .25, 0], [.25, 1, .25], [0, .25, 0]])):
            rgb = np.zeros((16, 16, 3), np.float64); y, x = 8 + py, 8 + px
            mosaic = np.zeros((16, 16)); mosaic[y, x] = 200
            out = bayer_demosaic(mosaic)
            np.testing.assert_allclose(out[y - 1:y + 2, x - 1:x + 2, channel] / 200, expected, atol=1e-12)
            others = [c for c in range(3) if c != channel]
            self.assertEqual(float(np.abs(out[..., others]).max()), 0.0)

    def test_red_dot_grid_gets_false_color_and_phase_follows_roi_origin(self):
        dots = np.zeros((40, 64, 3), np.uint8)
        for y in range(3, 40, 3):
            for x in range(3, 64, 3):
                dots[y:y + 2, x:x + 2] = (20, 30, 240)  # BGR 的红点
        a, report = apply_bayer(dots, np.random.default_rng(0), sample(), (0, 0, 64, 40), self.options, "train")
        b, _ = apply_bayer(dots, np.random.default_rng(0), sample(), (1, 0, 64, 40), self.options, "train")
        gained_blue = a[..., 0].astype(int) - dots[..., 0].astype(int)
        self.assertGreater(int(gained_blue.max()), 10)                  # 蓝通道比原图多出来：伪色
        self.assertGreater(int(((gained_blue > 8) & (dots.sum(-1) == 0)).sum()), 100)  # 暗背景上也出现蓝色
        self.assertLess(int((a[..., 2].astype(int) - dots[..., 2].astype(int)).min()), -100)  # 红点被抹掉一部分
        self.assertGreater(float(np.abs(a.astype(int) - dots).mean()), 1.0)
        self.assertGreater(float(np.abs(a.astype(int) - b.astype(int)).mean()), 1.0)  # 奇偶相位不同，结果不同
        self.assertEqual(report["origin_parity"], [0, 0])
        self.assertGreater(report["changed_values"], 0)
        again, _ = apply_bayer(dots, np.random.default_rng(0), sample(), (0, 0, 64, 40), self.options, "train")
        np.testing.assert_array_equal(a, again)

    def test_noise_attenuation_matches_the_bilinear_kernels_whatever_the_noise_order(self):
        # 每个传感器像素独立加噪声，和先在 RGB 三个通道里独立加噪声再取样，在数学上等价；去马赛克后的噪声标准差
        # 是原来的 0.75（红、蓝）和 0.79（绿）倍，等于双线性核的理论值。这里固定这个事实，免得有人再把噪声搬进马赛克。
        rng = np.random.default_rng(0)
        h, w = 240, 280
        core = (slice(8, -8), slice(8, -8))
        in_mosaic = bayer_demosaic(rng.standard_normal((h, w)))[core].reshape(-1, 3).std(0)
        rgb_first = bayer_demosaic(bayer_mosaic(rng.standard_normal((h, w, 3))))[core].reshape(-1, 3).std(0)
        expected = np.sqrt([(1 + .5 + .5 + .25) / 4, (1 + .25) / 2, (1 + .5 + .5 + .25) / 4])  # R、G、B
        np.testing.assert_allclose(in_mosaic, expected, atol=0.02)
        np.testing.assert_allclose(rgb_first, expected, atol=0.02)
        # 去马赛克后相邻像素的噪声相关（空间相关的彩色噪声），不再是独立噪声
        noise = bayer_demosaic(rng.standard_normal((h, w)))[core][..., 0]
        noise = noise - noise.mean()
        self.assertGreater(float((noise[:, :-1] * noise[:, 1:]).mean() / (noise ** 2).mean()), 0.2)

    def test_geometry_is_not_moved(self):
        image = np.zeros((41, 41, 3), np.uint8); image[18:23, 18:23] = 200
        out, _ = apply_bayer(image, np.random.default_rng(0), sample(), (0, 0, 41, 41), self.options, "train")
        yy, xx = np.mgrid[:41, :41]
        for c in range(3):
            weight = out[..., c].astype(float)
            self.assertAlmostEqual((weight * xx).sum() / weight.sum(), 20, delta=0.05)
            self.assertAlmostEqual((weight * yy).sum() / weight.sum(), 20, delta=0.05)

    def test_off_real_and_validation_paths_are_exact_noops(self):
        image = np.full((32, 40, 3), 77, np.uint8); s = sample()
        for source, split, prob in (("real", "train", 1), ("sim", "val", 1), ("sim", "train", 0)):
            s.source = source; self.options["prob"] = prob
            rng = np.random.default_rng(2); state = copy.deepcopy(rng.bit_generator.state)
            out, meta = apply_bayer(image, rng, s, (0, 0, 40, 32), self.options, split)
            np.testing.assert_array_equal(out, image); self.assertIsNone(meta)
            self.assertEqual(state, rng.bit_generator.state)


class ConfigTest(unittest.TestCase):
    def test_invalid_augmentation_settings_fail_early(self):
        for override in ({"effects": {"motion_blur": {"kernel": "wrong"}}},
                         {"roi": {"tracker": {"search_prob": 1, "patrol_prob": 1}}},
                         {"roi": {"tracker": {"lost_time_ms": 0}}},
                         {"effects": {"moire": {"pixel_aperture": [0, 1]}}},
                         {"effects": {"moire": {"strength": [0, 2]}}},
                         {"effects": {"moire": {"harmonics": 100}}},
                         {"effects": {"bayer": {"pattern": "bggr"}}},
                         {"effects": {"bayer": {"demosaic": "ahd"}}},
                         {"effects": {"bayer": {"prob": 2}}},
                         {"effects": {"flicker": {"duty": [0, 1]}}},
                         {"effects": {"flicker": {"refresh_hz": [5, 50]}}},
                         {"effects": {"flicker": {"scan_span": [0, 2]}}},
                         {"effects": {"flicker": {"scan_direction": "left"}}},
                         {"effects": {"flicker": {"rows": 1}}}):
            with self.assertRaises(ValueError):
                configuration(override)


if __name__ == "__main__":
    unittest.main()
