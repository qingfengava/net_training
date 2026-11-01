"""Training hyper-parameter files: settings that would make runs silently incomparable are refused."""
import unittest
from pathlib import Path

import numpy as np
import yaml

from nt.train import ALBUMENTATIONS_ALLOWED, build_albumentations, check_hyp, load_hyp

ROOT = Path(__file__).resolve().parents[1]


class TrainConfigTest(unittest.TestCase):
    def test_every_hyp_file_pins_the_optimizer(self):
        for path in sorted((ROOT / "configs" / "train").glob("*.yaml")):
            hyp = load_hyp(path)
            check_hyp(hyp)  # raises SystemExit on optimizer auto or an SGD bias warmup with Adam
            self.assertNotEqual(str(hyp["optimizer"]).lower(), "auto", path.name)

    def test_default_reproduces_what_auto_chose_for_the_baselines(self):
        hyp = load_hyp(ROOT / "configs" / "train" / "pose_default.yaml")
        from nt.spec import LabelSpec

        nc = LabelSpec.load().n_classes
        self.assertEqual((hyp["optimizer"], hyp["momentum"], hyp["warmup_bias_lr"]), ("AdamW", 0.9, 0.0))
        self.assertEqual(hyp["lr0"], round(0.002 * 5 / (4 + nc), 6))  # Ultralytics' auto fit for nc classes

    def test_refusals(self):
        for bad in ({"optimizer": "auto"}, {}, {"optimizer": "AdamW", "warmup_bias_lr": 0.1}):
            with self.assertRaises(SystemExit):
                check_hyp(bad)
        check_hyp({"optimizer": "MuSGD", "lr0": 0.01, "warmup_bias_lr": 0.1})
        with self.assertRaises(SystemExit):  # a shifted motion-blur kernel moves the content
            check_hyp({"optimizer": "AdamW", "warmup_bias_lr": 0.0, "albumentations": [{"name": "MotionBlur", "p": 0.1}]})


class AlbumentationsTest(unittest.TestCase):
    """With albumentations installed, Ultralytics applies its default list (which has ToGray) unless given one."""

    def ultralytics_transforms(self, augmentations):
        from ultralytics.data.augment import Albumentations

        a = Albumentations(p=1.0, transforms=augmentations)
        self.assertIsNotNone(a.transform, "albumentations is not importable")
        return a, [type(x).__name__ for x in a.transform.transforms]

    def test_default_hyp_passes_an_empty_list_so_nothing_is_applied(self):
        hyp = load_hyp(ROOT / "configs" / "train" / "pose_default.yaml")
        aug = build_albumentations(hyp.get("albumentations"))
        self.assertEqual(aug, [])
        a, names = self.ultralytics_transforms(aug)
        self.assertEqual(names, [])
        img = np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)
        np.testing.assert_array_equal(a.transform(image=img)["image"], img)
        _, stock = self.ultralytics_transforms(None)
        self.assertIn("ToGray", stock)  # what would run if nt.train passed nothing

    def test_albu_preset_is_pixel_level_and_keeps_color(self):
        hyp = load_hyp(ROOT / "configs" / "train" / "pose_albu.yaml")
        self.assertEqual(hyp["optimizer"], "AdamW")  # inherited from pose_default.yaml
        aug = build_albumentations(hyp["albumentations"])
        a, names = self.ultralytics_transforms(aug)
        self.assertTrue(names and set(names) <= ALBUMENTATIONS_ALLOWED, names)
        self.assertFalse(a.contains_spatial)  # Ultralytics then touches only the image, never the keypoints
        img = np.zeros((96, 128, 3), np.uint8)
        img[30:60, 40:90] = (255, 90, 20)  # a blue patch, BGR
        for _ in range(20):
            out = a.transform(image=img)["image"].astype(int)
            patch = out[35:55, 45:85]
            self.assertGreater(patch[..., 0].mean(), patch[..., 2].mean() + 100)  # still blue, not gray or red

    def test_albu_preset_keeps_a_bright_patch_in_place(self):
        """A filter that shifts the image moves every armor off its labeled corners."""
        aug = build_albumentations(load_hyp(ROOT / "configs" / "train" / "pose_albu.yaml")["albumentations"])
        import albumentations as A

        for t in aug:
            if type(t).__name__ not in {"MotionBlur", "GaussianBlur", "Defocus", "Sharpen"}:
                continue  # the others act on each pixel's value alone
            comp = A.Compose([t])
            comp.set_random_seed(0)
            for k in range(40):
                img = np.zeros((96, 96, 3), np.uint8)
                img[30 + k % 5:39 + k % 5, 30 + k % 7:43 + k % 7] = 255
                out = img
                for _ in range(10):  # p < 1: retry until the transform fires
                    out = comp(image=img)["image"]
                    if not np.array_equal(out, img):
                        break
                g0, g1 = img[..., 0].astype(float), out[..., 0].astype(float)
                ys, xs = np.mgrid[:96, :96]
                c0 = np.array([(xs * g0).sum(), (ys * g0).sum()]) / g0.sum()
                c1 = np.array([(xs * g1).sum(), (ys * g1).sum()]) / g1.sum()
                self.assertLess(np.abs(c1 - c0).max(), 0.05, type(t).__name__)

    def test_transforms_that_move_pixels_or_change_hue_are_refused(self):
        for name in ("ToGray", "HueSaturationValue", "ChannelShuffle", "HorizontalFlip", "Affine", "CLAHE", "Downscale", "Blur"):
            with self.assertRaises(SystemExit, msg=name):
                check_hyp({"optimizer": "AdamW", "warmup_bias_lr": 0.0, "albumentations": [{"name": name, "p": 0.1}]})


if __name__ == "__main__":
    unittest.main()
