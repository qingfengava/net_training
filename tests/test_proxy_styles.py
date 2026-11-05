"""代理屏复用 A/B 规格、保持几何比例，并接入 Recorder PNG 的自动标注流程。"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt import proxy, session
from nt.canonical import read_samples
from nt.spec import LabelSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = LabelSpec.load()
A = ROOT / "configs/screens/rm27_hypothesis_a.toml"
B = ROOT / "configs/screens/rm27_hypothesis_b.toml"


class ProxyStylesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_bitmap_keeps_physical_aspect_with_non_square_cells(self):
        # 2:1 的实心图案，占 2:1 面板半高；应覆盖面板中心的一半宽和一半高。
        bitmap = self.root / "bar.png"
        cv2.imwrite(str(bitmap), np.full((8, 16), 255, np.uint8))
        for cols in (40, 80):
            mask = proxy._bitmap_mask(bitmap, cols, 20, 0.5, 2.0)
            expected = np.zeros((20, cols), bool)
            expected[5:15, cols // 4:cols * 3 // 4] = True
            np.testing.assert_array_equal(mask, expected)

    def test_a_font_and_real_pattern_are_selected(self):
        digit = proxy.render(self.root / "a1", SPEC, "one", "red", screen_spec=A)
        sentry = proxy.render(self.root / "as", SPEC, "sentry", "blue", screen_spec=A)
        self.assertEqual((digit["grid"]["cols"], digit["grid"]["rows"]), (34, 28))
        self.assertEqual(digit["content"]["kind"], "text")
        self.assertEqual(sentry["content"]["kind"], "bitmap")
        self.assertEqual(digit["screen_spec"]["sha256"], hashlib.sha256(A.read_bytes()).hexdigest())
        # A 的占屏高是 0.725，字库画布为 10×14，数字 1 的实际笔画范围为 6×14。
        ys, xs = np.where(np.array(digit["content"]["lit_mask"]))
        self.assertEqual((xs.max() - xs.min() + 1, ys.max() - ys.min() + 1), (6, 14))

    def test_b_assets_geometry_and_rendered_dots(self):
        for label in ("one", "three", "four", "sentry", "outpost", "base"):
            with self.subTest(label=label):
                out = self.root / label
                meta = proxy.render(out, SPEC, label, "red", screen_spec=B, display=(960, 540))
                self.assertEqual((meta["grid"]["cols"], meta["grid"]["rows"], meta["grid"]["dot"]), (64, 48, "square"))
                bitmap = ROOT / "configs/screens/patterns_b" / (label + ".png")
                self.assertEqual(meta["content"]["bitmap"]["sha256"], hashlib.sha256(bitmap.read_bytes()).hexdigest())
                self.assertAlmostEqual(meta["panel_aspect"], 117 / 100, delta=0.003)
                self.assertIsNone(meta["panel_size_mm"])
                self.assertIsNone(meta["grid"]["pitch_px"])
                image = cv2.imread(str(out / "proxy.png"))
                mask = np.array(meta["content"]["lit_mask"], bool)
                x0, y0 = meta["panel_corners"][0]
                px, py = meta["grid"]["pitch_xy_px"]
                yy, xx = np.indices(mask.shape)
                pixels = image[(y0 + (yy + 0.5) * py).astype(int), (x0 + (xx + 0.5) * px).astype(int)]
                np.testing.assert_array_equal(pixels[:, :, 2] > 100, mask)

    def test_overrides_and_invalid_inputs(self):
        meta = proxy.render(self.root / "override", SPEC, "three", "blue", screen_spec=B,
                            cols=40, rows=30, dot="round", fill=0.5, glyph_height=0.6, panel_aspect=1.5)
        self.assertEqual(meta["grid"]["cols"], 40)
        self.assertEqual(meta["grid"]["dot"], "round")
        self.assertAlmostEqual(meta["panel_aspect"], 1.5, delta=0.003)
        for args in ({"text": "4", "screen_spec": B}, {"cols": 0}, {"fill": 1.5},
                     {"panel_aspect": float("nan")}, {"bitmap": self.root / "missing.png"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                proxy.render(self.root / "bad", SPEC, "three", "red", **args)
            self.assertFalse((self.root / "bad").exists())

    def test_recorder_import_proxy_label_and_ingest(self):
        out = self.root / "proxy"
        meta = proxy.render(out, SPEC, "three", "red", screen_spec=B, display=(960, 540))
        display = cv2.imread(str(out / "proxy.png"))
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [959, 0], [959, 539], [0, 539]]),
                                       np.float32([[100, 100], [860, 120], [850, 600], [110, 620]]))
        frame = cv2.warpPerspective(display, H, (960, 720), borderValue=(20, 20, 20))
        recording = self.root / "recording"
        (recording / "frames").mkdir(parents=True)
        cv2.imwrite(str(recording / "frames/frame_000000000.png"), frame)
        record = {"write_index": 0, "camera_frame_id": 1234, "time_offset_s": 0.1234567,
                  "width": 960, "height": 720, "channels": 3, "pixel_format": "BGR", "mode": "png_sequence",
                  "image": "frames/frame_000000000.png", "pose": {"qw": 1, "qx": 0, "qy": 0, "qz": 0}}
        (recording / "frames.jsonl").write_text(json.dumps(record) + "\n")
        capture = session.create(recording, self.root / "sessions", "proxy_b", "val")
        dataset = self.root / "dataset"
        result = proxy.label_session(capture, out / "proxy.json", dataset, SPEC)
        self.assertEqual(result["counts"]["accepted_samples"], 1)
        self.assertEqual(result["role"], "val")
        sample = next(read_samples(dataset / "labels.jsonl"))
        self.assertEqual(sample.scene["camera_frame_id"], 1234)
        self.assertEqual(sample.scene["t_s"], 0.1234567)
        self.assertEqual(sample.scene["q_wxyz"], [1, 0, 0, 0])
        gt = cv2.perspectiveTransform((np.array(meta["panel_corners"], float) - 0.5).reshape(-1, 1, 2), H).reshape(4, 2) + 0.5
        np.testing.assert_allclose(sample.armors[0].corners, gt, atol=0.7)


if __name__ == "__main__":
    unittest.main()
