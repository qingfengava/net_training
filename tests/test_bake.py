import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt.bake import apply_effects, bake, choose_roi, draw_distractors, load_config, split_of
from nt.canonical import Armor, Sample, dumps, sample_to_dict
from nt.ingest import ingest
from nt.spec import LabelSpec
from test_geometry import rect

SPEC = LabelSpec.load()
SPEC_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "label_spec.yaml")
W, H = 1440, 1080

NO_EFFECTS = {
    "effects": {
        "gain": [1.0, 1.0], "black_level": [0.0, 0.0], "white_balance": [1.0, 1.0],
        "bloom": {"prob": 0.0}, "motion_blur": {"prob": 0.0}, "defocus": {"prob": 0.0},
        "read_noise": [0.0, 0.0], "shot_noise": [0.0, 0.0],
    },
    "distractors": {"prob": 0.0},
}


def make_canonical(tmp: Path, n: int = 3, quad=None) -> Path:
    """A canonical dataset of black frames with one white filled quad, labeled exactly with its corners."""
    quad = quad or rect(700, 500, 100, 40, 0)
    raw = tmp / "raw"
    (raw / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        cv2.fillPoly(img, [np.round(np.array(quad)).astype(np.int32)], (255, 255, 255))
        cv2.imwrite(str(raw / "images" / f"s{i:06d}.png"), img)
        armor = Armor(color=1, label=3, corners=[list(p) for p in quad], vis=[2, 2, 2, 2], type="small", dist_m=5.0, yaw_deg=10.0, occ=0.0)
        rows.append(dumps(sample_to_dict(Sample(id=f"s{i}", image=f"images/s{i:06d}.png", w=W, h=H, source="sim", armors=[armor]))))
    (raw / "labels.jsonl").write_text("\n".join(rows) + "\n")
    ingest(raw, tmp / "canon", "canonical-jsonl", "sim", "t", SPEC)
    return tmp / "canon"


def read_label(path: Path):
    f = [float(v) for v in path.read_text().split()]
    return f[0], [(f[5 + 3 * i], f[6 + 3 * i]) for i in range(4)]


class BakeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_bake(self, canon, name, override, seed=0, copies=1):
        cfg = load_config(override={"val_fraction": 0.0, **override})
        return bake([(canon, 1)], self.tmp / name, SPEC, cfg, copies, seed, 2, SPEC_PATH)

    def test_full_frame_label_follows_the_resize(self):
        canon = make_canonical(self.tmp)
        self.run_bake(canon, "full", {**NO_EFFECTS, "roi": {"full_frame_prob": 1.0}})
        out = self.tmp / "full"
        img = cv2.imread(str(out / "images/train/t_000000_c0.png"))
        self.assertEqual(img.shape, (576, 768, 3))
        cid, pts = read_label(out / "labels/train/t_000000_c0.txt")
        self.assertEqual(cid, SPEC.class_id(1, 3))
        for (nx, ny), (ox, oy) in zip(pts, rect(700, 500, 100, 40, 0)):
            # Labels are written with 6 decimals of a fraction, so 1e-6 * 768 px is the resolution.
            self.assertAlmostEqual(nx * 768, ox * 768 / W, delta=1e-3)
            self.assertAlmostEqual(ny * 576, oy * 576 / H, delta=1e-3)
        cx, cy = 700 * 768 / W, 500 * 576 / H
        self.assertGreater(img[int(cy), int(cx)].mean(), 200)  # the quad is where the label says
        self.assertLess(img[int(cy) + 40, int(cx)].mean(), 20)

    def test_roi_crop_maps_labels_into_the_crop(self):
        canon = make_canonical(self.tmp)
        roi_cfg = {"full_frame_prob": 0.0, "armor_width_multiple": [6.0, 6.0], "center_jitter": 0.0}
        self.run_bake(canon, "roi", {**NO_EFFECTS, "roi": roi_cfg})
        out = self.tmp / "roi"
        meta = json.loads((out / "meta.jsonl").read_text().splitlines()[0])
        x0, y0, rw, rh = meta["roi"]
        self.assertEqual(meta["roi_mode"], "roi")
        self.assertEqual(rw * 3, rh * 4)
        self.assertAlmostEqual(meta["scale"], 768 / rw)
        img = cv2.imread(str(out / "images/train/t_000000_c0.png"))
        _, pts = read_label(out / "labels/train/t_000000_c0.txt")
        self.assertTrue(all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts))
        cx = sum(p[0] for p in pts) / 4 * 768
        cy = sum(p[1] for p in pts) / 4 * 576
        self.assertGreater(img[int(cy), int(cx)].mean(), 200)
        # Same transform recorded for later evaluation in original pixels.
        ox, oy = meta["armors"][0]["corners_orig"][0]
        nx, ny = meta["armors"][0]["corners_net"][0]
        self.assertAlmostEqual((ox - x0) * meta["scale"], nx, delta=1e-2)
        self.assertAlmostEqual((oy - y0) * meta["scale"], ny, delta=1e-2)

    def test_same_seed_same_pixels_different_seed_different_pixels(self):
        canon = make_canonical(self.tmp, n=2)
        self.run_bake(canon, "a", {}, seed=5)
        self.run_bake(canon, "b", {}, seed=5)
        self.run_bake(canon, "c", {}, seed=6)
        read = lambda d: cv2.imread(str(self.tmp / d / "images/train/t_000000_c0.png"))
        self.assertTrue(np.array_equal(read("a"), read("b")))
        self.assertFalse(np.array_equal(read("a"), read("c")))
        ma = [json.loads(ln) for ln in (self.tmp / "a" / "meta.jsonl").read_text().splitlines()]
        mb = [json.loads(ln) for ln in (self.tmp / "b" / "meta.jsonl").read_text().splitlines()]
        self.assertEqual(ma, mb)

    def test_output_is_never_overwritten_and_manifest_records_inputs(self):
        canon = make_canonical(self.tmp, n=2)
        m = self.run_bake(canon, "out", {}, copies=2)
        self.assertEqual(m["images"], {"train": 4})
        self.assertEqual(m["sources"][0]["samples"], 2)
        self.assertEqual(m["label_spec_sha256"], SPEC.sha256)
        with self.assertRaises(FileExistsError):
            self.run_bake(canon, "out", {})
        self.assertIn("kpt_shape: [4, 3]", (self.tmp / "out" / "data.yaml").read_text())


class PiecesTest(unittest.TestCase):
    def test_identity_effects_keep_the_image(self):
        cfg = load_config(override=NO_EFFECTS)
        img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
        out, used = apply_effects(img, np.random.default_rng(1), cfg)
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)
        self.assertEqual(used["gain"], 1.0)

    def test_default_effects_change_the_image_and_record_parameters(self):
        cfg = load_config()
        img = np.full((90, 120, 3), 128, np.uint8)
        out, used = apply_effects(img, np.random.default_rng(3), cfg)
        self.assertEqual((out.dtype, out.shape), (np.uint8, img.shape))
        self.assertFalse(np.array_equal(out, img))
        self.assertIn("noise", used)

    def test_distractors_never_touch_armor_boxes(self):
        cfg = load_config(override={"distractors": {"prob": 1.0, "count": [3, 3]}})
        avoid = (170.0, 120.0, 230.0, 180.0)
        drew_any = False
        for seed in range(40):
            img = np.zeros((300, 400, 3), np.uint8)
            placed = draw_distractors(img, np.random.default_rng(seed), cfg, [avoid], 50.0)
            drew_any |= bool(placed)
            self.assertEqual(int(img[120:180, 170:230].sum()), 0, f"seed {seed} painted over the armor")
        self.assertTrue(drew_any)

    def test_roi_is_four_by_three_and_inside_the_image(self):
        cfg = load_config()
        rng = np.random.default_rng(0)
        for _ in range(200):
            boxes = [(float(rng.uniform(50, 1300)), float(rng.uniform(50, 1000)), 0.0, 0.0)]
            boxes = [(b[0], b[1], b[0] + 40, b[1] + 16) for b in boxes]
            (x, y, w, h), how = choose_roi(rng, W, H, boxes, cfg)
            self.assertEqual(w * 3, h * 4)
            self.assertTrue(0 <= x and 0 <= y and x + w <= W and y + h <= H, (x, y, w, h, how))

    def test_split_is_stable_and_roughly_proportional(self):
        ids = [f"id{i}" for i in range(2000)]
        val = sum(split_of(i, 0.1) == "val" for i in ids)
        self.assertTrue(150 < val < 250, val)
        self.assertEqual([split_of(i, 0.1) for i in ids[:50]], [split_of(i, 0.1) for i in ids[:50]])



class LensTest(unittest.TestCase):
    """The distorted image and the mapped labels must describe the same view."""

    CAM = load_config()["lens"]["cameras"][0]

    def params(self, scale=1.0):
        return {"camera": 0, "dist": [v * scale for v in self.CAM["dist"]], "k": self.CAM["k"], "size": self.CAM["size"]}

    def test_a_bright_dot_lands_where_its_label_goes(self):
        from nt.bake import apply_lens

        img = np.zeros((1080, 1440, 3), np.uint8)
        dots = [(20, 15), (700, 520), (1400, 1050), (1100, 80), (60, 900)]
        for x, y in dots:
            img[y - 2:y + 3, x - 2:x + 3] = 255  # 5x5 block centred on pixel (x, y): its centre is (x + .5, y + .5)
        for scale in (1.0, 1.5):
            out, mapped, zoom = apply_lens(img, [[[x + 0.5, y + 0.5] for x, y in dots]], self.params(scale))
            self.assertGreater(zoom, 1.0)  # barrel distortion: zoomed in so no border pixel is invented
            g = out[..., 0].astype(np.float64)
            moved = 0.0
            for (x, y), (mx, my) in zip(dots, mapped[0]):
                ys, xs = np.mgrid[int(my) - 12:int(my) + 13, int(mx) - 12:int(mx) + 13]
                ys, xs = np.clip(ys, 0, 1079), np.clip(xs, 0, 1439)
                w = g[ys, xs]
                c = np.array([(xs * w).sum() / w.sum() + 0.5, (ys * w).sum() / w.sum() + 0.5])
                self.assertLess(np.abs(c - [mx, my]).max(), 0.15, (scale, x, y))
                moved = max(moved, abs(mx - x - 0.5), abs(my - y - 0.5))
            # these lenses are mild: about 1.5% at the corners, mostly undone by the zoom; still not a no-op
            self.assertGreater(moved, 1.0)

    def test_off_by_default_and_never_for_real_or_already_distorted_frames(self):
        from nt.bake import lens_params
        from nt.canonical import Camera, Sample

        sim = Sample("s", "x.png", 1440, 1080, "sim", camera=Camera(1000, 1000, 720, 540))
        rng = np.random.default_rng(0)
        state = rng.bit_generator.state
        self.assertIsNone(lens_params(rng, load_config(), sim))
        self.assertEqual(rng.bit_generator.state, state)  # nothing drawn: old bakes stay bit-identical
        on = load_config(Path(__file__).resolve().parents[1] / "configs" / "bake" / "lens.yaml")
        self.assertEqual(on["lens"]["prob"], 0.8)
        hits = sum(lens_params(np.random.default_rng(i), on, sim) is not None for i in range(200))
        self.assertTrue(130 < hits < 190, hits)
        real = Sample("r", "x.png", 1440, 1080, "real", camera=Camera(1000, 1000, 720, 540))
        distorted = Sample("d", "x.png", 1440, 1080, "sim", camera=Camera(1000, 1000, 720, 540, [-0.1, 0, 0, 0, 0]))
        for s in (real, distorted):
            self.assertIsNone(lens_params(np.random.default_rng(0), {"lens": {**on["lens"], "prob": 1.0}}, s))

    def test_bake_with_lens_keeps_labels_on_the_plate(self):
        canon = make_canonical(Path(self._tmp.name), n=2)
        m = bake([(canon, 1)], Path(self._tmp.name) / "b", SPEC, load_config(override={"lens": {"prob": 1.0},
                 "roi": {"full_frame_prob": 1.0}, "distractors": {"prob": 0.0}}), 1, 0, 1, str(SPEC_PATH))
        metas = [json.loads(ln) for ln in (Path(self._tmp.name) / "b" / "meta.jsonl").read_text().splitlines()]
        self.assertTrue(all("lens" in mm["effects"] for mm in metas))
        self.assertEqual(sum(m["images"].values()), 2)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
