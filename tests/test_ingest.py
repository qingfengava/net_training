import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt.canonical import read_manifest, read_samples
from nt.convert import canonical_to_yolo
from nt.ingest import ingest
from nt.spec import LabelSpec
from nt.validators import validate_dataset
from test_geometry import rect

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SPEC = LabelSpec.load()


def old_corners(label_file: Path, w=1440, h=1080):
    """Pixel corners per armor from an old Daedalus label file, as sets for order-free comparison."""
    out = []
    for line in label_file.read_text().splitlines():
        f = line.split()
        # The old exporter wrote whole pixels as 6-decimal fractions, so compare to 0.01 px, not 0.001.
        out.append({(round(float(f[3 + 2 * i]) * w, 2), round(float(f[4 + 2 * i]) * h, 2)) for i in range(4)})
    return out


def write_yolo_dataset(root: Path, n_images: int, identical: bool = False):
    """Tiny ultralytics-pose dataset with one correctly ordered armor per image."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    w, h = 300, 200
    pts = rect(150, 100, 60, 25, 4)
    cls = SPEC.class_id(0, 1)
    row = [str(cls), "0.5", "0.5", "0.25", "0.2"]
    for x, y in pts:
        row += [f"{x / w:.6f}", f"{y / h:.6f}", "2"]
    for i in range(n_images):
        rng = np.random.default_rng(0 if identical else i)
        cv2.imwrite(str(root / "images" / f"img{i}.png"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        (root / "labels" / f"img{i}.txt").write_text(" ".join(row) + "\n")


class IngestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_daedalus_old_fixture_is_accepted_and_reordered(self):
        dst = self.tmp / "ds"
        m = ingest(FIXTURES / "daedalus_old", dst, "daedalus-old", "sim", "dae", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["counts"]["rejected_samples"], m["counts"]["armors"]), (3, 0, 6))
        self.assertTrue(any("reordered corners" in n for n in m["notes"]))
        self.assertEqual(m["label_spec_sha256"], SPEC.sha256)
        self.assertEqual(read_manifest(dst / "manifest.json").counts["accepted_samples"], 3)
        report = validate_dataset(dst, SPEC)
        self.assertTrue(report.ok, [i.msg for i in report.errors])
        for i, s in enumerate(read_samples(dst / "labels.jsonl"), 1):
            expected = old_corners(FIXTURES / "daedalus_old" / "label" / f"frame_{i:06d}.txt")
            for a, exp in zip(s.armors, expected):
                got = {(round(x, 2), round(y, 2)) for x, y in a.corners}
                self.assertEqual(got, exp)

    def test_gate_refuses_shifted_corner_order(self):
        # The M0 fixture keeps the old exporter's unstable start corner on purpose.
        m = ingest(FIXTURES / "m0_smoke", self.tmp / "ds", "ultralytics-pose", "sim", "m0", SPEC)
        self.assertEqual(m["counts"]["accepted_samples"], 0)
        self.assertEqual(m["counts"]["rejected_samples"], 3)
        self.assertIn("E_QUAD_ORDER", m["rejected"])
        rejected = [json.loads(ln) for ln in (self.tmp / "ds" / "rejected.jsonl").read_text().splitlines()]
        self.assertEqual(len(rejected), 3)

    def test_destination_is_immutable(self):
        dst = self.tmp / "ds"
        ingest(FIXTURES / "daedalus_old", dst, "daedalus-old", "sim", "dae", SPEC)
        with self.assertRaises(FileExistsError):
            ingest(FIXTURES / "daedalus_old", dst, "daedalus-old", "sim", "dae", SPEC)

    def test_dry_run_writes_nothing(self):
        dst = self.tmp / "out" / "ds"
        m = ingest(FIXTURES / "daedalus_old", dst, "daedalus-old", "sim", "dae", SPEC, dry_run=True)
        self.assertEqual(m["counts"]["accepted_samples"], 3)
        self.assertFalse(dst.exists())
        self.assertFalse((self.tmp / "out").exists())

    def test_near_duplicates_warn_or_drop(self):
        src = self.tmp / "src"
        write_yolo_dataset(src, 2, identical=True)
        m = ingest(src, self.tmp / "a", "ultralytics-pose", "exchange", "x", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["issues"].get("W_NEAR_DUPLICATE_FRAME")), (2, 1))
        m = ingest(src, self.tmp / "b", "ultralytics-pose", "exchange", "x", SPEC, drop_near_duplicates=True)
        self.assertEqual((m["counts"]["accepted_samples"], m["rejected"].get("E_NEAR_DUPLICATE")), (1, 1))

    def test_yolo_round_trip_keeps_corners_and_classes(self):
        canon = self.tmp / "canon"
        ingest(FIXTURES / "daedalus_old", canon, "daedalus-old", "sim", "dae", SPEC)
        stats = canonical_to_yolo(canon, self.tmp / "yolo", SPEC)
        self.assertEqual((stats["samples"], stats["armors"], stats["skipped_untrained_colors"]), (3, 6, {}))
        again = ingest(self.tmp / "yolo", self.tmp / "canon2", "ultralytics-pose", "sim", "dae", SPEC)
        self.assertEqual(again["counts"]["rejected_samples"], 0)
        first = list(read_samples(canon / "labels.jsonl"))
        second = list(read_samples(self.tmp / "canon2" / "labels.jsonl"))
        for s1, s2 in zip(first, second):
            for a1, a2 in zip(s1.armors, s2.armors):
                self.assertEqual((a1.color, a1.label), (a2.color, a2.label))
                for p1, p2 in zip(a1.corners, a2.corners):
                    self.assertAlmostEqual(p1[0], p2[0], delta=2e-3)
                    self.assertAlmostEqual(p1[1], p2[1], delta=2e-3)

    def test_untrained_colors_are_counted_on_export(self):
        canon = self.tmp / "canon"
        ingest(FIXTURES / "daedalus_old", canon, "daedalus-old", "sim", "dae", SPEC)
        lines = (canon / "labels.jsonl").read_text().splitlines()
        edited = []
        for ln in lines:
            d = json.loads(ln)
            d["armors"][0]["color"] = 3  # purple has no training class (gray does: it is the off class)
            edited.append(json.dumps(d))
        (canon / "labels.jsonl").write_text("\n".join(edited) + "\n")
        stats = canonical_to_yolo(canon, self.tmp / "yolo", SPEC)
        self.assertEqual((stats["armors"], stats["skipped_untrained_colors"]), (3, {"purple": 3}))


def write_sim_export(root: Path, n: int = 2, declared_w: int = 300):
    """A miniature of what Daedalus writes: labels.jsonl with pixel corners, vis flags, scene data, images/."""
    from nt.canonical import Armor, Camera, Sample, dumps, sample_to_dict

    (root / "images").mkdir(parents=True)
    rows = []
    for i in range(n):
        rng = np.random.default_rng(100 + i)
        cv2.imwrite(str(root / "images" / f"s{i:06d}.png"), rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
        armor = Armor(
            color=1, label=3, corners=[list(p) for p in rect(150, 100, 60, 25, 4)], vis=[2, 2, 1, 2],
            type="small", occ=0.1, dist_m=4.2, yaw_deg=12.0, pitch_deg=-3.0,
        )
        sample = Sample(
            id=f"s{i:06d}", image=f"images/s{i:06d}.png", w=declared_w, h=200, source="sim", armors=[armor],
            camera=Camera(fx=1500.0, fy=1500.0, cx=150.0, cy=100.0), origin=f"seq{i}", scene={"seed": 7, "attempt": i},
        )
        rows.append(dumps(sample_to_dict(sample)))
    (root / "labels.jsonl").write_text("\n".join(rows) + "\n")


class CanonicalJsonlIngestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_simulator_export_keeps_pixels_vis_camera_and_scene(self):
        src = self.tmp / "raw"
        write_sim_export(src)
        m = ingest(src, self.tmp / "ds", "canonical-jsonl", "sim", "sim", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["counts"]["armors"]), (2, 2))
        first = next(read_samples(self.tmp / "ds" / "labels.jsonl"))
        a = first.armors[0]
        self.assertEqual((a.vis, a.occ, a.dist_m, a.yaw_deg), ([2, 2, 1, 2], 0.1, 4.2, 12.0))
        self.assertAlmostEqual(a.corners[0][0], rect(150, 100, 60, 25, 4)[0][0], delta=1e-3)  # pixels, not rescaled
        self.assertEqual((first.camera.fx, first.scene["seed"], first.origin), (1500.0, 7, "seq0"))
        self.assertTrue(validate_dataset(self.tmp / "ds", SPEC).ok)

    def test_declared_size_is_checked_against_the_image(self):
        src = self.tmp / "raw"
        write_sim_export(src, declared_w=301)
        m = ingest(src, self.tmp / "ds", "canonical-jsonl", "sim", "sim", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["rejected"].get("E_SIZE_MISMATCH")), (0, 2))

    def test_link_hard_links_images(self):
        src = self.tmp / "raw"
        write_sim_export(src, n=1)
        ingest(src, self.tmp / "ds", "canonical-jsonl", "sim", "sim", SPEC, link=True)
        self.assertEqual((self.tmp / "ds" / "images" / "sim_000000.png").stat().st_nlink, 2)


if __name__ == "__main__":
    unittest.main()
