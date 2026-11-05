"""Recorder PNG 的像素、时间、姿态、异常恢复及标注入库验证。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nt import annotate, recorder, session
from nt.canonical import read_samples
from nt.spec import LabelSpec


class RecorderPngTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / "recording"
        (self.src / "frames").mkdir(parents=True)
        self.bgr = np.full((24, 32, 3), [17, 3, 251], np.uint8)
        self.times = [0.00123456789, 0.04456789, 0.1387654321, 0.2023456]
        self.ids = [7, 8, 12, 13]
        self.rows = []
        for i in range(4):
            # 文件名字典序与写入顺序相反，导入必须按 JSONL 配对。
            image = f"frames/f{9-i}.png"
            cv2.imwrite(str(self.src / image), self.bgr)
            self.rows.append({"write_index": i, "camera_frame_id": self.ids[i],
                              "time_offset_s": self.times[i], "width": 32, "height": 24,
                              "channels": 3, "pixel_format": "RGB", "mode": "png_sequence",
                              "image": image, "pose": {"qw": 2, "qx": 0, "qy": 0, "qz": 0}})
        self.save(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, rows):
        (self.src / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def create(self, name="s", src=None, **kwargs):
        return session.create(src or self.src, self.root / "sessions", name, "val", **kwargs)

    def test_pixels_metadata_and_continuous_frames_survive(self):
        for i, src in enumerate([self.src, self.src / "frames", self.src / "frames.jsonl"]):
            out = self.create(f"s{i}", src=src)
            meta = session.load(out)
            self.assertEqual(meta["source"]["time_base"], "recorder_jsonl")
            self.assertEqual(meta["selection"]["dedupe_hamming"], -1)
            self.assertEqual(len(meta["frames"]), 4)
            self.assertEqual([f["t_s"] for f in meta["frames"]], self.times)
            self.assertEqual([f["camera_frame_id"] for f in meta["frames"]], self.ids)
            self.assertEqual(meta["source"]["camera_frame_ids"]["missing_between_known_ids"], 3)
            for frame, row in zip(meta["frames"], self.rows):
                self.assertEqual(frame["q_wxyz"], [1, 0, 0, 0])
                self.assertEqual((frame["source_pixel_format"], frame["pixel_format"]), ("RGB", "BGR"))
                np.testing.assert_array_equal(cv2.imread(str(out / frame["image"])), self.bgr)
                self.assertEqual((out / frame["image"]).read_bytes(), (self.src / row["image"]).read_bytes())
        with self.assertRaises(FileExistsError):
            self.create("s0")

    def test_selection_and_ignoring_poses_keep_original_identity(self):
        meta = session.load(self.create(every=2, start_s=0.04, end_s=0.19, poses_path=False))
        self.assertEqual(len(meta["frames"]), 1)
        frame = meta["frames"][0]
        self.assertEqual((frame["index"], frame["write_index"], frame["camera_frame_id"]), (2, 2, 12))
        self.assertEqual(frame["t_s"], self.times[2])
        self.assertNotIn("q_wxyz", frame)
        with self.assertRaisesRegex(ValueError, "--poses"):
            self.create("poses", poses_path=self.src / "poses.txt")
        self.assertEqual(len(session.load(self.create("dedup", dedupe_hamming=0))["frames"]), 1)

    def test_invalid_records_do_not_create_a_session(self):
        cases = [
            {"write_index": 0}, {"camera_frame_id": 7}, {"camera_frame_id": -2},
            {"time_offset_s": self.times[0]}, {"time_offset_s": float("nan")},
            {"pose": {"qw": 0, "qx": 0, "qy": 0, "qz": 0}},
            {"image": "frames/missing.png"}, {"image": "../outside.png"},
            {"image": str((self.src / self.rows[0]["image"]).resolve())},
            {"image": self.rows[0]["image"]}, {"width": 31}, {"channels": 1}, {"mode": "mjpg"},
        ]
        for change in cases:
            with self.subTest(change=change):
                rows = copy.deepcopy(self.rows)
                rows[1].update(change)
                self.save(rows)
                with self.assertRaises(ValueError):
                    self.create(every=2)
                self.assertFalse((self.root / "sessions/s").exists())

    def test_actual_png_header_checked_even_when_frame_is_not_selected(self):
        cv2.imwrite(str(self.src / self.rows[1]["image"]), np.zeros((20, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "PNG 头"):
            self.create(every=2)

    def test_symlink_outside_recording_is_rejected(self):
        outside = self.root / "outside.png"
        cv2.imwrite(str(outside), self.bgr)
        link = self.src / "frames/outside.png"
        link.symlink_to(outside)
        self.rows[1]["image"] = "frames/outside.png"
        self.save(self.rows)
        with self.assertRaisesRegex(ValueError, "越出"):
            self.create()

    def test_truncated_tail_recovers_only_complete_records(self):
        manifest = self.src / "frames.jsonl"
        original = manifest.read_text()
        cv2.imwrite(str(self.src / "frames/orphan.tmp.png"), self.bgr)
        tails = ('{"write_index":', '{"image":"frames/frame_', '{"time_offset_s":-',
                 '{"time_offset_s":1e-', '{"time_offset_s":1.', '{"pose":{"qw":1,')
        for i, tail in enumerate(tails):
            with self.subTest(tail=tail):
                manifest.write_text(original + tail)
                meta = session.load(self.create(f"tail{i}"))
                self.assertEqual(len(meta["frames"]), 4)
                self.assertEqual(meta["source"]["unlisted_pngs"], 1)
                self.assertTrue(any("截断" in w for w in meta["warnings"]))
        for text in (original + "broken\n", original.splitlines()[0] + "\nbroken\n" + original,
                     original + "broken", original + '{"write_index":garbage',
                     original + '{"write_index":1.}', original + '{}garbage', original + '{}'):
            manifest.write_text(text)
            with self.assertRaises(ValueError):
                self.create("bad")

    def test_gray_and_unknown_camera_ids(self):
        gray = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
        for row in self.rows:
            row.update(pixel_format="GRAY", channels=1, camera_frame_id=-1)
            cv2.imwrite(str(self.src / row["image"]), gray)
        self.save(self.rows)
        out = self.create()
        meta = session.load(out)
        self.assertEqual(meta["source"]["camera_frame_ids"]["unknown_records"], 4)
        np.testing.assert_array_equal(cv2.imread(str(out / meta["frames"][0]["image"]), -1), gray)

    def test_annotation_roundtrip_preserves_capture_metadata(self):
        out = self.create()
        project = self.root / "annot"
        spec = LabelSpec.load()
        annotate.export(project, spec, session_dir=out)
        for path in project.glob("batch_*/*.json"):
            if path.name == "batch.json":
                continue
            doc = json.loads(path.read_text())
            doc["flags"]["reviewed"] = True
            path.write_text(json.dumps(doc))
        dataset = self.root / "dataset"
        result = annotate.import_project(project, dataset, spec)
        self.assertEqual(result["counts"]["accepted_samples"], 4)
        samples = list(read_samples(dataset / "labels.jsonl"))
        self.assertEqual([s.scene["camera_frame_id"] for s in samples], self.ids)
        self.assertEqual([s.scene["t_s"] for s in samples], self.times)
        self.assertEqual(samples[0].scene["q_wxyz"], [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
