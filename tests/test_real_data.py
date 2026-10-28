"""M6 end to end on synthetic data: session -> pre-labels -> labelme project -> import; roles in bake; mining;
proxy labeling; replay. Nothing is trained; the models are fixed-output ONNX graphs from m6_fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from m6_fixtures import FRAME_H, FRAME_W, PLATE_FRAME, PLATE_NET, SECOND_NET, det_row, fake_model, write_frames
from nt import annotate, mine, proxy, replay
from nt import session as sess
from nt.bake import bake, load_config, split_of
from nt.canonical import Camera, read_manifest, read_samples
from nt.detector import ArmorModel, letterbox
from nt.spec import LabelSpec

SPEC = LabelSpec.load()
ROOT = Path(__file__).resolve().parents[1]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def session(self, name="s1", role="train", n=6, camera=None):
        src = self.tmp / f"src_{name}"
        write_frames(src, n)
        return sess.create(src, self.tmp / "sessions", name, role, camera_path=camera, dedupe_hamming=-1)


class SessionTest(Base):
    def test_create_is_immutable_and_records_frames(self):
        out = self.session(n=5)
        meta = sess.load(out)
        self.assertEqual((meta["role"], len(meta["frames"]), meta["width"], meta["height"]), ("train", 5, FRAME_W, FRAME_H))
        self.assertTrue(all((out / f["image"]).exists() for f in meta["frames"]))
        with self.assertRaises(FileExistsError):
            self.session(n=5)

    def test_camera_from_a_sleepy_config_and_size_check(self):
        cfg = self.tmp / "cam.yaml"
        cfg.write_text(
            "camera:\n  camera_info:\n    image_width: 256\n    image_height: 192\n"
            "    camera_matrix: {rows: 3, cols: 3, data: [300, 0, 128, 0, 301, 96, 0, 0, 1]}\n"
            "    distortion_model: plumb_bob\n    distortion_coefficients: {rows: 1, cols: 5, data: [-0.1, 0.05, 0, 0, 0]}\n")
        meta = sess.load(self.session(camera=cfg))
        self.assertEqual(meta["camera"]["fx"], 300.0)
        self.assertEqual(meta["camera"]["dist"][:2], [-0.1, 0.05])
        bad = self.tmp / "cam2.yaml"
        bad.write_text(cfg.read_text().replace("image_width: 256", "image_width: 1440"))
        with self.assertRaises(ValueError):
            self.session(name="s2", camera=bad)

    def test_video_source_and_duplicate_filter(self):
        path = self.tmp / "v.avi"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (FRAME_W, FRAME_H))
        frames = write_frames(self.tmp / "vf", 4)
        for p in frames:
            img = cv2.imread(str(p))
            vw.write(img)
            vw.write(img)  # every frame twice
        vw.release()
        out = sess.create(path, self.tmp / "sessions", "vid", "val", dedupe_hamming=2)
        meta = sess.load(out)
        self.assertEqual(meta["selection"]["kept"], 4)
        self.assertEqual(meta["selection"]["dropped_near_duplicates"], 4)
        self.assertTrue(any("lossy" in w for w in meta["warnings"]))
        self.assertEqual(meta["frames"][1]["t_s"], 0.2)


class DetectorTest(Base):
    def test_letterbox_and_both_layouts_map_back_to_frame_pixels(self):
        img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        _, s, pl, pt = letterbox(img, 128, 96)
        self.assertEqual((s, pl, pt), (0.5, 0, 0))
        for kind, cls, want in (("praysky", 1, (3, "small")), ("nt_contract", 5, (5, None))):
            m = ArmorModel(fake_model(self.tmp / f"{kind}.onnx", [det_row(0.9, cls, 1, PLATE_NET)], kind))
            (d,) = m.detect(img, 0.25)
            self.assertEqual((d.label, d.type, d.color), (*want, 1))
            np.testing.assert_allclose(d.corners, np.array(PLATE_FRAME), atol=1e-4)

    def test_praysky_rm26_large_class_has_no_type_and_tiles_do_not_duplicate(self):
        img = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        m = ArmorModel(fake_model(self.tmp / "p.onnx", [det_row(0.9, 2, 0, PLATE_NET)]))  # class 2 is s1_o1
        (d,) = m.detect(img, 0.25)
        self.assertEqual((d.label, d.type), (1, None))
        # tiles see the same constant output at other offsets; the merge keeps one detection per place
        self.assertLessEqual(len(m.detect(img, 0.25, tiles=2)), 5)


class AnnotationLoopTest(Base):
    def test_export_review_import_and_refusals(self):
        s = self.session(n=4)
        rows = [det_row(0.9, 1, 1, PLATE_NET), det_row(0.3, 0, 0, SECOND_NET)]  # one plain, one "?" pre-label
        model = fake_model(self.tmp / "m.onnx", rows)
        proj = self.tmp / "annot" / "p"
        info = annotate.export(proj, SPEC, session_dir=s, model_path=model, batch_size=3)
        self.assertEqual((info["batches"], info["counts"]["pre_labels"], info["counts"]["unresolved"]), (2, 8, 4))

        with self.assertRaises(ValueError):  # nothing reviewed yet
            annotate.import_project(proj, self.tmp / "ds" / "p_v1", SPEC)
        problems, counts = annotate.frame_problems(proj, SPEC, include_unreviewed=True)
        self.assertEqual(sum(1 for _, c, _ in problems if c == "E_LM_UNRESOLVED"), 4)

        jsons = sorted(j for j in proj.glob("batch_*/*.json") if j.name != "batch.json")
        for k, j in enumerate(jsons):
            d = json.loads(j.read_text())
            d["flags"]["reviewed"] = True
            if k < 3:  # resolve the "?" by deleting it; the last frame keeps it and must be refused
                d["shapes"] = [x for x in d["shapes"] if not x["label"].startswith("?")]
            j.write_text(json.dumps(d))
        m = annotate.import_project(proj, self.tmp / "ds" / "p_v1", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["counts"]["rejected_samples"]), (3, 1))
        self.assertEqual(m["rejected"], {"E_LM_UNRESOLVED": 1})
        self.assertEqual(m["role"], "train")
        samples = list(read_samples(self.tmp / "ds" / "p_v1" / "labels.jsonl"))
        # RM27 has small armor only, so every labelme armor comes back small
        self.assertEqual({(a.color, a.label, a.type) for x in samples for a in x.armors}, {(1, 3, "small")})
        np.testing.assert_allclose(samples[0].armors[0].corners, np.array(PLATE_FRAME), atol=1e-2)
        self.assertEqual(samples[0].scene["session"], "s1")

    def test_neg_session_refuses_labeled_frames(self):
        s = self.session(name="n1", role="neg", n=2)
        proj = self.tmp / "annot" / "n"
        annotate.export(proj, SPEC, session_dir=s, model_path=fake_model(self.tmp / "m.onnx", [det_row(0.9, 1, 1, PLATE_NET)]))
        for j in proj.glob("batch_*/*.json"):
            if j.name != "batch.json":
                d = json.loads(j.read_text())
                d["flags"]["reviewed"] = True
                j.write_text(json.dumps(d))
        m = annotate.import_project(proj, self.tmp / "ds" / "n_v1", SPEC)
        self.assertEqual(m["rejected"], {"E_NEG_HAS_ARMOR": 2})

    def test_re_review_export_keeps_labels_and_flags_unlabeled_detections(self):
        s = self.session(n=2)
        proj = self.tmp / "annot" / "p"
        annotate.export(proj, SPEC, session_dir=s, model_path=fake_model(self.tmp / "m.onnx", [det_row(0.9, 1, 1, PLATE_NET)]))
        for j in proj.glob("batch_*/*.json"):
            if j.name != "batch.json":
                d = json.loads(j.read_text())
                d["flags"]["reviewed"] = True
                j.write_text(json.dumps(d))
        annotate.import_project(proj, self.tmp / "ds" / "v1", SPEC)
        frames = self.tmp / "frames.txt"
        frames.write_text("s1_000000  # comment\n")
        model2 = fake_model(self.tmp / "m2.onnx", [det_row(0.9, 1, 1, PLATE_NET), det_row(0.9, 0, 0, SECOND_NET)])
        info = annotate.export(self.tmp / "annot" / "r", SPEC, dataset_dir=self.tmp / "ds" / "v1", model_path=model2, frames_file=frames)
        self.assertEqual((info["frames"], info["counts"]["existing_armors"], info["counts"]["unresolved"]), (1, 1, 1))


class RolesInBakeTest(Base):
    def test_only_train_and_auto_reach_training(self):
        self.assertEqual({split_of(f"x{i}", 0.5, "val") for i in range(50)}, {"val"})
        self.assertEqual({split_of(f"x{i}", 0.5, "train") for i in range(50)}, {"train"})
        self.assertEqual({split_of(f"x{i}", 0.5, "neg") for i in range(50)}, {"val"})
        self.assertEqual({split_of(f"x{i}", 0.5, "auto") for i in range(50)}, {"train", "val"})

    def test_val_session_is_baked_once_into_val(self):
        s = self.session(name="v1", role="val", n=3)
        proj = self.tmp / "annot" / "v"
        annotate.export(proj, SPEC, session_dir=s, model_path=fake_model(self.tmp / "m.onnx", [det_row(0.9, 1, 1, PLATE_NET)]))
        for j in proj.glob("batch_*/*.json"):
            if j.name != "batch.json":
                d = json.loads(j.read_text())
                d["flags"]["reviewed"] = True
                j.write_text(json.dumps(d))
        annotate.import_project(proj, self.tmp / "ds" / "v_v1", SPEC)
        self.assertEqual(read_manifest(self.tmp / "ds" / "v_v1" / "manifest.json").role, "val")
        cfg = load_config(override={"size": [128, 96], "roi": {"full_frame_prob": 1.0}})
        m = bake([(self.tmp / "ds" / "v_v1", 2)], self.tmp / "baked", SPEC, cfg, copies=3, seed=0, workers=1,
                 spec_path=str(ROOT / "configs" / "label_spec.yaml"))
        self.assertEqual(m["images"], {"val": 3})
        self.assertEqual(m["sources"][0]["role"], "val")


class MiningTest(unittest.TestCase):
    def test_flicker_flip_and_uncertain(self):
        from nt.detector import Detection

        plate = np.array(PLATE_FRAME)

        def d(conf=0.9, label=3, shift=0.0):
            return Detection(conf, 1, label, plate + shift)

        frames = [{"id": f"f{i}", "index": i} for i in range(5)]
        dets = [[d()], [], [d()], [d(label=4)], [d(conf=0.3)]]
        rows = mine.score_session(frames, dets, SPEC, FRAME_W, FRAME_H, 0.5, 0.5)
        self.assertEqual(rows[1]["reasons"], {"flicker": 1})
        self.assertEqual(rows[3]["reasons"].get("flip"), 1)
        self.assertEqual(rows[4]["reasons"].get("uncertain"), 1)
        chosen = mine.select(rows, top=2, min_gap=2)
        self.assertEqual(len(chosen), 2)
        self.assertGreaterEqual(abs(chosen[0]["index"] - chosen[1]["index"]), 2)


class ProxyTest(Base):
    def test_render_and_label_a_warped_view(self):
        meta = proxy.render(self.tmp / "px", SPEC, "three", "red", display=(960, 540))
        disp = cv2.imread(str(self.tmp / "px" / "proxy.png"))
        W, H = 960, 720
        src = np.float32([[0, 0], [959, 0], [959, 539], [0, 539]])
        dst = np.float32([[150, 140], [820, 180], [800, 560], [170, 590]])
        Ht = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(disp, Ht, (W, H), flags=cv2.INTER_LINEAR, borderValue=(20, 20, 20))
        gt = cv2.perspectiveTransform((np.array(meta["panel_corners"], float) - 0.5).reshape(-1, 1, 2), Ht).reshape(-1, 2) + 0.5
        corners, info = proxy.ProxyLabeler(meta).label(img)
        self.assertEqual(info["markers"], 4)
        self.assertLess(float(np.abs(corners - gt).max()), 0.5)

    def test_label_session_ingests_with_the_session_role(self):
        meta = proxy.render(self.tmp / "px", SPEC, "three", "red", display=(960, 540))
        disp = cv2.imread(str(self.tmp / "px" / "proxy.png"))
        src = self.tmp / "frames"
        src.mkdir()
        for i in range(3):
            Ht = cv2.getPerspectiveTransform(np.float32([[0, 0], [959, 0], [959, 539], [0, 539]]),
                                             np.float32([[100 + 10 * i, 100], [860, 120], [850, 600], [110, 620]]))
            cv2.imwrite(str(src / f"{i}.png"), cv2.warpPerspective(disp, Ht, (960, 720), borderValue=(20, 20, 20)))
        cv2.imwrite(str(src / "3.png"), np.zeros((720, 960, 3), np.uint8))  # no markers
        s = sess.create(src, self.tmp / "sessions", "px1", "hard", dedupe_hamming=-1)
        m = proxy.label_session(s, self.tmp / "px" / "proxy.json", self.tmp / "ds" / "px1", SPEC)
        self.assertEqual((m["counts"]["accepted_samples"], m["role"], m["proxy_skipped"]), (3, "hard", {"no_markers": 1}))
        a = next(read_samples(self.tmp / "ds" / "px1" / "labels.jsonl")).armors[0]
        self.assertEqual((a.color, a.label), (meta["color"], meta["label"]))


class ReplayTest(Base):
    def test_pnp_matches_a_known_pose_and_tracks_link(self):
        from nt.detector import Detection

        geometry = replay.load_geometry(ROOT / "configs" / "geometry.yaml", SPEC)
        K = np.array([[300.0, 0, 128], [0, 300.0, 96], [0, 0, 1]])
        obj = replay.object_points(0.133, 0.055)
        # plate faces the camera: its x (forward) is the camera's +z, its y (left) is -x, its z (up) is -y
        R = np.column_stack([(0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)])
        rvec = cv2.Rodrigues(R)[0]
        img_pts, _ = cv2.projectPoints(obj, rvec, np.array([0.0, 0.0, 2.0]), K, None)
        d = Detection(0.9, 1, 3, img_pts.reshape(4, 2) + 0.5)
        p = replay.solve(d, K, np.zeros(5), geometry)
        self.assertAlmostEqual(p["dist_m"], 2.0, places=3)
        self.assertLess(abs(p["yaw_deg"]), 1.0)
        tracks = replay.link_tracks([[d], [d], [], [d]])
        self.assertEqual([len(t) for t in tracks], [3])


if __name__ == "__main__":
    unittest.main()
