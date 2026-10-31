"""Export, contract and decoder-parity tests. They export random-initialized models, so nothing is trained.

The slow ones (a model export takes a few seconds) share one export per test class. The C++ comparison runs
only when a configured sleepy checkout is present (it compiles armor_infer.cpp from source).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nt.contract import (
    SLEEPY_CLASSES, SLEEPY_COLORS, check_model, check_values, compare_decoders, reference_decode, run_cpp,
)
from nt.spec import LabelSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = LabelSpec.load()
# Every yaml in configs/models: the two model families and the M5 attention variants of YOLO26n.
MODEL_YAMLS = sorted(p.stem for p in (ROOT / "configs" / "models").glob("*.yaml"))
SLEEPY_BUILT = all((Path.home() / "sleepy" / "build" / f).exists() for f in ("libsleepy_utils.so", "compile_commands.json"))


def row(number, color_slot, corners, conf=0.9):
    r = np.zeros(18, np.float32)
    r[:5] = [0, 0, 100, 50, conf]
    r[5] = number
    r[6 + color_slot] = 1.0
    for k, (x, y) in enumerate(corners):
        r[10 + 2 * k], r[11 + 2 * k] = x, y
    return r


class ReferenceDecoderTest(unittest.TestCase):
    def test_class_table_matches_label_ids(self):
        # The exported number ids are the spec's label ids, and sleepy's table must name them the same way.
        for label_id, name in SPEC.labels.items():
            self.assertEqual(SLEEPY_CLASSES[label_id], name)
        # the off class exports as sleepy's UNKNOWN with the color slot sleepy reads as NONE
        off_color, off_number, off_name = SPEC.off_class
        self.assertEqual((SLEEPY_CLASSES[off_number], off_name), ("unknown", "unknown"))
        self.assertEqual(SLEEPY_COLORS[off_color], "none")

    def test_off_armor_decodes_as_unknown_none(self):
        a = reference_decode(row(8, 2, [(20, 15), (20, 35), (80, 35), (80, 15)])[None], 0.2)[0]
        self.assertEqual((a["number"], a["color"]), ("unknown", "none"))

    def test_decodes_number_color_and_corners(self):
        corners = [(20, 15), (20, 35), (80, 35), (80, 15)]  # LT, LB, RB, RT
        a = reference_decode(row(4, 1, corners)[None], 0.2)[0]
        self.assertEqual((a["number"], a["color"]), ("four", "red"))
        self.assertEqual((a["lt"], a["lb"], a["rb"], a["rt"]), tuple(map(tuple, [corners[0], corners[1], corners[2], corners[3]])))

    def test_threshold_and_unknown_class_are_dropped(self):
        rows = np.stack([row(3, 0, [(0, 0)] * 4, conf=0.1), row(9, 0, [(0, 0)] * 4)])
        self.assertEqual(reference_decode(rows, 0.2), [])

    def test_decoder_keeps_the_network_corner_order(self):
        # A plate rolled by about 40 degrees: the old y-then-x sort turned its roles by 90 degrees.
        corners = [(789, 417), (799, 448), (912, 350), (904, 319)]
        a = reference_decode(row(3, 1, corners)[None], 0.2)[0]
        self.assertEqual((a["lt"], a["lb"], a["rb"], a["rt"]), tuple(map(tuple, corners)))


class ValueCheckTest(unittest.TestCase):
    def good(self):
        out = np.zeros((1, 30, 18), np.float32)
        out[0, :, 6] = 1.0
        return out

    def test_accepts_a_conforming_tensor(self):
        self.assertEqual(check_values(self.good(), SPEC, "x"), [])

    def test_rejects_wrong_shape_dtype_and_semantics(self):
        self.assertTrue(check_values(np.zeros((1, 30, 14), np.float32), SPEC, "shape"))
        self.assertTrue(check_values(self.good().astype(np.float64), SPEC, "dtype"))
        bad = self.good(); bad[0, 0, 4] = 1.5
        self.assertTrue(any("confidence" in p for p in check_values(bad, SPEC, "conf")))
        bad = self.good(); bad[0, 0, 5] = 2.5
        self.assertTrue(any("number" in p for p in check_values(bad, SPEC, "num")))
        bad = self.good(); bad[0, 0, 6:10] = [1, 1, 0, 0]
        self.assertTrue(any("one-hot" in p for p in check_values(bad, SPEC, "oh")))
        bad = self.good(); bad[0, 0, 6:10] = [0, 0, 0, 1]  # purple is not trained
        self.assertTrue(any("training_colors" in p for p in check_values(bad, SPEC, "purple")))
        bad = self.good(); bad[0, 0, 6:10] = [0, 0, 1, 0]  # gray (off) needs the off number 8
        self.assertTrue(any("off" in p for p in check_values(bad, SPEC, "gray")))
        ok = self.good(); ok[0, 0, 6:10] = [0, 0, 1, 0]; ok[0, 0, 5] = 8
        self.assertEqual(check_values(ok, SPEC, "off"), [])
        bad = self.good(); bad[0, 0, 12] = np.nan
        self.assertTrue(any("non-finite" in p for p in check_values(bad, SPEC, "nan")))

    def test_decoder_comparison_reports_differences(self):
        a = {"conf": 0.9, "number": "three", "color": "red", "lt": (1, 2), "lb": (1, 3), "rb": (4, 3), "rt": (4, 2)}
        self.assertEqual(compare_decoders([a], [dict(a)], "t"), [])
        self.assertTrue(compare_decoders([a], [dict(a, number="four")], "t"))
        self.assertTrue(compare_decoders([a], [dict(a, rt=(9, 9))], "t"))
        self.assertTrue(compare_decoders([a], [], "t"))


def export_random_model(name: str, out: Path):
    import torch
    from ultralytics import YOLO

    from nt import modules, offline
    from nt.export import export

    offline.apply()
    modules.register()
    m = YOLO(str(ROOT / "configs" / "models" / f"{name}.yaml"), task="pose")
    weights = out / f"{name}.pt"
    torch.save({"model": m.model.half().float(), "date": "random-init", "version": "test", "epoch": -1,
                "train_args": {}, "optimizer": None, "ema": None, "updates": 0, "best_fitness": None}, weights)
    return export(weights, out / f"{name}_onnx", SPEC)


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.cwd = os.getcwd()
        cls.paths = {n: export_random_model(n, cls.dir) for n in MODEL_YAMLS}

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls._tmp.cleanup()

    def test_every_model_yaml_conforms(self):
        for name, path in self.paths.items():
            self.assertEqual(check_model(path, SPEC), [], name)

    def test_contract_is_embedded_and_written_beside_the_model(self):
        import onnx

        for path in self.paths.values():
            meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
            embedded = json.loads(meta["nt_contract"])
            self.assertEqual(embedded, json.loads(path.with_name("contract.json").read_text()))
            self.assertEqual(embedded["label_spec_sha256"], SPEC.sha256)
            self.assertEqual(embedded["output"]["keypoint_order"], ["LT", "LB", "RB", "RT"])

    def test_remap_equals_a_numpy_computation_from_the_stock_graph(self):
        import onnxruntime as ort
        from ultralytics import YOLO

        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            m = YOLO(str(self.dir / "yolo26n-pose-armor.pt"), task="pose")
            stock_path = Path(m.export(format="onnx", imgsz=[576, 768], batch=1, half=False, dynamic=False,
                                       simplify=True, opset=17, max_det=30, nms=False, device="cpu", verbose=False))
        finally:
            os.chdir(cwd)
        stock_path = stock_path if stock_path.is_absolute() else self.dir / stock_path
        stock = ort.InferenceSession(str(stock_path), providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(self.paths["yolo26n-pose-armor"]), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(1)
        numbers = np.array([n for _, n in SPEC.classes], np.float32)
        slots = np.array([c for c, _ in SPEC.classes])
        seen = set()
        for trial in range(4):
            x = rng.random((1, 3, 576, 768), dtype=np.float32) * (0.2 + 0.25 * trial)
            s, n = stock.run(None, {"images": x})[0][0], new.run(None, {"images": x})[0][0]
            cid = s[:, 5].astype(int)
            seen |= set(cid.tolist())
            np.testing.assert_array_equal(n[:, :5], s[:, :5])
            # the number column is the label id, which has gaps (no robot 2 or 5)
            # number and color come from the class table, ids with gaps and the off class included
            np.testing.assert_array_equal(n[:, 5], numbers[cid])
            np.testing.assert_array_equal(n[:, 6:10], np.eye(4, dtype=np.float32)[slots[cid]])
            np.testing.assert_array_equal(n[:, 10:18], np.stack([s[:, [6 + 3 * k, 7 + 3 * k]] for k in range(4)], axis=1).reshape(30, 8))
        self.assertGreaterEqual(len({int(slots[c]) for c in seen}), 2, "at least two color slots must be exercised")

    @unittest.skipUnless(SLEEPY_BUILT, "needs a built sleepy checkout")
    def test_real_sleepy_decoder_agrees_with_the_python_reference(self):
        for name, path in self.paths.items():
            self.assertEqual(check_model(path, SPEC, cpp=True), [], name)

    def test_export_refuses_a_non_empty_directory(self):
        from nt.export import export

        with self.assertRaises(FileExistsError):
            export(self.dir / "yolo26n-pose-armor.pt", self.paths["yolo26n-pose-armor"].parent, SPEC)


@unittest.skipUnless(SLEEPY_BUILT, "needs a built sleepy checkout")
class RolledPlateTest(unittest.TestCase):
    """The real C++ decoder must hand back LT, LB, RB, RT as the network gave them, at any roll of the plate."""

    def test_every_roll_keeps_the_roles(self):
        rows, expected = [], []
        # sleepy's light-bar plate, plus a flatter 227 x 53 mm quad (its former large plate) where the old sort
        # failed from 13 degrees of roll on
        for w, h in ((133.0, 55.0), (227.0, 53.0)):
            for roll in range(-85, 90, 5):
                c, s = np.cos(np.radians(roll)), np.sin(np.radians(roll))
                # LT, LB, RB, RT of a plate rotated by `roll`, scaled to pixels, y pointing down
                local = [(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)]
                pts = [(400 + 0.8 * (c * x - s * y), 300 + 0.8 * (s * x + c * y)) for x, y in local]
                rows.append(row(3, 1, pts))
                expected.append(pts)
        out = np.stack(rows)
        real = run_cpp(out, 0.2)
        self.assertEqual(len(real), len(rows))
        for i, (a, pts) in enumerate(zip(real, expected)):
            got = [tuple(a[k]) for k in ("lt", "lb", "rb", "rt")]
            for g, p in zip(got, pts):
                self.assertAlmostEqual(g[0], p[0], places=2, msg=f"row {i}")
                self.assertAlmostEqual(g[1], p[1], places=2, msg=f"row {i}")
        self.assertEqual(compare_decoders(reference_decode(out, 0.2), real, "rolled"), [])


if __name__ == "__main__":
    unittest.main()
