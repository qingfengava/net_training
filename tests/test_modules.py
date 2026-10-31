"""Custom network modules (nt/modules.py) and the model yamls that use them."""
import unittest
from pathlib import Path

import torch

from nt.modules import CoordAtt

ROOT = Path(__file__).resolve().parents[1]


class CoordAttTest(unittest.TestCase):
    def test_keeps_the_shape_and_gates_per_position(self):
        m = CoordAtt(32).eval()
        x = torch.randn(2, 32, 9, 12)
        with torch.no_grad():
            y = m(x)
        self.assertEqual(y.shape, x.shape)
        gate = y / x
        # the gate is a product of a row and a column factor, each in (0, 1)
        self.assertTrue(bool(((gate > 0) & (gate < 1)).all()))
        self.assertFalse(torch.allclose(gate[:, :, 0, 0], gate[:, :, 8, 11]))


class ModelYamlTest(unittest.TestCase):
    def test_attention_variants_differ_only_in_layer_10(self):
        from nt import modules, offline

        offline.apply()
        modules.register()
        from ultralytics import YOLO

        layers = {}
        for name in ("yolo26n-pose-armor", "yolo26n-pose-armor-noattn", "yolo26n-pose-armor-coordatt"):
            net = YOLO(str(ROOT / "configs" / "models" / f"{name}.yaml"), task="pose").model.eval()
            with torch.no_grad():
                out = net(torch.zeros(1, 3, 576, 768))
            self.assertEqual(tuple((out[0] if isinstance(out, (list, tuple)) else out).shape), (1, 300, 18), name)
            layers[name] = [type(m).__name__ for m in net.model]
        self.assertEqual(layers["yolo26n-pose-armor"][10], "C2PSA")
        self.assertEqual(layers["yolo26n-pose-armor-noattn"][10], "Identity")
        self.assertEqual(layers["yolo26n-pose-armor-coordatt"][10], "CoordAtt")
        for name, types in layers.items():
            self.assertEqual(types[:10] + types[11:], layers["yolo26n-pose-armor"][:10] + layers["yolo26n-pose-armor"][11:], name)


if __name__ == "__main__":
    unittest.main()
