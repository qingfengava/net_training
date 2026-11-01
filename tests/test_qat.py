"""nt.qat: fake quantization that matches the exported QDQ graph, BN folding, and installation on the real model."""
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nt import qat

ROOT = Path(__file__).resolve().parents[1]


def ult_conv(c1, c2, k=3):
    from ultralytics.nn.modules.conv import Conv

    torch.manual_seed(0)
    m = Conv(c1, c2, k)
    with torch.no_grad():  # BN statistics that are not the identity, so the folding is actually tested
        m.bn.running_mean.uniform_(-0.5, 0.5)
        m.bn.running_var.uniform_(0.5, 2.0)
        m.bn.weight.uniform_(0.5, 1.5)
        m.bn.bias.uniform_(-0.2, 0.2)
    return m


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = ult_conv(3, 8)
        self.b = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return self.b(self.a(x))


class FakeQuantTest(unittest.TestCase):
    def test_straight_through_gradient_inside_the_range_only(self):
        fq = qat.FakeQuant("tensorrt")
        fq.train()
        fq(torch.tensor([-1.0, 1.0]))  # range [-1, 1]
        qat.freeze(fq)
        x = torch.tensor([-2.0, -0.3, 0.4, 3.0], requires_grad=True)
        y = fq(x)
        y.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 1.0, 1.0, 0.0])
        step = 1.0 / 127  # symmetric int8: scale = amax / 127, codes -128..127
        self.assertAlmostEqual(float(y[0].detach()), -128 * step, places=5)  # clipped at the lowest code
        self.assertAlmostEqual(float(y[3].detach()), 127 * step, places=5)
        self.assertLessEqual(abs(float(y[1].detach()) + 0.3), step / 2 + 1e-6)  # on the int8 grid

    def test_rknn_scheme_is_uint8_with_a_zero_point(self):
        fq = qat.FakeQuant("rknn")
        fq.train()
        fq(torch.tensor([0.0, 6.0]))  # a ReLU-like range
        scale, zp = fq.qparams()
        self.assertAlmostEqual(float(scale), 6.0 / 255, places=6)
        self.assertEqual(float(zp), 0.0)


class InstallTest(unittest.TestCase):
    def test_bn_folding_reproduces_conv_bn_when_quantization_is_off(self):
        net = Net().eval()
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            ref = net(x)
        q = copy.deepcopy(net)
        info = qat.install(q, "rknn")
        self.assertEqual((info["conv_bn"], info["conv"]), (1, 1))
        qat.set_enabled(q, False)
        with torch.no_grad():
            np.testing.assert_allclose(q(x).numpy(), ref.numpy(), rtol=1e-4, atol=1e-5)
        self.assertTrue(qat.is_quantized(q))
        with self.assertRaises(ValueError):
            qat.install(q, "rknn")

    def test_export_emits_qdq_and_onnxruntime_agrees_with_torch(self):
        import onnx
        import onnxruntime as ort

        for scheme in ("rknn", "tensorrt"):
            net = Net()
            qat.install(net, scheme)
            net.train()
            with torch.no_grad():
                for _ in range(3):  # observe some ranges
                    net(torch.randn(4, 3, 16, 16))
            qat.freeze(net)
            net.eval()
            x = torch.randn(1, 3, 16, 16)
            with torch.no_grad():
                ref = net(x).numpy()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "m.onnx"
                torch.onnx.export(net, x, str(path), opset_version=17, input_names=["x"], output_names=["y"], dynamo=False)
                ops = [n.op_type for n in onnx.load(str(path)).graph.node]
                self.assertGreaterEqual(ops.count("QuantizeLinear"), 3, scheme)  # 2 activations + weights
                self.assertGreaterEqual(ops.count("DequantizeLinear"), 3, scheme)
                y = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
            # same grids; tiny differences in rounding at a grid midpoint can move a value by one step
            step = float(np.abs(ref).max()) / 100
            self.assertLess(float(np.abs(y - ref).max()), step, scheme)

    def test_real_model_installs_survives_deepcopy_pickle_and_fuse(self):
        from nt import modules, offline

        offline.apply()
        modules.register()
        from ultralytics import YOLO

        cfg = str(ROOT / "configs" / "models" / "yolo26n-pose-armor.yaml")
        info = qat.install(YOLO(cfg, task="pose").model, "rknn")
        self.assertEqual(info["head_outputs"], 18)  # 9 one2one + 9 one2many head convolutions
        model = YOLO(cfg, task="pose").model
        info = qat.install(model, "tensorrt")
        self.assertGreater(info["conv_bn"], 100)
        self.assertEqual(info["head_outputs"], 0)  # float, as nt.quant's tensorrt PTQ leaves them
        model.train()
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64))
        ema = copy.deepcopy(model)
        self.assertTrue(qat.is_quantized(ema))
        buf = io.BytesIO()
        torch.save(ema, buf)
        buf.seek(0)
        back = torch.load(buf, weights_only=False)
        self.assertTrue(qat.is_quantized(back))
        back.eval()
        fused = back.fuse()
        with torch.no_grad():
            fused(torch.zeros(1, 3, 64, 64))  # forward_fuse of the quantized Conv still works


if __name__ == "__main__":
    unittest.main()
