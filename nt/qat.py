"""Quantization-aware training (QAT) for the armor pose models, in place, the way the deployed runtime quantizes.

    python -m nt.train --model runs/train/NAME/weights/last.pt --data baked/X --name NAME_qat \\
        --hyp configs/train/pose_qat.yaml --qat rknn|tensorrt [--qat-freeze 0.5]
    python -m nt.export --weights runs/train/NAME_qat/weights/last.pt --out exports/NAME_qat [--heads]

What gets fake-quantized (install()):
  * weights: int8, symmetric, per output channel. An Ultralytics Conv (conv + BN + act) is quantized with its BN
    folded in, using the BN running statistics, which stay frozen during QAT: that is exactly the weight that
    exists after nt.export fuses Conv and BN, so training sees the deployed weights.
  * activations: 8 bits per tensor at the input of every convolution. Scheme "rknn": uint8 with a zero point, as
    the RKNN SDK guide (2.2.5) recommends for QAT, and also at the output of every head convolution (the tensors
    the NPU hands to the CPU decode); scheme "tensorrt": int8 symmetric, because TensorRT's explicit quantization
    needs a zero point of 0 (RKNN also loads symmetric parameters), with the head outputs left in float as
    nt.quant's PTQ leaves them: TensorRT ends those convolutions in FP16 and rounding the class logits to 8 bits
    gains nothing there.
Ranges are a running min/max (momentum 0.05) updated in training mode and frozen by freeze() (nt.train does it
after --qat-freeze of the epochs). Gradients pass straight through inside the clip range and stop outside it.

Export: under torch.onnx export the same quantizers emit QuantizeLinear/DequantizeLinear pairs (opset >= 13) with
these scales, so the ONNX from nt.export is a QDQ model: RKNN loads it with build(do_quantization=False)
(optimization_level 2, see docs/rknn_int8_qat_notes.md), TensorRT builds it with --int8, ONNX Runtime runs it
on the CPU. nt.export turns onnxslim off for such models, because it would fold the weight Q/DQ into floats.

Not fake-quantized here: Concat, Upsample, Add, MaxPool, the attention MatMul and Softmax, a BatchNorm that is
not part of an Ultralytics Conv (the one inside CoordAtt), and the decode after the heads. The NPU quantizes some
of these by its own rules, so a QAT model still has to be checked on the target (nt.eval / rknn_check).
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

SCHEMES = {"rknn": (0, 255, False), "tensorrt": (-128, 127, True)}  # activation qmin, qmax, symmetric
W_QMIN, W_QMAX = -128, 127
MOMENTUM = 0.05
# Head convolutions whose outputs leave the NPU: box (cv2), class (cv3) and keypoints (cv4_kpts), both heads.
HEAD_CONV = re.compile(r"\.(one2one_)?(cv2\.\d+\.2|cv3\.\d+\.2|cv4_kpts\.\d+)$")


def _ste(x: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Quantize-dequantize with a straight-through gradient inside [qmin, qmax] and none outside."""
    lo, hi = (qmin - zp) * scale, (qmax - zp) * scale
    xc = torch.maximum(torch.minimum(x, hi), lo)
    xq = (torch.clamp(torch.round(x / scale + zp), qmin, qmax) - zp) * scale
    return xc + (xq - xc).detach()


class FakeQuant(nn.Module):
    """Per-tensor activation quantizer with a running min/max."""

    def __init__(self, scheme: str):
        super().__init__()
        self.qmin, self.qmax, self.symmetric = SCHEMES[scheme]
        self.register_buffer("lo", torch.zeros(()))
        self.register_buffer("hi", torch.zeros(()))
        # Float flags on purpose: Ultralytics' ModelEMA averages only floating-point state, so with bool flags the
        # EMA copy (the one that is validated and saved) would keep "not initialized" forever.
        self.register_buffer("initialized", torch.zeros(()))
        self.register_buffer("frozen", torch.zeros(()))
        self.enabled = True

    def _flag(self, buf: torch.Tensor) -> bool:
        return bool(buf > 0.5)

    @torch.no_grad()
    def _observe(self, x: torch.Tensor) -> None:
        lo, hi = x.detach().min().float(), x.detach().max().float()
        if not self._flag(self.initialized):
            self.lo.copy_(lo)
            self.hi.copy_(hi)
            self.initialized.fill_(1.0)
        else:
            self.lo.mul_(1 - MOMENTUM).add_(MOMENTUM * lo)
            self.hi.mul_(1 - MOMENTUM).add_(MOMENTUM * hi)

    def qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = torch.clamp(self.lo.float(), max=0.0), torch.clamp(self.hi.float(), min=0.0)
        if self.symmetric:
            scale = torch.clamp(torch.maximum(-lo, hi) / self.qmax, min=1e-8)
            return scale, torch.zeros_like(scale)
        scale = torch.clamp((hi - lo) / (self.qmax - self.qmin), min=1e-8)
        zp = torch.clamp(torch.round(self.qmin - lo / scale), self.qmin, self.qmax)
        return scale, zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if (self.training and not self._flag(self.frozen)) or not self._flag(self.initialized):
            self._observe(x)
        scale, zp = self.qparams()
        if torch.onnx.is_in_onnx_export():
            return torch.fake_quantize_per_tensor_affine(x, float(scale), int(zp), self.qmin, self.qmax)
        return _ste(x, scale.to(x.dtype), zp.to(x.dtype), self.qmin, self.qmax)


def fake_quant_weight(w: torch.Tensor) -> torch.Tensor:
    """int8 symmetric, one scale per output channel (dim 0)."""
    amax = w.detach().abs().flatten(1).amax(1).float().clamp(min=1e-8)
    scale = amax / W_QMAX
    if torch.onnx.is_in_onnx_export():
        s = torch.tensor(scale.tolist(), dtype=torch.float32)  # constants in the exported graph
        return torch.fake_quantize_per_channel_affine(w, s, torch.zeros(len(s), dtype=torch.int32), 0, W_QMIN, W_QMAX)
    shape = (-1,) + (1,) * (w.dim() - 1)
    s = scale.to(w.dtype).view(shape)
    return _ste(w, s, torch.zeros_like(s), W_QMIN, W_QMAX)


def _conv(x, conv: nn.Conv2d, weight, bias):
    return F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def _quantized_class(base: type, forward, forward_fuse=None) -> type:
    """A subclass of `base` that only overrides forward (and forward_fuse), so isinstance() checks still hold."""
    attrs = {"forward": forward, "__module__": __name__}
    if forward_fuse is not None:
        attrs["forward_fuse"] = forward_fuse
    return type(f"Q{base.__name__}", (base,), attrs)


def _convbn_forward(self, x):
    """Ultralytics Conv with BN folded from its running statistics (frozen), weights and input quantized."""
    x = self.nt_in(x)
    conv, bn = self.conv, self.bn
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * k.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * k + (conv.bias * k if conv.bias is not None else 0)
    return self.act(_conv(x, conv, fake_quant_weight(w) if self.nt_quant_weights else w, b))


def _convbn_forward_fused(self, x):
    """After nt.export fuses BN into conv.weight: the same quantization on the same numbers."""
    x = self.nt_in(x)
    w = self.conv.weight
    return self.act(_conv(x, self.conv, fake_quant_weight(w) if self.nt_quant_weights else w, self.conv.bias))


def _conv2d_forward(self, x):
    x = self.nt_in(x)
    y = _conv(x, self, fake_quant_weight(self.weight) if self.nt_quant_weights else self.weight, self.bias)
    out = getattr(self, "nt_out", None)
    return out(y) if out is not None else y


_CLASSES: dict[type, type] = {}
# Quantized subclasses are made on first use. Checkpoints pickle them as nt.qat.Q<Base>; the module __getattr__
# below rebuilds the same class when a fresh process unpickles one.
_BASES = {"QConv": ("ultralytics.nn.modules.conv", "Conv"), "QDWConv": ("ultralytics.nn.modules.conv", "DWConv"),
          "QConv2d": ("torch.nn", "Conv2d")}


def _class_for(base: type) -> type:
    if base not in _CLASSES:
        from ultralytics.nn.modules.conv import Conv

        if issubclass(base, Conv):
            cls = _quantized_class(base, _convbn_forward, _convbn_forward_fused)
        elif base is nn.Conv2d:
            cls = _quantized_class(base, _conv2d_forward)
        else:
            raise TypeError(f"no quantized version of {base.__name__}")
        _CLASSES[base] = cls
        globals()[cls.__name__] = cls
    return _CLASSES[base]


def __getattr__(name: str):
    if name in _BASES:
        import importlib

        mod, base = _BASES[name]
        return _class_for(getattr(importlib.import_module(mod), base))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _swap(m: nn.Module) -> None:
    m.__class__ = _class_for(type(m))


def install(model: nn.Module, scheme: str) -> dict:
    """Put fake quantization into an Ultralytics pose model in place. Returns what was quantized."""
    from ultralytics.nn.modules.conv import Conv

    if scheme not in SCHEMES:
        raise ValueError(f"scheme {scheme!r}: one of {sorted(SCHEMES)}")
    if is_quantized(model):
        raise ValueError("the model is quantized already")
    inner = set()
    n_convbn = n_conv = n_head = 0
    for _, m in list(model.named_modules()):
        if isinstance(m, Conv) and isinstance(getattr(m, "bn", None), nn.BatchNorm2d):
            inner.add(id(m.conv))
            m.nt_in = FakeQuant(scheme)
            m.nt_quant_weights = True
            _swap(m)
            n_convbn += 1
    for name, m in list(model.named_modules()):
        if type(m) is nn.Conv2d and id(m) not in inner:
            m.nt_in = FakeQuant(scheme)
            m.nt_quant_weights = True
            if scheme == "rknn" and HEAD_CONV.search("." + name):
                m.nt_out = FakeQuant(scheme)
                n_head += 1
            _swap(m)
            n_conv += 1
    info = {"scheme": scheme, "conv_bn": n_convbn, "conv": n_conv, "head_outputs": n_head,
            "weights": "int8 symmetric per output channel",
            "activations": {"rknn": "uint8 asymmetric per tensor", "tensorrt": "int8 symmetric per tensor"}[scheme]}
    model.nt_qat = info
    return info


def is_quantized(model: nn.Module) -> bool:
    return any(isinstance(m, FakeQuant) for m in model.modules())


def describe(model: nn.Module) -> dict | None:
    return getattr(model, "nt_qat", None) if is_quantized(model) else None


def freeze(model: nn.Module) -> int:
    """Stop updating the activation ranges. Returns how many quantizers were frozen."""
    n = 0
    for m in model.modules():
        if isinstance(m, FakeQuant):
            m.frozen.fill_(1.0)
            n += 1
    return n


def set_enabled(model: nn.Module, enabled: bool) -> None:
    """Switch fake quantization off (or on) everywhere, e.g. to compare with the float network."""
    for m in model.modules():
        if isinstance(m, FakeQuant):
            m.enabled = enabled
        if hasattr(m, "nt_quant_weights"):
            m.nt_quant_weights = enabled


def make_trainer(scheme: str, freeze_epoch: int):
    """An Ultralytics PoseTrainer that installs fake quantization right after the weights are loaded (so the EMA
    copy and the optimizer see the quantized model) and freezes the ranges at freeze_epoch."""
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.utils import LOGGER

    class QATPoseTrainer(PoseTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if not is_quantized(model):
                info = install(model, scheme)
                LOGGER.info(f"nt.qat: {info}")
            return model

        def _do_train(self, *args, **kwargs):
            def maybe_freeze(trainer):
                if trainer.epoch == freeze_epoch:
                    n = freeze(trainer.model) + (freeze(trainer.ema.ema) if trainer.ema else 0)
                    LOGGER.info(f"nt.qat: activation ranges frozen at epoch {trainer.epoch} ({n} quantizers)")
            self.add_callback("on_train_epoch_start", maybe_freeze)
            return super()._do_train(*args, **kwargs)

    return QATPoseTrainer
