#!/usr/bin/env python3
"""Run a serialized TensorRT engine on frames streamed through stdin; used by `nt.eval --backend trt-*`.

    NT_TRT_PYTHON=/usr/bin/python3  python3 -I scripts/trt_runner.py ENGINE

Runs under a Python that has the tensorrt bindings (on this machine the system Python 3.10 with TensorRT 10.15 from
the CUDA 13 apt packages), not in the training venv: that keeps TensorRT's CUDA runtime out of the process that
holds PyTorch's. Needs only tensorrt and numpy; device memory comes from libcudart through ctypes.

Protocol: for every frame the caller writes the engine's input tensor as raw float32 (C order), and reads back the
output tensor as raw float32. EOF on stdin ends the run. The engine must have one input and one output, float32.
"""
import ctypes
import ctypes.util
import sys

import numpy as np
import tensorrt as trt

H2D, D2H = 1, 2


def load_cudart():
    for name in ("libcudart.so.13", "libcudart.so.12", ctypes.util.find_library("cudart") or "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("libcudart not found")


def check(rc, what):
    if rc != 0:
        raise SystemExit(f"{what} failed with CUDA error {rc}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cudart = load_cudart()
    logger = trt.Logger(trt.Logger.WARNING)
    with open(sys.argv[1], "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise SystemExit("cannot deserialize the engine (built by another TensorRT version?)")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"expected one input and one output, got {inputs} and {outputs}")
    ctx = engine.create_execution_context()
    buffers = {}
    for n in names:
        if engine.get_tensor_dtype(n) != trt.DataType.FLOAT:
            raise SystemExit(f"{n} is {engine.get_tensor_dtype(n)}, expected float32")
        shape = tuple(engine.get_tensor_shape(n))
        if any(d < 0 for d in shape):
            raise SystemExit(f"{n} has a dynamic shape {shape}; export with a fixed shape")
        host = np.empty(shape, np.float32)
        ptr = ctypes.c_void_p()
        check(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(host.nbytes)), "cudaMalloc")
        ctx.set_tensor_address(n, ptr.value)
        buffers[n] = (host, ptr)
    stream = ctypes.c_void_p()
    check(cudart.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    inp, inp_ptr = buffers[inputs[0]]
    out, out_ptr = buffers[outputs[0]]
    src, dst = sys.stdin.buffer, sys.stdout.buffer
    while True:
        raw = src.read(inp.nbytes)
        if not raw:
            break
        if len(raw) != inp.nbytes:
            raise SystemExit(f"short frame: {len(raw)} of {inp.nbytes} bytes")
        inp[...] = np.frombuffer(raw, np.float32).reshape(inp.shape)
        check(cudart.cudaMemcpy(inp_ptr, inp.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(inp.nbytes), H2D), "copy in")
        if not ctx.execute_async_v3(stream.value):
            raise SystemExit("execute_async_v3 failed")
        check(cudart.cudaStreamSynchronize(stream), "synchronize")
        check(cudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), out_ptr, ctypes.c_size_t(out.nbytes), D2H), "copy out")
        dst.write(out.tobytes())
        dst.flush()
    for _, ptr in buffers.values():
        cudart.cudaFree(ptr)


if __name__ == "__main__":
    main()
