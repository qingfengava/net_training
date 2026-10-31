"""TensorRT latency of exported models with trtexec, for the M5 architecture A/B and later the robot computer.

    python -m nt.bench_trt exports/A/model.onnx exports/B/model.onnx ... [--rounds 5] [--seconds 10] [--fp32]
        [--out runs/bench/NAME.json]

Builds one engine per model (FP16 unless --fp32; kept in a temp directory), then measures the models in turn for
several rounds, so clock and temperature drift of a laptop GPU hit every model alike. Reports trtexec's GPU
compute time (the network alone) and latency (with the input upload and output download), as the median over
rounds of each round's median and 99th percentile. Weights do not change the speed, so random-init exports do.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

LINE = re.compile(r"\] (GPU Compute Time|Latency): min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, "
                  r"median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, "
                  r"percentile\(99%\) = ([\d.]+) ms")


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return res.stdout


def parse(text: str) -> dict:
    out = {}
    for m in LINE.finditer(text):
        key = "compute" if m.group(1) == "GPU Compute Time" else "latency"
        out[key] = {"median": float(m.group(5)), "p99": float(m.group(8)), "mean": float(m.group(4))}
    if set(out) != {"compute", "latency"}:
        raise RuntimeError("trtexec output has no timing summary")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", type=Path, nargs="+")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seconds", type=int, default=10, help="measurement time per model and round")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise SystemExit("trtexec not found")
    names = [m.parent.name if m.name == "model.onnx" else m.stem for m in args.models]
    if len(set(names)) != len(names):
        raise SystemExit(f"model names collide: {names}")
    precision = [] if args.fp32 else ["--fp16"]
    version = re.search(r"TensorRT v(\d+)", run([trtexec, "--help"]) or "")
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).strip()

    samples: dict[str, list[dict]] = {n: [] for n in names}
    with tempfile.TemporaryDirectory() as tmp:
        engines = {}
        for name, onnx in zip(names, args.models):
            engines[name] = Path(tmp) / f"{name}.engine"
            print(f"building {name}", flush=True)
            run([trtexec, f"--onnx={onnx}", *precision, f"--saveEngine={engines[name]}", "--skipInference"])
        for r in range(args.rounds):
            for name in names:
                t = parse(run([trtexec, f"--loadEngine={engines[name]}", "--warmUp=1000",
                               f"--duration={args.seconds}", "--useSpinWait"]))
                samples[name].append(t)
                print(f"round {r + 1} {name:28s} compute median {t['compute']['median']:.3f} ms  "
                      f"latency median {t['latency']['median']:.3f} ms", flush=True)

    summary = {n: {k: {s: round(statistics.median(x[k][s] for x in v), 4) for s in ("median", "p99")}
                   for k in ("compute", "latency")} for n, v in samples.items()}
    report = {"precision": "fp32" if args.fp32 else "fp16", "tensorrt": version.group(1) if version else None,
              "gpu": gpu, "rounds": args.rounds, "seconds": args.seconds,
              "models": {n: str(m) for n, m in zip(names, args.models)}, "summary": summary, "samples": samples}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
