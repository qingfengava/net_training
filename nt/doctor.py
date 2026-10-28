"""Read-only readiness check for the day the official armor style is published.

    python -m nt.doctor

It changes nothing and trains nothing. It reports, link by link, whether the pipeline can go from "new style"
to "trained model": environment, label spec, tests, the simulator exporter, the contract tools, and what is
still undecided. Anything marked TODO is a decision or an edit that needs a person; FAIL means something is
broken. The exit code is 1 only for FAIL.
"""
from __future__ import annotations

import importlib
import json
import shutil
from collections import Counter
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = Path.home() / "bevy_robomaster_simulator"
SLEEPY = Path.home() / "sleepy"

results: list[tuple[str, str, str]] = []


def add(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check_environment() -> None:
    try:
        import torch

        arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        add("OK" if torch.cuda.is_available() else "FAIL", "torch + CUDA",
            f"{torch.__version__}, {torch.cuda.get_device_name(0) if arch else 'no GPU'}")
    except Exception as e:  # noqa: BLE001
        add("FAIL", "torch import", repr(e))
    for mod in ("ultralytics", "onnx", "onnxruntime", "cv2", "yaml"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            add("FAIL", f"import {mod}", repr(e))
    lock = (ROOT / "requirements.lock").read_text() if (ROOT / "requirements.lock").exists() else ""
    add("OK" if "--hash=" in lock else "TODO", "requirements.lock has hashes",
        "" if "--hash=" in lock else "plain freeze only; add hashes before running on a rented machine")


def check_spec() -> None:
    from nt.spec import LabelSpec

    spec = LabelSpec.load()
    add("OK", "label_spec loads", f"{spec.n_classes} training classes, sha {spec.sha256[:10]}")
    if spec.corner_definition == "pending_official_style":
        add("TODO", "corner_definition", "still 'pending_official_style': decide what a corner is once the style is published")
    else:
        add("OK", "corner_definition", spec.corner_definition)
    add("OK" if list(spec.types) == ["small"] else "TODO", "armor sizes",
        f"types {list(spec.types)}" + ("" if list(spec.types) == ["small"] else "; RM27 has small armor modules only"))


def check_data_and_tools() -> None:
    from nt.canonical import read_manifest

    datasets = sorted(p for p in (ROOT / "datasets").glob("*/*") if (p / "manifest.json").exists())
    if not datasets:
        add("TODO", "canonical datasets", "none under datasets/")
    for d in datasets:
        m = read_manifest(d / "manifest.json")
        from nt.spec import LabelSpec

        stale = m.label_spec_sha256 != LabelSpec.load().sha256
        add("TODO" if stale else "OK", f"dataset {d.relative_to(ROOT)}",
            f"{m.counts.get('accepted_samples')} samples, corner_definition={m.corner_definition}"
            + ("; label_spec changed since ingest, re-ingest" if stale else ""))
    baked = sorted(p for p in (ROOT / "baked").glob("*") if (p / "bake_manifest.json").exists())
    for b in baked:
        add("OK", f"baked {b.name}", json.dumps(json.loads((b / "bake_manifest.json").read_text())["images"]))
    add("OK" if (ROOT / "scripts" / "pack_for_cloud.sh").exists() else "FAIL", "cloud packing script")


def check_real_data() -> None:
    labelme = ROOT / ".venv-label" / "bin" / "labelme"
    add("OK" if labelme.exists() else "TODO", "labelme for annotation", str(labelme) if labelme.exists() else "scripts/setup_labelme.sh")
    sessions = sorted(p for p in (ROOT / "sessions").glob("*") if (p / "session.json").exists())
    roles = Counter(json.loads((p / "session.json").read_text())["role"] for p in sessions)
    add("OK" if sessions else "TODO", "real capture sessions",
        ", ".join(f"{n} {r}" for r, n in sorted(roles.items())) if sessions else "none yet: record with the robot camera, then nt.session create")
    real = [read_manifest_safe(p) for p in sorted((ROOT / "datasets" / "real").glob("*/manifest.json"))]
    val = sum(1 for m in real if m and m.get("role") == "val")
    add("OK" if val else "TODO", "real_val dataset", f"{val} val datasets" if val else
        "none: the YOLO26n vs YOLOv8n decision and every model gate need a real val session")
    no_cam = [p.name for p in sessions if json.loads((p / "session.json").read_text()).get("camera") is None]
    if no_cam:
        add("TODO", "session intrinsics", f"no camera for {no_cam[:5]}: nt.replay cannot compute PnP jitter on them")


def read_manifest_safe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_simulator() -> None:
    if not SIM.exists():
        add("TODO", "Daedalus checkout", f"{SIM} not found")
        return
    binary = SIM / "target" / "release" / "daedalus"
    add("OK" if binary.exists() else "TODO", "Daedalus release binary", str(binary) if binary.exists() else "cargo build --release")
    status = subprocess.run(["git", "-C", str(SIM), "status", "--short", "src/dataset", "src/main.rs"], capture_output=True, text=True).stdout
    add("TODO" if status.strip() else "OK", "exporter changes committed", "src/dataset and src/main.rs have uncommitted changes" if status.strip() else "")
    screen_code = (SIM / "src" / "robomaster" / "armor" / "screen.rs").exists()
    specs = sorted((ROOT / "configs" / "screens").glob("*.toml"))
    add("OK" if screen_code and specs else "TODO", "simulator dot-matrix armor",
        f"style switch in Daedalus, specs: {[p.stem for p in specs]}" if screen_code and specs else "M7: screen style or spec missing")
    official = [p.stem for p in specs if "official" in p.stem]
    add("OK" if official else "TODO", "official screen spec", official[0] if official else
        "only hypothesis specs: copy configs/screens/rm27_hypothesis_a.toml to rm27_official_v1.toml once the style is published")


def check_contract_tools() -> None:
    add("OK" if (ROOT / "nt" / "export.py").exists() and (ROOT / "nt" / "contract.py").exists() else "FAIL", "export + contract modules")
    built = all((SLEEPY / "build" / f).exists() for f in ("libsleepy_utils.so", "compile_commands.json"))
    add("OK" if built else "TODO", "sleepy configured for decoder parity", "" if built else f"build {SLEEPY} to run the C++ parity check")
    binary = ROOT / "build" / "cpp_parity" / "armor_decode"
    add("OK" if binary.exists() else "TODO", "C++ parity harness built", "" if binary.exists() else "scripts/run_cpp_parity.sh build")
    source = SLEEPY / "src" / "tasks" / "auto_aim" / "armor_detect" / "armor_infer.cpp"
    resorts = source.exists() and "std::sort(pts.begin()" in source.read_text()
    add("TODO" if resorts else "OK", "sleepy AT decoder keeps the network corner order",
        "armor_infer.cpp still re-derives LT/RT/LB/RB by sorting y then x; wrong for about 12% of simulator armors "
        "(60% at 30-45 deg of roll)" if resorts else "the AT decoder assigns keypoints 0..3 as LT, LB, RB, RT (checked by a source grep; "
        "tests/test_export.py runs the real decoder)")
    add("TODO", "sleepy light-bar logic",
        "tracker UVL factors, ArmorTypeTraits dimensions, detect_lights/correct_corners and number warp assume light bars")


def check_disk_and_memory() -> None:
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    add("OK" if free_gb > 40 else "TODO", "free disk", f"{free_gb:.0f} GB")
    avail = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            avail = int(line.split()[1]) / 2**20
    add("OK" if avail >= 6 else "TODO", "available memory", f"{avail:.1f} GB; heavy jobs must go through scripts/run_capped.sh")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    for fn in (check_environment, check_spec, check_data_and_tools, check_real_data, check_simulator, check_contract_tools,
               check_disk_and_memory):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a checker that crashes is itself a finding
            add("FAIL", fn.__name__, repr(e))
    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        print(f"{status:4s} {name:{width}s}  {detail}")
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("OK", "TODO", "FAIL")}
    print(f"\n{counts['OK']} ok, {counts['TODO']} todo, {counts['FAIL']} fail")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
