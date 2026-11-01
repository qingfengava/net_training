"""Train on a rented GPU: pack a job here, run it there, take the results back here.

    python -m nt.cloud pack --name JOB --data datasets/sim/X[:REPEAT] [--data ...] [--models yolo26n-pose-armor[,..]]
                            [--seeds 0[,1]] [--epochs 100] [--hyp configs/train/pose_cloud.yaml] [--copies 2]
                            [--init runs/train/R/weights/last.pt] [--bake-limit N] [--fraction F] [--out cloud]
    python -m nt.cloud run job.json [--skip-tests] [--skip-smoke] [--dry-run]       (on the rented machine)
    python -m nt.cloud receive JOB_results.tgz [--dest .]                            (back here)

scripts/pack_for_cloud.sh, scripts/cloud_run.sh and scripts/receive_cloud_results.sh wrap these.

pack writes cloud/JOB/: the code (nt_code.tar.gz), one tar per canonical dataset (already ingested and checked
here; baking happens on the rented machine, where it is fast and its config can still change), job.json (what to
bake and train, the git commit, every hash), cloud_run.sh, README.md and SHA256SUMS. Nothing is uploaded.

run, inside the unpacked job directory after scripts/setup_env.sh: records the GPU and driver, runs the unit
tests and nt.smoke_m0, checks every dataset against the label spec, bakes, then for each model and seed trains,
evaluates, exports and checks the contract (the C++ parity needs sleepy, so it runs at receive). Steps that are
already complete are skipped, so after an interruption the same command continues. One failed run does not
stop the others. Ends with results/JOB_results.tgz and its .sha256.

receive verifies the archive, unpacks it without overwriting anything (and without trusting member paths), and
runs nt.contract --cpp on every exported model when a built sleepy is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB_VERSION = 1
CODE_PATHS = ["nt", "configs", "scripts", "tests", "docs", "export", "requirements.in", "requirements-nodeps.in",
              "requirements.lock", ".gitignore"]
MIN_DRIVER = (570, 26)  # CUDA 12.8 GA on Linux x86_64 (NVIDIA CUDA 12.8 release notes, table 3)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_info() -> dict:
    def git(*a):
        r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None  # porcelain lines start with a space
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": git("rev-parse", "HEAD"), "dirty": dirty.splitlines() if dirty else []}


# ---- pack ---------------------------------------------------------------------------------------------------

def _no_pycache(info: tarfile.TarInfo):
    return None if "__pycache__" in info.name or info.name.endswith(".pyc") else info


def pack(name: str, data: list[str], models: list[str], seeds: list[int], epochs: int, hyp: str, copies: int,
         out_root: Path, init: Path | None = None, bake_limit: int = 0, fraction: float | None = None,
         bake_config: str = "configs/bake/default.yaml") -> Path:
    from nt.canonical import read_manifest
    from nt.spec import LabelSpec

    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"job name {name!r}: letters, digits, '_' and '-' only")
    out = out_root / name
    if out.exists():
        raise FileExistsError(f"{out} exists; packs are never overwritten, use a new --name")
    spec = LabelSpec.load()
    for rel in (hyp, bake_config, *[f"configs/models/{m}.yaml" for m in models]):
        if not (ROOT / rel).exists():
            raise FileNotFoundError(f"{rel} does not exist in this checkout")
    from nt.train import check_hyp, load_hyp

    check_hyp(load_hyp(ROOT / hyp))
    datasets = []
    for i, item in enumerate(data):
        path_s, _, rep = item.partition(":")
        path = Path(path_s)
        man = read_manifest(path / "manifest.json")
        if man.label_spec_sha256 != spec.sha256:
            raise ValueError(f"{path} was ingested with another label_spec; re-ingest it before packing")
        arc = f"datasets/{man.source}/{path.resolve().name}"
        if any(d["path"] == arc for d in datasets):
            raise ValueError(f"two datasets would unpack to {arc}")
        datasets.append({"path": arc, "repeat": int(rep) if rep else 1, "archive": f"data_{i:02d}_{path.resolve().name}.tar",
                         "role": man.role, "samples": man.counts.get("accepted_samples"),
                         "manifest_sha256": _sha256(path / "manifest.json"), "_local": str(path)})
    git = _git_info()
    if git["dirty"]:
        print(f"warning: {len(git['dirty'])} uncommitted code files are packed as they are; job.json lists them", file=sys.stderr)

    tmp = out_root / f".{name}.tmp"
    if tmp.exists():
        raise FileExistsError(f"{tmp} is left over from a failed pack; remove it")
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(tmp / "nt_code.tar.gz", "w:gz") as t:
            for rel in CODE_PATHS:
                t.add(ROOT / rel, arcname=rel, filter=_no_pycache)
        for d in datasets:
            with tarfile.open(tmp / d["archive"], "w") as t:
                t.add(d.pop("_local"), arcname=d["path"])
        init_rel = None
        if init is not None:
            (tmp / "init").mkdir()
            # runs/train/R/weights/last.pt -> init/R-last.pt, so the run names say what they started from
            init_rel = f"init/{init.parent.parent.name}-{init.stem}.pt" if init.parent.name == "weights" else f"init/{init.name}"
            shutil.copy2(init, tmp / init_rel)
        shutil.copy2(ROOT / "scripts" / "cloud_run.sh", tmp / "cloud_run.sh")
        job = {
            "job_version": JOB_VERSION, "name": name, "created_utc": _utc(), "label_spec_sha256": spec.sha256,
            "code": {"archive": "nt_code.tar.gz", "git_commit": git["commit"], "git_dirty": git["dirty"]},
            "datasets": datasets,
            "bake": {"config": bake_config, "copies": copies, "seed": 0, "workers": 0, "limit": bake_limit},
            "train": {"models": [init_rel] if init_rel else models, "seeds": seeds, "epochs": epochs, "hyp": hyp,
                      "workers": None, "batch": None, "fraction": fraction},
        }
        (tmp / "job.json").write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n")
        (tmp / "README.md").write_text(_readme(job))
        files = sorted(p for p in tmp.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (tmp / "SHA256SUMS").write_text("".join(f"{_sha256(p)}  ./{p.relative_to(tmp)}\n" for p in files))
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


def _readme(job: dict) -> str:
    name = job["name"]
    runs = len(job["train"]["models"]) * len(job["train"]["seeds"])
    return f"""# 云端训练任务 {name}

{len(job['datasets'])} 个数据集，{runs} 次训练，每次 {job['train']['epochs']} 轮，超参数 `{job['train']['hyp']}`。
代码版本 {job['code']['git_commit'] or '未知'}{'（含未提交改动）' if job['code']['git_dirty'] else ''}。

## 在租用的机器上

需要：NVIDIA 驱动 >= 570.26（CUDA 12.8），能访问 PyPI 和 download.pytorch.org（或用 NT_PIP_INDEX 指定镜像），
Python 3.10（或用 NT_PYTHON 指定），内存 32 GB 以上（不够时会自动减少数据加载进程）。

    # 把整个目录传上去，然后在 tmux 或 nohup 里跑，断开 SSH 不会中断
    cd {name}
    nohup bash cloud_run.sh > cloud_run.log 2>&1 &
    tail -f cloud_run.log

中断后再执行同一条命令会接着跑，已完成的步骤会跳过。结束后下载 `results/{name}_results.tgz` 和同名的 `.sha256`。

## 回到本机

    scripts/receive_cloud_results.sh {name}_results.tgz

它会校验、解包到 runs/、exports/（不覆盖已有内容），并对每个导出的模型跑 sleepy 的 C++ 对拍。
"""


# ---- run ----------------------------------------------------------------------------------------------------

def _memory_limit_gb() -> float:
    """The smaller of physical memory and the tightest cgroup limit on the way up from this process."""
    limit = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            limit = int(line.split()[1]) / 2**20
    try:
        rel = next(ln.split("::", 1)[1] for ln in Path("/proc/self/cgroup").read_text().splitlines() if ln.startswith("0::"))
        d = Path("/sys/fs/cgroup") / rel.lstrip("/")
        while True:
            f = d / "memory.max"
            if f.exists() and f.read_text().strip() != "max":
                limit = min(limit, int(f.read_text()) / 2**30)
            if d == Path("/sys/fs/cgroup") or d == d.parent:
                break
            d = d.parent
    except (OSError, StopIteration, ValueError):
        pass
    return float(limit or 0.0)


class Runner:
    def __init__(self, job_path: Path, dry_run: bool = False):
        self.job_path = job_path.resolve()
        self.base = self.job_path.parent
        self.job = json.loads(self.job_path.read_text())
        if self.job.get("job_version") != JOB_VERSION:
            raise SystemExit(f"job_version {self.job.get('job_version')} is not {JOB_VERSION}")
        self.name = self.job["name"]
        self.dry = dry_run
        self.dir = self.base / "runs" / "cloud" / self.name
        self.logs = self.dir / "logs"
        self.status: dict = {"job": self.name, "steps": {}}
        self.baked = f"baked/{self.name}"

    # every step is a command line, so --dry-run can print the whole plan
    def step(self, key: str, cmd: list[str], done: bool = False) -> bool:
        if done:
            self.status["steps"][key] = {"status": "skipped (already done)"}
            print(f"[skip] {key}")
            return True
        print(f"[run ] {key}: {' '.join(cmd)}", flush=True)
        if self.dry:
            self.status["steps"][key] = {"status": "dry-run", "cmd": cmd}
            return True
        self.logs.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(self.logs / f"{len(self.status['steps']):02d}_{key.replace('/', '_')}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=self.base, stdout=log, stderr=subprocess.STDOUT).returncode
        self.status["steps"][key] = {"status": "ok" if rc == 0 else f"failed (exit {rc})", "seconds": round(time.time() - t0, 1)}
        self.save_status()
        print(f"       {self.status['steps'][key]['status']} in {time.time() - t0:.0f} s", flush=True)
        return rc == 0

    def save_status(self) -> None:
        if not self.dry:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "status.json").write_text(json.dumps(self.status, indent=2) + "\n")

    def environment(self) -> None:
        env = {"started_utc": _utc(), "python": sys.version.split()[0], "cpus": os.cpu_count(), "memory_gb": round(_memory_limit_gb(), 1)}
        try:
            line = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
            gpu, driver, vram = (x.strip() for x in line.split(","))
            env.update(gpu=gpu, driver=driver, vram=vram)
            if tuple(int(x) for x in driver.split(".")[:2]) < MIN_DRIVER:
                print(f"warning: driver {driver} is older than {MIN_DRIVER[0]}.{MIN_DRIVER[1]}, which CUDA 12.8 needs", file=sys.stderr)
        except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as e:
            if not self.dry:
                raise SystemExit(f"no usable NVIDIA GPU: {e}")
        self.status["environment"] = env
        print(f"[env ] {json.dumps(env)}")

    def check_datasets(self) -> None:
        from nt.spec import LabelSpec

        spec_sha = LabelSpec.load().sha256
        if spec_sha != self.job["label_spec_sha256"]:
            raise SystemExit("this code's label_spec differs from the one the job was packed with")
        for d in self.job["datasets"]:
            man = self.base / d["path"] / "manifest.json"
            if not man.exists() and not self.dry:
                raise SystemExit(f"{d['path']} is missing; cloud_run.sh unpacks {d['archive']}")
            if man.exists() and _sha256(man) != d["manifest_sha256"]:
                raise SystemExit(f"{d['path']}/manifest.json does not match the job")

    def workers(self) -> tuple[int, int | None]:
        """Bake workers, and train workers when memory forces fewer than the hyp file asks for."""
        mem = _memory_limit_gb()
        cpus = os.cpu_count() or 4
        bake = self.job["bake"]["workers"] or max(1, min(cpus, 16, int(mem) if mem else 16))  # about 1 GB each
        train = self.job["train"]["workers"]
        if train is None and mem:
            from nt.train import load_hyp

            want = int(load_hyp(self.base / self.job["train"]["hyp"]).get("workers", 2))
            fit = max(2, int((mem - 6) / 2.5))  # about 2.5 GB per worker: train + 2x validation processes
            if fit < want:
                print(f"note: {mem:.0f} GB of memory, so {fit} data-loader workers instead of {want}", flush=True)
                train = fit
        return bake, train

    def train_done(self, run: str, model: str, seed: int) -> bool:
        meta = self.base / "runs" / "train" / run / "nt_meta.json"
        bake_man = self.base / self.baked / "bake_manifest.json"
        if not meta.exists() or not bake_man.exists():
            return False
        m = json.loads(meta.read_text())
        eff = m["hyp"]["effective"]
        return (m["status"] == "completed" and eff.get("seed") == seed and eff.get("epochs") == self.job["train"]["epochs"]
                and m["model_yaml"]["sha256"] == _sha256(self.base / model)
                and m["data"]["bake_manifest_sha256"] == _sha256(bake_man))

    def run(self, skip_tests: bool, skip_smoke: bool) -> int:
        py = sys.executable
        self.environment()
        self.check_datasets()
        if not skip_tests and not self.step("unit_tests", [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]):
            print("unit tests failed; not training (see runs/cloud/JOB/logs, or rerun with --skip-tests)", file=sys.stderr)
            self.save_status()
            return 1
        if not skip_smoke and not self.step("smoke_m0", [py, "-m", "nt.smoke_m0"]):
            print("nt.smoke_m0 failed on this GPU; not training", file=sys.stderr)
            self.save_status()
            return 1
        bake_w, train_w = self.workers()
        b = self.job["bake"]
        bake_cmd = [py, "-m", "nt.bake", "--out", self.baked, "--config", b["config"], "--copies", str(b["copies"]),
                    "--seed", str(b["seed"]), "--workers", str(bake_w)]
        for d in self.job["datasets"]:
            bake_cmd += ["--src", f"{d['path']}:{d['repeat']}"]
        if b.get("limit"):
            bake_cmd += ["--limit", str(b["limit"])]
        if not self.step("bake", bake_cmd, done=(self.base / self.baked / "bake_manifest.json").exists()):
            self.save_status()
            return 1
        t = self.job["train"]
        failed = 0
        for model in t["models"]:
            model_path = model if model.startswith("init/") else f"configs/models/{model}.yaml"
            for seed in t["seeds"]:
                run = f"{self.name}_{Path(model).stem}_s{seed}"
                cmd = [py, "-m", "nt.train", "--model", model_path, "--data", self.baked, "--name", run,
                       "--hyp", t["hyp"], "--epochs", str(t["epochs"]), "--seed", str(seed)]
                for key, val in (("--workers", t["workers"] or train_w), ("--batch", t["batch"]), ("--fraction", t["fraction"])):
                    if val is not None:
                        cmd += [key, str(val)]
                done = not self.dry and self.train_done(run, model_path, seed)
                run_dir = self.base / "runs" / "train" / run
                if not done and run_dir.exists() and not self.dry:
                    run_dir.rename(run_dir.with_name(f"{run}.failed-{datetime.now():%Y%m%d-%H%M%S}"))
                if not self.step(f"train/{run}", cmd, done=done):
                    failed += 1
                    continue
                weights = f"runs/train/{run}/weights/last.pt"
                ok = self.step(f"eval/{run}", [py, "-m", "nt.eval", "--weights", weights, "--data", self.baked,
                                               "--out", f"runs/eval/{run}.json"], done=done and (self.base / f"runs/eval/{run}.json").exists())
                ok &= self.step(f"export/{run}", [py, "-m", "nt.export", "--weights", weights, "--out", f"exports/{run}"],
                                done=(self.base / f"exports/{run}/model.onnx").exists())
                ok &= self.step(f"contract/{run}", [py, "-m", "nt.contract", f"exports/{run}/model.onnx"])
                failed += not ok
                self.save_status()
        self.status["finished_utc"] = _utc()
        self.save_status()
        if not self.dry:
            self.results()
        self.summary()
        return 1 if failed else 0

    def results(self) -> None:
        out = self.base / "results"
        out.mkdir(exist_ok=True)
        tgz = out / f"{self.name}_results.tgz"
        runs = [f"{self.name}_{Path(m).stem}_s{s}" for m in self.job["train"]["models"] for s in self.job["train"]["seeds"]]
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(self.job_path, arcname=f"runs/cloud/{self.name}/job.json")
            if self.dir.exists():
                tar.add(self.dir, arcname=str(self.dir.relative_to(self.base)))
            bake_man = self.base / self.baked / "bake_manifest.json"
            if bake_man.exists():  # beside the logs: a baked/ dir with no images would look like a real bake here
                tar.add(bake_man, arcname=f"runs/cloud/{self.name}/bake_manifest.json")
            for run in runs:
                for p in [self.base / "runs" / "train" / run, self.base / "runs" / "eval" / f"{run}.json", self.base / "exports" / run]:
                    if p.exists():
                        tar.add(p, arcname=str(p.relative_to(self.base)))
        tgz.with_name(tgz.name + ".sha256").write_text(f"{_sha256(tgz)}  {tgz.name}\n")
        print(f"[done] {tgz} ({tgz.stat().st_size / 1e6:.0f} MB); download it with its .sha256")

    def summary(self) -> None:
        for run in [f"{self.name}_{Path(m).stem}_s{s}" for m in self.job["train"]["models"] for s in self.job["train"]["seeds"]]:
            f = self.base / "runs" / "eval" / f"{run}.json"
            if f.exists():
                e = json.loads(f.read_text())["heads"]
                h = e.get("e2e") or next(iter(e.values()))
                print(f"  {run}: AP50 {h['armor_ap50']}, class acc {h['at_conf']['class_acc']}, "
                      f"corner p95 {h['corner_error']['orig_px']['p95']} px (sim val)")
            else:
                print(f"  {run}: no evaluation")


# ---- receive ------------------------------------------------------------------------------------------------

def receive(tgz: Path, dest: Path) -> int:
    side = tgz.with_name(tgz.name + ".sha256")
    if side.exists():
        want = side.read_text().split()[0]
        if _sha256(tgz) != want:
            raise SystemExit(f"{tgz} does not match {side}; download it again")
    else:
        print(f"warning: no {side.name}, the archive is not verified", file=sys.stderr)
    with tarfile.open(tgz) as tar:
        members = tar.getmembers()
        tops = set()
        for m in members:
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
                raise SystemExit(f"refusing archive member {m.name!r}")
            if parts[0] not in ("runs", "exports") or (parts[0] == "runs" and (len(parts) < 3 or parts[1] not in ("train", "eval", "cloud"))):
                raise SystemExit(f"unexpected archive member {m.name!r}")
            tops.add(Path(*parts[:3 if parts[0] == "runs" else 2]))
        clash = [str(p) for p in sorted(tops) if (dest / p).exists()]
        if clash:
            raise SystemExit(f"these exist already, nothing was unpacked: {clash}")
        tar.extractall(dest, members=members, filter="data")
    print(f"unpacked {len(members)} files into {dest}")
    exports = sorted(p.parent for p in dest.glob("exports/*/model.onnx") if Path("exports") / p.parent.name in tops)
    from nt.contract import check_model
    from nt.spec import LabelSpec

    sleepy = all((Path.home() / "sleepy" / "build" / f).exists() for f in ("libsleepy_utils.so", "compile_commands.json"))
    bad = 0
    for e in exports:
        problems = check_model(e / "model.onnx", LabelSpec.load(), cpp=sleepy)
        print(f"  {e.name}: {'OK' if not problems else problems}{'' if sleepy else ' (no sleepy build: C++ parity skipped)'}")
        bad += bool(problems)
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--name", required=True)
    p.add_argument("--data", action="append", required=True, help="canonical dataset dir, optionally :REPEAT")
    p.add_argument("--models", default="yolo26n-pose-armor", help="comma-separated configs/models/*.yaml stems")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hyp", default="configs/train/pose_cloud.yaml")
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--bake-config", default="configs/bake/default.yaml")
    p.add_argument("--init", type=Path, default=None, help="start from these weights (fine-tune) instead of --models")
    p.add_argument("--bake-limit", type=int, default=0, help="bake only N samples per dataset (rehearsals)")
    p.add_argument("--fraction", type=float, default=None, help="train on this share of the images (rehearsals)")
    p.add_argument("--out", type=Path, default=ROOT / "cloud")
    r = sub.add_parser("run")
    r.add_argument("job", type=Path)
    r.add_argument("--skip-tests", action="store_true")
    r.add_argument("--skip-smoke", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    v = sub.add_parser("receive")
    v.add_argument("archive", type=Path)
    v.add_argument("--dest", type=Path, default=ROOT)
    args = ap.parse_args()
    if args.cmd == "pack":
        out = pack(args.name, args.data, [m for m in args.models.split(",") if m], [int(s) for s in args.seeds.split(",")],
                   args.epochs, args.hyp, args.copies, args.out, args.init, args.bake_limit, args.fraction, args.bake_config)
        size = sum(p.stat().st_size for p in out.iterdir())
        print(f"wrote {out} ({size / 1e9:.2f} GB); read {out}/README.md, then upload the whole directory")
    elif args.cmd == "run":
        sys.exit(Runner(args.job, args.dry_run).run(args.skip_tests, args.skip_smoke))
    else:
        sys.exit(receive(args.archive, args.dest))


if __name__ == "__main__":
    main()
