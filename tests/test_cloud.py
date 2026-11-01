"""Cloud job round trip without a cloud: pack -> unpacked job directory -> planned run -> receive."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from nt.cloud import Runner, pack, receive
from nt.ingest import ingest
from nt.spec import LabelSpec

ROOT = Path(__file__).resolve().parents[1]
SPEC = LabelSpec.load()


class CloudTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.ds = self.tmp / "datasets" / "sim" / "dae"
        ingest(ROOT / "tests" / "fixtures" / "daedalus_old", self.ds, "daedalus-old", "sim", "dae", SPEC)

    def tearDown(self):
        self._tmp.cleanup()

    def pack(self, name="j1", **kw):
        return pack(name, [f"{self.ds}:2"], ["yolo26n-pose-armor", "yolo26n-pose-armor-coordatt"], [0, 1], 100,
                    "configs/train/pose_cloud.yaml", 2, self.tmp / "cloud", **kw)

    def test_pack_contents_and_checksums(self):
        out = self.pack()
        names = {p.name for p in out.iterdir()}
        self.assertEqual(names, {"nt_code.tar.gz", "data_00_dae.tar", "job.json", "cloud_run.sh", "README.md", "SHA256SUMS"})
        for line in (out / "SHA256SUMS").read_text().splitlines():
            digest, rel = line.split("  ")
            self.assertEqual(hashlib.sha256((out / rel).read_bytes()).hexdigest(), digest, rel)
        job = json.loads((out / "job.json").read_text())
        self.assertEqual(job["datasets"][0]["path"], "datasets/sim/dae")
        self.assertEqual(job["datasets"][0]["repeat"], 2)
        self.assertEqual(job["label_spec_sha256"], SPEC.sha256)
        with tarfile.open(out / "data_00_dae.tar") as t:
            self.assertIn("datasets/sim/dae/manifest.json", t.getnames())
        with tarfile.open(out / "nt_code.tar.gz") as t:
            code = t.getnames()
        self.assertIn("nt/cloud.py", code)
        self.assertIn("requirements.lock", code)
        self.assertIn("export/cpp_parity/armor_decode.cpp", code)  # the C++ parity tests need it
        self.assertFalse(any("__pycache__" in n for n in code))
        with self.assertRaises(FileExistsError):
            self.pack()

    def test_pack_refuses_a_dataset_ingested_with_another_label_spec(self):
        man = json.loads((self.ds / "manifest.json").read_text())
        man["label_spec_sha256"] = "0" * 64
        (self.ds / "manifest.json").write_text(json.dumps(man))
        with self.assertRaises(ValueError):
            self.pack("j2")

    def test_unpacked_job_plans_every_step_in_order(self):
        out = self.pack()
        job_dir = self.tmp / "remote"
        job_dir.mkdir()
        for f in out.iterdir():
            (job_dir / f.name).write_bytes(f.read_bytes())
        with tarfile.open(job_dir / "data_00_dae.tar") as t:
            t.extractall(job_dir, filter="data")
        (job_dir / "configs" / "train").mkdir(parents=True)
        for f in ("pose_cloud.yaml", "pose_default.yaml"):  # what nt_code.tar.gz would unpack
            (job_dir / "configs" / "train" / f).write_text((ROOT / "configs" / "train" / f).read_text())
        r = Runner(job_dir / "job.json", dry_run=True)
        self.assertEqual(r.run(skip_tests=False, skip_smoke=False), 0)
        steps = list(r.status["steps"])
        self.assertEqual(steps[:3], ["unit_tests", "smoke_m0", "bake"])
        runs = [s.split("/", 1)[1] for s in steps if s.startswith("train/")]
        self.assertEqual(runs, ["j1_yolo26n-pose-armor_s0", "j1_yolo26n-pose-armor_s1",
                                "j1_yolo26n-pose-armor-coordatt_s0", "j1_yolo26n-pose-armor-coordatt_s1"])
        for run in runs:  # each run is trained, evaluated on last.pt, exported and contract-checked
            self.assertEqual([s for s in steps if s.endswith("/" + run)],
                             [f"train/{run}", f"eval/{run}", f"export/{run}", f"contract/{run}"])
        bake = r.status["steps"]["bake"]["cmd"]
        self.assertIn("datasets/sim/dae:2", bake)
        train = r.status["steps"][f"train/{runs[1]}"]["cmd"]
        self.assertEqual(train[train.index("--seed") + 1], "1")
        self.assertEqual(train[train.index("--hyp") + 1], "configs/train/pose_cloud.yaml")


class ReceiveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def archive(self, members: dict[str, bytes], name="j_results.tgz", checksum=True) -> Path:
        path = self.tmp / name
        with tarfile.open(path, "w:gz") as t:
            for arc, data in members.items():
                info = tarfile.TarInfo(arc)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        if checksum:
            path.with_name(path.name + ".sha256").write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        return path

    def test_unpacks_and_refuses_to_overwrite(self):
        tgz = self.archive({"runs/train/r1/nt_meta.json": b"{}", "runs/eval/r1.json": b"{}", "runs/cloud/j/status.json": b"{}"})
        dest = self.tmp / "dest"
        self.assertEqual(receive(tgz, dest), 0)
        self.assertTrue((dest / "runs" / "train" / "r1" / "nt_meta.json").exists())
        with self.assertRaises(SystemExit):
            receive(tgz, dest)

    def test_refuses_unsafe_or_foreign_members_and_bad_checksums(self):
        for bad in ("../evil.txt", "runs/../../evil.txt", "nt/train.py", "/etc/x", "runs/x"):
            with self.assertRaises(SystemExit, msg=bad):
                receive(self.archive({bad: b"x"}, name="b.tgz"), self.tmp / "d")
        tgz = self.archive({"runs/eval/r1.json": b"{}"}, name="c.tgz")
        tgz.with_name("c.tgz.sha256").write_text("0" * 64 + "  c.tgz\n")
        with self.assertRaises(SystemExit):
            receive(tgz, self.tmp / "d2")
        self.assertFalse((self.tmp / "d2").exists())


if __name__ == "__main__":
    unittest.main()
