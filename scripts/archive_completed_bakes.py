#!/usr/bin/env python3
"""归档本次已完成增强实验的重建材料，按明确名单清理派生烘焙；默认仅预览。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "style_b_v3_base", "style_b_v3_roi", "style_b_v3_moire", "style_b_v3_both",
    "style_b_v3_stress_val", "style_b_v4_bayer", "style_b_v4_bayer_val",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="新数据审计通过后执行归档和清理")
    args = ap.parse_args()
    archive = ROOT / "runs/baked_manifests/cleanup_field_refresh_20261106"
    entries = []
    for name in NAMES:
        path = ROOT / "baked" / name
        if not path.exists():
            continue
        assert not path.is_symlink() and path.resolve().parent == (ROOT / "baked").resolve()
        manifest = json.loads((path / "bake_manifest.json").read_text())
        for source in manifest["sources"]:
            source_path = ROOT / source["path"]
            assert (source_path / "labels.jsonl").is_file()
            assert sha256(source_path / "manifest.json") == source["manifest_sha256"]
        files = [p for p in path.rglob("*") if p.is_file()]
        entries.append({"name": name, "path": str(path), "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files),
            "images": manifest["images"], "files": len(files)})
    print(json.dumps({"apply": args.apply, "planned": entries,
        "reclaim_gib": sum(e["allocated_bytes"] for e in entries) / 1024**3}, ensure_ascii=False, indent=2))
    if not args.apply:
        return
    audit = json.loads((ROOT / "runs/verification/field_refresh_20261106/audit.json").read_text())
    assert audit["negative_guard"] == "all passed" and audit["train_val_identical_source_images"] == 0
    archive.mkdir(parents=True, exist_ok=False)
    # 所有目录先完成归档并校验，之后才清理图片及标签；原始数据、权重和实验报告都不在名单内。
    for entry in entries:
        path = Path(entry["path"])
        dst = archive / entry["name"]
        dst.mkdir()
        metadata = {}
        for file in path.iterdir():
            if not file.is_file():
                assert file.name in {"images", "labels"}, f"未知子目录，停止：{file}"
                continue
            digest = sha256(file)
            if file.name == "meta.jsonl":
                target = dst / "meta.jsonl.gz"
                with file.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as compressed:
                    shutil.copyfileobj(src, compressed)
                with gzip.open(target, "rb") as compressed:
                    assert hashlib.sha256(compressed.read()).hexdigest() == digest
            else:
                target = dst / file.name
                shutil.copy2(file, target)
                assert sha256(target) == digest
            metadata[file.name] = {"sha256": digest, "archived_as": target.name}
        manifest = json.loads((path / "bake_manifest.json").read_text())
        (dst / "config.yaml").write_text(yaml.safe_dump(manifest["config"], sort_keys=False, allow_unicode=True))
        command = [sys.executable, "-m", "nt.bake"]
        for source in manifest["sources"]:
            command.extend(["--src", f"{source['path']}:{source['repeat']}"] if source["repeat"] != 1
                else ["--src", source["path"]])
        command.extend(["--out", entry["path"], "--config", str(dst / "config.yaml"),
            "--copies", str(manifest["copies"]), "--seed", str(manifest["seed"]), "--workers", "3"])
        entry["metadata"] = metadata
        entry["rebuild_argv"] = command
        (dst / "archive.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
    for entry in entries:
        shutil.rmtree(entry["path"])
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "removed": entries,
        "reclaimed_bytes": sum(e["allocated_bytes"] for e in entries),
        "remaining_free_bytes": shutil.disk_usage(ROOT).free}
    (archive / "cleanup.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"归档并清理完成：{archive}")


if __name__ == "__main__":
    main()
