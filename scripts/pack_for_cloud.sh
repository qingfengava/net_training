#!/usr/bin/env bash
# Pack the code and one or more baked datasets for a cloud GPU machine. Nothing is uploaded from here.
#
#   scripts/pack_for_cloud.sh baked/simm3b_c2 [baked/other ...]
#
# Writes cloud/nt_code.tar.gz, cloud/<baked name>.tar (PNG is already compressed, so no gzip) and
# cloud/SHA256SUMS. Copy the cloud/ directory to the machine however you like, then there:
#
#   mkdir -p ~/net_training && cd ~/net_training && cp -r <cloud dir>/. . 
#   sha256sum -c SHA256SUMS
#   tar xzf nt_code.tar.gz && mkdir -p baked && for t in *.tar; do tar xf "$t" -C baked; done
#   nvidia-smi                       # the cu128 wheels want an NVIDIA driver >= 570 (from memory, verify)
#   bash scripts/setup_env.sh && . scripts/env.sh
#   python -m nt.smoke_m0            # once per GPU type: checks the architecture and AMP training
#   python -m nt.train --model configs/models/yolo26n-pose-armor.yaml --data baked/<name> --name y26n \
#       --epochs 100 --batch 64 --workers 8
#   python -m nt.train --model configs/models/yolov8n-pose-armor.yaml --data baked/<name> --name y8n ...
#   python -m nt.eval --weights runs/train/y26n/weights/best.pt --data baked/<name> --out runs/eval/y26n.json
#   tar czf results.tgz runs/train/*/weights runs/train/*/results.csv runs/train/*/nt_meta.json runs/eval
#
# nt.train rewrites the dataset path for the machine it runs on, so the baked directories need no editing.
# The baked data.yaml and meta.jsonl carry no secrets; they describe simulator images only.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $# -ge 1 ]] || { echo "usage: pack_for_cloud.sh BAKED_DIR [BAKED_DIR ...]" >&2; exit 2; }
mkdir -p cloud

for d in "$@"; do
    [[ -f $d/bake_manifest.json && -f $d/data.yaml ]] || { echo "$d is not a baked dataset" >&2; exit 2; }
done

tar --exclude=__pycache__ --exclude='*.pyc' -czf cloud/nt_code.tar.gz \
    nt configs scripts tests requirements.in requirements.lock .gitignore
for d in "$@"; do
    name=$(basename "$d")
    tar -cf "cloud/$name.tar" -C "$(dirname "$d")" "$name"
done
(cd cloud && sha256sum nt_code.tar.gz ./*.tar > SHA256SUMS)
ls -la cloud
