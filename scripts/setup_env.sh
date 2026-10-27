#!/usr/bin/env bash
# Create .venv and install the pinned training environment. Safe to re-run.
# Needs network access to pypi.org and download.pytorch.org. Does not touch ~/.local or system packages.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TORCH_INDEX=https://download.pytorch.org/whl/cu128

[[ -d $ROOT/.venv ]] || python3 -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
. "$ROOT/.venv/bin/activate"

python -m pip install --quiet pip==26.2.2
# torch first, from the cu128 index only, so ultralytics does not pull a CPU or cu126 build.
python -m pip install --index-url "$TORCH_INDEX" torch==2.11.0 torchvision==0.26.0
python -m pip install -r "$ROOT/requirements.in"

python -m pip freeze --all >"$ROOT/requirements.lock"
echo "env ready, lock written to requirements.lock"
