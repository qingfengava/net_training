#!/usr/bin/env bash
# Create .venv and install the training environment. Safe to re-run.
#
#   scripts/setup_env.sh            install exactly requirements.lock, every file checked against its sha256
#                                   (a rented machine, a fresh checkout, the cloud pack)
#   scripts/setup_env.sh --resolve  resolve requirements.in + requirements-nodeps.in again and rewrite
#                                   requirements.lock with hashes (only after changing a pin, on this machine)
#
# Env:
#   NT_PYTHON       interpreter for the venv, default python3; the lock was made on Python 3.10
#   NT_PIP_INDEX    PyPI mirror, default https://pypi.org/simple; a mirror is safe because every file is hash-checked
#   NT_TORCH_INDEX  PyTorch wheel index, default https://download.pytorch.org/whl/cu128
# Needs network access to both indexes. Does not touch ~/.local or system packages.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${NT_PYTHON:-python3}
PIP_INDEX=${NT_PIP_INDEX:-https://pypi.org/simple}
TORCH_INDEX=${NT_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
MODE=${1:-lock}
[[ $MODE == lock || $MODE == --resolve ]] || { echo "usage: setup_env.sh [--resolve]" >&2; exit 2; }

if [[ ! -d $ROOT/.venv ]]; then
    "$PY" -m venv "$ROOT/.venv" || {
        echo "cannot create a venv with $PY; install python3.10-venv or point NT_PYTHON at a Python 3.10" >&2; exit 1; }
fi
# shellcheck disable=SC1091
. "$ROOT/.venv/bin/activate"
pyver=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[[ $pyver == 3.10 ]] || echo "warning: Python $pyver; requirements.lock was made on 3.10, other versions are untested" >&2

if [[ $MODE == lock ]]; then
    grep -q -- '--hash=sha256:' "$ROOT/requirements.lock" || { echo "requirements.lock has no hashes; run --resolve here first" >&2; exit 1; }
    # --no-deps: the lock is the complete environment, and albumentations' declared opencv-python-headless must
    # not be pulled in (see requirements-nodeps.in). --only-binary: every locked package has a wheel.
    python -m pip install --quiet --require-hashes --no-deps --only-binary=:all: \
        --index-url "$PIP_INDEX" --extra-index-url "$TORCH_INDEX" -r "$ROOT/requirements.lock"
    # pip check: the only expected complaint is the headless OpenCV that albumentations declares
    if problems=$(python -m pip check 2>&1); then :; fi
    other=$(grep -v 'requires opencv-python-headless, which is not installed' <<<"$problems" | grep -v '^No broken requirements' || true)
    [[ -z $other ]] || { echo "pip check:"; echo "$other"; exit 1; } >&2
    echo "env ready from requirements.lock (hash-checked)"
    exit 0
fi

python -m pip install --quiet --index-url "$PIP_INDEX" pip==26.2.2
# torch first, from the cu128 index only, so ultralytics does not pull a CPU or cu126 build.
python -m pip install --index-url "$TORCH_INDEX" torch==2.11.0 torchvision==0.26.0
python -m pip install --index-url "$PIP_INDEX" -r "$ROOT/requirements.in"
python -m pip install --index-url "$PIP_INDEX" --no-deps -r "$ROOT/requirements-nodeps.in"
freeze=$(mktemp)
python -m pip freeze --all >"$freeze"
python -I "$ROOT/scripts/hash_lock.py" "$freeze" >"$ROOT/requirements.lock.new"
mv "$ROOT/requirements.lock.new" "$ROOT/requirements.lock"
rm -f "$freeze"
echo "env resolved, requirements.lock rewritten with hashes"
