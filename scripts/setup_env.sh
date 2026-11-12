#!/usr/bin/env bash
# Create .venv and install the training environment with uv. Safe to re-run.
#
#   scripts/setup_env.sh            install exactly requirements.lock, every file checked against its sha256
#                                   (a rented machine, a fresh checkout, the cloud pack)
#   scripts/setup_env.sh --resolve  resolve requirements.in + requirements-nodeps.in again and rewrite
#                                   requirements.lock with hashes (only after changing a pin, on this machine)
#
# Env:
#   NT_PYTHON         interpreter for the venv, default /usr/bin/python3.10; the lock was made on Python 3.10
#   NT_PIP_INDEX      PyPI mirror, default https://mirrors.aliyun.com/pypi/simple; a mirror is safe because
#                     every file is hash-checked
#   NT_TORCH_LINKS    PyTorch cu128 wheel listing, default https://mirrors.aliyun.com/pytorch-wheels/cu128/
#   NT_TORCH_ROOT     extra flat wheel listing for the PyTorch build of triton, default
#                     https://mirrors.aliyun.com/pytorch-wheels/ (requirements.lock pins that exact file)
# Needs network access to both indexes. Does not touch ~/.local or system packages.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${NT_PYTHON:-/usr/bin/python3.10}
PIP_INDEX=${NT_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
TORCH_LINKS=${NT_TORCH_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu128/}
TORCH_ROOT=${NT_TORCH_ROOT:-https://mirrors.aliyun.com/pytorch-wheels/}
MODE=${1:-lock}
UV=${UV:-uv}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
[[ $MODE == lock || $MODE == --resolve ]] || { echo "usage: setup_env.sh [--resolve]" >&2; exit 2; }
command -v "$UV" >/dev/null || { echo "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

if [[ ! -d $ROOT/.venv ]]; then
    "$UV" venv --python "$PY" "$ROOT/.venv"
fi
venv_pip=("$UV" pip)
pyver=$("$ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
[[ $pyver == 3.10 ]] || echo "warning: Python $pyver; requirements.lock was made on 3.10, other versions are untested" >&2

if [[ $MODE == lock ]]; then
    grep -q -- '--hash=sha256:' "$ROOT/requirements.lock" || { echo "requirements.lock has no hashes; run --resolve here first" >&2; exit 1; }
    # --no-deps: the lock is the complete environment, and albumentations' declared opencv-python-headless must
    # not be pulled in (see requirements-nodeps.in). --only-binary: every locked package has a wheel.
    "${venv_pip[@]}" install --python "$ROOT/.venv/bin/python" --require-hashes --no-deps --only-binary=:all: \
        --default-index "$PIP_INDEX" --find-links "$TORCH_LINKS" --find-links "$TORCH_ROOT" \
        -r "$ROOT/requirements.lock"
    # uv pip check: the only expected complaint is the headless OpenCV that albumentations declares
    if problems=$("${venv_pip[@]}" check --python "$ROOT/.venv/bin/python" 2>&1); then :; fi
    other=$(grep -v 'opencv-python-headless' <<<"$problems" \
        | grep -vE '^(Checked [0-9]+ packages|Found [0-9]+ incompatibilit)' || true)
    [[ -z $other ]] || { echo "uv pip check:"; echo "$other"; exit 1; } >&2
    echo "env ready from requirements.lock (hash-checked)"
    exit 0
fi

# torch first, from the cu128 listing only, so ultralytics does not pull a CPU or cu126 build.
"${venv_pip[@]}" install --python "$ROOT/.venv/bin/python" --default-index "$PIP_INDEX" --find-links "$TORCH_LINKS" \
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"
"${venv_pip[@]}" install --python "$ROOT/.venv/bin/python" --default-index "$PIP_INDEX" -r "$ROOT/requirements.in"
"${venv_pip[@]}" install --python "$ROOT/.venv/bin/python" --default-index "$PIP_INDEX" --no-deps -r "$ROOT/requirements-nodeps.in"
freeze=$(mktemp)
"${venv_pip[@]}" freeze --python "$ROOT/.venv/bin/python" >"$freeze"
python3 -I "$ROOT/scripts/hash_lock.py" "$freeze" >"$ROOT/requirements.lock.new"
mv "$ROOT/requirements.lock.new" "$ROOT/requirements.lock"
rm -f "$freeze"
echo "env resolved, requirements.lock rewritten with hashes"
