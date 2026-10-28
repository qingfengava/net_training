#!/usr/bin/env bash
# Create .venv-label with the pinned labelme GUI. Safe to re-run.
#
# labelme lives apart from the training venv on purpose: through osam it requires the CPU `onnxruntime`
# package, which installs the same `onnxruntime` module as the `onnxruntime-gpu` the training tools use, so
# whichever is installed last silently replaces the other.
#
#   scripts/setup_labelme.sh
#   .venv-label/bin/labelme annot/PROJECT/batch_000 --config annot/PROJECT/labelme_config.yaml --nodata --autosave
#
# Annotators on another machine only need `pip install labelme==6.3.1` and a batch folder.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=$ROOT/.venv-label

[[ -d $VENV ]] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet pip==26.2.2
"$VENV/bin/python" -m pip install labelme==6.3.1
"$VENV/bin/python" -m pip freeze --all >"$ROOT/requirements-label.lock"
echo "labelme ready: $VENV/bin/labelme (lock in requirements-label.lock)"
