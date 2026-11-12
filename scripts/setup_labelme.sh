#!/usr/bin/env bash
# Create .venv-label with the pinned labelme GUI, using uv. Safe to re-run.
#
# labelme lives apart from the training venv on purpose: through osam it requires the CPU `onnxruntime`
# package, which installs the same `onnxruntime` module as the `onnxruntime-gpu` the training tools use, so
# whichever is installed last silently replaces the other.
#
#   scripts/setup_labelme.sh
#   .venv-label/bin/labelme annot/PROJECT/batch_000 --config annot/PROJECT/labelme_config.yaml --nodata --autosave
#
# Annotators on another machine only need `pip install labelme==6.3.1` (or uv) and a batch folder.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=$ROOT/.venv-label
PY=${NT_PYTHON:-/usr/bin/python3.10}
INDEX=${NT_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}
UV=${UV:-uv}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
command -v "$UV" >/dev/null || { echo "需要 uv：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

[[ -d $VENV ]] || "$UV" venv --python "$PY" "$VENV"
# labelme 6.3.1 经 osam 用到 onnxruntime，但 osam 只把 onnxruntime 放在 [cpu] extra 里，labelme 自己
# 不声明它；不显式安装时 labelme 启动即在 import onnxruntime 处失败。
"$UV" pip install --python "$VENV/bin/python" --default-index "$INDEX" labelme==6.3.1 "onnxruntime==1.23.2"
"$UV" pip freeze --python "$VENV/bin/python" >"$ROOT/requirements-label.lock"
echo "labelme ready: $VENV/bin/labelme (lock in requirements-label.lock)"
