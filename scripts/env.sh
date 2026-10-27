# Source this file: . scripts/env.sh
# Activates the venv, keeps Ultralytics settings inside the project, and stops it from going online.
NT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
. "$NT_ROOT/.venv/bin/activate"
export NT_ROOT
mkdir -p "$NT_ROOT/.yolo_config"
export YOLO_CONFIG_DIR="$NT_ROOT/.yolo_config"
export YOLO_OFFLINE=true       # skips the DNS online check and every AutoUpdate/download path
export YOLO_AUTOINSTALL=false  # never pip install behind our back; versions come from requirements.lock
export PYTHONPATH="$NT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
