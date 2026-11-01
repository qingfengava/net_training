#!/usr/bin/env bash
# Pack a training job for a rented GPU machine into cloud/NAME/. Nothing is uploaded from here.
#
#   scripts/pack_for_cloud.sh --name simoff_v1 --data datasets/sim/simoff [--data datasets/real/X:3 ...] \
#       [--models yolo26n-pose-armor[,yolo26n-pose-armor-coordatt]] [--seeds 0[,1]] [--epochs 100] \
#       [--hyp configs/train/pose_cloud.yaml] [--copies 2] [--init runs/train/R/weights/last.pt]
#
# The datasets are canonical ones (already ingested and looked at with nt.viz); they are baked on the rented
# machine. Upload the whole cloud/NAME directory and follow its README.md. Options: python -m nt.cloud pack -h.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/scripts/env.sh"
exec python -m nt.cloud pack "$@"
