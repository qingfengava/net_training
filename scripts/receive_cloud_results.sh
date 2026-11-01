#!/usr/bin/env bash
# Take a finished cloud job back: verify JOB_results.tgz against its .sha256, unpack it into runs/ and exports/
# without overwriting anything, and run nt.contract --cpp (sleepy's real decoder) on every exported model.
#
#   scripts/receive_cloud_results.sh path/to/JOB_results.tgz [--dest DIR]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/scripts/env.sh"
exec python -m nt.cloud receive "$@"
