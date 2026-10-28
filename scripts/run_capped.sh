#!/usr/bin/env bash
# Run any heavy job in a memory-capped, low-priority cgroup scope, so a runaway job is killed alone instead of
# the kernel picking VS Code or the desktop. This machine has 15 GB and the editor plus rust-analyzer take
# 5 GB or more, so keep the cap well below the free memory.
#
#   scripts/run_capped.sh MEMORY_GB -- command args...
#   scripts/run_capped.sh 5 -- python -m nt.train --model ... --workers 2
#
# MemorySwapMax=0 makes the job fail fast at the cap instead of dragging the whole machine into swap.
set -euo pipefail

GB=${1:?usage: run_capped.sh MEMORY_GB -- command args...}
shift
[[ ${1:-} == -- ]] && shift
[[ $# -gt 0 ]] || { echo "no command given" >&2; exit 2; }

AVAIL_GB=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
if (( AVAIL_GB < GB )); then
    echo "only ${AVAIL_GB} GB available, less than the requested cap of ${GB} GB; close something or lower the cap" >&2
    exit 3
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
. "$ROOT/scripts/env.sh"
exec systemd-run --user --scope -q -p "MemoryMax=${GB}G" -p MemorySwapMax=0 -p CPUWeight=50 nice -n 10 "$@"
