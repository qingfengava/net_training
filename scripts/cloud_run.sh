#!/usr/bin/env bash
# Run a packed training job on a rented GPU machine. Lives in the job directory made by
# scripts/pack_for_cloud.sh; upload that whole directory, then run this inside it, best under tmux or nohup:
#
#   nohup bash cloud_run.sh > cloud_run.log 2>&1 &        extra flags go to `nt.cloud run` (--skip-tests ...)
#
# Checks every file against SHA256SUMS, unpacks the code and the datasets once, installs the hash-locked
# environment (scripts/setup_env.sh; NT_PIP_INDEX / NT_TORCH_INDEX / NT_PYTHON select mirrors and the
# interpreter), then runs the job. Running it again continues where it stopped.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
[[ -f job.json && -f SHA256SUMS ]] || { echo "run this inside a job directory from scripts/pack_for_cloud.sh" >&2; exit 2; }
if [[ ! -f .verified ]]; then
    echo "checking SHA256SUMS ..."
    sha256sum --quiet -c SHA256SUMS
    touch .verified
fi
[[ -f .unpacked_code ]] || { tar xzf nt_code.tar.gz && touch .unpacked_code; }
for t in data_*.tar; do
    [[ -f ".unpacked_${t%.tar}" ]] || { echo "unpacking $t"; tar xf "$t" && touch ".unpacked_${t%.tar}"; }
done
bash scripts/setup_env.sh
# shellcheck disable=SC1091
. scripts/env.sh
exec python -m nt.cloud run job.json "$@"
