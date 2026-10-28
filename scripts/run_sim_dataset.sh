#!/usr/bin/env bash
# Run the Daedalus dataset exporter headless and bounded.
#
#   scripts/run_sim_dataset.sh CONFIG.toml [TIMEOUT_SECONDS]
#
# Safety, copied from sleepy's script/sim_gtsam_check/run_session.sh: no window (the hybrid-GPU laptop has a
# history of freezing the desktop in windowed mode), a 4 GB memory cap with no swap, low CPU priority, a hard
# timeout, and a refusal to start next to another simulator. The output directory comes from the config and
# must not exist or must be empty.
#
# Env: SIM_DIR  simulator checkout, default ~/bevy_robomaster_simulator
set -euo pipefail

CONFIG=${1:?usage: run_sim_dataset.sh CONFIG.toml [TIMEOUT_SECONDS]}
TIMEOUT=${2:-1800}
SIM_DIR=${SIM_DIR:-$HOME/bevy_robomaster_simulator}
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")

[[ -x $SIM_DIR/target/release/daedalus ]] || { echo "build first: cd $SIM_DIR && cargo build --release" >&2; exit 1; }
if pgrep -x daedalus >/dev/null; then
    echo "a daedalus process is already running; stop it first" >&2
    exit 1
fi

SIM_ENV=(
    BEVY_ASSET_ROOT="$SIM_DIR"
    LD_LIBRARY_PATH="$SIM_DIR/target/release/deps:$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"
    DAEDALUS_DATASET="$CONFIG"
    DAEDALUS_HEADLESS=1
    DAEDALUS_MAX_FPS=0
    DAEDALUS_DISABLE_TALOS=1
)

# The simulator reads config.toml from its working directory.
cd "$SIM_DIR"
exec env "${SIM_ENV[@]}" systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
    nice -n 10 timeout --signal=INT --kill-after=20 "$TIMEOUT" "$SIM_DIR/target/release/daedalus"
