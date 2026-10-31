#!/usr/bin/env bash
# M5 attention A/B: train the three YOLO26n structures (C2PSA, no attention, CoordAtt) with two seeds each on one
# baked dataset, one run at a time, then export each run and check it against the contract and sleepy's decoder.
#
#   scripts/m5_attention_ab.sh BAKED_DIR TAG [EPOCHS] [MEM_GB]
#   scripts/m5_attention_ab.sh baked/simm7_c2 simm7            # runs/train/y26n_simm7_{c2psa,noattn,coordatt}_s{0,1}
#
# Safe to run again: a run whose nt_meta.json says completed with the same yaml, data, epochs and seed is skipped,
# and a left-over unfinished run directory is moved aside to NAME.failed-<time>, never deleted. A run that fails
# (for example killed at the memory cap) does not stop the queue. When less memory is free than MEM_GB the
# queue waits for up to two hours before it gives that run up. Then compare with nt.eval and nt.bench_trt.
set -uo pipefail

BAKED=${1:?usage: m5_attention_ab.sh BAKED_DIR TAG [EPOCHS] [MEM_GB]}
TAG=${2:?usage: m5_attention_ab.sh BAKED_DIR TAG [EPOCHS] [MEM_GB]}
EPOCHS=${3:-100}
MEM=${4:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# shellcheck disable=SC1091
. scripts/env.sh
[[ -f $BAKED/bake_manifest.json ]] || { echo "$BAKED is not a baked dataset" >&2; exit 2; }

# seed 0 of every structure first, so an interrupted night still leaves one complete comparison
RUNS=(c2psa:0 noattn:0 coordatt:0 c2psa:1 noattn:1 coordatt:1)
declare -A YAML=([c2psa]=yolo26n-pose-armor [noattn]=yolo26n-pose-armor-noattn [coordatt]=yolo26n-pose-armor-coordatt)

done_already() {  # name yaml seed
    python3 - "$1" "$2" "$3" "$EPOCHS" "$BAKED" <<'EOF'
import hashlib, json, sys
from pathlib import Path
name, yaml, seed, epochs, baked = sys.argv[1:]
meta = Path("runs/train") / name / "nt_meta.json"
if not meta.exists():
    sys.exit(1)
m = json.loads(meta.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ok = (m["status"] == "completed" and m["hyp"]["effective"]["epochs"] == int(epochs)
      and m["hyp"]["effective"]["seed"] == int(seed) and m["model_yaml"]["sha256"] == sha(f"configs/models/{yaml}.yaml")
      and m["data"]["bake_manifest_sha256"] == sha(f"{baked}/bake_manifest.json"))
sys.exit(0 if ok else 1)
EOF
}

for spec in "${RUNS[@]}"; do
    arch=${spec%%:*}; seed=${spec##*:}; yaml=${YAML[$arch]}
    name="y26n_${TAG}_${arch}_s${seed}"
    log="runs/train_${name}.log"
    if done_already "$name" "$yaml" "$seed"; then
        echo "$(date +%H:%M) skip $name (completed)"
    else
        if [[ -e runs/train/$name ]]; then
            mv "runs/train/$name" "runs/train/$name.failed-$(date +%Y%m%d-%H%M%S)"
            echo "$(date +%H:%M) moved an unfinished runs/train/$name aside"
        fi
        for attempt in $(seq 1 24); do
            echo "$(date +%H:%M) train $name ($yaml, seed $seed, $EPOCHS epochs)"
            scripts/run_capped.sh "$MEM" -- python -m nt.train --model "configs/models/$yaml.yaml" --data "$BAKED" \
                --name "$name" --epochs "$EPOCHS" --seed "$seed" --workers 2 > "$log" 2>&1
            rc=$?
            [[ $rc -eq 3 ]] || break  # 3: run_capped found less free memory than the cap
            echo "$(date +%H:%M) not enough free memory for $MEM GB, waiting 5 min ($attempt/24)"
            sleep 300
        done
        echo "$(date +%H:%M) $name exit=$rc"
        done_already "$name" "$yaml" "$seed" || { echo "$(date +%H:%M) $name did not complete, see $log"; continue; }
    fi
    if [[ ! -f exports/$name/model.onnx ]]; then
        scripts/run_capped.sh 4 -- python -m nt.export --weights "runs/train/$name/weights/last.pt" --out "exports/$name" > "runs/export_${name}.log" 2>&1 \
            || echo "$(date +%H:%M) $name export failed, see runs/export_${name}.log"
    fi
    [[ -f exports/$name/model.onnx ]] && echo "$(date +%H:%M) $(scripts/run_capped.sh 4 -- python -m nt.contract "exports/$name/model.onnx" --cpp 2>&1 | tail -1)"
    if [[ ! -f runs/eval/$name.json ]]; then
        scripts/run_capped.sh 5 -- python -m nt.eval --weights "runs/train/$name/weights/last.pt" --data "$BAKED" --out "runs/eval/$name.json" 2>&1 \
            | grep -E '^\[(e2e|nms)\]' | sed "s/^/$(date +%H:%M) $name /"
    fi
done
echo "$(date +%H:%M) queue finished"
