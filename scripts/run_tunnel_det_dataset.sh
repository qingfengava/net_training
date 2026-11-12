#!/usr/bin/env bash
# 依次渲染 configs/sim/tunnel_<名字>.toml 描述的隧道数据集，输出目录取配置里的 out_dir。
# 用法：scripts/run_tunnel_det_dataset.sh [名字...]（不带参数时渲染检测器的 6 个数据集）
#
# - 开跑前按配置估算帧数和磁盘占用（每帧约 7 MB：PNG + 32 位深度），空间不够就停下。
# - 已有 done.json 且没有丢帧的数据集跳过。
# - 输出目录已存在但不完整（没有 done.json 或有丢帧）时不会自动删除，只报告并跳过，由人决定怎么处理。
# - 一个数据集失败不影响后面的，最后统一汇总。
set -uo pipefail
NT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DAEDALUS=${DAEDALUS_ROOT:-/home/robo/bevy_robomaster_simulator}
MB_PER_FRAME=${MB_PER_FRAME:-7}
RESERVE_GB=${RESERVE_GB:-20}   # 渲染完成后至少还要剩这么多，留给转换和训练

# 从配置读出 out_dir 和预计帧数：runs × (fps × 平均接近时长 + 1)，速度在 [lo, hi] 内均匀分布
read_cfg() {
  python3 -I - "$1" <<'EOF'
import math, sys, toml   # 系统 Python 3.10 没有 tomllib
c = toml.load(sys.argv[1])
t = c.get("tunnel", {})
lo, hi = t.get("speed_mps", [1.5, 2.5])
inv_v = 1.0 / lo if hi <= lo else math.log(hi / lo) / (hi - lo)
travel = t.get("start_dist_m", 4.5) - t.get("reference_offset_m", 0.25) - t.get("end_ref_dist_m", 0.2)
frames = int(t.get("runs", 8) * (t.get("fps", 60.0) * travel * inv_v + 1))
print(c["out_dir"], frames)
EOF
}

sets=("$@")
[ ${#sets[@]} -gt 0 ] || sets=(det_train_cad det_train_proc det_train_neg det_val_proc det_val_stress det_val_neg)
cd "$DAEDALUS"
failed=()

for s in "${sets[@]}"; do
  cfg="$NT_ROOT/configs/sim/tunnel_$s.toml"
  [ -f "$cfg" ] || { echo "!! $s：没有配置 $cfg"; failed+=("$s"); continue; }
  read -r out est < <(read_cfg "$cfg") || { echo "!! $s：配置读取失败"; failed+=("$s"); continue; }
  if [ -f "$out/done.json" ]; then
    lost=$(python3 -c "import json;print(json.load(open('$out/done.json')).get('lost',0))" 2>/dev/null || echo 1)
    if [ "$lost" = "0" ]; then
      echo "跳过 $s：已完成且没有丢帧"
      continue
    fi
    echo "!! $s：$out 上一次丢了 $lost 帧，需要人工处理后再重渲染"
    failed+=("$s"); continue
  fi
  if [ -e "$out" ]; then
    echo "!! $s：$out 已存在但没有 done.json（上次中断），需要人工处理后再重渲染"
    failed+=("$s"); continue
  fi

  need_mb=$(( est * MB_PER_FRAME ))
  mkdir -p "$(dirname "$out")"
  free_mb=$(df -Pm "$(dirname "$out")" | awk 'NR==2{print $4}')
  if [ $(( free_mb - RESERVE_GB * 1024 )) -lt "$need_mb" ]; then
    echo "!! $s：磁盘空间不够（需要约 $need_mb MB + 余量 ${RESERVE_GB} GB，现有 $free_mb MB），已停止"
    failed+=("$s")
    break
  fi

  echo "== $s → $out（预计 $est 帧、约 ${need_mb} MB，剩余 ${free_mb} MB）"
  BEVY_ASSET_ROOT="$DAEDALUS" DAEDALUS_DATASET="$cfg" DAEDALUS_HEADLESS=1 \
    DAEDALUS_MAX_FPS=0 DAEDALUS_DISABLE_TALOS=1 cargo run --release --features talos \
    > "/tmp/tunnel_render_$s.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out/done.json" ]; then
    echo "!! $s：渲染失败（退出码 $rc），日志 /tmp/tunnel_render_$s.log"
    failed+=("$s")
    continue
  fi
  python3 -c "import json;print(json.load(open('$out/done.json')))"
  lost=$(python3 -c "import json;print(json.load(open('$out/done.json')).get('lost',0))")
  if [ "$lost" != "0" ]; then
    echo "!! $s：丢了 $lost 帧（多半是写盘失败），不能使用"
    failed+=("$s")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "失败或跳过的数据集：${failed[*]}"
  exit 1
fi
echo "全部完成"
