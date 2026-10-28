#!/usr/bin/env bash
# Build export/cpp_parity/armor_decode.cpp together with sleepy's armor_infer.cpp and run it on a raw output tensor.
#
#   scripts/run_cpp_parity.sh build              compile (needs sleepy configured: build/compile_commands.json, libsleepy_utils.so)
#   scripts/run_cpp_parity.sh run OUT.f32 [ROWS] [CONF]
#
# SLEEPY_DIR defaults to ~/sleepy. Nothing in that directory is modified; the binary goes to build/cpp_parity/.
# The include and define flags are copied from sleepy's own compile_commands.json for armor_infer.cpp, so the
# harness sees the same headers and configuration as the real decoder.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SLEEPY_DIR=${SLEEPY_DIR:-$HOME/sleepy}
BIN=$ROOT/build/cpp_parity/armor_decode
LIB=$SLEEPY_DIR/build

build() {
    [[ -f $LIB/libsleepy_utils.so && -f $LIB/compile_commands.json ]] || { echo "build sleepy first: $LIB/libsleepy_utils.so or compile_commands.json is missing" >&2; exit 1; }
    mkdir -p "$ROOT/build/cpp_parity"
    # Pull compiler, defines and include paths for armor_infer.cpp out of sleepy's compile database.
    mapfile -t FLAGS < <(python3 -I - "$LIB/compile_commands.json" <<'EOF'
import json, shlex, sys
for e in json.load(open(sys.argv[1])):
    if e["file"].endswith("armor_infer.cpp"):
        toks = shlex.split(e["command"])
        print(toks[0])
        keep, skip = [], False
        for t in toks[1:]:
            if skip:
                skip = False
                continue
            if t in ("-o", "-c"):
                skip = t == "-o"
                continue
            if t.endswith(".cpp") or t.endswith(".o"):
                continue
            if t.startswith(("-D", "-I", "-isystem", "-std=", "-O")) or t in ("-isystem",):
                keep.append(t)
            elif keep and keep[-1] == "-isystem":
                keep.append(t)
        print("\n".join(keep))
        break
EOF
)
    CXX=${FLAGS[0]}
    # Compile the decoder from the checkout's current source instead of linking libsleepy_auto_aim.so: the
    # built library can be older than the source, and a parity test must see the code as it is now.
    "$CXX" "${FLAGS[@]:1}" -fPIC -o "$BIN" "$ROOT/export/cpp_parity/armor_decode.cpp" \
        "$SLEEPY_DIR/src/tasks/auto_aim/armor_detect/armor_infer.cpp" \
        -L"$LIB" -lsleepy_utils \
        -L/usr/local/lib -lopencv_core -lyaml-cpp -lspdlog -lfmt \
        -Wl,-rpath,"$LIB" -Wl,-rpath,/usr/local/lib
    echo "built $BIN"
}

case "${1:-}" in
    build) build ;;
    run)
        shift
        SRC=$SLEEPY_DIR/src/tasks/auto_aim/armor_detect/armor_infer.cpp
        [[ -x $BIN && $BIN -nt $SRC && $BIN -nt $ROOT/export/cpp_parity/armor_decode.cpp ]] || build
        exec "$BIN" "$@"
        ;;
    *) echo "usage: run_cpp_parity.sh build | run OUT.f32 [ROWS] [CONF]" >&2; exit 2 ;;
esac
