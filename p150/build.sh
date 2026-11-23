#!/usr/bin/env bash
set -euo pipefail
# Pure C++20 TT-Metalium Build Script for 2x Tenstorrent Blackhole P150A (FABRIC_1D TP2) — Hy4 770B

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

TT_TOOL="<NVME_STRIPE_0>/toolchains"
TT_METAL_RT="$TT_TOOL/ttnn-0.80.0rc6"
TT_METAL_SRC="<HY4_770B_WORKSPACE_ROOT>/engine/engine/experimental_native_tt/tt-metal-src"
TT_DEPS="<HY4_770B_WORKSPACE_ROOT>/engine/engine/experimental_native_tt/deps"
LLVM_BIN="$TT_TOOL/llvm-20/bin"
LLVM_LIB="$TT_TOOL/llvm-20/lib"
LLVM_DEPS_LIB="$TT_TOOL/llvm-deps/usr/lib/x86_64-linux-gnu"
HWLOC_LIB="<OPERATOR_HOME>/.local/state/p150-recovery-20261107/runtime/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$LLVM_LIB:$LLVM_DEPS_LIB:$TT_METAL_RT/build/lib:$TT_TOOL/ttnn.libs:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/bin" "$ROOT/build/p150"

CXXFLAGS=(
  -std=c++20 -O3 -Wall -Wextra -Werror -Wno-unused-parameter
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
  -isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11
  -mavx2 -mf16c -mfma -fPIC -DNTEST -DFMT_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL
  -DHY4_770B_P150_TT_METAL=1
  -I"$ROOT/src" -I"$ROOT/p150"
  -isystem "$TT_METAL_SRC"
  -isystem "$TT_METAL_SRC/ttnn/api"
  -isystem "$TT_METAL_SRC/ttnn/cpp"
  -isystem "$TT_METAL_SRC/tt_metal/api"
  -isystem "$TT_METAL_SRC/tt_metal"
  -isystem "$TT_METAL_SRC/tt_metal/hostdevcommon/api"
  -isystem "$TT_METAL_SRC/tt_stl"
  -isystem "$TT_METAL_SRC/tt_metal/third_party/umd/device/api"
  -isystem "$TT_METAL_SRC/tt_metal/third_party/tracy/public"
  -isystem "$TT_DEPS/fmt-11.1.4/include"
  -isystem "$TT_DEPS/reflect-1.2.6"
  -isystem "$TT_DEPS/enchantum-8ca5b0eb7e7ebe0252e5bc6915083f1dd1b8294e/enchantum/include"
  -isystem "$TT_DEPS/json-3.11.3/include"
  -isystem "$TT_DEPS/tt-logger-1.1.8/include"
  -isystem "$TT_DEPS/spdlog-1.15.2/include"
)

LDFLAGS=(
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
  -L"$TT_METAL_RT/build/lib" -Wl,-rpath,"$TT_METAL_RT/build/lib"
  -L"$TT_TOOL/ttnn.libs" -Wl,-rpath,"$TT_TOOL/ttnn.libs"
  -L"$HWLOC_LIB" -Wl,-rpath,"$HWLOC_LIB"
  -l:_ttnncpp.so -ltt_metal -ltt_stl -l:libhwloc.so.15 -pthread -lrt
)

TARGET="${1:-p150_hy4_770b_worker}"
echo "[P150-770B-BUILD] Compiling $TARGET with clang++-20 + TT-Metalium (Blackhole FABRIC_1D)..."
"$LLVM_BIN/clang++" "${CXXFLAGS[@]}" "$HERE/$TARGET.cpp" -o "$ROOT/bin/$TARGET.new" "${LDFLAGS[@]}"
mv -f "$ROOT/bin/$TARGET.new" "$ROOT/bin/$TARGET"
echo "[P150-770B-BUILD] Built $ROOT/bin/$TARGET OK"
