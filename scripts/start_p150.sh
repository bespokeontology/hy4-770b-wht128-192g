#!/usr/bin/env bash
set -euo pipefail

TT_TOOL="<NVME_STRIPE_0>/toolchains"
HWLOC_LIB="<OPERATOR_HOME>/.local/state/p150-recovery-20261107/runtime/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$TT_TOOL/llvm-20/lib:$TT_TOOL/llvm-deps/usr/lib/x86_64-linux-gnu:$TT_TOOL/ttnn-0.80.0rc6/build/lib:$TT_TOOL/ttnn.libs:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
export TT_METAL_RUNTIME_ROOT="$TT_TOOL/ttnn-0.80.0rc6"
export TT_METAL_HOME="$TT_TOOL/ttnn-0.80.0rc6"
export ARCH_NAME="blackhole"

exec <HY4_770B_WORKSPACE_ROOT>/bin/p150_hy4_770b_worker \
  --weight-root <HY4_770B_WEIGHT_ROOT> \
  --kernels-dir <HY4_770B_WORKSPACE_ROOT>/p150/kernels \
  --shm /hy4_770b_p150_ipc \
  "$@"
