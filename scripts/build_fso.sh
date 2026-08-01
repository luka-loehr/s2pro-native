#!/usr/bin/env bash
# Build the two fish-scales-ops objects this project links for the FP8 path:
#   $FSO_OUT/runner_$ARCH.o  — CUTLASS block-scale GEMM runner (torch-free)
#   $FSO_OUT/quant_$ARCH.o   — activation/weight quantize + UE8M0 repack
#
# Run inside a CUDA 13 container (see docs/SPARK.md). No PyTorch required.
# Usage: ARCH=121a FSO_DIR=/path/to/fish-scales-ops scripts/build_fso.sh
set -euo pipefail

ARCH="${ARCH:-121a}"
FSO_DIR="${FSO_DIR:-3rdparty/fish-scales-ops}"
FSO_OUT="${FSO_OUT:-$FSO_DIR/../build}"

if [ ! -d "$FSO_DIR" ]; then
    git clone --recursive --depth 1 --shallow-submodules \
        https://github.com/fishaudio/fish-scales-ops.git "$FSO_DIR"
fi

CSRC="$FSO_DIR/csrc"
CUTLASS="$FSO_DIR/3rdparty/cutlass"
mkdir -p "$FSO_OUT"

FLAGS=(-gencode=arch=compute_${ARCH},code=sm_${ARCH}
       -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda
       -Xcompiler=-Wno-psabi --diag-suppress=20012,20013,20014,177,20050
       -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
       -DENABLE_BF16 -DENABLE_FP8 -DFSO_JIT_INCLUDE_DIRS_DEFAULT=\"\"
       -DCOMPILE_HOPPER_TMA_GEMMS -DCOMPILE_HOPPER_TMA_GROUPED_GEMMS
       -I"$CSRC/gemm/include" -I"$CSRC/common/compat/include"
       -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include")

echo "[fso] runner.cu -> runner_${ARCH}.o"
nvcc -c "$CSRC/gemm/src/runner.cu"        -o "$FSO_OUT/runner_${ARCH}.o" "${FLAGS[@]}"
echo "[fso] quant_kernels.cu -> quant_${ARCH}.o"
nvcc -c "$CSRC/gemm/ops/quant_kernels.cu" -o "$FSO_OUT/quant_${ARCH}.o"  "${FLAGS[@]}"
ls -la "$FSO_OUT"
