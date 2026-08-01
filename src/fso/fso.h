/* s2pro-native — private extern-C surface of the fish-scales-ops FP8 shim.
 * Private header (serve builder): consumed by src/sched/gemm.c only.
 *
 * Backing implementation: src/fso/fso_wrap.cpp, linked against the prebuilt
 * sm_121a objects (runner_121a.o + quant_121a.o, see docs/SPARK.md). Call
 * sequence and shapes mirror the proven harness (~/fso-sm121-test/harness3.cu)
 * exactly:
 *   weights  (once, load): fp8bs_quantize_128x128 -> repack_ue8m0_..._sm120
 *   acts     (per call):   fp8bs_quantize_1x128_packed (use_ue8m0=true)
 *   gemm     (per call):   CutlassFp8BlockScaleGemmRunner::gemm
 *
 * Constraints (validated here, callers fall back to BF16 on violation):
 *   weight quant: N % 128 == 0 && K % 128 == 0
 *   act quant packed + sfb repack: K % 512 == 0  (Kb=K/128 must pack by 4)
 */
#pragma once

#include <stddef.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when the device compute capability is 12.x (GB10 sm_121). Cached after
 * the first call. 0 => the CUTLASS FP8 path must not be used. */
int s2p_fso_arch_ok(void);

/* Buffer sizing (bytes) so callers own all allocations.
 * Packed scale layouts are the sm_120/121 sfb layout the runner consumes. */
size_t s2p_fso_weight_fp8_bytes(int N, int K);      /* N*K e4m3           */
size_t s2p_fso_sfb_bytes(int N, int K);             /* packed wgt scales  */
size_t s2p_fso_weight_scratch_bytes(int N, int K);  /* f32 pre-repack     */
size_t s2p_fso_act_fp8_bytes(int M, int K);         /* M*K e4m3           */
size_t s2p_fso_sa_bytes(int M, int K);              /* packed act scales  */

/* Quantize a [N, K] row-major BF16 weight to 128x128 block-scaled e4m3 and
 * repack the scales into the sm_120/121 sfb layout. scratch_f32 holds the
 * intermediate float scales (s2p_fso_weight_scratch_bytes). Once at load. */
s2p_status s2p_fso_quant_weight_128x128(void* w_fp8, void* sfb_packed,
                                        void* scratch_f32, const void* w_bf16,
                                        int N, int K, cudaStream_t stream);

/* Quantize [M, K] row-major BF16 activations to 1x128 block-scaled e4m3 with
 * UE8M0-packed scales. Per forward call. */
s2p_status s2p_fso_quant_act_1x128_packed(void* x_fp8, void* sa_packed,
                                          const void* x_bf16, int M, int K,
                                          cudaStream_t stream);

/* y[M,N] bf16 = x_fp8[M,K] * w_fp8[N,K]^T with block scales. Inputs are
 * pre-quantized (the two calls above); output is BF16. */
s2p_status s2p_fso_gemm_fp8(void* y_bf16, const void* x_fp8,
                            const void* sa_packed, const void* w_fp8,
                            const void* sfb_packed, int M, int N, int K,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif
