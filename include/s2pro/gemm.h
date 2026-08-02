/* s2pro-native — linear layers: cuBLAS BF16 default, fish-scales-ops FP8 opt-in.
 * Contract header: frozen.
 *
 * Convention (row-major): x [M, in], w [out, in], y [M, out]  =>  y = x * w^T.
 * FP8 path: weights pre-quantized 128x128 block-scale ONCE at load
 * (s2p_linear_prepare_fp8), activations quantized 1x128 per call inside
 * s2p_linear_forward. Proven on GB10 sm_121: cos 0.9993 vs BF16 on the
 * shapes that matter (see docs/SPARK.md benchmark table).
 */
#pragma once

#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/tensor.h"
#include "s2pro/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    S2P_GEMM_BF16 = 0,   /* cuBLAS BF16, FP32 accumulate (default) */
    S2P_GEMM_FP8  = 1,   /* fish-scales-ops block-scaled FP8 (S2P_FP8=1) */
    S2P_GEMM_INT8 = 2,   /* per-out-channel weight-only INT8 (S2P_INT8=1):
                          * decode-M GEMV reads int8 weights + f32 row scales,
                          * activations stay BF16, FP32 accumulate; larger M
                          * dequantizes into a shared scratch and uses cuBLAS */
} s2p_gemm_mode;

/* Largest M served by the INT8 GEMV kernel; beyond this the INT8 path
 * dequantizes to BF16 scratch + cuBLAS (prefill / big lockstep batches). */
#define S2P_INT8_GEMV_MAX_M 8

/* Global gemm context (cublas handle + fp8 scratch). One per process. */
s2p_status    s2p_gemm_init(int max_m);
void          s2p_gemm_shutdown(void);
s2p_gemm_mode s2p_gemm_mode_from_env(void); /* env S2P_FP8=1 -> FP8 */
int           s2p_fso_available(void);      /* 1 if FP8 kernels linked+arch ok */

typedef struct {
    int        in_features;
    int        out_features;
    s2p_tensor w_bf16;        /* [out, in] device BF16 (freed by prepare_int8
                               * unless S2P_INT8_KEEP_BF16=1) */
    /* FP8 sidecar (present when fp8_ready): */
    s2p_tensor w_fp8;         /* [out, in] e4m3 */
    s2p_tensor w_scales;      /* int32-packed UE8M0 sfb, sm_120/121 layout */
    int        fp8_ready;
    /* INT8/INT4 sidecar (present when int8_ready): */
    s2p_tensor w_int8;        /* [out, in] int8, round-to-nearest */
    s2p_tensor w_iscale;      /* q_group==0: [out] f32 per-out-channel;
                               * q_group>0:  [out, in/q_group] f32 group-wise */
    int        int8_ready;
    int        q_group;       /* 0 = per-channel (INT8 path), else group size */
} s2p_linear;

/* Which module a linear belongs to; drives the mixed-precision policy under
 * S2P_INT4=1 (backbone -> group-wise INT4, fast-AR + head stay per-channel
 * INT8 — the small decoder run 9x per frame is the tensor most damaged by
 * 4-bit; S2P_INT4_ALL=1 forces group-wise INT4 everywhere for A/B). */
typedef enum {
    S2P_QSITE_BACKBONE = 0,
    S2P_QSITE_FASTAR   = 1,
} s2p_qsite;

s2p_status s2p_linear_from_st(s2p_linear* lin, s2p_st* st, const char* name,
                              int in_features, int out_features,
                              cudaStream_t stream);
s2p_status s2p_linear_prepare_fp8(s2p_linear* lin, cudaStream_t stream);
/* Quantize w_bf16 -> INT8 sidecar, then FREE w_bf16 (that is the RAM win;
 * env S2P_INT8_KEEP_BF16=1 keeps it for A/B). Synchronizes `stream`. */
s2p_status s2p_linear_prepare_int8(s2p_linear* lin, cudaStream_t stream);
/* Same, but the scheme (per-channel INT8 vs group-wise INT4) is chosen by the
 * mixed-precision policy from `site` and the S2P_INT4* envs (see s2p_qsite).
 * Group size: S2P_INT4_GROUP (default 32); per-group MSE clip search:
 * S2P_INT4_MSE (default 1). */
s2p_status s2p_linear_prepare_int8_site(s2p_linear* lin, s2p_qsite site,
                                        cudaStream_t stream);
s2p_status s2p_linear_forward(const s2p_linear* lin, const void* x_bf16,
                              void* y_bf16, int M, s2p_gemm_mode mode,
                              cudaStream_t stream);
void       s2p_linear_free(s2p_linear* lin);

/* Raw BF16 GEMM for non-weight matmuls (tied lm-head etc.):
 * y[M,N] = x[M,K] * w[N,K]^T via cuBLAS, FP32 accumulate. */
s2p_status s2p_gemm_bf16(const void* x, const void* w, void* y, int M, int N,
                         int K, cudaStream_t stream);

/* Raw INT8 primitives (src/sched/int8_gemv.cu), exposed for the tied lm-head
 * which quantizes the embedding table outside any s2p_linear. */
s2p_status s2p_int8_quant(void* w_i8, float* scales, const void* w_bf16,
                          int N, int K, cudaStream_t stream);
/* y[M,N] bf16 = x[M,K] bf16 * (w_i8[N,K] * scales[N])^T; M <= 8 and
 * K % 512 == 0 (one warp per row, int8x16 vector tiles). Every model K
 * (2560 / 4096 / 9728) satisfies this; other shapes take the dequant path. */
s2p_status s2p_int8_gemv(void* y_bf16, const void* x_bf16, const void* w_i8,
                         const float* scales, int M, int N, int K,
                         cudaStream_t stream);
/* w_bf16[N,K] = w_i8[N,K] * scales[N] (prefill fallback dequant). */
s2p_status s2p_int8_dequant(void* w_bf16, const void* w_i8,
                            const float* scales, int N, int K,
                            cudaStream_t stream);

/* Group-wise low-bit primitives (INT4 value precision in the int8 container).
 * scales f32 [N, K/G]; G power of two, >= 16, dividing K. levels = 7 for
 * 4-bit symmetric (15 levels); mse != 0 grid-searches the per-group clip
 * range for min MSE instead of plain absmax RTN. */
s2p_status s2p_intq_quant(void* w_i8, float* scales, const void* w_bf16,
                          int N, int K, int G, int levels, int mse,
                          cudaStream_t stream);
s2p_status s2p_intq_gemv(void* y_bf16, const void* x_bf16, const void* w_i8,
                         const float* scales, int M, int N, int K, int G,
                         cudaStream_t stream);
s2p_status s2p_intq_dequant(void* w_bf16, const void* w_i8,
                            const float* scales, int N, int K, int G,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif
