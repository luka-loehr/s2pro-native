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
} s2p_gemm_mode;

/* Global gemm context (cublas handle + fp8 scratch). One per process. */
s2p_status    s2p_gemm_init(int max_m);
void          s2p_gemm_shutdown(void);
s2p_gemm_mode s2p_gemm_mode_from_env(void); /* env S2P_FP8=1 -> FP8 */
int           s2p_fso_available(void);      /* 1 if FP8 kernels linked+arch ok */

typedef struct {
    int        in_features;
    int        out_features;
    s2p_tensor w_bf16;        /* [out, in] device BF16, always present */
    /* FP8 sidecar (present when fp8_ready): */
    s2p_tensor w_fp8;         /* [out, in] e4m3 */
    s2p_tensor w_scales;      /* int32-packed UE8M0 sfb, sm_120/121 layout */
    int        fp8_ready;
} s2p_linear;

s2p_status s2p_linear_from_st(s2p_linear* lin, s2p_st* st, const char* name,
                              int in_features, int out_features,
                              cudaStream_t stream);
s2p_status s2p_linear_prepare_fp8(s2p_linear* lin, cudaStream_t stream);
s2p_status s2p_linear_forward(const s2p_linear* lin, const void* x_bf16,
                              void* y_bf16, int M, s2p_gemm_mode mode,
                              cudaStream_t stream);
void       s2p_linear_free(s2p_linear* lin);

/* Raw BF16 GEMM for non-weight matmuls (tied lm-head etc.):
 * y[M,N] = x[M,K] * w[N,K]^T via cuBLAS, FP32 accumulate. */
s2p_status s2p_gemm_bf16(const void* x, const void* w, void* y, int M, int N,
                         int K, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
