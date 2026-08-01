/* s2pro-native — slow-AR private CUDA launchers (implemented in kernels.cu).
 * Private header: src/slowar only. Extends the frozen include/s2pro/kernels.h
 * with the batched-decode variants (per-row positions + per-row KV caches)
 * that lockstep multi-session decode needs, plus small glue kernels.
 *
 * C-compat shim: cuda_bf16.h defines __nv_bfloat16 only under C++, but the
 * frozen kernels.h (and src/fastar/fastar.h) name it in prototypes. From C11
 * we provide a layout-identical POD (one unsigned short) BEFORE including
 * them; only pointers to it cross the ABI, so this is exact.
 */
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#if !defined(__cplusplus) && !defined(S2P_SLOWAR_BF16_SHIM)
#define S2P_SLOWAR_BF16_SHIM
typedef struct {
    unsigned short x;
} __nv_bfloat16;
#endif

#include "s2pro/kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Split fused qkv [rows, q_width + 2*kv_width] into contiguous q [rows,
 * q_width], k [rows, kv_width], v [rows, kv_width] (concat order q,k,v). */
cudaError_t s2pk_qkv_split(const __nv_bfloat16* qkv, __nv_bfloat16* q,
                           __nv_bfloat16* k, __nv_bfloat16* v, int rows,
                           int q_width, int kv_width, cudaStream_t st);

/* RoPE with per-row positions (lockstep batch decode; same bf16-truncated
 * table math as the contract s2pk_rope). pos is a device array [rows]. */
cudaError_t s2pk_rope_pos(__nv_bfloat16* q, __nv_bfloat16* k, int rows,
                          int q_heads, int kv_heads, int head_dim,
                          const int32_t* pos, float rope_base,
                          cudaStream_t st);

/* KV append with per-row caches + positions: row r writes its k/v vector at
 * position pos[r] of k_caches[r]/v_caches[r] (each [kv_heads,max_seq,hd]). */
cudaError_t s2pk_kv_append_ptrs(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                __nv_bfloat16* const* k_caches,
                                __nv_bfloat16* const* v_caches, int rows,
                                int kv_heads, int head_dim, const int32_t* pos,
                                int max_seq, cudaStream_t st);

/* Causal GQA decode attention with per-row caches: row r attends over
 * cache[0..pos[r]] of its own cache (append first). q/out [rows, q_heads*hd]. */
cudaError_t s2pk_attention_ptrs(const __nv_bfloat16* q,
                                const __nv_bfloat16* const* k_caches,
                                const __nv_bfloat16* const* v_caches,
                                __nv_bfloat16* out, int rows, int q_heads,
                                int kv_heads, int head_dim, const int32_t* pos,
                                int max_seq, cudaStream_t st);

/* In-place x[i] = bf16(f32(x[i]) / divisor). Replicates the reference prefill
 * VQ scale `x / sqrt(num_codebooks+1)` (a DIVISION — decode uses a multiply;
 * the two round differently in bf16, so both forms exist). */
cudaError_t s2pk_scale_div(__nv_bfloat16* x, int64_t n, float divisor,
                           cudaStream_t st);

/* In-place row-masked x[r,:] = bf16(f32(x[r,:]) * scale) where mask[r] != 0.
 * Replicates the reference decode VQ scale `(embed + vq_sum) * (1/sqrt(11))`
 * applied only at semantic-token rows. */
cudaError_t s2pk_scale_rows_masked(__nv_bfloat16* x, const uint8_t* mask,
                                   int rows, int dim, float scale,
                                   cudaStream_t st);

/* dst[r*stride] = src[r] — writes the sampled semantic id into slot 0 of each
 * frame's out_codes row before the fast-AR fills slots 1..9. */
cudaError_t s2pk_i32_scatter_stride(int32_t* dst, const int32_t* src, int rows,
                                    int stride, cudaStream_t st);

#ifdef __cplusplus
}
#endif
