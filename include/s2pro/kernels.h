/* s2pro-native — shared CUDA kernel launchers (BF16 transformer primitives).
 * Contract header: frozen. Implemented in src/slowar/kernels.cu, used by both
 * slow-AR and fast-AR (fast-AR: no qk-norm — pass qnorm/knorm = NULL).
 *
 * All pointers are device pointers. All launchers return cudaGetLastError()
 * mapped to s2p_status via S2P_CUDA_TRY at the call site convention:
 * they return cudaError_t so callers own the mapping.
 */
#pragma once

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y = rmsnorm(x) * w, row-wise over width; rows x width, eps = S2P_NORM_EPS */
cudaError_t s2pk_rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* w,
                          __nv_bfloat16* y, int rows, int width, float eps,
                          cudaStream_t st);

/* In-place per-head RMSNorm on q or k: [rows, heads, head_dim] */
cudaError_t s2pk_qk_norm(__nv_bfloat16* qk, const __nv_bfloat16* w, int rows,
                         int heads, int head_dim, float eps, cudaStream_t st);

/* In-place RoPE on q and k, positions pos..pos+rows-1.
 * MUST replicate the reference bf16 truncation behavior (PORTING pitfall):
 * cos/sin computed in fp32, cast to bf16, applied in bf16. */
cudaError_t s2pk_rope(__nv_bfloat16* q, __nv_bfloat16* k, int rows,
                      int q_heads, int kv_heads, int head_dim, int pos,
                      float rope_base, cudaStream_t st);

/* Append k,v ([rows, kv_heads*head_dim] each) to cache at position pos.
 * Cache layout: [kv_heads, max_seq, head_dim] per tensor. */
cudaError_t s2pk_kv_append(const __nv_bfloat16* k, const __nv_bfloat16* v,
                           __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                           int rows, int kv_heads, int head_dim, int pos,
                           int max_seq, cudaStream_t st);

/* Causal GQA attention over the cache, decode (rows small) and prefill.
 * q [rows, q_heads*head_dim]; out same shape. seq_len = pos+rows. */
cudaError_t s2pk_attention(const __nv_bfloat16* q,
                           const __nv_bfloat16* k_cache,
                           const __nv_bfloat16* v_cache, __nv_bfloat16* out,
                           int rows, int q_heads, int kv_heads, int head_dim,
                           int pos, int max_seq, cudaStream_t st);

/* h = silu(gate) * up where gu = [gate || up] is [rows, 2*ffn] fused. */
cudaError_t s2pk_silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* h, int rows,
                          int ffn, cudaStream_t st);

/* y += x (residual add), n elements */
cudaError_t s2pk_add(__nv_bfloat16* y, const __nv_bfloat16* x, int64_t n,
                     cudaStream_t st);

/* Gather rows from an embedding table: out[i] = table[ids[i]] */
cudaError_t s2pk_embed(const __nv_bfloat16* table, const int64_t* ids,
                       __nv_bfloat16* out, int rows, int dim, cudaStream_t st);

/* Sampling (slow-AR semantic head, PORTING §sampling — two-softmax order,
 * repetition penalty over a trailing window of semantic history):
 * logits [vocab] bf16 -> sampled id. State lives host-side in the session. */
cudaError_t s2pk_sample_prepare(const __nv_bfloat16* logits, float* scratch,
                                int vocab, cudaStream_t st);

/* argmax over logits [rows, vocab] -> ids[rows] (fast-AR residuals) */
cudaError_t s2pk_argmax(const __nv_bfloat16* logits, int32_t* ids, int rows,
                        int vocab, cudaStream_t st);

#ifdef __cplusplus
}
#endif
