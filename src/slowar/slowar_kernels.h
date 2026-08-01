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

/* Split-K flash-decode variant of s2pk_attention_ptrs for small `rows`:
 * grid (rows, kv_heads, seq-splits of 64), each block loads its K/V tile
 * into shared memory COALESCED once and scores all q_heads/kv_heads queries
 * of the group against it; a combine kernel merges the per-split partials
 * (log-sum-exp). Same f32 accumulation from bf16 as the single-pass kernel,
 * different summation order (parity-gated, like the INT8 GEMMs).
 * Requirements: head_dim == 128, q_heads/kv_heads == 4.
 * `max_len` = max over rows of pos[r]+1 (host-known); `part` is device
 * scratch of at least rows*q_heads*ceil(max_len/64)*(head_dim+2) floats. */
cudaError_t s2pk_attention_decode(const __nv_bfloat16* q,
                                  const __nv_bfloat16* const* k_caches,
                                  const __nv_bfloat16* const* v_caches,
                                  __nv_bfloat16* out, int rows, int q_heads,
                                  int kv_heads, int head_dim,
                                  const int32_t* pos, int max_seq,
                                  int max_len, float* part, cudaStream_t st);

/* ---- device-side semantic sampler (kernels.cu, exact port of sampling.c) */

/* Per-session sampler state, DEVICE-resident. Field semantics identical to
 * the host s2ps_sampler; initialized once per session by H2D copy. */
typedef struct {
    int64_t  prev[64];       /* rep/RAS window, oldest first */
    uint64_t count;          /* uncapped emitted count == step counter */
    uint64_t rng[4];         /* xoshiro256** (unseeded path) */
    float    temperature, top_p, rep_penalty;
    int32_t  window, seeded;
    uint32_t seed31;
    uint32_t pad_;
} s2ps_dev_state;

/* One block per row: gathers the 4097-candidate slice straight from the full
 * logits row, applies penalty/top-k/two-softmax/draw exactly like
 * s2ps_sample, updates the state in place, and writes token/sem/eos:
 *   out_tok[r] (i64), out_sem[r] (i32), out_codes[r*10+0] = sem, out_eos[r].
 * Greedy (temperature 0) is bit-identical to the host sampler; sampled draws
 * differ only by device expf/logf ULPs (parity-gated). */
cudaError_t s2pk_sample(const __nv_bfloat16* logits, int64_t vocab_stride,
                        s2ps_dev_state* const* states, int rows,
                        int64_t* out_tok, int32_t* out_sem,
                        int32_t* out_codes, uint8_t* out_eos,
                        cudaStream_t st);

/* Pack fast-AR residual stage [9][rows] into frame-major codes [rows][10]
 * (cb0 already written by s2pk_sample). */
cudaError_t s2pk_pack_frame(const int32_t* stage, int rows, int32_t* codes,
                            cudaStream_t st);

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
