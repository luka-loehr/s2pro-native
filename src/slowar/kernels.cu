/* s2pro-native — shared BF16 transformer primitives (contract: kernels.h)
 * plus slow-AR private batched-decode kernels (slowar_kernels.h).
 *
 * Numerics contract (docs/PORTING.md §8):
 *  - RoPE: interleaved gpt-fast pairs (2i,2i+1); angles in fp32; cos/sin
 *    ROUNDED THROUGH BF16 before use; butterfly in fp32 with per-op IEEE
 *    rounding (__fmul_rn/__fadd_rn — no FMA contraction so the reference
 *    mul-then-add rounding sequence is preserved); result rounded to bf16.
 *  - RMSNorm: eps INSIDE the rsqrt, mean over the last dim, fp32 math,
 *    weight multiply in fp32, single final rounding (nn.RMSNorm semantics).
 *  - SwiGLU: silu(gate) rounds to bf16 BEFORE the up-multiply (two torch ops,
 *    two roundings), matching `F.silu(gate) * up`.
 *  - Elementwise adds compute in fp32 and round once (torch bf16 add).
 *  - Attention: fp32 accumulate online-softmax over the bf16 cache; ties and
 *    exp/rsqrt ulp differences are within the free-running parity budget.
 *  - argmax: lowest index wins on exact ties (torch.argmax semantics) — this
 *    IS load-bearing for teacher-forced residual parity.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "s2pro/kernels.h"
#include "slowar_kernels.h"

#define S2PK_THREADS 256
#define S2PK_MAX_GRID 4096

static inline int s2pk_blocks(int64_t total, int threads) {
    int64_t b = (total + threads - 1) / threads;
    if (b > S2PK_MAX_GRID) b = S2PK_MAX_GRID;
    if (b < 1) b = 1;
    return (int)b;
}

__device__ __forceinline__ float s2pk_b2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 s2pk_f2b(float v) {
    return __float2bfloat16(v); /* round-to-nearest-even, as torch .to(bf16) */
}

/* ---------------------------------------------------------------- rms_norm */

static __global__ void k_rms_norm(const __nv_bfloat16* x,
                                  const __nv_bfloat16* w, __nv_bfloat16* y,
                                  int width, float eps) {
    __shared__ float red[S2PK_THREADS];
    const int row = blockIdx.x;
    const size_t base = (size_t)row * width;
    float s = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = s2pk_b2f(x[base + i]);
        s += v * v;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = s2pk_b2f(x[base + i]);
        y[base + i] = s2pk_f2b(v * inv * s2pk_b2f(w[i]));
    }
}

extern "C" cudaError_t s2pk_rms_norm(const __nv_bfloat16* x,
                                     const __nv_bfloat16* w, __nv_bfloat16* y,
                                     int rows, int width, float eps,
                                     cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    k_rms_norm<<<rows, S2PK_THREADS, 0, st>>>(x, w, y, width, eps);
    return cudaGetLastError();
}

/* ----------------------------------------------------------------- qk_norm */

static __global__ void k_qk_norm(__nv_bfloat16* qk, const __nv_bfloat16* w,
                                 int head_dim, float eps) {
    __shared__ float red[S2PK_THREADS];
    const size_t base = (size_t)blockIdx.x * head_dim; /* block = row*head */
    float s = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = s2pk_b2f(qk[base + i]);
        s += v * v;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = s2pk_b2f(qk[base + i]);
        qk[base + i] = s2pk_f2b(v * inv * s2pk_b2f(w[i]));
    }
}

extern "C" cudaError_t s2pk_qk_norm(__nv_bfloat16* qk, const __nv_bfloat16* w,
                                    int rows, int heads, int head_dim,
                                    float eps, cudaStream_t st) {
    if (rows <= 0 || w == NULL) return cudaSuccess; /* fast-AR passes NULL */
    k_qk_norm<<<rows * heads, 128, 0, st>>>(qk, w, head_dim, eps);
    return cudaGetLastError();
}

/* -------------------------------------------------------------------- rope */

/* One tensor (q or k). pos_arr == NULL -> positions pos0..pos0+rows-1. */
static __global__ void k_rope(__nv_bfloat16* x, int rows, int heads,
                              int head_dim, int pos0, const int32_t* pos_arr,
                              float rope_base) {
    const int pairs = head_dim / 2;
    const int64_t total = (int64_t)rows * heads * pairs;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int r = (int)(idx / ((int64_t)heads * pairs));
        const int rem = (int)(idx % ((int64_t)heads * pairs));
        const int h = rem / pairs;
        const int i = rem % pairs;
        const int p = pos_arr ? pos_arr[r] : pos0 + r;
        /* inv_freq = 1 / base^(2i/dim), all fp32 like the reference table. */
        const float ex = __fdiv_rn((float)(2 * i), (float)head_dim);
        const float inv = __fdiv_rn(1.0f, powf(rope_base, ex));
        const float ang = __fmul_rn((float)p, inv);
        /* PORTING pitfall 1: cos/sin rounded through bf16 before use. */
        const float c = s2pk_b2f(s2pk_f2b(cosf(ang)));
        const float s = s2pk_b2f(s2pk_f2b(sinf(ang)));
        const size_t o = ((size_t)r * heads + h) * head_dim + 2 * i;
        const float x0 = s2pk_b2f(x[o]);
        const float x1 = s2pk_b2f(x[o + 1]);
        x[o] = s2pk_f2b(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
        x[o + 1] = s2pk_f2b(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
    }
}

static cudaError_t rope_launch(__nv_bfloat16* q, __nv_bfloat16* k, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               int pos0, const int32_t* pos_arr,
                               float rope_base, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t tq = (int64_t)rows * q_heads * (head_dim / 2);
    k_rope<<<s2pk_blocks(tq, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        q, rows, q_heads, head_dim, pos0, pos_arr, rope_base);
    int64_t tk = (int64_t)rows * kv_heads * (head_dim / 2);
    k_rope<<<s2pk_blocks(tk, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        k, rows, kv_heads, head_dim, pos0, pos_arr, rope_base);
    return cudaGetLastError();
}

extern "C" cudaError_t s2pk_rope(__nv_bfloat16* q, __nv_bfloat16* k, int rows,
                                 int q_heads, int kv_heads, int head_dim,
                                 int pos, float rope_base, cudaStream_t st) {
    return rope_launch(q, k, rows, q_heads, kv_heads, head_dim, pos, NULL,
                       rope_base, st);
}

extern "C" cudaError_t s2pk_rope_pos(__nv_bfloat16* q, __nv_bfloat16* k,
                                     int rows, int q_heads, int kv_heads,
                                     int head_dim, const int32_t* pos,
                                     float rope_base, cudaStream_t st) {
    return rope_launch(q, k, rows, q_heads, kv_heads, head_dim, 0, pos,
                       rope_base, st);
}

/* --------------------------------------------------------------- kv append */

static __global__ void k_kv_append(const __nv_bfloat16* k,
                                   const __nv_bfloat16* v,
                                   __nv_bfloat16* k_cache,
                                   __nv_bfloat16* v_cache,
                                   __nv_bfloat16* const* k_caches,
                                   __nv_bfloat16* const* v_caches, int rows,
                                   int kv_heads, int head_dim, int pos0,
                                   const int32_t* pos_arr, int max_seq) {
    const int64_t total = (int64_t)rows * kv_heads * head_dim;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int r = (int)(idx / ((int64_t)kv_heads * head_dim));
        const int rem = (int)(idx % ((int64_t)kv_heads * head_dim));
        const int h = rem / head_dim;
        const int d = rem % head_dim;
        const int p = pos_arr ? pos_arr[r] : pos0 + r;
        __nv_bfloat16* kc = k_caches ? k_caches[r] : k_cache;
        __nv_bfloat16* vc = v_caches ? v_caches[r] : v_cache;
        const size_t src = (size_t)r * kv_heads * head_dim + rem;
        const size_t dst = ((size_t)h * max_seq + p) * head_dim + d;
        kc[dst] = k[src];
        vc[dst] = v[src];
    }
}

extern "C" cudaError_t s2pk_kv_append(const __nv_bfloat16* k,
                                      const __nv_bfloat16* v,
                                      __nv_bfloat16* k_cache,
                                      __nv_bfloat16* v_cache, int rows,
                                      int kv_heads, int head_dim, int pos,
                                      int max_seq, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * kv_heads * head_dim;
    k_kv_append<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        k, v, k_cache, v_cache, NULL, NULL, rows, kv_heads, head_dim, pos,
        NULL, max_seq);
    return cudaGetLastError();
}

extern "C" cudaError_t s2pk_kv_append_ptrs(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* const* k_caches, __nv_bfloat16* const* v_caches, int rows,
    int kv_heads, int head_dim, const int32_t* pos, int max_seq,
    cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * kv_heads * head_dim;
    k_kv_append<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        k, v, NULL, NULL, k_caches, v_caches, rows, kv_heads, head_dim, 0, pos,
        max_seq);
    return cudaGetLastError();
}

/* --------------------------------------------------------------- attention */

/* Single-pass online-softmax causal GQA over a [kv_heads,max_seq,head_dim]
 * cache. Block = (row, q_head); 128 threads: thread j scores kv position
 * tile_start+j, thread d accumulates output dim d. fp32 accumulate.
 * Serves prefill (rows=L, pos0 base, row i sees 0..pos0+i) and decode
 * (per-row caches + per-row pos, row r sees 0..pos[r]).
 * Requires head_dim <= 128 (S2P_HEAD_DIM == 128). */
#define S2PK_ATTN_TILE 128

static __global__ void k_attention(const __nv_bfloat16* q,
                                   const __nv_bfloat16* k_cache0,
                                   const __nv_bfloat16* v_cache0,
                                   const __nv_bfloat16* const* k_caches,
                                   const __nv_bfloat16* const* v_caches,
                                   __nv_bfloat16* out, int q_heads,
                                   int kv_heads, int head_dim, int pos0,
                                   const int32_t* pos_arr, int max_seq) {
    __shared__ float sq[S2PK_ATTN_TILE];  /* scaled query vector */
    __shared__ float sp[S2PK_ATTN_TILE];  /* tile probabilities */
    __shared__ float red[S2PK_ATTN_TILE]; /* reduction scratch */
    __shared__ float s_tile[2];           /* [0]=tile max, [1]=tile sum */

    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int len = (pos_arr ? pos_arr[row] : pos0 + row) + 1;
    const int kvh = h / (q_heads / kv_heads);

    const __nv_bfloat16* kc =
        (k_caches ? k_caches[row] : k_cache0) + (size_t)kvh * max_seq * head_dim;
    const __nv_bfloat16* vc =
        (v_caches ? v_caches[row] : v_cache0) + (size_t)kvh * max_seq * head_dim;

    const float scale = rsqrtf((float)head_dim);
    if (tid < head_dim)
        sq[tid] = s2pk_b2f(q[((size_t)row * q_heads + h) * head_dim + tid]) *
                  scale;
    __syncthreads();

    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int t0 = 0; t0 < len; t0 += S2PK_ATTN_TILE) {
        const int j = t0 + tid;
        float s = -INFINITY;
        if (j < len) {
            const __nv_bfloat16* kr = kc + (size_t)j * head_dim;
            float d = 0.f;
            for (int e = 0; e < head_dim; e++)
                d = fmaf(sq[e], s2pk_b2f(kr[e]), d);
            s = d;
        }
        red[tid] = s;
        __syncthreads();
        for (int off = S2PK_ATTN_TILE >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        if (tid == 0) s_tile[0] = red[0];
        __syncthreads();
        const float m_new = fmaxf(m, s_tile[0]);
        const float corr = (m == -INFINITY) ? 0.f : expf(m - m_new);
        const float p = (s == -INFINITY) ? 0.f : expf(s - m_new);
        sp[tid] = p;
        red[tid] = p;
        __syncthreads();
        for (int off = S2PK_ATTN_TILE >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) s_tile[1] = red[0];
        __syncthreads();
        l = l * corr + s_tile[1];
        if (tid < head_dim) {
            acc *= corr;
            const int lim = min(S2PK_ATTN_TILE, len - t0);
            const __nv_bfloat16* vb = vc + (size_t)t0 * head_dim;
            for (int j2 = 0; j2 < lim; j2++)
                acc = fmaf(sp[j2], s2pk_b2f(vb[(size_t)j2 * head_dim + tid]),
                           acc);
        }
        m = m_new;
        __syncthreads(); /* shared reuse in next tile */
    }
    if (tid < head_dim)
        out[((size_t)row * q_heads + h) * head_dim + tid] = s2pk_f2b(acc / l);
}

extern "C" cudaError_t s2pk_attention(const __nv_bfloat16* q,
                                      const __nv_bfloat16* k_cache,
                                      const __nv_bfloat16* v_cache,
                                      __nv_bfloat16* out, int rows,
                                      int q_heads, int kv_heads, int head_dim,
                                      int pos, int max_seq, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim > S2PK_ATTN_TILE) return cudaErrorInvalidValue;
    dim3 grid(rows, q_heads);
    k_attention<<<grid, S2PK_ATTN_TILE, 0, st>>>(q, k_cache, v_cache, NULL,
                                                 NULL, out, q_heads, kv_heads,
                                                 head_dim, pos, NULL, max_seq);
    return cudaGetLastError();
}

extern "C" cudaError_t s2pk_attention_ptrs(
    const __nv_bfloat16* q, const __nv_bfloat16* const* k_caches,
    const __nv_bfloat16* const* v_caches, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, const int32_t* pos, int max_seq,
    cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim > S2PK_ATTN_TILE) return cudaErrorInvalidValue;
    dim3 grid(rows, q_heads);
    k_attention<<<grid, S2PK_ATTN_TILE, 0, st>>>(q, NULL, NULL, k_caches,
                                                 v_caches, out, q_heads,
                                                 kv_heads, head_dim, 0, pos,
                                                 max_seq);
    return cudaGetLastError();
}

/* ---------------------------------------------------------------- silu_mul */

static __global__ void k_silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* h,
                                  int64_t total, int ffn) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / ffn;
        const int c = (int)(idx % ffn);
        const float g = s2pk_b2f(gu[(size_t)r * 2 * ffn + c]);
        const float u = s2pk_b2f(gu[(size_t)r * 2 * ffn + ffn + c]);
        /* Two torch ops = two roundings: silu(g) -> bf16, then * up -> bf16 */
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
        const float sil = s2pk_b2f(s2pk_f2b(__fmul_rn(g, sig)));
        h[idx] = s2pk_f2b(__fmul_rn(sil, u));
    }
}

extern "C" cudaError_t s2pk_silu_mul(const __nv_bfloat16* gu, __nv_bfloat16* h,
                                     int rows, int ffn, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * ffn;
    k_silu_mul<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        gu, h, total, ffn);
    return cudaGetLastError();
}

/* --------------------------------------------------------------------- add */

static __global__ void k_add(__nv_bfloat16* y, const __nv_bfloat16* x,
                             int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        y[i] = s2pk_f2b(__fadd_rn(s2pk_b2f(y[i]), s2pk_b2f(x[i])));
}

extern "C" cudaError_t s2pk_add(__nv_bfloat16* y, const __nv_bfloat16* x,
                                int64_t n, cudaStream_t st) {
    if (n <= 0) return cudaSuccess;
    k_add<<<s2pk_blocks(n, S2PK_THREADS), S2PK_THREADS, 0, st>>>(y, x, n);
    return cudaGetLastError();
}

/* ------------------------------------------------------------------- embed */

static __global__ void k_embed(const __nv_bfloat16* table, const int64_t* ids,
                               __nv_bfloat16* out, int64_t total, int dim) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / dim;
        const int d = (int)(idx % dim);
        out[idx] = table[(size_t)ids[r] * dim + d];
    }
}

extern "C" cudaError_t s2pk_embed(const __nv_bfloat16* table,
                                  const int64_t* ids, __nv_bfloat16* out,
                                  int rows, int dim, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * dim;
    k_embed<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        table, ids, out, total, dim);
    return cudaGetLastError();
}

/* ---------------------------------------------------------- sample_prepare */

/* Widen bf16 logits to fp32 device scratch. The v1 sampler downloads the
 * candidate slice host-side, but the contract keeps a device fp32 staging
 * path open for later device-side sampling. */
static __global__ void k_widen(const __nv_bfloat16* logits, float* scratch,
                               int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        scratch[i] = s2pk_b2f(logits[i]);
}

extern "C" cudaError_t s2pk_sample_prepare(const __nv_bfloat16* logits,
                                           float* scratch, int vocab,
                                           cudaStream_t st) {
    if (vocab <= 0) return cudaSuccess;
    k_widen<<<s2pk_blocks(vocab, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        logits, scratch, vocab);
    return cudaGetLastError();
}

/* ------------------------------------------------------------------ argmax */

static __global__ void k_argmax(const __nv_bfloat16* logits, int32_t* ids,
                                int vocab) {
    __shared__ float rv[S2PK_THREADS];
    __shared__ int32_t ri[S2PK_THREADS];
    const int row = blockIdx.x;
    const size_t base = (size_t)row * vocab;
    float bv = -INFINITY;
    int32_t bi = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = s2pk_b2f(logits[base + i]);
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    rv[threadIdx.x] = bv;
    ri[threadIdx.x] = bi;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            const float ov = rv[threadIdx.x + off];
            const int32_t oi = ri[threadIdx.x + off];
            if (ov > rv[threadIdx.x] ||
                (ov == rv[threadIdx.x] && oi < ri[threadIdx.x])) {
                rv[threadIdx.x] = ov;
                ri[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ids[row] = ri[0];
}

extern "C" cudaError_t s2pk_argmax(const __nv_bfloat16* logits, int32_t* ids,
                                   int rows, int vocab, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    k_argmax<<<rows, S2PK_THREADS, 0, st>>>(logits, ids, vocab);
    return cudaGetLastError();
}

/* --------------------------------------------------------------- qkv split */

static __global__ void k_qkv_split(const __nv_bfloat16* qkv, __nv_bfloat16* q,
                                   __nv_bfloat16* k, __nv_bfloat16* v,
                                   int64_t total, int q_width, int kv_width) {
    const int w = q_width + 2 * kv_width;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / w;
        const int c = (int)(idx % w);
        const __nv_bfloat16 val = qkv[idx];
        if (c < q_width)
            q[r * q_width + c] = val;
        else if (c < q_width + kv_width)
            k[r * kv_width + (c - q_width)] = val;
        else
            v[r * kv_width + (c - q_width - kv_width)] = val;
    }
}

extern "C" cudaError_t s2pk_qkv_split(const __nv_bfloat16* qkv,
                                      __nv_bfloat16* q, __nv_bfloat16* k,
                                      __nv_bfloat16* v, int rows, int q_width,
                                      int kv_width, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * (q_width + 2 * kv_width);
    k_qkv_split<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        qkv, q, k, v, total, q_width, kv_width);
    return cudaGetLastError();
}

/* ----------------------------------------------------------- VQ scale glue */

static __global__ void k_scale_div(__nv_bfloat16* x, int64_t n, float div) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        x[i] = s2pk_f2b(__fdiv_rn(s2pk_b2f(x[i]), div));
}

extern "C" cudaError_t s2pk_scale_div(__nv_bfloat16* x, int64_t n,
                                      float divisor, cudaStream_t st) {
    if (n <= 0) return cudaSuccess;
    k_scale_div<<<s2pk_blocks(n, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        x, n, divisor);
    return cudaGetLastError();
}

static __global__ void k_scale_rows_masked(__nv_bfloat16* x,
                                           const uint8_t* mask, int64_t total,
                                           int dim, float scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        if (mask[idx / dim])
            x[idx] = s2pk_f2b(__fmul_rn(s2pk_b2f(x[idx]), scale));
    }
}

extern "C" cudaError_t s2pk_scale_rows_masked(__nv_bfloat16* x,
                                              const uint8_t* mask, int rows,
                                              int dim, float scale,
                                              cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * dim;
    k_scale_rows_masked<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0,
                          st>>>(x, mask, total, dim, scale);
    return cudaGetLastError();
}

/* ----------------------------------------------------------- misc scatter */

static __global__ void k_i32_scatter_stride(int32_t* dst, const int32_t* src,
                                            int rows, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) dst[(size_t)i * stride] = src[i];
}

extern "C" cudaError_t s2pk_i32_scatter_stride(int32_t* dst,
                                               const int32_t* src, int rows,
                                               int stride, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    const int blocks = (rows + 127) / 128;
    k_i32_scatter_stride<<<blocks, 128, 0, st>>>(dst, src, rows, stride);
    return cudaGetLastError();
}
