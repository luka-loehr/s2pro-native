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

/* ---------------------------------------------- split-K flash decode ----- */
/* One block = (row, kv_head, split of 64 positions). The K tile (64x128
 * bf16, 16 KB) is loaded into shared memory coalesced ONCE and scored
 * against the 4 q_heads sharing this kv_head; then the same shared buffer
 * is reloaded with the V tile for the weighted accumulation. Per-split
 * partials (m, l, acc[128]) go to `part`; k_attn_combine merges them. */
#define S2PK_SPLIT 64
#define S2PK_GROUP 4
#define S2PK_HD 128

static __global__ void k_attn_split(const __nv_bfloat16* __restrict__ q,
                                    const __nv_bfloat16* const* k_caches,
                                    const __nv_bfloat16* const* v_caches,
                                    int q_heads, const int32_t* __restrict__
                                    pos, int max_seq, int n_splits,
                                    float* __restrict__ part) {
    const int row = blockIdx.x;
    const int kvh = blockIdx.y;
    const int spl = blockIdx.z;
    const int tid = threadIdx.x;              /* 128 threads */
    const int len = pos[row] + 1;
    const int start = spl * S2PK_SPLIT;
    const int qh0 = kvh * S2PK_GROUP;

    /* header slot even when the split is past this row's length */
    float* pbase = part + (((size_t)row * q_heads + qh0) * n_splits + spl) *
                              (S2PK_HD + 2);
    if (start >= len) {
        if (tid < S2PK_GROUP) {
            float* ph = pbase + (size_t)tid * n_splits * (S2PK_HD + 2);
            ph[0] = -INFINITY;
            ph[1] = 0.f;
        }
        return;
    }
    const int cnt = min(S2PK_SPLIT, len - start);

    __shared__ float tile[S2PK_SPLIT * S2PK_HD / 2]; /* bf16x2 as float bits */
    __shared__ float sq[S2PK_GROUP][S2PK_HD];
    __shared__ float sc[S2PK_GROUP][S2PK_SPLIT];
    __shared__ float red[128];
    __shared__ float hm[S2PK_GROUP], hs[S2PK_GROUP];

    const float scale = rsqrtf((float)S2PK_HD);
    /* queries, pre-scaled */
    for (int i = tid; i < S2PK_GROUP * S2PK_HD; i += blockDim.x)
        sq[i / S2PK_HD][i % S2PK_HD] =
            s2pk_b2f(q[((size_t)row * q_heads + qh0 + i / S2PK_HD) * S2PK_HD +
                       i % S2PK_HD]) *
            scale;

    /* K tile, coalesced (each thread streams consecutive bf16x2 pairs) */
    const __nv_bfloat16* kc =
        k_caches[row] + ((size_t)kvh * max_seq + start) * S2PK_HD;
    const uint32_t* ksrc = (const uint32_t*)kc;
    uint32_t* tdst = (uint32_t*)tile;
    for (int i = tid; i < cnt * S2PK_HD / 2; i += blockDim.x) tdst[i] = ksrc[i];
    __syncthreads();

    /* scores: thread t handles position t (t < cnt) for each q head */
    if (tid < cnt) {
        const __nv_bfloat162* kr =
            (const __nv_bfloat162*)(tdst + (size_t)tid * (S2PK_HD / 2));
        for (int g = 0; g < S2PK_GROUP; g++) {
            float d = 0.f;
            const float* qv = sq[g];
            for (int e = 0; e < S2PK_HD / 2; e++) {
                float2 kf = __bfloat1622float2(kr[e]);
                d = fmaf(qv[2 * e], kf.x, d);
                d = fmaf(qv[2 * e + 1], kf.y, d);
            }
            sc[g][tid] = d;
        }
    }
    __syncthreads();

    /* per-head max + exp-sum over the tile */
    for (int g = 0; g < S2PK_GROUP; g++) {
        float v = (tid < cnt) ? sc[g][tid] : -INFINITY;
        red[tid] = v;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        const float m = red[0];
        __syncthreads();
        float p = (tid < cnt) ? expf(sc[g][tid] - m) : 0.f;
        if (tid < S2PK_SPLIT) sc[g][tid] = p;
        red[tid] = p;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) {
            hm[g] = m;
            hs[g] = red[0];
        }
        __syncthreads();
    }

    /* V tile replaces K in shared memory */
    const __nv_bfloat16* vc =
        v_caches[row] + ((size_t)kvh * max_seq + start) * S2PK_HD;
    const uint32_t* vsrc = (const uint32_t*)vc;
    __syncthreads();
    for (int i = tid; i < cnt * S2PK_HD / 2; i += blockDim.x) tdst[i] = vsrc[i];
    __syncthreads();

    /* weighted V: thread t accumulates output dim t for each head */
    for (int g = 0; g < S2PK_GROUP; g++) {
        float acc = 0.f;
        for (int j = 0; j < cnt; j++) {
            const __nv_bfloat16* vr =
                (const __nv_bfloat16*)(tdst + (size_t)j * (S2PK_HD / 2));
            acc = fmaf(sc[g][j], s2pk_b2f(vr[tid]), acc);
        }
        float* ph = pbase + (size_t)g * n_splits * (S2PK_HD + 2);
        if (tid == 0) {
            ph[0] = hm[g];
            ph[1] = hs[g];
        }
        ph[2 + tid] = acc;
    }
}

/* Merge the per-split partials: out = sum_s exp(m_s - M) * acc_s / l. */
static __global__ void k_attn_combine(const float* __restrict__ part,
                                      __nv_bfloat16* __restrict__ out,
                                      int q_heads, int n_splits) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;              /* 128 threads = head_dim */
    const float* pb =
        part + (((size_t)row * q_heads + h) * n_splits) * (S2PK_HD + 2);
    float M = -INFINITY;
    for (int s = 0; s < n_splits; s++)
        M = fmaxf(M, pb[(size_t)s * (S2PK_HD + 2)]);
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < n_splits; s++) {
        const float* ph = pb + (size_t)s * (S2PK_HD + 2);
        if (ph[1] == 0.f) continue;
        const float w = expf(ph[0] - M);
        l = fmaf(ph[1], w, l);
        acc = fmaf(ph[2 + tid], w, acc);
    }
    out[((size_t)row * q_heads + h) * S2PK_HD + tid] = s2pk_f2b(acc / l);
}

extern "C" cudaError_t s2pk_attention_decode(
    const __nv_bfloat16* q, const __nv_bfloat16* const* k_caches,
    const __nv_bfloat16* const* v_caches, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, const int32_t* pos, int max_seq,
    int max_len, float* part, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim != S2PK_HD || q_heads != kv_heads * S2PK_GROUP || !part)
        return cudaErrorInvalidValue;
    const int n_splits = (max_len + S2PK_SPLIT - 1) / S2PK_SPLIT;
    dim3 g1((unsigned)rows, (unsigned)kv_heads, (unsigned)n_splits);
    k_attn_split<<<g1, 128, 0, st>>>(q, k_caches, v_caches, q_heads, pos,
                                     max_seq, n_splits, part);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return ce;
    dim3 g2((unsigned)rows, (unsigned)q_heads);
    k_attn_combine<<<g2, S2PK_HD, 0, st>>>(part, out, q_heads, n_splits);
    return cudaGetLastError();
}

/* ----------------------------------------------------- INT8 KV cache ---- */
/* Payload i8 [kv_heads,max_seq,head_dim], scales f16 [kv_heads,max_seq,
 * head_dim/32]. Quantization on append: per 32-group absmax/127 rounded
 * through f16, values rounded to nearest-even and clamped to ±127; dequant
 * multiplies by exactly the stored f16 scale. Warp == group when
 * head_dim == 128 and blockDim == 128 (the only served shape). */

/* Bit width is a template parameter: BW==8 stores one int8 per element,
 * BW==4 packs two 4-bit values per byte (element 2b in the low nibble,
 * 2b+1 in the high). Scales stay f16 per 32-element group either way, so
 * only the payload plane halves. */
template <int BW>
static __global__ void k_kv_append8(const __nv_bfloat16* k,
                                    const __nv_bfloat16* v, int8_t* kc0,
                                    int8_t* vc0, __half* ks0, __half* vs0,
                                    int8_t* const* kcs, int8_t* const* vcs,
                                    __half* const* kss, __half* const* vss,
                                    int kv_heads, int head_dim, int pos0,
                                    const int32_t* pos_arr, int max_seq) {
    const int r = blockIdx.x / kv_heads;
    const int h = blockIdx.x % kv_heads;
    const int d = threadIdx.x;           /* 0..head_dim-1 */
    const int g = d >> 5;                /* 32-group == warp */
    const int p = pos_arr ? pos_arr[r] : pos0 + r;
    int8_t* kc = kcs ? kcs[r] : kc0;
    int8_t* vc = vcs ? vcs[r] : vc0;
    __half* ks = kss ? kss[r] : ks0;
    __half* vs = vss ? vss[r] : vs0;
    const size_t src = ((size_t)r * kv_heads + h) * head_dim + d;
    const size_t dst = ((size_t)h * max_seq + p) * head_dim + d;
    const size_t sdst = ((size_t)h * max_seq + p) * (head_dim >> 5) + g;

    const int QMAX = (BW == 8) ? 127 : 7;
    const size_t pdst = (BW == 8) ? dst
                                  : ((size_t)h * max_seq + p) * (head_dim / 2)
                                        + (d >> 1);

    float kx = s2pk_b2f(k[src]);
    float a = fabsf(kx);
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    __half hsc = __float2half_rn(a / (float)QMAX);
    float sc = __half2float(hsc);
    int q8 = (sc > 0.f) ? __float2int_rn(kx / sc) : 0;
    q8 = max(-QMAX, min(QMAX, q8));
    if (BW == 8) {
        kc[dst] = (int8_t)q8;
    } else {
        /* thread d (even) owns the byte holding d and d+1 */
        int hi = __shfl_down_sync(0xffffffffu, q8, 1);
        if ((d & 1) == 0)
            kc[pdst] = (int8_t)((q8 & 0xF) | ((hi & 0xF) << 4));
    }
    if ((d & 31) == 0) ks[sdst] = hsc;

    float vx = s2pk_b2f(v[src]);
    a = fabsf(vx);
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    hsc = __float2half_rn(a / (float)QMAX);
    sc = __half2float(hsc);
    q8 = (sc > 0.f) ? __float2int_rn(vx / sc) : 0;
    q8 = max(-QMAX, min(QMAX, q8));
    if (BW == 8) {
        vc[dst] = (int8_t)q8;
    } else {
        int hi = __shfl_down_sync(0xffffffffu, q8, 1);
        if ((d & 1) == 0)
            vc[pdst] = (int8_t)((q8 & 0xF) | ((hi & 0xF) << 4));
    }
    if ((d & 31) == 0) vs[sdst] = hsc;
}

extern "C" cudaError_t s2pk_kv_append8(const __nv_bfloat16* k,
                                       const __nv_bfloat16* v, int8_t* k_cache,
                                       int8_t* v_cache, void* k_scale,
                                       void* v_scale, int rows, int kv_heads,
                                       int head_dim, int pos, int max_seq,
                                       int bw, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim != 128) return cudaErrorInvalidValue;
    if (bw == 4)
        k_kv_append8<4><<<rows * kv_heads, head_dim, 0, st>>>(
            k, v, k_cache, v_cache, (__half*)k_scale, (__half*)v_scale, NULL,
            NULL, NULL, NULL, kv_heads, head_dim, pos, NULL, max_seq);
    else
        k_kv_append8<8><<<rows * kv_heads, head_dim, 0, st>>>(
            k, v, k_cache, v_cache, (__half*)k_scale, (__half*)v_scale, NULL,
            NULL, NULL, NULL, kv_heads, head_dim, pos, NULL, max_seq);
    return cudaGetLastError();
}

extern "C" cudaError_t s2pk_kv_append8_ptrs(
    const __nv_bfloat16* k, const __nv_bfloat16* v, int8_t* const* k_caches,
    int8_t* const* v_caches, void* const* k_scales, void* const* v_scales,
    int rows, int kv_heads, int head_dim, const int32_t* pos, int max_seq,
    int bw, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim != 128) return cudaErrorInvalidValue;
    if (bw == 4)
        k_kv_append8<4><<<rows * kv_heads, head_dim, 0, st>>>(
            k, v, NULL, NULL, NULL, NULL, k_caches, v_caches,
            (__half* const*)k_scales, (__half* const*)v_scales, kv_heads,
            head_dim, 0, pos, max_seq);
    else
        k_kv_append8<8><<<rows * kv_heads, head_dim, 0, st>>>(
            k, v, NULL, NULL, NULL, NULL, k_caches, v_caches,
            (__half* const*)k_scales, (__half* const*)v_scales, kv_heads,
            head_dim, 0, pos, max_seq);
    return cudaGetLastError();
}

/* Single-pass online-softmax attention over the INT8 cache; structure and
 * f32 accumulation order identical to k_attention, K/V dequantized in
 * registers (per-group partial dot x f16 scale). */
template <int BW>
static __global__ void k_attention8(const __nv_bfloat16* q,
                                    const int8_t* kc0, const int8_t* vc0,
                                    const __half* ks0, const __half* vs0,
                                    __nv_bfloat16* out, int q_heads,
                                    int kv_heads, int head_dim, int pos0,
                                    int max_seq) {
    __shared__ float sq[S2PK_ATTN_TILE];
    __shared__ float sp[S2PK_ATTN_TILE];
    __shared__ float red[S2PK_ATTN_TILE];
    __shared__ float s_tile[2];

    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int len = pos0 + row + 1;
    const int kvh = h / (q_heads / kv_heads);
    const int ng = head_dim >> 5;

    const int elems = (BW == 8) ? head_dim : head_dim / 2; /* bytes/vector */
    const int8_t* kc = kc0 + (size_t)kvh * max_seq * elems;
    const int8_t* vc = vc0 + (size_t)kvh * max_seq * elems;
    const __half* ks = ks0 + (size_t)kvh * max_seq * ng;
    const __half* vs = vs0 + (size_t)kvh * max_seq * ng;

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
            const int8_t* kr = kc + (size_t)j * elems;
            const __half* kr_s = ks + (size_t)j * ng;
            float d = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                float pp = 0.f;
                const int base = gi << 5;
                for (int e = 0; e < 32; e++) {
                    float kv;
                    if (BW == 8) {
                        kv = (float)kr[base + e];
                    } else {
                        const int idx = base + e;
                        const int8_t byte = kr[idx >> 1];
                        const int nib = (idx & 1) ? (byte >> 4) : (byte & 0xF);
                        kv = (float)(int8_t)((nib << 4) >> 4);
                    }
                    pp = fmaf(sq[base + e], kv, pp);
                }
                d = fmaf(pp, __half2float(kr_s[gi]), d);
            }
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
            const int8_t* vb = vc + (size_t)t0 * elems;
            const __half* vbs = vs + (size_t)t0 * ng;
            const int vg = tid >> 5;
            for (int j2 = 0; j2 < lim; j2++) {
                float vv;
                if (BW == 8) {
                    vv = (float)vb[(size_t)j2 * elems + tid];
                } else {
                    const int8_t byte = vb[(size_t)j2 * elems + (tid >> 1)];
                    const int nib = (tid & 1) ? (byte >> 4) : (byte & 0xF);
                    vv = (float)(int8_t)((nib << 4) >> 4);
                }
                acc = fmaf(sp[j2] * __half2float(vbs[(size_t)j2 * ng + vg]),
                           vv, acc);
            }
        }
        m = m_new;
        __syncthreads();
    }
    if (tid < head_dim)
        out[((size_t)row * q_heads + h) * head_dim + tid] = s2pk_f2b(acc / l);
}

extern "C" cudaError_t s2pk_attention8(const __nv_bfloat16* q,
                                       const int8_t* k_cache,
                                       const int8_t* v_cache,
                                       const void* k_scale,
                                       const void* v_scale,
                                       __nv_bfloat16* out, int rows,
                                       int q_heads, int kv_heads,
                                       int head_dim, int pos, int max_seq,
                                       int bw, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim > S2PK_ATTN_TILE || (head_dim & 31) != 0)
        return cudaErrorInvalidValue;
    dim3 grid(rows, q_heads);
    if (bw == 4)
        k_attention8<4><<<grid, S2PK_ATTN_TILE, 0, st>>>(
            q, k_cache, v_cache, (const __half*)k_scale,
            (const __half*)v_scale, out, q_heads, kv_heads, head_dim, pos,
            max_seq);
    else
        k_attention8<8><<<grid, S2PK_ATTN_TILE, 0, st>>>(
            q, k_cache, v_cache, (const __half*)k_scale,
            (const __half*)v_scale, out, q_heads, kv_heads, head_dim, pos,
            max_seq);
    return cudaGetLastError();
}

/* Split-K flash decode over the INT8 cache: K/V tiles staged in shared
 * memory as int8 + f16 scales (half the bf16 tile traffic); partial layout
 * and the combine kernel are shared with the bf16 path. */
template <int BW>
static __global__ void k_attn_split8(const __nv_bfloat16* __restrict__ q,
                                     const int8_t* const* k_caches,
                                     const int8_t* const* v_caches,
                                     const __half* const* k_scales,
                                     const __half* const* v_scales,
                                     int q_heads,
                                     const int32_t* __restrict__ pos,
                                     int max_seq, int n_splits,
                                     float* __restrict__ part) {
    const int row = blockIdx.x;
    const int kvh = blockIdx.y;
    const int spl = blockIdx.z;
    const int tid = threadIdx.x; /* 128 threads */
    const int len = pos[row] + 1;
    const int start = spl * S2PK_SPLIT;
    const int qh0 = kvh * S2PK_GROUP;
    const int NG = S2PK_HD >> 5; /* 4 scale groups */

    float* pbase = part + (((size_t)row * q_heads + qh0) * n_splits + spl) *
                              (S2PK_HD + 2);
    if (start >= len) {
        if (tid < S2PK_GROUP) {
            float* ph = pbase + (size_t)tid * n_splits * (S2PK_HD + 2);
            ph[0] = -INFINITY;
            ph[1] = 0.f;
        }
        return;
    }
    const int cnt = min(S2PK_SPLIT, len - start);

    /* u32-typed shared arrays so the coalesced u32 tile copies are aligned;
     * aliased as int8/f16 for the compute reads. */
    const int EL = (BW == 8) ? S2PK_HD : S2PK_HD / 2; /* bytes per vector */
    /* row stride EL+4: at stride 128 B every lane of the K-score phase
     * (lane t walks row t) lands on the same bank — a 32-way conflict on
     * every read. One u32 of skew makes the word stride 33 (coprime with
     * 32), spreading rows across all banks; same values, same order. */
    const int ELP = EL + 4;
    __shared__ uint32_t tile8_u[S2PK_SPLIT * (S2PK_HD + 4) / 4];
    __shared__ uint32_t stile_u[S2PK_SPLIT * (S2PK_HD >> 5) / 2]; /* scales */
    int8_t* tile8 = (int8_t*)tile8_u;
    __half* stile = (__half*)stile_u;
    __shared__ float sq[S2PK_GROUP][S2PK_HD];
    __shared__ float sc[S2PK_GROUP][S2PK_SPLIT];
    __shared__ float red[128];
    __shared__ float hm[S2PK_GROUP], hs[S2PK_GROUP];

    const float scale = rsqrtf((float)S2PK_HD);
    for (int i = tid; i < S2PK_GROUP * S2PK_HD; i += blockDim.x)
        sq[i / S2PK_HD][i % S2PK_HD] =
            s2pk_b2f(q[((size_t)row * q_heads + qh0 + i / S2PK_HD) * S2PK_HD +
                       i % S2PK_HD]) *
            scale;

    /* K tile + K scales, coalesced as u32 words */
    const int8_t* kc = k_caches[row] + ((size_t)kvh * max_seq + start) * EL;
    const __half* ksrow = k_scales[row] + ((size_t)kvh * max_seq + start) * NG;
    {
        const uint32_t* src = (const uint32_t*)kc;
        const int wpr = EL / 4; /* payload words per row */
        for (int i = tid; i < cnt * wpr; i += blockDim.x)
            tile8_u[(i / wpr) * (ELP / 4) + i % wpr] = src[i];
        const uint32_t* ssrc = (const uint32_t*)ksrow;
        for (int i = tid; i < cnt * NG / 2; i += blockDim.x)
            stile_u[i] = ssrc[i];
    }
    __syncthreads();

    if (tid < cnt) {
        const int8_t* kr = tile8 + (size_t)tid * ELP;
        const __half* krs = stile + (size_t)tid * NG;
        for (int g = 0; g < S2PK_GROUP; g++) {
            const float* qv = sq[g];
            float d = 0.f;
            for (int gi = 0; gi < NG; gi++) {
                float pp = 0.f;
                const int base = gi << 5;
                for (int e = 0; e < 32; e++) {
                    float kv;
                    if (BW == 8) {
                        kv = (float)kr[base + e];
                    } else {
                        const int idx = base + e;
                        const int8_t byte = kr[idx >> 1];
                        const int nib = (idx & 1) ? (byte >> 4) : (byte & 0xF);
                        kv = (float)(int8_t)((nib << 4) >> 4);
                    }
                    pp = fmaf(qv[base + e], kv, pp);
                }
                d = fmaf(pp, __half2float(krs[gi]), d);
            }
            sc[g][tid] = d;
        }
    }
    __syncthreads();

    for (int g = 0; g < S2PK_GROUP; g++) {
        float v = (tid < cnt) ? sc[g][tid] : -INFINITY;
        red[tid] = v;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        const float m = red[0];
        __syncthreads();
        float p = (tid < cnt) ? expf(sc[g][tid] - m) : 0.f;
        if (tid < S2PK_SPLIT) sc[g][tid] = p;
        red[tid] = p;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) {
            hm[g] = m;
            hs[g] = red[0];
        }
        __syncthreads();
    }

    /* V tile + V scales replace K in shared memory */
    const int8_t* vc = v_caches[row] + ((size_t)kvh * max_seq + start) * EL;
    const __half* vsrow = v_scales[row] + ((size_t)kvh * max_seq + start) * NG;
    __syncthreads();
    {
        const uint32_t* src = (const uint32_t*)vc;
        const int wpr = EL / 4;
        for (int i = tid; i < cnt * wpr; i += blockDim.x)
            tile8_u[(i / wpr) * (ELP / 4) + i % wpr] = src[i];
        const uint32_t* ssrc = (const uint32_t*)vsrow;
        for (int i = tid; i < cnt * NG / 2; i += blockDim.x)
            stile_u[i] = ssrc[i];
    }
    __syncthreads();

    for (int g = 0; g < S2PK_GROUP; g++) {
        float acc = 0.f;
        const int vg = tid >> 5;
        for (int j = 0; j < cnt; j++) {
            float vv;
            if (BW == 8) {
                vv = (float)tile8[(size_t)j * ELP + tid];
            } else {
                const int8_t byte = tile8[(size_t)j * ELP + (tid >> 1)];
                const int nib = (tid & 1) ? (byte >> 4) : (byte & 0xF);
                vv = (float)(int8_t)((nib << 4) >> 4);
            }
            acc = fmaf(sc[g][j] * __half2float(stile[(size_t)j * NG + vg]),
                       vv, acc);
        }
        float* ph = pbase + (size_t)g * n_splits * (S2PK_HD + 2);
        if (tid == 0) {
            ph[0] = hm[g];
            ph[1] = hs[g];
        }
        ph[2 + tid] = acc;
    }
}

extern "C" cudaError_t s2pk_attention8_decode(
    const __nv_bfloat16* q, const int8_t* const* k_caches,
    const int8_t* const* v_caches, const void* const* k_scales,
    const void* const* v_scales, __nv_bfloat16* out, int rows, int q_heads,
    int kv_heads, int head_dim, const int32_t* pos, int max_seq, int max_len,
    float* part, int bw, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    if (head_dim != S2PK_HD || q_heads != kv_heads * S2PK_GROUP || !part)
        return cudaErrorInvalidValue;
    const int n_splits = (max_len + S2PK_SPLIT - 1) / S2PK_SPLIT;
    dim3 g1((unsigned)rows, (unsigned)kv_heads, (unsigned)n_splits);
    if (bw == 4)
        k_attn_split8<4><<<g1, 128, 0, st>>>(
            q, k_caches, v_caches, (const __half* const*)k_scales,
            (const __half* const*)v_scales, q_heads, pos, max_seq, n_splits,
            part);
    else
        k_attn_split8<8><<<g1, 128, 0, st>>>(
            q, k_caches, v_caches, (const __half* const*)k_scales,
            (const __half* const*)v_scales, q_heads, pos, max_seq, n_splits,
            part);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return ce;
    dim3 g2((unsigned)rows, (unsigned)q_heads);
    k_attn_combine<<<g2, S2PK_HD, 0, st>>>(part, out, q_heads, n_splits);
    return cudaGetLastError();
}

/* ------------------------------------------------- device semantic sampler */
/* Exact port of s2ps_sample (sampling.c): candidate gather, RAS detect,
 * repetition penalty, top-30, top-p over the un-temperatured softmax, second
 * softmax with temperature, seeded hash-Gumbel or xoshiro inverse-CDF draw,
 * in-place history/state update. Constants mirror slowar_internal.h. */
#define SMP_N_SEM 4096
#define SMP_N_CAND 4097
#define SMP_CAND_EOS 4096
#define SMP_TOP_K 30
#define SMP_RAS_TEMP 1.0f
#define SMP_RAS_TOP_P 0.9f
#define SMP_SEM_START 151678
#define SMP_SEM_END 155773
#define SMP_EOS 151645

static __device__ uint64_t smp_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static __device__ uint64_t smp_xoshiro_next(uint64_t s[4]) {
    const uint64_t result = smp_rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = smp_rotl64(s[3], 45);
    return result;
}

static __global__ void k_sample(const __nv_bfloat16* __restrict__ logits,
                                int64_t vocab_stride,
                                const __nv_bfloat16* __restrict__ eos_logits,
                                s2ps_dev_state* const* __restrict__ states,
                                int64_t* __restrict__ out_tok,
                                int32_t* __restrict__ out_sem,
                                int32_t* __restrict__ out_codes,
                                uint8_t* __restrict__ out_eos) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    s2ps_dev_state* sp = states[row];
    const __nv_bfloat16* lrow = logits + (size_t)row * vocab_stride;

    __shared__ float cand[SMP_N_CAND];
    __shared__ float rv[128];
    __shared__ int ri[128];
    __shared__ float tv[SMP_TOP_K];
    __shared__ int ti[SMP_TOP_K];

    /* compact mode (eos_logits != NULL): `logits` holds only the 4096
     * semantic rows (stride = vocab_stride = 4096) and the EOS logit
     * arrives separately — same values as the full-vocab gather, so the
     * sampled trajectory is bit-identical. */
    if (eos_logits != NULL) {
        for (int i = tid; i < SMP_N_SEM; i += blockDim.x)
            cand[i] = s2pk_b2f(lrow[i]);
        if (tid == 0) cand[SMP_CAND_EOS] = s2pk_b2f(eos_logits[row]);
    } else {
        for (int i = tid; i < SMP_N_SEM; i += blockDim.x)
            cand[i] = s2pk_b2f(lrow[SMP_SEM_START + i]);
        if (tid == 0) cand[SMP_CAND_EOS] = s2pk_b2f(lrow[SMP_EOS]);
    }
    __syncthreads();

    const int greedy = sp->temperature == 0.0f;
    const int capped = sp->count < (uint64_t)sp->window ? (int)sp->count
                                                        : sp->window;
    __shared__ float sh_temp, sh_topp;
    if (tid == 0) {
        /* RAS detect: duplicate among the clamped last 4 of the window */
        int use_ras = 0;
        if (!greedy && capped >= 4) {
            int64_t last4[4];
            for (int r = 0; r < 4; r++) {
                int idx = capped - (4 - r);
                if (idx < 0) idx = 0;
                last4[r] = sp->prev[idx];
            }
            for (int a = 0; a < 3; a++)
                for (int b = a + 1; b < 4; b++)
                    if (last4[b] < last4[a]) {
                        int64_t t = last4[a];
                        last4[a] = last4[b];
                        last4[b] = t;
                    }
            for (int a = 0; a < 3; a++)
                if (last4[a] == last4[a + 1]) use_ras = 1;
        }
        sh_temp = use_ras ? SMP_RAS_TEMP : sp->temperature;
        sh_topp = use_ras ? SMP_RAS_TOP_P : sp->top_p;

        /* repetition penalty: gather originals, then scatter */
        if (capped > 0) {
            int ridx[64];
            float rval[64];
            int rn = 0;
            for (int j = 0; j < capped; j++) {
                const int64_t tok = sp->prev[j];
                int ci;
                if (tok >= SMP_SEM_START && tok <= SMP_SEM_END)
                    ci = (int)(tok - SMP_SEM_START);
                else if (tok == SMP_EOS)
                    ci = SMP_CAND_EOS;
                else
                    continue;
                const float v = cand[ci];
                ridx[rn] = ci;
                rval[rn] = v < 0.f ? v * sp->rep_penalty
                                   : v / sp->rep_penalty;
                rn++;
            }
            for (int j = 0; j < rn; j++) cand[ridx[j]] = rval[j];
        }
    }
    __syncthreads();

    /* top-30 by iterative argmax (ties -> lowest index, matching torch.topk
     * descending-value / ascending-index order) */
    for (int k = 0; k < SMP_TOP_K; k++) {
        float bv = -INFINITY;
        int bi = SMP_N_CAND;
        for (int i = tid; i < SMP_N_CAND; i += blockDim.x) {
            const float v = cand[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        rv[tid] = bv;
        ri[tid] = bi;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) {
                if (rv[tid + off] > rv[tid] ||
                    (rv[tid + off] == rv[tid] && ri[tid + off] < ri[tid])) {
                    rv[tid] = rv[tid + off];
                    ri[tid] = ri[tid + off];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            tv[k] = rv[0];
            ti[k] = ri[0];
            cand[ri[0]] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid != 0) return;

    /* ---- serial tail on thread 0 (30 elements) ---- */
    int choice = 0;
    if (!greedy) {
        float e[SMP_TOP_K], sum = 0.f;
        const float m = tv[0];
        for (int j = 0; j < SMP_TOP_K; j++) {
            e[j] = expf(tv[j] - m);
            sum += e[j];
        }
        int masked[SMP_TOP_K] = {0};
        float cum = 0.f;
        for (int j = 0; j < SMP_TOP_K; j++) {
            cum += e[j] / sum;
            if (j > 0 && cum > sh_topp) masked[j] = 1;
        }
        const float t = sh_temp < 1e-5f ? 1e-5f : sh_temp;
        float p2[SMP_TOP_K], s2 = 0.f;
        const float m2 = tv[0] / t;
        for (int j = 0; j < SMP_TOP_K; j++) {
            p2[j] = masked[j] ? 0.f : expf(tv[j] / t - m2);
            s2 += p2[j];
        }
        for (int j = 0; j < SMP_TOP_K; j++) p2[j] /= s2;

        if (sp->seeded) {
            const uint64_t step_seed = ((uint64_t)sp->seed31 * 19349663ULL) ^
                                       (sp->count * 73856093ULL);
            float best = -INFINITY;
            for (int j = 0; j < SMP_TOP_K; j++) {
                const uint64_t hashed = (step_seed * 8589934591ULL) ^
                                        ((uint64_t)j * 479001599ULL);
                float u =
                    (float)(uint32_t)(hashed & 0xFFFFFFULL) / 16777216.0f;
                if (u < 1e-10f) u = 1e-10f;
                const float g = -logf(-logf(u));
                const float pert = logf(p2[j] + 1e-10f) + g;
                if (pert > best) {
                    best = pert;
                    choice = j;
                }
            }
        } else {
            const double r =
                (double)(smp_xoshiro_next(sp->rng) >> 11) * 0x1.0p-53;
            double cum2 = 0.0;
            int lastnz = 0;
            choice = -1;
            for (int j = 0; j < SMP_TOP_K; j++) {
                if (p2[j] <= 0.f) continue;
                lastnz = j;
                cum2 += (double)p2[j];
                if (r < cum2) {
                    choice = j;
                    break;
                }
            }
            if (choice < 0) choice = lastnz;
        }
    }

    const int ci = ti[choice];
    const int64_t tok = ci == SMP_CAND_EOS ? (int64_t)SMP_EOS
                                           : (int64_t)(SMP_SEM_START + ci);
    if (tok == SMP_EOS) {
        out_tok[row] = tok;
        out_sem[row] = 0;
        out_codes[(size_t)row * 10] = 0;
        out_eos[row] = 1;
        return;
    }
    out_tok[row] = tok;
    out_sem[row] = ci;
    out_codes[(size_t)row * 10] = ci;
    out_eos[row] = 0;
    if (sp->count < (uint64_t)sp->window) {
        sp->prev[sp->count] = tok;
    } else {
        for (int j = 0; j < sp->window - 1; j++) sp->prev[j] = sp->prev[j + 1];
        sp->prev[sp->window - 1] = tok;
    }
    sp->count++;
}

extern "C" cudaError_t s2pk_sample(const __nv_bfloat16* logits,
                                   int64_t vocab_stride,
                                   const __nv_bfloat16* eos_logits,
                                   s2ps_dev_state* const* states, int rows,
                                   int64_t* out_tok, int32_t* out_sem,
                                   int32_t* out_codes, uint8_t* out_eos,
                                   cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    k_sample<<<rows, 128, 0, st>>>(logits, vocab_stride, eos_logits,
                                   states, out_tok,
                                   out_sem, out_codes, out_eos);
    return cudaGetLastError();
}

static __global__ void k_pack_frame(const int32_t* __restrict__ stage,
                                    int rows, int32_t* __restrict__ codes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 9) return;
    int cb = i / rows;          /* 0..8 -> codebook cb+1 */
    int b = i % rows;
    codes[(size_t)b * 10 + cb + 1] = stage[(size_t)cb * rows + b];
}

extern "C" cudaError_t s2pk_pack_frame(const int32_t* stage, int rows,
                                       int32_t* codes, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int n = rows * 9;
    k_pack_frame<<<(n + 127) / 128, 128, 0, st>>>(stage, rows, codes);
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

/* INT8-sidecar embedding lookup (S2P_EMBED_BF16=0 default in INT8 mode):
 * out = bf16(i8[id][d] * row_scale[id]) — the bf16 table is dropped and
 * the per-row-quantized tied-head sidecar serves lookups too. */
static __global__ void k_embed_i8(const int8_t* table, const float* scales,
                                  const int64_t* ids, __nv_bfloat16* out,
                                  int64_t total, int dim) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = idx / dim;
        const int d = (int)(idx % dim);
        const int64_t id = ids[r];
        out[idx] = s2pk_f2b((float)table[(size_t)id * dim + d] * scales[id]);
    }
}

extern "C" cudaError_t s2pk_embed_i8(const int8_t* table, const float* scales,
                                     const int64_t* ids, __nv_bfloat16* out,
                                     int rows, int dim, cudaStream_t st) {
    if (rows <= 0) return cudaSuccess;
    int64_t total = (int64_t)rows * dim;
    k_embed_i8<<<s2pk_blocks(total, S2PK_THREADS), S2PK_THREADS, 0, st>>>(
        table, scales, ids, out, total, dim);
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
