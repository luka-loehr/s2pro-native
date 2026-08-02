/* s2pro-native — per-out-channel weight-only INT8: quantize / GEMV / dequant.
 *
 * The decode workload is M<=8 rows against [N,K] weights, purely bandwidth
 * bound: reading int8 instead of bf16 halves the weight stream, which is the
 * entire speedup (7.76 GB instead of 15.5 GB per frame across backbone,
 * tied head and fast-AR cascade — see README.md §Performance).
 *
 * Layout contract (matches gemm.h): w row-major [N, K], x row-major [M, K],
 * y row-major [M, N], y = x * (w * scale[n])^T, FP32 accumulate.
 *
 * GEMV: one warp per output row. Each lane loads an int8x16 tile of the
 * weight row (int4 vector load) and the matching 16 bf16 activations; a warp
 * covers 512 K-elements per iteration, hence the K % 512 == 0 requirement.
 * Activations are tiny (M*K bf16) and stay resident in L1/L2; DRAM traffic
 * is the weight stream. Quantize/dequant are one-block-per-row and accept
 * any K.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "s2pro/status.h"
#include "s2pro/gemm.h"

#define QUANT_THREADS 256
#define GEMV_WARPS 4

static __global__ void k_quant(int8_t* q, float* scales,
                               const __nv_bfloat16* w, int K, int levels) {
    const int n = blockIdx.x;
    const __nv_bfloat16* row = w + (size_t)n * K;
    __shared__ float red[QUANT_THREADS];

    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(row[k])));
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = QUANT_THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = red[0] > 0.0f ? red[0] / (float)levels : 1.0f;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) scales[n] = scale;

    int8_t* qrow = q + (size_t)n * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int v = __float2int_rn(__bfloat162float(row[k]) * inv);
        v = v > levels ? levels : (v < -levels ? -levels : v);
        qrow[k] = (int8_t)v;
    }
}

static __global__ void k_gemv(__nv_bfloat16* y, const __nv_bfloat16* x,
                              const int8_t* w, const float* scales, int M,
                              int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS + warp;
    if (n >= N) return;
    const int8_t* wrow = w + (size_t)n * K;

    float acc[S2P_INT8_GEMV_MAX_M];
#pragma unroll
    for (int m = 0; m < S2P_INT8_GEMV_MAX_M; m++) acc[m] = 0.0f;

    for (int k0 = lane * 16; k0 < K; k0 += 32 * 16) {
        const int4 wv = *(const int4*)(wrow + k0);
        const int8_t* wq = (const int8_t*)&wv;
        for (int m = 0; m < M; m++) { /* M uniform across the warp */
            const __nv_bfloat16* xp = x + (size_t)m * K + k0;
            const int4 xa = *(const int4*)xp;
            const int4 xb = *(const int4*)(xp + 8);
            const __nv_bfloat16* x16a = (const __nv_bfloat16*)&xa;
            const __nv_bfloat16* x16b = (const __nv_bfloat16*)&xb;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++)
                s += (float)wq[j] * __bfloat162float(x16a[j]);
#pragma unroll
            for (int j = 0; j < 8; j++)
                s += (float)wq[8 + j] * __bfloat162float(x16b[j]);
            acc[m] += s;
        }
    }

#pragma unroll
    for (int m = 0; m < S2P_INT8_GEMV_MAX_M; m++)
        for (int off = 16; off > 0; off >>= 1)
            acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);

    if (lane == 0) {
        const float sc = scales[n];
        for (int m = 0; m < M; m++)
            y[(size_t)m * N + n] = __float2bfloat16(acc[m] * sc);
    }
}

static __global__ void k_dequant(__nv_bfloat16* o, const int8_t* w,
                                 const float* scales, int K) {
    const int n = blockIdx.x;
    const float sc = scales[n];
    const int8_t* wrow = w + (size_t)n * K;
    __nv_bfloat16* orow = o + (size_t)n * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        orow[k] = __float2bfloat16((float)wrow[k] * sc);
}

/* ── Group-wise low-bit variants (INT4 value precision, int8 container) ──
 *
 * One scale per G consecutive K-elements instead of per whole row. At 4 bits
 * the per-channel scheme audibly drifts over autoregressive generation (the
 * absmax of a 2560..9728-wide row is a terrible shared step size); group-wise
 * scales are the standard fix (GPTQ/AWQ/llama.cpp all quantize in groups of
 * 32..128). Symmetric, with an optional per-group MSE scale search: absmax is
 * only the largest candidate, shrinking the clip range trades rare saturation
 * for a finer step over the bulk of the distribution.
 *
 * Layout: scales f32 [N, K/G] row-major. G must be a power of two, >= 16
 * (so every 16-wide GEMV tile lies inside one group) and divide K. */

#define GQ_NCAND 32 /* MSE search grid: cand 0 == plain absmax RTN */

static __global__ void k_quant_g(int8_t* q, float* scales,
                                 const __nv_bfloat16* w, int K, int G,
                                 int levels, int mse) {
    /* one block per group: blockIdx.x = n * (K/G) + g */
    const int ng = K / G;
    const int n = blockIdx.x / ng;
    const int g = blockIdx.x - n * ng;
    const __nv_bfloat16* wp = w + (size_t)n * K + (size_t)g * G;
    __shared__ float red[QUANT_THREADS];

    float amax = 0.0f;
    for (int k = threadIdx.x; k < G; k += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(wp[k])));
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    amax = red[0];
    float scale = amax > 0.0f ? amax / (float)levels : 1.0f;

    if (mse && amax > 0.0f) {
        /* grid-search the clip range for min Σ(w - Q(w))² */
        float best_err = 3.4e38f, best_s = scale;
        for (int c = 0; c < GQ_NCAND; c++) {
            const float s = scale * (1.0f - (float)c / (2.0f * GQ_NCAND));
            const float inv = 1.0f / s;
            float err = 0.0f;
            for (int k = threadIdx.x; k < G; k += blockDim.x) {
                const float v = __bfloat162float(wp[k]);
                int qi = __float2int_rn(v * inv);
                qi = qi > levels ? levels : (qi < -levels ? -levels : qi);
                const float d = v - (float)qi * s;
                err += d * d;
            }
            __syncthreads();
            red[threadIdx.x] = err;
            __syncthreads();
            for (int r = blockDim.x / 2; r > 0; r >>= 1) {
                if (threadIdx.x < r) red[threadIdx.x] += red[threadIdx.x + r];
                __syncthreads();
            }
            if (red[0] < best_err) { best_err = red[0]; best_s = s; }
        }
        scale = best_s;
    }

    if (threadIdx.x == 0) scales[(size_t)n * ng + g] = scale;
    const float inv = 1.0f / scale;
    int8_t* qp = q + (size_t)n * K + (size_t)g * G;
    for (int k = threadIdx.x; k < G; k += blockDim.x) {
        int v = __float2int_rn(__bfloat162float(wp[k]) * inv);
        v = v > levels ? levels : (v < -levels ? -levels : v);
        qp[k] = (int8_t)v;
    }
}

static __global__ void k_gemv_g(__nv_bfloat16* y, const __nv_bfloat16* x,
                                const int8_t* w, const float* scales, int M,
                                int N, int K, int gshift) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * GEMV_WARPS + warp;
    if (n >= N) return;
    const int8_t* wrow = w + (size_t)n * K;
    const float* srow = scales + (size_t)n * (K >> gshift);

    float acc[S2P_INT8_GEMV_MAX_M];
#pragma unroll
    for (int m = 0; m < S2P_INT8_GEMV_MAX_M; m++) acc[m] = 0.0f;

    for (int k0 = lane * 16; k0 < K; k0 += 32 * 16) {
        const int4 wv = *(const int4*)(wrow + k0);
        const int8_t* wq = (const int8_t*)&wv;
        const float gs = srow[k0 >> gshift]; /* 16-tile ⊂ one group (G>=16) */
        for (int m = 0; m < M; m++) { /* M uniform across the warp */
            const __nv_bfloat16* xp = x + (size_t)m * K + k0;
            const int4 xa = *(const int4*)xp;
            const int4 xb = *(const int4*)(xp + 8);
            const __nv_bfloat16* x16a = (const __nv_bfloat16*)&xa;
            const __nv_bfloat16* x16b = (const __nv_bfloat16*)&xb;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; j++)
                s += (float)wq[j] * __bfloat162float(x16a[j]);
#pragma unroll
            for (int j = 0; j < 8; j++)
                s += (float)wq[8 + j] * __bfloat162float(x16b[j]);
            acc[m] += s * gs;
        }
    }

#pragma unroll
    for (int m = 0; m < S2P_INT8_GEMV_MAX_M; m++)
        for (int off = 16; off > 0; off >>= 1)
            acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);

    if (lane == 0)
        for (int m = 0; m < M; m++)
            y[(size_t)m * N + n] = __float2bfloat16(acc[m]);
}

static __global__ void k_dequant_g(__nv_bfloat16* o, const int8_t* w,
                                   const float* scales, int K, int gshift) {
    const int n = blockIdx.x;
    const int8_t* wrow = w + (size_t)n * K;
    const float* srow = scales + (size_t)n * (K >> gshift);
    __nv_bfloat16* orow = o + (size_t)n * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        orow[k] = __float2bfloat16((float)wrow[k] * srow[k >> gshift]);
}

static int gshift_of(int G) {
    if (G < 16 || (G & (G - 1)) != 0) return -1;
    int s = 0;
    while ((1 << s) < G) s++;
    return s;
}

extern "C" s2p_status s2p_intq_quant(void* w_i8, float* scales,
                                     const void* w_bf16, int N, int K, int G,
                                     int levels, int mse,
                                     cudaStream_t stream) {
    if (!w_i8 || !scales || !w_bf16 || N <= 0 || K <= 0 || levels <= 0)
        return S2P_ERR_INVALID;
    if (gshift_of(G) < 0 || K % G != 0) return S2P_ERR_INVALID;
    const int threads = G < QUANT_THREADS ? G : QUANT_THREADS;
    k_quant_g<<<(unsigned)N * (K / G), threads, 0, stream>>>(
        (int8_t*)w_i8, scales, (const __nv_bfloat16*)w_bf16, K, G, levels,
        mse);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] intq quant launch (N=%d,K=%d,G=%d): %s\n", N,
                K, G, cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

extern "C" s2p_status s2p_intq_gemv(void* y_bf16, const void* x_bf16,
                                    const void* w_i8, const float* scales,
                                    int M, int N, int K, int G,
                                    cudaStream_t stream) {
    if (!y_bf16 || !x_bf16 || !w_i8 || !scales || M <= 0 || N <= 0 || K <= 0)
        return S2P_ERR_INVALID;
    const int gs = gshift_of(G);
    if (M > S2P_INT8_GEMV_MAX_M || K % 512 != 0 || gs < 0 || K % G != 0)
        return S2P_ERR_INVALID;
    const int blocks = (N + GEMV_WARPS - 1) / GEMV_WARPS;
    k_gemv_g<<<blocks, GEMV_WARPS * 32, 0, stream>>>(
        (__nv_bfloat16*)y_bf16, (const __nv_bfloat16*)x_bf16,
        (const int8_t*)w_i8, scales, M, N, K, gs);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] intq gemv launch (M=%d,N=%d,K=%d): %s\n", M,
                N, K, cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

extern "C" s2p_status s2p_intq_dequant(void* w_bf16, const void* w_i8,
                                       const float* scales, int N, int K,
                                       int G, cudaStream_t stream) {
    if (!w_bf16 || !w_i8 || !scales || N <= 0 || K <= 0) return S2P_ERR_INVALID;
    const int gs = gshift_of(G);
    if (gs < 0 || K % G != 0) return S2P_ERR_INVALID;
    k_dequant_g<<<N, QUANT_THREADS, 0, stream>>>(
        (__nv_bfloat16*)w_bf16, (const int8_t*)w_i8, scales, K, gs);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] intq dequant launch (N=%d,K=%d): %s\n", N, K,
                cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

extern "C" s2p_status s2p_int8_quant(void* w_i8, float* scales,
                                     const void* w_bf16, int N, int K,
                                     cudaStream_t stream) {
    if (!w_i8 || !scales || !w_bf16 || N <= 0 || K <= 0) return S2P_ERR_INVALID;
    k_quant<<<N, QUANT_THREADS, 0, stream>>>(
        (int8_t*)w_i8, scales, (const __nv_bfloat16*)w_bf16, K, 127);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] int8 quant launch (N=%d,K=%d): %s\n", N, K,
                cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

extern "C" s2p_status s2p_int8_gemv(void* y_bf16, const void* x_bf16,
                                    const void* w_i8, const float* scales,
                                    int M, int N, int K, cudaStream_t stream) {
    if (!y_bf16 || !x_bf16 || !w_i8 || !scales || M <= 0 || N <= 0 || K <= 0)
        return S2P_ERR_INVALID;
    if (M > S2P_INT8_GEMV_MAX_M || K % 512 != 0) return S2P_ERR_INVALID;
    const int blocks = (N + GEMV_WARPS - 1) / GEMV_WARPS;
    k_gemv<<<blocks, GEMV_WARPS * 32, 0, stream>>>(
        (__nv_bfloat16*)y_bf16, (const __nv_bfloat16*)x_bf16,
        (const int8_t*)w_i8, scales, M, N, K);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] int8 gemv launch (M=%d,N=%d,K=%d): %s\n", M,
                N, K, cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

extern "C" s2p_status s2p_int8_dequant(void* w_bf16, const void* w_i8,
                                       const float* scales, int N, int K,
                                       cudaStream_t stream) {
    if (!w_bf16 || !w_i8 || !scales || N <= 0 || K <= 0) return S2P_ERR_INVALID;
    k_dequant<<<N, QUANT_THREADS, 0, stream>>>(
        (__nv_bfloat16*)w_bf16, (const int8_t*)w_i8, scales, K);
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] int8 dequant launch (N=%d,K=%d): %s\n", N, K,
                cudaGetErrorString(ce));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}
