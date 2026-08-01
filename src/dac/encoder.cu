/* s2pro-native — encode-path VQ kernels (voice cloning).
 *
 * Mirrors dac.nn.quantize.VectorQuantize at inference:
 *   z_e   = in_proj(residual)                       (1x1 conv, done via
 *                                                    s2pdk_conv1d in dac.c)
 *   idx   = argmax(-dist) over the L2-NORMALIZED z_e vs L2-NORMALIZED
 *           codebook (F.normalize eps 1e-12), dist computed exactly as
 *           |e|^2 - 2 e.c + |c|^2 on the normalized vectors; first index
 *           wins ties (strict comparison).
 *   z_q8  = z_e + (codebook[idx] - z_e)             (RAW codebook rows;
 *           literal straight-through arithmetic for float parity)
 *   z_q   = out_proj(z_q8)                          (1x1 conv in dac.c)
 * The residual cascade (9 codebooks) and the semantic/residual split are
 * orchestrated host-side in dac.c. The encoder conv stack reuses the shared
 * kernels in decoder.cu — no encoder-specific conv kernels are needed.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "dac_internal.h"

/* ze [8,T] channels-first; cb [n,8]; idx [T]. One thread per t. */
__global__ void k_vq_nearest(const float* __restrict__ ze, int T,
                             const float* __restrict__ cb, int n,
                             int32_t* __restrict__ idx) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float e[8];
    float en2 = 0.f;
    #pragma unroll
    for (int d = 0; d < 8; d++) {
        e[d] = ze[(size_t)d * T + t];
        en2 += e[d] * e[d];
    }
    float einv = 1.0f / fmaxf(sqrtf(en2), 1e-12f);   /* F.normalize(p=2) */
    float es2 = 0.f;
    #pragma unroll
    for (int d = 0; d < 8; d++) {
        e[d] *= einv;
        es2 += e[d] * e[d];
    }

    float best = -INFINITY;    /* of (-dist); strict > keeps first index */
    int besti = 0;
    for (int i = 0; i < n; i++) {
        const float* cr = cb + (size_t)i * 8;
        float cn2 = 0.f;
        #pragma unroll
        for (int d = 0; d < 8; d++) cn2 += cr[d] * cr[d];
        float cinv = 1.0f / fmaxf(sqrtf(cn2), 1e-12f);
        float dot = 0.f, cs2 = 0.f;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            float cv = cr[d] * cinv;
            dot += e[d] * cv;
            cs2 += cv * cv;
        }
        float negdist = -(es2 - 2.0f * dot + cs2);
        if (negdist > best) { best = negdist; besti = i; }
    }
    idx[t] = besti;
}

extern "C" cudaError_t s2pdk_vq_nearest(const float* ze, int T, const float* cb,
                                        int n, int32_t* idx, cudaStream_t st) {
    k_vq_nearest<<<(T + 127) / 128, 128, 0, st>>>(ze, T, cb, n, idx);
    return cudaPeekAtLastError();
}

/* zq8[d,t] = ze[d,t] + (cb[idx[t]][d] - ze[d,t])  — literal straight-through */
__global__ void k_vq_dequant(const float* __restrict__ ze,
                             const int32_t* __restrict__ idx,
                             const float* __restrict__ cb, int T,
                             float* __restrict__ zq8) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* cr = cb + (size_t)idx[t] * 8;
    #pragma unroll
    for (int d = 0; d < 8; d++) {
        float zev = ze[(size_t)d * T + t];
        zq8[(size_t)d * T + t] = zev + (cr[d] - zev);
    }
}

extern "C" cudaError_t s2pdk_vq_dequant(const float* ze, const int32_t* idx,
                                        const float* cb, int T, float* zq8,
                                        cudaStream_t st) {
    k_vq_dequant<<<(T + 127) / 128, 128, 0, st>>>(ze, idx, cb, T, zq8);
    return cudaPeekAtLastError();
}
