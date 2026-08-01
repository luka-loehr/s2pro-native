/* s2pro-native — RVQ from_indices: 10 codebooks -> summed 1024-dim latent.
 *
 * Mirrors DownsampleResidualVectorQuantize.decode steps 2..4 (rvq.py):
 *   z = semantic.from_codes(codes[0]) + sum_i residual_i.from_codes(codes[1+i])
 * where each from_codes = out_proj(codebook_lookup(idx)), out_proj being a
 * weight-norm-folded 1x1 conv 8->1024 with bias. Index clamping is done on
 * the HOST (dac.c) on a copy — the reference clamps in place; we never mutate
 * caller memory. codes are [10,T] codebook-major int32 on device.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include "dac_internal.h"

/* z[c,t] = sum_q ( ob_q[c] + sum_d ow_q[c,d] * cb_q[codes[q,t], d] ) */
__global__ void k_rvq_from_indices(const int32_t* __restrict__ codes, int T,
                                   s2pdk_rvq_tabs tabs,
                                   float* __restrict__ z) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;                       /* output channel 0..1023 */
    if (t >= T) return;
    float acc = 0.f;
    #pragma unroll
    for (int q = 0; q < S2P_NUM_CODEBOOKS; q++) {
        const s2pdk_rvq_tab* tb = &tabs.t[q];
        int idx = codes[(size_t)q * T + t];
        const float* e = tb->cb + (size_t)idx * 8;
        const float* wr = tb->ow + (size_t)c * 8;
        float s = tb->ob[c];
        #pragma unroll
        for (int d = 0; d < 8; d++) s += wr[d] * e[d];
        acc += s;
    }
    z[(size_t)c * T + t] = acc;
}

extern "C" cudaError_t s2pdk_rvq_from_indices(const int32_t* codes, int T,
                                              s2pdk_rvq_tabs tabs, float* z,
                                              cudaStream_t st) {
    dim3 grid((T + 127) / 128, 1024);
    k_rvq_from_indices<<<grid, 128, 0, st>>>(codes, T, tabs, z);
    return cudaPeekAtLastError();
}
