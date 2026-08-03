/* s2pro-native — DAC shared CUDA kernels (conv stacks, transformer, ConvNeXt).
 *
 * Correctness-first kernels mirroring the PyTorch reference ops exactly:
 *   - causal Conv1d: manual left pad (zeros), dynamic right pad handled by
 *     bounds checks (PORTING.md pitfall 11); out_len = ceil(Tin/stride).
 *   - causal ConvTranspose1d: raw output cropped by (k - stride) on the RIGHT.
 *   - Snake1d: x + sin^2(alpha*x)/(alpha+1e-9), per-channel alpha.
 *   - RMSNorm (nn.RMSNorm semantics, eps INSIDE the sqrt, f32).
 *   - LayerNorm (biased variance), exact-erf GELU, SiLU.
 *   - interleaved (gpt-fast) RoPE with bf16-rounded cos/sin (table built on
 *     the host, values already rounded; butterfly in f32).
 *   - windowed causal SDPA in f32 (softmax with running max subtraction).
 * All launchers are extern "C", enqueue on the given stream, and return
 * cudaPeekAtLastError() so hosts can check every launch.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "dac_internal.h"

/* FP16 weight storage (S2P_DAC_F32=1 reverts): the conv/matmul launchers
 * receive void* weights and dispatch on this flag; kernels convert at the
 * shared-memory staging (or first read), so inner-loop arithmetic and
 * accumulation order are IDENTICAL to the f32 path — only the rounded
 * weight values differ. Biases, Snake alphas, norms, layer scales, the
 * codebooks and the mono output conv stay f32 (0.04 % of the bytes;
 * measured insurance, see docs/DAC-KERNELS.md). */
static int g_w16 = 0;
extern "C" void s2pdk_weights_f16(int on) { g_w16 = on; }

__global__ void k_f32_to_f16(const float* __restrict__ src,
                             __half* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = __float2half_rn(src[i]);
}

extern "C" cudaError_t s2pdk_f32_to_f16(const float* src, void* dst,
                                        int64_t n, cudaStream_t st) {
    int blocks = (int)((n + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    k_f32_to_f16<<<blocks, 256, 0, st>>>(src, (__half*)dst, n);
    return cudaPeekAtLastError();
}

#define S2PDK_MAX_ATTN_WINDOW 1024   /* shared-mem score buffer bound */

/* ------------------------- per-kernel-type profiler ----------------------
 * S2P_DAC_PROF=1: every conv launcher brackets its kernel with events;
 * s2pdk_prof_dump() (called by hosts after a decode) syncs, sums per type
 * and prints. Diagnostic only — events serialize nothing but add launch
 * overhead, so keep it off for real measurements of totals. */
#define PROF_MAX 4096
static struct {
    int         inited, on, n;
    cudaEvent_t ev[PROF_MAX][2];
    int         type[PROF_MAX];
    int         dim[PROF_MAX][5]; /* cin, cout, k, stride_or_dil, tout */
} g_prof;

static int prof_on(void) {
    if (!g_prof.inited) {
        g_prof.inited = 1;
        const char* e = getenv("S2P_DAC_PROF");
        g_prof.on = e && e[0] == '1' && e[1] == '\0';
    }
    return g_prof.on;
}

static int prof_begin(int type, cudaStream_t st, int cin, int cout, int k,
                      int sd, int tout) {
    if (!prof_on() || g_prof.n >= PROF_MAX) return -1;
    int i = g_prof.n++;
    g_prof.type[i] = type;
    g_prof.dim[i][0] = cin; g_prof.dim[i][1] = cout; g_prof.dim[i][2] = k;
    g_prof.dim[i][3] = sd;  g_prof.dim[i][4] = tout;
    if (!g_prof.ev[i][0]) {
        cudaEventCreate(&g_prof.ev[i][0]);
        cudaEventCreate(&g_prof.ev[i][1]);
    }
    cudaEventRecord(g_prof.ev[i][0], st);
    return i;
}

static void prof_end(int i, cudaStream_t st) {
    if (i >= 0) cudaEventRecord(g_prof.ev[i][1], st);
}

extern "C" void s2pdk_prof_dump(void) {
    if (!prof_on() || g_prof.n == 0) return;
    cudaDeviceSynchronize();
    static const char* names[] = {"conv1d", "dwconv", "tconv"};
    float sum[3] = {0, 0, 0};
    int   cnt[3] = {0, 0, 0};
    for (int i = 0; i < g_prof.n; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, g_prof.ev[i][0], g_prof.ev[i][1]);
        sum[g_prof.type[i]] += ms;
        cnt[g_prof.type[i]]++;
    }
    for (int t = 0; t < 3; t++)
        fprintf(stderr, "[dac-prof] %-7s %4d launches  %8.2f ms\n", names[t],
                cnt[t], sum[t]);
    for (int i = 0; i < g_prof.n; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, g_prof.ev[i][0], g_prof.ev[i][1]);
        if (ms < 20.0f) continue; /* only the heavy launches */
        fprintf(stderr, "[dac-prof]   %-6s cin=%-5d cout=%-5d k=%-3d sd=%-2d "
                        "tout=%-7d %8.2f ms\n", names[g_prof.type[i]],
                g_prof.dim[i][0], g_prof.dim[i][1], g_prof.dim[i][2],
                g_prof.dim[i][3], g_prof.dim[i][4], ms);
    }
    g_prof.n = 0;
}

/* ------------------------------ conv ------------------------------------- */

/* out[co,to] = b[co] + sum_ci sum_k w[co,ci,k] * in[ci, to*stride - P + k*dil]
 * (reads outside [0,Tin) are zero: covers causal left pad + dynamic right pad)
 *
 * Output-channel tiling: one thread accumulates S2PDK_CO_TILE output
 * channels for its time step, so every input load is amortized CO_TILE-fold
 * (the naive one-block-per-co layout re-read the whole input plane once per
 * output channel — measured 18 GB of traffic on a 192x192 pointwise conv
 * where 0.2 GB is inherent). Each output's accumulation order stays exactly
 * ci-outer/kk-inner — bit-identical to the untiled kernel. */
#define S2PDK_CO_TILE 16

#define S2PDK_CONV_K_MAX    7  /* largest decoder conv kernel width */
#define S2PDK_CONV_CI_CHUNK 64 /* smem: 16*64*7*4 = 28 KB */

template <typename WT>
__global__ void k_conv1d(const float* __restrict__ in, int cin, int tin,
                         const WT* __restrict__ w,
                         const float* __restrict__ b, int cout,
                         int k, int dil, int stride, int leftpad,
                         float* __restrict__ out, int tout) {
    __shared__ float sw[S2PDK_CO_TILE * S2PDK_CONV_CI_CHUNK *
                        S2PDK_CONV_K_MAX];
    const int to = blockIdx.x * blockDim.x + threadIdx.x;
    const int co0 = blockIdx.y * S2PDK_CO_TILE;
    const int nc =
        (cout - co0) < S2PDK_CO_TILE ? (cout - co0) : S2PDK_CO_TILE;
    const int active = (to < tout);

    float acc[S2PDK_CO_TILE];
    for (int c = 0; c < nc; c++) acc[c] = b ? b[co0 + c] : 0.f;

    const int t0 = to * stride - leftpad;
    for (int ci0 = 0; ci0 < cin; ci0 += S2PDK_CONV_CI_CHUNK) {
        const int nci = (cin - ci0) < S2PDK_CONV_CI_CHUNK
                            ? (cin - ci0)
                            : S2PDK_CONV_CI_CHUNK;
        __syncthreads();
        for (int c = 0; c < nc; c++) { /* contiguous nci*k floats per c */
            const WT* src = w + ((size_t)(co0 + c) * cin + ci0) * k;
            for (int idx = threadIdx.x; idx < nci * k; idx += blockDim.x)
                sw[(size_t)c * S2PDK_CONV_CI_CHUNK * k + idx] = (float)src[idx];
        }
        __syncthreads();
        if (!active) continue;
        for (int ci = 0; ci < nci; ci++) {
            const float* inr = in + (size_t)(ci0 + ci) * tin;
            const float* swc = sw + (size_t)ci * k;
            for (int kk = 0; kk < k; kk++) {
                int ti = t0 + kk * dil;
                if (ti < 0 || ti >= tin) continue;
                const float v = inr[ti];
                if (nc == S2PDK_CO_TILE) {
#pragma unroll
                    for (int c = 0; c < S2PDK_CO_TILE; c++)
                        acc[c] +=
                            swc[(size_t)c * S2PDK_CONV_CI_CHUNK * k + kk] * v;
                } else {
                    for (int c = 0; c < nc; c++)
                        acc[c] +=
                            swc[(size_t)c * S2PDK_CONV_CI_CHUNK * k + kk] * v;
                }
            }
        }
    }
    if (active)
        for (int c = 0; c < nc; c++)
            out[(size_t)(co0 + c) * tout + to] = acc[c];
}

/* 2D-register-blocked kernel for k in [2, S2PDK_CONV_K_MAX]: each thread
 * accumulates S2PDK_CONV_TSUB time positions x S2PDK_CONV_CO output
 * channels (16 accumulators), weights stream through smem in ci-chunks,
 * inputs ride L1 (adjacent threads read adjacent taps — measured
 * effectively free on the reference kernel). Per-output accumulation
 * order stays ci-outer/kk-inner across ascending chunks — bit-identical. */
#define S2PDK_CONV_CO   8
#define S2PDK_CONV_TSUB 2

template <typename WT>
__global__ void k_conv1d_wide(const float* __restrict__ in, int cin, int tin,
                              const WT* __restrict__ w,
                              const float* __restrict__ b, int cout,
                              int k, int dil, int stride, int leftpad,
                              float* __restrict__ out, int tout) {
    __shared__ float sw[S2PDK_CONV_CO * S2PDK_CONV_CI_CHUNK *
                        S2PDK_CONV_K_MAX];
    const int t_a = blockIdx.x * (blockDim.x * S2PDK_CONV_TSUB) + threadIdx.x;
    const int t_b = t_a + blockDim.x;
    const int co0 = blockIdx.y * S2PDK_CONV_CO;
    const int nc = (cout - co0) < S2PDK_CONV_CO ? (cout - co0)
                                                : S2PDK_CONV_CO;
    const int act_a = (t_a < tout), act_b = (t_b < tout);

    float acc_a[S2PDK_CONV_CO], acc_b[S2PDK_CONV_CO];
    for (int c = 0; c < nc; c++) acc_a[c] = acc_b[c] = b ? b[co0 + c] : 0.f;

    const int t0a = t_a * stride - leftpad;
    const int t0b = t_b * stride - leftpad;
    for (int ci0 = 0; ci0 < cin; ci0 += S2PDK_CONV_CI_CHUNK) {
        const int nci = (cin - ci0) < S2PDK_CONV_CI_CHUNK
                            ? (cin - ci0)
                            : S2PDK_CONV_CI_CHUNK;
        __syncthreads();
        for (int c = 0; c < nc; c++) {
            const WT* src = w + ((size_t)(co0 + c) * cin + ci0) * k;
            for (int idx = threadIdx.x; idx < nci * k; idx += blockDim.x)
                sw[(size_t)c * S2PDK_CONV_CI_CHUNK * k + idx] = (float)src[idx];
        }
        __syncthreads();
        if (!act_a) continue; /* t_b > t_a: both inactive */
        for (int ci = 0; ci < nci; ci++) {
            const float* inr = in + (size_t)(ci0 + ci) * tin;
            const float* swc = sw + (size_t)ci * k;
            for (int kk = 0; kk < k; kk++) {
                const int tia = t0a + kk * dil;
                const int tib = t0b + kk * dil;
                const float va =
                    (tia >= 0 && tia < tin) ? inr[tia] : 0.f;
                const float vb =
                    (act_b && tib >= 0 && tib < tin) ? inr[tib] : 0.f;
#pragma unroll
                for (int c = 0; c < S2PDK_CONV_CO; c++) {
                    const float wv =
                        swc[(size_t)c * S2PDK_CONV_CI_CHUNK * k + kk];
                    acc_a[c] += wv * va;
                    acc_b[c] += wv * vb;
                }
            }
        }
    }
    for (int c = 0; c < nc; c++) {
        if (act_a) out[(size_t)(co0 + c) * tout + t_a] = acc_a[c];
        if (act_b) out[(size_t)(co0 + c) * tout + t_b] = acc_b[c];
    }
}

/* reference kernel (one block row per output channel): serves k >
 * S2PDK_CONV_K_MAX — encoder strided convs go up to k=16 — and stands as
 * the bit-exactness reference for the tiled kernel. */
template <typename WT>
__global__ void k_conv1d_ref(const float* __restrict__ in, int cin, int tin,
                             const WT* __restrict__ w,
                             const float* __restrict__ b, int cout,
                             int k, int dil, int stride, int leftpad,
                             float* __restrict__ out, int tout) {
    int to = blockIdx.x * blockDim.x + threadIdx.x;
    int co = blockIdx.y;
    if (to >= tout || co >= cout) return;
    float acc = b ? b[co] : 0.f;
    const WT* wrow = w + (size_t)co * cin * k;
    int t0 = to * stride - leftpad;
    for (int ci = 0; ci < cin; ci++) {
        const float* inr = in + (size_t)ci * tin;
        const WT* wr = wrow + (size_t)ci * k;
        for (int kk = 0; kk < k; kk++) {
            int ti = t0 + kk * dil;
            if (ti >= 0 && ti < tin) acc += (float)wr[kk] * inr[ti];
        }
    }
    out[(size_t)co * tout + to] = acc;
}

/* f32-forced variant for tensors shared with non-launcher consumers
 * (the VQ out_proj stays f32 for rvq.cu's direct reads). */
extern "C" cudaError_t s2pdk_conv1d_w32(const float* in, int cin, int tin,
                                        const void* w, const float* b,
                                        int cout, int k, int dil, int stride,
                                        int leftpad, float* out, int tout,
                                        cudaStream_t st) {
    dim3 grid((tout + 255) / 256, cout);
    k_conv1d_ref<<<grid, 256, 0, st>>>(in, cin, tin, (const float*)w, b,
                                       cout, k, dil, stride, leftpad, out,
                                       tout);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t s2pdk_conv1d(const float* in, int cin, int tin,
                                    const void* w, const float* b, int cout,
                                    int k, int dil, int stride, int leftpad,
                                    float* out, int tout, cudaStream_t st) {
    int pi = prof_begin(0, st, cin, cout, k, dil, tout);
    if (0) { /* superseded by k_conv1d_wide below; kept for A/B */
        /* pointwise conv: the co-tiled kernel amortizes the input plane
         * 16-fold — measured 4x (81 -> 20 ms on 192ch x 122880). For k > 1
         * the tiled variant LOST to the reference (smem/FMA throughput per
         * output dominates, not bandwidth; a GEMM-grade 2D register tile
         * would be needed — roadmap), so wider kernels keep the reference
         * one-block-row-per-channel layout. */
        dim3 grid((tout + 255) / 256,
                  (cout + S2PDK_CO_TILE - 1) / S2PDK_CO_TILE);
        if (g_w16)
            k_conv1d<<<grid, 256, 0, st>>>(in, cin, tin, (const __half*)w, b,
                                           cout, k, dil, stride, leftpad, out,
                                           tout);
        else
            k_conv1d<<<grid, 256, 0, st>>>(in, cin, tin, (const float*)w, b,
                                           cout, k, dil, stride, leftpad, out,
                                           tout);
    } else if (k <= S2PDK_CONV_K_MAX) {
        dim3 grid((tout + 256 * S2PDK_CONV_TSUB - 1) /
                      (256 * S2PDK_CONV_TSUB),
                  (cout + S2PDK_CONV_CO - 1) / S2PDK_CONV_CO);
        if (g_w16)
            k_conv1d_wide<<<grid, 256, 0, st>>>(in, cin, tin, (const __half*)w,
                                                b, cout, k, dil, stride,
                                                leftpad, out, tout);
        else
            k_conv1d_wide<<<grid, 256, 0, st>>>(in, cin, tin, (const float*)w,
                                                b, cout, k, dil, stride,
                                                leftpad, out, tout);
    } else {
        dim3 grid((tout + 255) / 256, cout);
        if (g_w16)
            k_conv1d_ref<<<grid, 256, 0, st>>>(in, cin, tin, (const __half*)w,
                                               b, cout, k, dil, stride,
                                               leftpad, out, tout);
        else
            k_conv1d_ref<<<grid, 256, 0, st>>>(in, cin, tin, (const float*)w,
                                               b, cout, k, dil, stride,
                                               leftpad, out, tout);
    }
    prof_end(pi, st);
    return cudaPeekAtLastError();
}

/* depthwise causal conv (ConvNeXt dwconv): w [C,1,K] flattened [C*K] */
template <typename WT>
__global__ void k_dwconv1d(const float* __restrict__ in, int c, int tin,
                           const WT* __restrict__ w,
                           const float* __restrict__ b, int k, int leftpad,
                           float* __restrict__ out, int tout) {
    int to = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.y;
    if (to >= tout || ch >= c) return;
    float acc = b ? b[ch] : 0.f;
    const float* inr = in + (size_t)ch * tin;
    const WT* wr = w + (size_t)ch * k;
    int t0 = to - leftpad;
    for (int kk = 0; kk < k; kk++) {
        int ti = t0 + kk;
        if (ti >= 0 && ti < tin) acc += (float)wr[kk] * inr[ti];
    }
    out[(size_t)ch * tout + to] = acc;
}

extern "C" cudaError_t s2pdk_dwconv1d(const float* in, int c, int tin,
                                      const void* w, const float* b, int k,
                                      int leftpad, float* out, int tout,
                                      cudaStream_t st) {
    dim3 grid((tout + 255) / 256, c);
    int pi = prof_begin(1, st, c, c, k, 1, tout);
    if (g_w16)
        k_dwconv1d<<<grid, 256, 0, st>>>(in, c, tin, (const __half*)w, b, k,
                                         leftpad, out, tout);
    else
        k_dwconv1d<<<grid, 256, 0, st>>>(in, c, tin, (const float*)w, b, k,
                                         leftpad, out, tout);
    prof_end(pi, st);
    return cudaPeekAtLastError();
}

/* causal ConvTranspose1d: raw_out[j] = sum over (ti,k) with j = ti*stride + k;
 * causal crop keeps raw indices [0, tin*stride). w layout [Cin,Cout,K].
 *
 * Phase-partitioned + output-channel-tiled: all outputs with the same
 * to % stride share the same valid tap set, so a block serves ONE phase —
 * no masked lanes (the earlier co-tiled attempts idled (stride-1)/stride
 * of every warp), input reads coalesce (ti = j - m is consecutive across
 * threads), and the per-(kk, ci-chunk) weight slice streams through an
 * 8 KB smem tile. Per-output accumulation order stays exactly
 * kk-ascending / ci-ascending — bit-identical to the reference kernel. */
#define S2PDK_TC_CI_CHUNK 128 /* smem: 128*16*4 = 8 KB */

#define S2PDK_TC_TSUB 2

template <typename WT>
__global__ void k_tconv1d(const float* __restrict__ in, int cin, int tin,
                          const WT* __restrict__ w,
                          const float* __restrict__ b, int cout,
                          int k, int stride, float* __restrict__ out,
                          int tout) {
    __shared__ float sw[S2PDK_TC_CI_CHUNK * S2PDK_CO_TILE];
    const int j_a = blockIdx.x * (blockDim.x * S2PDK_TC_TSUB) + threadIdx.x;
    const int j_b = j_a + blockDim.x;
    const int phase = blockIdx.y;
    const int co0 = blockIdx.z * S2PDK_CO_TILE;
    const int to_a = j_a * stride + phase;
    const int to_b = j_b * stride + phase;
    const int nc =
        (cout - co0) < S2PDK_CO_TILE ? (cout - co0) : S2PDK_CO_TILE;
    const int act_a = (to_a < tout), act_b = (to_b < tout);

    float acc_a[S2PDK_CO_TILE], acc_b[S2PDK_CO_TILE];
    for (int c = 0; c < nc; c++) acc_a[c] = acc_b[c] = b ? b[co0 + c] : 0.f;

    /* valid taps: kk = phase + m*stride, m = 0.., kk < k; ti = j - m */
    for (int kk = phase, m = 0; kk < k; kk += stride, m++) {
        const int ti_a = j_a - m, ti_b = j_b - m;
        const int va_ok = act_a && ti_a >= 0 && ti_a < tin;
        const int vb_ok = act_b && ti_b >= 0 && ti_b < tin;
        for (int ci0 = 0; ci0 < cin; ci0 += S2PDK_TC_CI_CHUNK) {
            const int nci = (cin - ci0) < S2PDK_TC_CI_CHUNK
                                ? (cin - ci0)
                                : S2PDK_TC_CI_CHUNK;
            __syncthreads();
            /* sw[ci][c] = w[(ci0+ci)*cout*k + (co0+c)*k + kk] */
            for (int idx = threadIdx.x; idx < nci * S2PDK_CO_TILE;
                 idx += blockDim.x) {
                const int ci = idx / S2PDK_CO_TILE;
                const int c = idx % S2PDK_CO_TILE;
                sw[idx] =
                    (c < nc)
                        ? (float)w[((size_t)(ci0 + ci) * cout + co0 + c) * k +
                                   kk]
                        : 0.f;
            }
            __syncthreads();
            if (!va_ok && !vb_ok) continue;
            for (int ci = 0; ci < nci; ci++) {
                const float* inr = in + (size_t)(ci0 + ci) * tin;
                const float va = va_ok ? inr[ti_a] : 0.f;
                const float vb = vb_ok ? inr[ti_b] : 0.f;
                const float* swc = sw + (size_t)ci * S2PDK_CO_TILE;
#pragma unroll
                for (int c = 0; c < S2PDK_CO_TILE; c++) {
                    acc_a[c] += swc[c] * va;
                    acc_b[c] += swc[c] * vb;
                }
            }
        }
    }
    for (int c = 0; c < nc; c++) {
        if (act_a) out[(size_t)(co0 + c) * tout + to_a] = acc_a[c];
        if (act_b) out[(size_t)(co0 + c) * tout + to_b] = acc_b[c];
    }
}

extern "C" cudaError_t s2pdk_tconv1d(const float* in, int cin, int tin,
                                     const void* w, const float* b, int cout,
                                     int k, int stride, float* out, int tout,
                                     cudaStream_t st) {
    const int tphase = (tout + stride - 1) / stride;
    dim3 grid((tphase + 256 * S2PDK_TC_TSUB - 1) / (256 * S2PDK_TC_TSUB),
              stride, (cout + S2PDK_CO_TILE - 1) / S2PDK_CO_TILE);
    int pi = prof_begin(2, st, cin, cout, k, stride, tout);
    if (g_w16)
        k_tconv1d<<<grid, 256, 0, st>>>(in, cin, tin, (const __half*)w, b,
                                        cout, k, stride, out, tout);
    else
        k_tconv1d<<<grid, 256, 0, st>>>(in, cin, tin, (const float*)w, b,
                                        cout, k, stride, out, tout);
    prof_end(pi, st);
    return cudaPeekAtLastError();
}

/* ------------------------------ elementwise ------------------------------ */

__global__ void k_snake(const float* __restrict__ in, float* __restrict__ out,
                        const float* __restrict__ alpha, int c, int t) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)c * t;
    if (i >= n) return;
    int ch = (int)(i / t);
    float a = alpha[ch];
    float x = in[i];
    float s = sinf(a * x);
    out[i] = x + (1.0f / (a + 1e-9f)) * s * s;
}

extern "C" cudaError_t s2pdk_snake(const float* in, float* out,
                                   const float* alpha, int c, int t,
                                   cudaStream_t st) {
    int64_t n = (int64_t)c * t;
    k_snake<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(in, out, alpha, c, t);
    return cudaPeekAtLastError();
}

__global__ void k_tanh_ip(float* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}
extern "C" cudaError_t s2pdk_tanh_ip(float* x, int64_t n, cudaStream_t st) {
    k_tanh_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n);
    return cudaPeekAtLastError();
}

__global__ void k_add_ip(float* x, const float* y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
extern "C" cudaError_t s2pdk_add_ip(float* x, const float* y, int64_t n,
                                    cudaStream_t st) {
    k_add_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, y, n);
    return cudaPeekAtLastError();
}

__global__ void k_sub_ip(float* x, const float* y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] -= y[i];
}
extern "C" cudaError_t s2pdk_sub_ip(float* x, const float* y, int64_t n,
                                    cudaStream_t st) {
    k_sub_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, y, n);
    return cudaPeekAtLastError();
}

__global__ void k_transpose(const float* __restrict__ in, int rows, int cols,
                            float* __restrict__ out) {
    /* out[c,r] = in[r,c]; simple tiled transpose */
    __shared__ float tile[32][33];
    int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
    int r = r0 + threadIdx.y, c = c0 + threadIdx.x;
    if (r < rows && c < cols) tile[threadIdx.y][threadIdx.x] = in[(size_t)r * cols + c];
    __syncthreads();
    int rr = c0 + threadIdx.y, cc = r0 + threadIdx.x;   /* transposed coords */
    if (rr < cols && cc < rows) out[(size_t)rr * rows + cc] = tile[threadIdx.x][threadIdx.y];
}
extern "C" cudaError_t s2pdk_transpose(const float* in, int rows, int cols,
                                       float* out, cudaStream_t st) {
    dim3 grid((cols + 31) / 32, (rows + 31) / 32), blk(32, 32);
    k_transpose<<<grid, blk, 0, st>>>(in, rows, cols, out);
    return cudaPeekAtLastError();
}

__global__ void k_silu_mul_ip(float* a, const float* b, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    a[i] = x / (1.0f + expf(-x)) * b[i];
}
extern "C" cudaError_t s2pdk_silu_mul_ip(float* a, const float* b, int64_t n,
                                         cudaStream_t st) {
    k_silu_mul_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(a, b, n);
    return cudaPeekAtLastError();
}

__global__ void k_gelu_ip(float* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
}
extern "C" cudaError_t s2pdk_gelu_ip(float* x, int64_t n, cudaStream_t st) {
    k_gelu_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, n);
    return cudaPeekAtLastError();
}

__global__ void k_colscale_ip(float* x, const float* g, int t, int c) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)t * c) return;
    x[i] *= g[i % c];
}
extern "C" cudaError_t s2pdk_colscale_ip(float* x, const float* g, int t, int c,
                                         cudaStream_t st) {
    int64_t n = (int64_t)t * c;
    k_colscale_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, g, t, c);
    return cudaPeekAtLastError();
}

__global__ void k_scale_add_ip(float* x, const float* y, const float* g,
                               int t, int c) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)t * c) return;
    x[i] += g[i % c] * y[i];
}
extern "C" cudaError_t s2pdk_scale_add_ip(float* x, const float* y,
                                          const float* g, int t, int c,
                                          cudaStream_t st) {
    int64_t n = (int64_t)t * c;
    k_scale_add_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, y, g, t, c);
    return cudaPeekAtLastError();
}

/* ------------------------------ norms ------------------------------------ */

/* RMSNorm rows of [t,c]: y = x * rsqrt(mean(x^2)+eps) * w. In-place safe. */
__global__ void k_rmsnorm(const float* __restrict__ in,
                          const float* __restrict__ w, float eps, int c,
                          float* __restrict__ out) {
    int row = blockIdx.x;
    const float* xr = in + (size_t)row * c;
    float* yr = out + (size_t)row * c;
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float v = xr[i];
        s += v * v;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)c + eps);
    for (int i = threadIdx.x; i < c; i += blockDim.x)
        yr[i] = xr[i] * inv * w[i];
}
extern "C" cudaError_t s2pdk_rmsnorm(const float* in, const float* w, float eps,
                                     int t, int c, float* out,
                                     cudaStream_t st) {
    k_rmsnorm<<<t, 256, 0, st>>>(in, w, eps, c, out);
    return cudaPeekAtLastError();
}

/* LayerNorm rows of [t,c] (biased var), in place. */
__global__ void k_layernorm_ip(float* __restrict__ x,
                               const float* __restrict__ w,
                               const float* __restrict__ b, float eps, int c) {
    int row = blockIdx.x;
    float* xr = x + (size_t)row * c;
    __shared__ float red[256];
    __shared__ float red2[256];
    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float v = xr[i];
        s += v; s2 += v * v;
    }
    red[threadIdx.x] = s; red2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) {
            red[threadIdx.x] += red[threadIdx.x + o];
            red2[threadIdx.x] += red2[threadIdx.x + o];
        }
        __syncthreads();
    }
    float mean = red[0] / (float)c;
    float var = red2[0] / (float)c - mean * mean;
    float inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < c; i += blockDim.x)
        xr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}
extern "C" cudaError_t s2pdk_layernorm_ip(float* x, const float* w,
                                          const float* b, float eps, int t,
                                          int c, cudaStream_t st) {
    k_layernorm_ip<<<t, 256, 0, st>>>(x, w, b, eps, c);
    return cudaPeekAtLastError();
}

/* ------------------------------ matmul ----------------------------------- */

/* out[m,n] = a[m,k] @ w[n,k]^T (+ bias[n]); torch Linear layout. Tiled 16x16. */
#define MM_TILE 16
template <typename WT>
__global__ void k_matmul(const float* __restrict__ a, int m, int k,
                         const WT* __restrict__ w, int n,
                         const float* __restrict__ bias,
                         float* __restrict__ out) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Ws[MM_TILE][MM_TILE + 1];
    int row = blockIdx.y * MM_TILE + threadIdx.y;   /* m index */
    int col = blockIdx.x * MM_TILE + threadIdx.x;   /* n index */
    float acc = 0.f;
    for (int k0 = 0; k0 < k; k0 += MM_TILE) {
        int ka = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < m && ka < k) ? a[(size_t)row * k + ka] : 0.f;
        int kw = k0 + threadIdx.y;
        int wn = blockIdx.x * MM_TILE + threadIdx.x;
        Ws[threadIdx.y][threadIdx.x] =
            (wn < n && kw < k) ? (float)w[(size_t)wn * k + kw] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < MM_TILE; kk++)
            acc += As[threadIdx.y][kk] * Ws[kk][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n)
        out[(size_t)row * n + col] = bias ? acc + bias[col] : acc;
}
extern "C" cudaError_t s2pdk_matmul(const float* a, int m, int k,
                                    const void* w, int n, const float* bias,
                                    float* out, cudaStream_t st) {
    dim3 grid((n + MM_TILE - 1) / MM_TILE, (m + MM_TILE - 1) / MM_TILE);
    dim3 blk(MM_TILE, MM_TILE);
    if (g_w16)
        k_matmul<<<grid, blk, 0, st>>>(a, m, k, (const __half*)w, n, bias,
                                       out);
    else
        k_matmul<<<grid, blk, 0, st>>>(a, m, k, (const float*)w, n, bias,
                                       out);
    return cudaPeekAtLastError();
}

/* ------------------------------ RoPE ------------------------------------- */

/* Interleaved (gpt-fast) RoPE, adjacent pairs (2i,2i+1), NOT rotate-half.
 * tab [maxT, hd/2, 2] f32 holds bf16-rounded (cos,sin); butterfly in f32.
 * t0 offsets the table row (absolute position of x row 0) so incremental
 * chunks rotate exactly like their whole-buffer positions. */
__global__ void k_rope_ip(float* __restrict__ x, int t0, int t, int heads,
                          int hd, int row_stride,
                          const float* __restrict__ tab) {
    int pairs = hd / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)t * heads * pairs;
    if (i >= n) return;
    int p = (int)(i % pairs);
    int h = (int)((i / pairs) % heads);
    int tt = (int)(i / ((int64_t)pairs * heads));
    float* xp = x + (size_t)tt * row_stride + (size_t)h * hd + 2 * p;
    float c = tab[((size_t)(t0 + tt) * pairs + p) * 2 + 0];
    float s = tab[((size_t)(t0 + tt) * pairs + p) * 2 + 1];
    float x0 = xp[0], x1 = xp[1];
    xp[0] = x0 * c - x1 * s;
    xp[1] = x1 * c + x0 * s;
}
extern "C" cudaError_t s2pdk_rope_ip(float* x, int t, int heads, int hd,
                                     int row_stride, const float* tab,
                                     cudaStream_t st) {
    int64_t n = (int64_t)t * heads * (hd / 2);
    k_rope_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, 0, t, heads, hd,
                                                           row_stride, tab);
    return cudaPeekAtLastError();
}
extern "C" cudaError_t s2pdk_rope_ip_off(float* x, int t0, int t, int heads,
                                         int hd, int row_stride,
                                         const float* tab, cudaStream_t st) {
    int64_t n = (int64_t)t * heads * (hd / 2);
    k_rope_ip<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(x, t0, t, heads, hd,
                                                           row_stride, tab);
    return cudaPeekAtLastError();
}

/* ------------------------------ attention -------------------------------- */

/* Windowed causal SDPA, f32, one block per (query t, head). Keys limited to
 * [max(0, t-window+1), t] — identical to the reference bool-mask + -inf. */
__global__ void k_sdpa(const float* __restrict__ q, const float* __restrict__ k,
                       const float* __restrict__ v, int qkv_stride, int t,
                       int hd, int window, float* __restrict__ out,
                       int out_stride) {
    int tq = blockIdx.x;
    int h = blockIdx.y;
    int ks = tq - window + 1;
    if (ks < 0) ks = 0;
    int cnt = tq - ks + 1;

    extern __shared__ float sh[];        /* scores[cnt] then red[blockDim] */
    float* scores = sh;
    float* red = sh + window;

    const float* qp = q + (size_t)tq * qkv_stride + (size_t)h * hd;
    const float scale = rsqrtf((float)hd);

    /* scores */
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
        const float* kp = k + (size_t)(ks + j) * qkv_stride + (size_t)h * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) dot += qp[d] * kp[d];
        scores[j] = dot * scale;
    }
    __syncthreads();

    /* max */
    float m = -INFINITY;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x)
        m = fmaxf(m, scores[j]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    /* exp + sum */
    float s = 0.f;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) {
        float e = expf(scores[j] - m);
        scores[j] = e;
        s += e;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();

    /* weighted V: threads over head dim */
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        const float* vp = v + (size_t)ks * qkv_stride + (size_t)h * hd + d;
        for (int j = 0; j < cnt; j++)
            acc += scores[j] * vp[(size_t)j * qkv_stride];
        out[(size_t)tq * out_stride + (size_t)h * hd + d] = acc * inv;
    }
}
extern "C" cudaError_t s2pdk_sdpa(const float* q, const float* k,
                                  const float* v, int qkv_stride, int t,
                                  int heads, int hd, int window, float* out,
                                  int out_stride, cudaStream_t st) {
    if (window <= 0 || window > S2PDK_MAX_ATTN_WINDOW)
        return cudaErrorInvalidValue;
    dim3 grid(t, heads);
    int threads = 128;
    size_t shm = (size_t)(window + threads) * sizeof(float);
    k_sdpa<<<grid, threads, shm, st>>>(q, k, v, qkv_stride, t, hd, window, out,
                                       out_stride);
    return cudaPeekAtLastError();
}

/* Incremental windowed SDPA: tn new queries (strided rows), keys/values in a
 * contiguous [hl+tn, heads*hd] buffer whose first hl rows are the retained
 * history. Query j sits at absolute-relative kv row hl+j and attends the
 * causal window ending there — the SAME key set, dot order, and reduction
 * pattern as k_sdpa over the full sequence, hence bit-identical. */
__global__ void k_sdpa_inc(const float* __restrict__ q, int q_stride,
                           const float* __restrict__ kv_k,
                           const float* __restrict__ kv_v, int kv_stride,
                           int hl, int hd, int window,
                           float* __restrict__ out, int out_stride) {
    int j = blockIdx.x;                  /* new-query index */
    int h = blockIdx.y;
    int tq = hl + j;                     /* kv row of this query */
    int ks = tq - window + 1;
    if (ks < 0) ks = 0;
    int cnt = tq - ks + 1;

    extern __shared__ float sh[];
    float* scores = sh;
    float* red = sh + window;

    const float* qp = q + (size_t)j * q_stride + (size_t)h * hd;
    const float scale = rsqrtf((float)hd);

    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        const float* kp = kv_k + (size_t)(ks + i) * kv_stride + (size_t)h * hd;
        float dot = 0.f;
        for (int d = 0; d < hd; d++) dot += qp[d] * kp[d];
        scores[i] = dot * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int i = threadIdx.x; i < cnt; i += blockDim.x)
        m = fmaxf(m, scores[i]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float s = 0.f;
    for (int i = threadIdx.x; i < cnt; i += blockDim.x) {
        float e = expf(scores[i] - m);
        scores[i] = e;
        s += e;
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();

    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        const float* vp = kv_v + (size_t)ks * kv_stride + (size_t)h * hd + d;
        for (int i = 0; i < cnt; i++)
            acc += scores[i] * vp[(size_t)i * kv_stride];
        out[(size_t)j * out_stride + (size_t)h * hd + d] = acc * inv;
    }
}
extern "C" cudaError_t s2pdk_sdpa_inc(const float* q, int q_stride,
                                      const float* kv_k, const float* kv_v,
                                      int kv_stride, int hl, int tn, int heads,
                                      int hd, int window, float* out,
                                      int out_stride, cudaStream_t st) {
    if (window <= 0 || window > S2PDK_MAX_ATTN_WINDOW)
        return cudaErrorInvalidValue;
    dim3 grid(tn, heads);
    int threads = 128;
    size_t shm = (size_t)(window + threads) * sizeof(float);
    k_sdpa_inc<<<grid, threads, shm, st>>>(q, q_stride, kv_k, kv_v, kv_stride,
                                           hl, hd, window, out, out_stride);
    return cudaPeekAtLastError();
}

