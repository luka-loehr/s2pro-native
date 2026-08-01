/* s2pro-native — the ONLY C++ translation unit in the project.
 *
 * Extern-C shim over fish-scales-ops (CUTLASS FP8 block-scale GEMM). Links
 * against the prebuilt sm_121a objects listed in docs/SPARK.md:
 *   /fso/build/runner_121a.o   CutlassFp8BlockScaleGemmRunner
 *   /fso/build/quant_121a.o    quantize + repack kernels (torch-free)
 * Usage mirrors the numerically verified harness at
 * ~/fso-sm121-test/harness3.cu on the box — do not deviate from that call
 * sequence; it is the only proven path on GB10.
 *
 * Compiled with nvcc (needs FSO + cutlass include dirs, see Makefile
 * FSO_INC). Everything exported here is C-linkage, declared in src/fso/fso.h.
 */
#include "blockscale_gemm/runner.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <mutex>

#include "s2pro/status.h"
#include "fso.h"

/* Proven symbols from quant_121a.o (exact signatures from harness3.cu). */
namespace blockscale_gemm {
namespace detail {
void fp8bs_quantize_1x128_packed(__nv_fp8_e4m3*, int32_t*,
                                 __nv_bfloat16 const*, int M, int K,
                                 cudaStream_t, bool use_ue8m0);
void fp8bs_quantize_128x128(__nv_fp8_e4m3*, float*, __nv_bfloat16 const*,
                            int N, int K, cudaStream_t);
void repack_ue8m0_scales_sfb_for_sm120(int32_t*, float const*, int N_pad,
                                       int N_blocks_in, int K_blocks_per_row,
                                       cudaStream_t);
} /* namespace detail */
} /* namespace blockscale_gemm */

namespace det = blockscale_gemm::detail;
using RunnerQ = tensorrt_llm::kernels::blockscale_gemm::
    CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3,
                                   __nv_bfloat16>;

namespace {

/* One process-global runner; construction + configureWorkspace(nullptr) is
 * exactly what the harness does. C++11 magic statics make this thread-safe. */
RunnerQ& runner() {
    static RunnerQ r = [] {
        RunnerQ rr;
        rr.configureWorkspace(nullptr);
        return rr;
    }();
    return r;
}

inline int ceil4(int v) { return (v + 3) / 4 * 4; }

inline s2p_status last_cuda(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[s2pro] fso %s: CUDA error: %s\n", what,
                     cudaGetErrorString(e));
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

} /* namespace */

extern "C" {

int s2p_fso_arch_ok(void) {
    static int cached = -1;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (cached < 0) {
        cudaDeviceProp p;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
            (void)cudaGetLastError(); /* clear */
            cached = 0;
        } else {
            /* The prebuilt objects carry sm_121a SASS only: require cc 12.x. */
            cached = (p.major == 12) ? 1 : 0;
            if (!cached)
                std::fprintf(stderr,
                             "[s2pro] fso: device cc %d.%d != 12.x — FP8 "
                             "path disabled, falling back to BF16\n",
                             p.major, p.minor);
        }
    }
    return cached;
}

/* -- sizing (bytes), matching the harness allocations exactly ------------- */

size_t s2p_fso_weight_fp8_bytes(int N, int K) {
    return (size_t)N * (size_t)K; /* e4m3 = 1 byte */
}

size_t s2p_fso_sfb_bytes(int N, int K) {
    /* int32-packed UE8M0: Npad rows, Kb/4 int32 per row (Kb = K/128). */
    int Npad = ceil4(N), Kb = K / 128;
    return (size_t)Npad * (size_t)(Kb / 4) * 4u;
}

size_t s2p_fso_weight_scratch_bytes(int N, int K) {
    /* float scales pre-repack: (N/128) x Kb */
    return (size_t)(N / 128) * (size_t)(K / 128) * 4u;
}

size_t s2p_fso_act_fp8_bytes(int M, int K) {
    return (size_t)M * (size_t)K;
}

size_t s2p_fso_sa_bytes(int M, int K) {
    int Mpad = ceil4(M), Kb = K / 128;
    return (size_t)Mpad * (size_t)(Kb / 4) * 4u;
}

/* -- quantization --------------------------------------------------------- */

s2p_status s2p_fso_quant_weight_128x128(void* w_fp8, void* sfb_packed,
                                        void* scratch_f32, const void* w_bf16,
                                        int N, int K, cudaStream_t stream) {
    if (!w_fp8 || !sfb_packed || !scratch_f32 || !w_bf16 || N <= 0 || K <= 0)
        return S2P_ERR_INVALID;
    if (N % 128 != 0 || K % 128 != 0 || K % 512 != 0)
        return S2P_ERR_INVALID; /* sfb repack packs Kb by 4 => K%512 */
    if (!s2p_fso_arch_ok()) return S2P_ERR_UNSUPPORTED;

    det::fp8bs_quantize_128x128((__nv_fp8_e4m3*)w_fp8, (float*)scratch_f32,
                                (__nv_bfloat16 const*)w_bf16, N, K, stream);
    S2P_TRY(last_cuda("fp8bs_quantize_128x128"));
    det::repack_ue8m0_scales_sfb_for_sm120((int32_t*)sfb_packed,
                                           (float const*)scratch_f32, ceil4(N),
                                           N / 128, K / 128, stream);
    return last_cuda("repack_ue8m0_scales_sfb_for_sm120");
}

s2p_status s2p_fso_quant_act_1x128_packed(void* x_fp8, void* sa_packed,
                                          const void* x_bf16, int M, int K,
                                          cudaStream_t stream) {
    if (!x_fp8 || !sa_packed || !x_bf16 || M <= 0 || K <= 0)
        return S2P_ERR_INVALID;
    if (K % 512 != 0) return S2P_ERR_INVALID;
    if (!s2p_fso_arch_ok()) return S2P_ERR_UNSUPPORTED;

    det::fp8bs_quantize_1x128_packed((__nv_fp8_e4m3*)x_fp8,
                                     (int32_t*)sa_packed,
                                     (__nv_bfloat16 const*)x_bf16, M, K,
                                     stream, /*use_ue8m0=*/true);
    return last_cuda("fp8bs_quantize_1x128_packed");
}

/* -- gemm ----------------------------------------------------------------- */

s2p_status s2p_fso_gemm_fp8(void* y_bf16, const void* x_fp8,
                            const void* sa_packed, const void* w_fp8,
                            const void* sfb_packed, int M, int N, int K,
                            cudaStream_t stream) {
    if (!y_bf16 || !x_fp8 || !sa_packed || !w_fp8 || !sfb_packed || M <= 0 ||
        N <= 0 || K <= 0)
        return S2P_ERR_INVALID;
    if (N % 128 != 0 || K % 512 != 0) return S2P_ERR_INVALID;
    if (!s2p_fso_arch_ok()) return S2P_ERR_UNSUPPORTED;

    /* Harness call: run.gemm(aq, K, wq, K, y, N, M, N, K, sa, sb, st). */
    runner().gemm((__nv_fp8_e4m3 const*)x_fp8, K, (__nv_fp8_e4m3 const*)w_fp8,
                  K, (__nv_bfloat16*)y_bf16, N, M, N, K,
                  (float const*)sa_packed, (float const*)sfb_packed, stream);
    return last_cuda("CutlassFp8BlockScaleGemmRunner::gemm");
}

} /* extern "C" */
