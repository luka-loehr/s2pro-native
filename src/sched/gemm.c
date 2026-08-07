/* s2pro-native — linear layers: cuBLAS BF16 default, fish-scales-ops FP8
 * opt-in (S2P_FP8=1), per-out-channel weight-only INT8 opt-in (S2P_INT8=1).
 * Implements include/s2pro/gemm.h.
 *
 * Convention (row-major): x [M, in], w [out, in], y [M, out] => y = x * w^T.
 * The cuBLAS call is the exact one proven in ~/fso-sm121-test/harness3.cu:
 *   cublasGemmEx(OP_T, OP_N, N, M, K, w(ld K), x(ld K), y(ld N),
 *                CUDA_R_16BF in/out, CUBLAS_COMPUTE_32F).
 * FP8 weights are quantized ONCE at load (s2p_linear_prepare_fp8);
 * activations are quantized per call from a process-global scratch that grows
 * on demand. Any shape the FP8 path cannot serve (K % 512 != 0) falls back
 * to BF16 with a logged warning — never an error.
 * INT8 weights are quantized ONCE at load (s2p_linear_prepare_int8), which
 * then FREES the BF16 copy — that is the sub-9-GB RAM lever. Decode M<=8
 * runs the int8 GEMV; larger M (prefill) dequantizes the layer into a shared
 * scratch and reuses the proven cuBLAS BF16 call, trading one-time prefill
 * bandwidth for the per-frame halving that decides RTF. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "s2pro/status.h"
#include "s2pro/gemm.h"
#include "s2pro/tensor.h"
#include "s2pro/safetensors.h"
#include "../fso/fso.h"

#define S2P_CUBLAS_TRY(expr)                                                   \
    do {                                                                       \
        cublasStatus_t _cs = (expr);                                           \
        if (_cs != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[s2pro] cuBLAS error %s:%d: status %d\n",         \
                    __FILE__, __LINE__, (int)_cs);                             \
            return S2P_ERR_CUDA;                                               \
        }                                                                      \
    } while (0)

/* Process-global gemm context. The cuBLAS handle is not safe for concurrent
 * calls, so every use is serialized by g.mu (in practice only the scheduler
 * worker thread issues GEMMs after load). */
static struct {
    int             inited;
    cublasHandle_t  handle;
    pthread_mutex_t mu;
    int             max_m;         /* sizing hint for the act scratch */
    /* FP8 activation scratch (device), grown on demand: */
    void*  act_fp8;    size_t act_fp8_bytes;
    void*  act_scales; size_t act_scales_bytes;
    /* INT8 prefill dequant scratch (device), grown on demand: */
    void*  deq;        size_t deq_bytes;
} g = { 0, NULL, PTHREAD_MUTEX_INITIALIZER, 0, NULL, 0, NULL, 0, NULL, 0 };

s2p_status s2p_gemm_init(int max_m) {
    pthread_mutex_lock(&g.mu);
    if (g.inited) {
        pthread_mutex_unlock(&g.mu);
        return S2P_OK;
    }
    cublasStatus_t cs = cublasCreate(&g.handle);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        pthread_mutex_unlock(&g.mu);
        fprintf(stderr, "[s2pro] cublasCreate failed: status %d\n", (int)cs);
        return S2P_ERR_CUDA;
    }
    g.max_m = max_m > 0 ? max_m : 32;
    g.inited = 1;
    pthread_mutex_unlock(&g.mu);
    return S2P_OK;
}

void s2p_gemm_shutdown(void) {
    pthread_mutex_lock(&g.mu);
    if (g.inited) {
        if (g.act_fp8) cudaFree(g.act_fp8);
        if (g.act_scales) cudaFree(g.act_scales);
        if (g.deq) cudaFree(g.deq);
        g.act_fp8 = NULL;
        g.act_scales = NULL;
        g.deq = NULL;
        g.act_fp8_bytes = g.act_scales_bytes = g.deq_bytes = 0;
        cublasDestroy(g.handle);
        g.handle = NULL;
        g.inited = 0;
    }
    pthread_mutex_unlock(&g.mu);
}

s2p_gemm_mode s2p_gemm_mode_from_env(void) {
    const char* i8 = getenv("S2P_INT8");
    if (i8 && i8[0] == '1' && i8[1] == '\0') return S2P_GEMM_INT8;
    const char* v = getenv("S2P_FP8");
    if (v && v[0] == '1' && v[1] == '\0') {
        if (!s2p_fso_available()) {
            fprintf(stderr, "[s2pro] S2P_FP8=1 but FP8 path unavailable "
                            "(arch != cc 12.x) — using BF16\n");
            return S2P_GEMM_BF16;
        }
        return S2P_GEMM_FP8;
    }
    return S2P_GEMM_BF16;
}

int s2p_fso_available(void) {
    /* Kernels are always linked (Makefile links the prebuilt FSO objects);
     * availability is purely the runtime arch gate. */
    return s2p_fso_arch_ok();
}

/* The cublas call itself; caller holds g.mu. */
static s2p_status gemm_bf16_locked(const void* x, const void* w, void* y,
                                   int M, int N, int K, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cs = cublasSetStream(g.handle, stream);
    if (cs == CUBLAS_STATUS_SUCCESS)
        cs = cublasGemmEx(g.handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                          w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, y,
                          CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                          CUBLAS_GEMM_DEFAULT);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[s2pro] cublasGemmEx(M=%d,N=%d,K=%d) failed: %d\n",
                M, N, K, (int)cs);
        return S2P_ERR_CUDA;
    }
    return S2P_OK;
}

/* y[M,N] = x[M,K] * w[N,K]^T, BF16 in/out, FP32 accumulate. */
s2p_status s2p_gemm_bf16(const void* x, const void* w, void* y, int M, int N,
                         int K, cudaStream_t stream) {
    if (!x || !w || !y || M <= 0 || N <= 0 || K <= 0) return S2P_ERR_INVALID;
    if (!g.inited) return S2P_ERR_STATE;
    pthread_mutex_lock(&g.mu);
    s2p_status rc = gemm_bf16_locked(x, w, y, M, N, K, stream);
    pthread_mutex_unlock(&g.mu);
    return rc;
}

s2p_status s2p_linear_from_st(s2p_linear* lin, s2p_st* st, const char* name,
                              int in_features, int out_features,
                              cudaStream_t stream) {
    if (!lin || !st || !name || in_features <= 0 || out_features <= 0)
        return S2P_ERR_INVALID;
    memset(lin, 0, sizeof(*lin));
    lin->in_features = in_features;
    lin->out_features = out_features;
    int64_t shape[2] = { (int64_t)out_features, (int64_t)in_features };
    S2P_TRY(s2p_st_load_device(st, name, S2P_DT_BF16, 2, shape, &lin->w_bf16,
                               stream));
    lin->fp8_ready = 0;
    return S2P_OK;
}

s2p_status s2p_linear_prepare_fp8(s2p_linear* lin, cudaStream_t stream) {
    if (!lin || !lin->w_bf16.data) return S2P_ERR_INVALID;
    if (lin->fp8_ready) return S2P_OK;
    int N = lin->out_features, K = lin->in_features;
    if (!s2p_fso_available()) {
        fprintf(stderr, "[s2pro] fp8 prepare [%d,%d]: FP8 path unavailable, "
                        "layer stays BF16\n", N, K);
        return S2P_OK;
    }
    if (N % 128 != 0 || K % 512 != 0) {
        /* K % 512 covers the act-scale repack constraint (docs/SPARK.md). */
        fprintf(stderr, "[s2pro] fp8 prepare [%d,%d]: shape not FP8-servable "
                        "(need N%%128==0 && K%%512==0), layer stays BF16\n",
                N, K);
        return S2P_OK;
    }

    int64_t fp8_shape[2] = { (int64_t)N, (int64_t)K };
    S2P_TRY(s2p_tensor_device_alloc(&lin->w_fp8, S2P_DT_I8, 2, fp8_shape));
    int64_t sc_elems = (int64_t)(s2p_fso_sfb_bytes(N, K) / sizeof(int32_t));
    s2p_status rc = s2p_tensor_device_alloc(&lin->w_scales, S2P_DT_I32, 1,
                                            &sc_elems);
    if (rc != S2P_OK) {
        s2p_tensor_free(&lin->w_fp8);
        return rc;
    }

    void* scratch = NULL;
    cudaError_t ce = cudaMalloc(&scratch, s2p_fso_weight_scratch_bytes(N, K));
    if (ce != cudaSuccess) {
        s2p_tensor_free(&lin->w_fp8);
        s2p_tensor_free(&lin->w_scales);
        fprintf(stderr, "[s2pro] fp8 prepare: scratch alloc failed: %s\n",
                cudaGetErrorString(ce));
        return S2P_ERR_OOM;
    }
    rc = s2p_fso_quant_weight_128x128(lin->w_fp8.data, lin->w_scales.data,
                                      scratch, lin->w_bf16.data, N, K, stream);
    if (rc == S2P_OK) {
        ce = cudaStreamSynchronize(stream); /* scratch is freed next */
        if (ce != cudaSuccess) {
            fprintf(stderr, "[s2pro] fp8 prepare sync: %s\n",
                    cudaGetErrorString(ce));
            rc = S2P_ERR_CUDA;
        }
    }
    cudaFree(scratch);
    if (rc != S2P_OK) {
        s2p_tensor_free(&lin->w_fp8);
        s2p_tensor_free(&lin->w_scales);
        return rc;
    }
    lin->fp8_ready = 1;
    return S2P_OK;
}

static int int8_keep_bf16(void) {
    const char* v = getenv("S2P_INT8_KEEP_BF16");
    return v && v[0] == '1' && v[1] == '\0';
}

/* ── INT4 mixed-precision policy (S2P_INT4=1 on top of S2P_INT8=1) ──
 * Backbone linears go group-wise INT4; the fast-AR (run 9x per frame, the
 * tensor 4-bit damages most: argmax 1/9 vs oracle when naive-quantized) and
 * the tied lm-head sidecar stay per-channel INT8. S2P_INT4_ALL=1 forces
 * group-wise INT4 on the fast-AR too, for A/B listening. */
static int env_flag(const char* name) {
    const char* v = getenv(name);
    return v && v[0] == '1' && v[1] == '\0';
}

static int int4_group(void) {
    static int g = 0;
    if (g == 0) {
        const char* v = getenv("S2P_INT4_GROUP");
        g = v ? atoi(v) : 32;
        if (g < 16 || g > 512 || (g & (g - 1)) != 0) {
            fprintf(stderr, "[s2pro] S2P_INT4_GROUP=%s invalid, using 32\n",
                    v);
            g = 32;
        }
    }
    return g;
}

static int int4_mse(void) {
    const char* v = getenv("S2P_INT4_MSE");
    return v ? (v[0] == '1' && v[1] == '\0') : 1; /* default ON */
}

static int int4_packed(void) {
    const char* v = getenv("S2P_INT4_PACKED");
    return v ? (v[0] == '1' && v[1] == '\0') : 1; /* default ON */
}

static int int4_wanted(s2p_qsite site) {
    static int banner = 0;
    if (!env_flag("S2P_INT4")) return 0;
    if (!banner) {
        banner = 1;
        fprintf(stderr, "[s2pro] S2P_INT4=1: group-wise 4-bit weights "
                        "(g=%d, mse=%d, fastar=%s, %s)\n",
                int4_group(), int4_mse(),
                env_flag("S2P_INT4_ALL") ? "int4" : "int8",
                int4_packed() ? "packed" : "int8 container");
    }
    if (site == S2P_QSITE_FASTAR) return env_flag("S2P_INT4_ALL");
    return 1;
}

s2p_status s2p_linear_prepare_int8(s2p_linear* lin, cudaStream_t stream) {
    if (!lin) return S2P_ERR_INVALID;
    if (lin->int8_ready) return S2P_OK; /* incl. qcache-loaded linears */
    if (!lin->w_bf16.data) return S2P_ERR_INVALID;
    int N = lin->out_features, K = lin->in_features;
    if (K % 512 != 0) {
        /* GEMV tile requirement; no such layer exists in S2-Pro. */
        fprintf(stderr, "[s2pro] int8 prepare [%d,%d]: K %% 512 != 0, layer "
                        "stays BF16\n", N, K);
        return S2P_OK;
    }

    int64_t qshape[2] = { (int64_t)N, (int64_t)K };
    S2P_TRY(s2p_tensor_device_alloc(&lin->w_int8, S2P_DT_I8, 2, qshape));
    int64_t sshape[1] = { (int64_t)N };
    s2p_status rc = s2p_tensor_device_alloc(&lin->w_iscale, S2P_DT_F32, 1,
                                            sshape);
    if (rc == S2P_OK)
        rc = s2p_int8_quant(lin->w_int8.data, (float*)lin->w_iscale.data,
                            lin->w_bf16.data, N, K, stream);
    if (rc == S2P_OK) {
        /* the BF16 source is freed (or reused by the caller) next */
        cudaError_t ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[s2pro] int8 prepare sync: %s\n",
                    cudaGetErrorString(ce));
            rc = S2P_ERR_CUDA;
        }
    }
    if (rc != S2P_OK) {
        s2p_tensor_free(&lin->w_int8);
        s2p_tensor_free(&lin->w_iscale);
        return rc;
    }
    lin->int8_ready = 1;
    if (!int8_keep_bf16()) s2p_tensor_free(&lin->w_bf16);
    return S2P_OK;
}

/* Group-wise INT4 variant of prepare_int8; same lifecycle (frees the BF16
 * source unless S2P_INT8_KEEP_BF16=1). */
static s2p_status prepare_int4_group(s2p_linear* lin, cudaStream_t stream) {
    if (!lin) return S2P_ERR_INVALID;
    if (lin->int8_ready) return S2P_OK; /* incl. qcache-loaded linears */
    if (!lin->w_bf16.data) return S2P_ERR_INVALID;
    int N = lin->out_features, K = lin->in_features;
    const int G = int4_group();
    if (K % 512 != 0 || K % G != 0) {
        fprintf(stderr, "[s2pro] int4 prepare [%d,%d]: shape not servable "
                        "(K %% 512 or K %% %d), layer stays BF16\n", N, K, G);
        return S2P_OK;
    }

    int64_t qshape[2] = { (int64_t)N, (int64_t)K };
    S2P_TRY(s2p_tensor_device_alloc(&lin->w_int8, S2P_DT_I8, 2, qshape));
    int64_t sshape[1] = { (int64_t)N * (K / G) };
    s2p_status rc = s2p_tensor_device_alloc(&lin->w_iscale, S2P_DT_F16, 1,
                                            sshape);
    if (rc == S2P_OK)
        rc = s2p_intq_quant(lin->w_int8.data, lin->w_iscale.data,
                            lin->w_bf16.data, N, K, G, 7, int4_mse(), stream);
    if (rc == S2P_OK) {
        cudaError_t ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[s2pro] int4 prepare sync: %s\n",
                    cudaGetErrorString(ce));
            rc = S2P_ERR_CUDA;
        }
    }
    if (rc == S2P_OK && int4_packed()) {
        /* pack two nibbles per byte, then drop the int8 container — that is
         * the INT4 bandwidth/RAM win (outputs stay bit-identical). */
        int64_t pshape[2] = { (int64_t)N, (int64_t)K / 2 };
        rc = s2p_tensor_device_alloc(&lin->w_pack, S2P_DT_I8, 2, pshape);
        if (rc == S2P_OK)
            rc = s2p_int4_pack(lin->w_pack.data, lin->w_int8.data, N, K,
                               stream);
        if (rc == S2P_OK) {
            cudaError_t ce = cudaStreamSynchronize(stream);
            if (ce != cudaSuccess) {
                fprintf(stderr, "[s2pro] int4 pack sync: %s\n",
                        cudaGetErrorString(ce));
                rc = S2P_ERR_CUDA;
            }
        }
        if (rc == S2P_OK) {
            s2p_tensor_free(&lin->w_int8);
            lin->q_packed = 1;
        }
    }
    if (rc != S2P_OK) {
        s2p_tensor_free(&lin->w_int8);
        s2p_tensor_free(&lin->w_iscale);
        s2p_tensor_free(&lin->w_pack);
        return rc;
    }
    lin->int8_ready = 1;
    lin->q_group = G;
    if (!int8_keep_bf16()) s2p_tensor_free(&lin->w_bf16);
    return S2P_OK;
}

s2p_status s2p_linear_prepare_int8_site(s2p_linear* lin, s2p_qsite site,
                                        cudaStream_t stream) {
    if (int4_wanted(site)) return prepare_int4_group(lin, stream);
    return s2p_linear_prepare_int8(lin, stream);
}

/* Ensure the dequant scratch covers one [N,K] BF16 layer. g.mu held. */
s2p_status s2p_int4p_gemm_tc(void* y_bf16, const void* x_bf16,
                             const void* w_pack, const void* scales_f16,
                             int M, int N, int K, int G,
                             cudaStream_t stream);

static int prefill_tc_on(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_PREFILL_TC");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static s2p_status deq_scratch_reserve(size_t bytes, cudaStream_t stream) {
    if (bytes <= g.deq_bytes) return S2P_OK;
    S2P_CUDA_TRY(cudaStreamSynchronize(stream));
    if (g.deq) { cudaFree(g.deq); g.deq = NULL; g.deq_bytes = 0; }
    cudaError_t ce = cudaMalloc(&g.deq, bytes);
    if (ce != cudaSuccess) return S2P_ERR_OOM;
    g.deq_bytes = bytes;
    return S2P_OK;
}

/* Ensure the global activation scratch covers (M, K). Called with g.mu held.
 * Growth syncs the stream first so in-flight work never loses its buffer. */
static s2p_status act_scratch_reserve(int M, int K, cudaStream_t stream) {
    int cap_m = M > g.max_m ? M : g.max_m;
    size_t need_fp8 = s2p_fso_act_fp8_bytes(cap_m, K);
    size_t need_sc = s2p_fso_sa_bytes(cap_m, K);
    if (need_fp8 <= g.act_fp8_bytes && need_sc <= g.act_scales_bytes)
        return S2P_OK;
    S2P_CUDA_TRY(cudaStreamSynchronize(stream));
    if (g.act_fp8) { cudaFree(g.act_fp8); g.act_fp8 = NULL; g.act_fp8_bytes = 0; }
    if (g.act_scales) { cudaFree(g.act_scales); g.act_scales = NULL; g.act_scales_bytes = 0; }
    cudaError_t ce = cudaMalloc(&g.act_fp8, need_fp8);
    if (ce != cudaSuccess) return S2P_ERR_OOM;
    ce = cudaMalloc(&g.act_scales, need_sc);
    if (ce != cudaSuccess) {
        cudaFree(g.act_fp8);
        g.act_fp8 = NULL;
        return S2P_ERR_OOM;
    }
    g.act_fp8_bytes = need_fp8;
    g.act_scales_bytes = need_sc;
    return S2P_OK;
}

s2p_status s2p_linear_forward(const s2p_linear* lin, const void* x_bf16,
                              void* y_bf16, int M, s2p_gemm_mode mode,
                              cudaStream_t stream) {
    if (!lin || !x_bf16 || !y_bf16 || M <= 0) return S2P_ERR_INVALID;
    if (!g.inited) return S2P_ERR_STATE;
    int N = lin->out_features, K = lin->in_features;

    if (mode == S2P_GEMM_FP8 && lin->fp8_ready) {
        pthread_mutex_lock(&g.mu);
        s2p_status rc = act_scratch_reserve(M, K, stream);
        if (rc == S2P_OK)
            rc = s2p_fso_quant_act_1x128_packed(g.act_fp8, g.act_scales,
                                                x_bf16, M, K, stream);
        if (rc == S2P_OK)
            rc = s2p_fso_gemm_fp8(y_bf16, g.act_fp8, g.act_scales,
                                  lin->w_fp8.data, lin->w_scales.data, M, N,
                                  K, stream);
        pthread_mutex_unlock(&g.mu);
        return rc;
    }
    if (mode == S2P_GEMM_INT8 && lin->int8_ready) {
        if (M <= S2P_INT8_GEMV_MAX_M) {
            if (lin->q_packed)
                return s2p_int4p_gemv(y_bf16, x_bf16, lin->w_pack.data,
                                      lin->w_iscale.data, M, N, K,
                                      lin->q_group, stream);
            if (lin->q_group > 0)
                return s2p_intq_gemv(y_bf16, x_bf16, lin->w_int8.data,
                                     lin->w_iscale.data, M, N, K,
                                     lin->q_group, stream);
            return s2p_int8_gemv(y_bf16, x_bf16, lin->w_int8.data,
                                 (const float*)lin->w_iscale.data, M, N, K,
                                 stream);
        }
        /* S2P_PREFILL_TC=1: chunk-sized prefills (M <= 128) run the
         * tensor-core GEMM over the packed weights directly — no
         * N*K*2 B dequant round-trip per linear per chunk. Different
         * summation order than cuBLAS, so parity-gated and off by
         * default (audit P1-5). */
        if (prefill_tc_on() && lin->q_packed && M <= 128 && K % 64 == 0)
            return s2p_int4p_gemm_tc(y_bf16, x_bf16, lin->w_pack.data,
                                     lin->w_iscale.data, M, N, K,
                                     lin->q_group, stream);
        /* prefill / oversize batch: dequant into the shared scratch, then
         * the proven cuBLAS call. Lock held across both so a concurrent
         * reserve can never free the scratch under the GEMM. */
        pthread_mutex_lock(&g.mu);
        s2p_status rc = deq_scratch_reserve(
            (size_t)N * K * sizeof(uint16_t), stream);
        if (rc == S2P_OK) {
            if (lin->q_packed)
                rc = s2p_int4p_dequant(g.deq, lin->w_pack.data,
                                       lin->w_iscale.data, N, K,
                                       lin->q_group, stream);
            else if (lin->q_group > 0)
                rc = s2p_intq_dequant(g.deq, lin->w_int8.data,
                                      lin->w_iscale.data, N, K,
                                      lin->q_group, stream);
            else
                rc = s2p_int8_dequant(g.deq, lin->w_int8.data,
                                      (const float*)lin->w_iscale.data, N,
                                      K, stream);
        }
        if (rc == S2P_OK)
            rc = gemm_bf16_locked(x_bf16, g.deq, y_bf16, M, N, K, stream);
        pthread_mutex_unlock(&g.mu);
        return rc;
    }
    /* BF16 default; also the silent fallback for FP8/INT8-unready layers. */
    if (!lin->w_bf16.data) {
        fprintf(stderr, "[s2pro] forward fallback without BF16 copy "
                        "[N=%d,K=%d] M=%d mode=%d ready=%d grp=%d pk=%d\n",
                N, K, M, (int)mode, lin->int8_ready, lin->q_group,
                lin->q_packed);
        return S2P_ERR_STATE; /* dropped by prepare_int8 */
    }
    return s2p_gemm_bf16(x_bf16, lin->w_bf16.data, y_bf16, M, N, K, stream);
}

void s2p_linear_free(s2p_linear* lin) {
    if (!lin) return;
    s2p_tensor_free(&lin->w_bf16);
    s2p_tensor_free(&lin->w_fp8);
    s2p_tensor_free(&lin->w_scales);
    s2p_tensor_free(&lin->w_int8);
    s2p_tensor_free(&lin->w_iscale);
    s2p_tensor_free(&lin->w_pack);
    lin->fp8_ready = 0;
    lin->int8_ready = 0;
    lin->q_group = 0;
    lin->q_packed = 0;
}
