/* s2pro-native — fast-AR audio decoder implementation (see fastar.h).
 *
 * Mirrors fish_speech/models/text2semantic/audio_decoder.py +
 * sglang_model.py::_decode_codebooks (fast half) exactly:
 *   per frame: reset_caches(); prime KV pos 0 with the final-normed slow
 *   hidden (project_in = Identity, logits DISCARDED — PORTING pitfall 6);
 *   then for cb_idx 1..9: forward one embeddings(prev_code) token at KV pos
 *   cb_idx, final RMSNorm -> untied output head -> argmax over [:4096].
 * The recurrence between steps flows ONLY through the KV cache; each step's
 * input is a fresh embedding row, never the previous residual stream.
 *
 * Uses the shared primitive launchers from include/s2pro/kernels.h with
 * qk-norm SKIPPED (fast-AR has none) and s2p_linear GEMMs. Private kernels
 * here: qkv split (fused [B,6144] -> contiguous q/k/v), i32->i64 id widen
 * with defensive 0..4095 clamp, and the 10-codebook VQ embedding sum.
 */
#include <stdlib.h>
#include <string.h>

#include "fastar.h"
#include "s2pro/kernels.h"
#include "s2pro/tensor.h"

/* parity dump hooks (src/slowar/debug_dump.c); no-ops without S2P_DUMP_DIR */
extern "C" {
const char* s2psl_dump_dir(void);
void s2psl_dump_vec_bf16(const char* name, const void* dev_bf16, int64_t n,
                         cudaStream_t st);
}

/* ---- dims (compile-time, verified against checkpoint config.json) ---- */
#define FA_DIM      S2P_DIM            /* 2560 */
#define FA_LAYERS   S2P_FAST_LAYERS    /* 4 */
#define FA_QH       S2P_SLOW_Q_HEADS   /* 32 (same head layout as backbone) */
#define FA_KVH      S2P_SLOW_KV_HEADS  /* 8 */
#define FA_HD       S2P_HEAD_DIM       /* 128 */
#define FA_QW       S2P_Q_WIDTH        /* 4096 */
#define FA_KVW      S2P_KV_WIDTH       /* 1024 */
#define FA_QKVW     S2P_QKV_WIDTH      /* 6144 */
#define FA_FFN      S2P_FFN_DIM        /* 9728 */
#define FA_CB       S2P_CB_SIZE        /* 4096 */
#define FA_NCB      S2P_NUM_CODEBOOKS  /* 10 */
#define FA_SEQ      S2P_FAST_SEQ       /* 11 = KV depth */
#define FA_KVSZ     (FA_KVH * FA_SEQ * FA_HD) /* elems per cache tensor */

typedef struct {
    s2p_linear wqkv;   /* [6144, 2560] */
    s2p_linear wo;     /* [2560, 4096] */
    s2p_linear w13;    /* fused [2*9728, 2560]: rows 0..9727 = w1, rest = w3 */
    s2p_linear w2;     /* [2560, 9728] */
    s2p_tensor attn_norm; /* [2560] */
    s2p_tensor ffn_norm;  /* [2560] */
} s2pfa_layer;

struct s2pfa {
    s2p_gemm_mode mode;
    s2pfa_layer   layers[FA_LAYERS];
    s2p_tensor    norm;          /* final RMSNorm weight [2560] */
    s2p_linear    output;        /* untied head [4096, 2560] */
    s2p_tensor    cb_emb;        /* codebook_embeddings [40960, 2560] */
    s2p_tensor    emb;           /* embeddings [4096, 2560] */
    /* decode workspace (device), sized for max_b rows */
    int            max_b;
    __nv_bfloat16 *x, *t, *qkv, *q, *k, *v, *attn, *proj, *gu, *hff, *logits;
    int64_t       *ids64;        /* [B] embed gather ids */
    int32_t       *sem_dev;      /* [B] uploaded semantic codes */
    int32_t       *stage;        /* [9*B] residuals, step-major (argmax dst) */
    int32_t       *stage_host;   /* pinned [9*B] */
    int32_t       *sem_host;     /* pinned [B] */
    __nv_bfloat16 *kv_slab;      /* [FA_LAYERS][max_b][2][FA_KVSZ] */
    size_t         kv_slab_bytes;
    /* vq scratch (grow-on-demand) */
    int32_t       *vq_dev;
    size_t         vq_cap;       /* elements */
};

/* ---------------------------------------------------------------- kernels */

/* Sum the 10 codebook embeddings of frame t (f32 accumulate, one bf16 round)
 * and add onto rows[t] with one bf16 add — matches torch's f32-accumulated
 * bf16 sum(dim=1) followed by a bf16 elementwise add. */
static __global__ void k_vq_embed_add(const __nv_bfloat16* __restrict__ table,
                                      const int32_t* __restrict__ codes, int T,
                                      __nv_bfloat16* __restrict__ rows) {
    int t = blockIdx.x;
    if (t >= T) return;
    int32_t idx[FA_NCB];
#pragma unroll
    for (int cb = 0; cb < FA_NCB; cb++) {
        int32_t c = codes[(size_t)cb * T + t];
        if (c < 0) c = 0;
        if (c >= FA_CB) c = FA_CB - 1;
        idx[cb] = cb * FA_CB + c;
    }
    for (int d = threadIdx.x; d < FA_DIM; d += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for (int cb = 0; cb < FA_NCB; cb++)
            acc += __bfloat162float(table[(size_t)idx[cb] * FA_DIM + d]);
        __nv_bfloat16 s = __float2bfloat16(acc); /* torch sum() output round */
        size_t o = (size_t)t * FA_DIM + d;
        rows[o] = __float2bfloat16(__bfloat162float(rows[o]) +
                                   __bfloat162float(s));
    }
}

/* Fused qkv [B, 6144] -> contiguous q [B,4096], k [B,1024], v [B,1024]. */
static __global__ void k_qkv_split(const __nv_bfloat16* __restrict__ qkv,
                                   __nv_bfloat16* __restrict__ q,
                                   __nv_bfloat16* __restrict__ k,
                                   __nv_bfloat16* __restrict__ v, int B) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = (int64_t)B * FA_QKVW;
    if (i >= n) return;
    int b = (int)(i / FA_QKVW);
    int c = (int)(i % FA_QKVW);
    __nv_bfloat16 x = qkv[i];
    if (c < FA_QW)
        q[(size_t)b * FA_QW + c] = x;
    else if (c < FA_QW + FA_KVW)
        k[(size_t)b * FA_KVW + (c - FA_QW)] = x;
    else
        v[(size_t)b * FA_KVW + (c - FA_QW - FA_KVW)] = x;
}

/* i32 codes -> i64 embed ids, clamped to the 4096-row table (the reference
 * clamps sem_id into [0, codebook_size-1]; argmax outputs are in range). */
static __global__ void k_widen_ids(const int32_t* __restrict__ src,
                                   int64_t* __restrict__ dst, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    int32_t c = src[i];
    if (c < 0) c = 0;
    if (c >= FA_CB) c = FA_CB - 1;
    dst[i] = (int64_t)c;
}

/* ------------------------------------------------------------- load utils */

static s2p_status fa_check_view(const s2p_st_view* v, int64_t d0, int64_t d1) {
    if (v->dtype != S2P_DT_BF16) return S2P_ERR_UNSUPPORTED;
    if (d1 < 0) { /* 1-D */
        if (v->ndim != 1 || v->shape[0] != d0) return S2P_ERR_FORMAT;
    } else {
        if (v->ndim != 2 || v->shape[0] != d0 || v->shape[1] != d1)
            return S2P_ERR_FORMAT;
    }
    return S2P_OK;
}

/* Build the fused [w1; w3] gate/up linear from the two checkpoint tensors.
 * Row-concatenation is numerically identical to two separate GEMMs and lands
 * gate||up per output row exactly as s2pk_silu_mul expects. 9728 % 128 == 0,
 * so FP8 128x128 weight blocks never straddle the w1/w3 boundary. */
static s2p_status fa_load_fused_w13(s2p_linear* lin, s2p_st* st,
                                    const char* n1, const char* n3,
                                    cudaStream_t stream) {
    s2p_st_view v1, v3;
    S2P_TRY(s2p_st_find(st, n1, &v1));
    S2P_TRY(s2p_st_find(st, n3, &v3));
    S2P_TRY(fa_check_view(&v1, FA_FFN, FA_DIM));
    S2P_TRY(fa_check_view(&v3, FA_FFN, FA_DIM));

    memset(lin, 0, sizeof(*lin));
    lin->in_features  = FA_DIM;
    lin->out_features = 2 * FA_FFN;
    int64_t shape[2] = { 2 * FA_FFN, FA_DIM };
    S2P_TRY(s2p_tensor_device_alloc(&lin->w_bf16, S2P_DT_BF16, 2, shape));
    size_t half = (size_t)FA_FFN * FA_DIM * sizeof(__nv_bfloat16);
    S2P_CUDA_TRY(cudaMemcpyAsync(lin->w_bf16.data, v1.data, half,
                                 cudaMemcpyHostToDevice, stream));
    S2P_CUDA_TRY(cudaMemcpyAsync((char*)lin->w_bf16.data + half, v3.data, half,
                                 cudaMemcpyHostToDevice, stream));
    return S2P_OK;
}

static s2p_status fa_alloc_bf16(__nv_bfloat16** p, size_t elems) {
    S2P_CUDA_TRY(cudaMalloc((void**)p, elems * sizeof(__nv_bfloat16)));
    return S2P_OK;
}

/* ------------------------------------------------------------------ load */

s2p_status s2pfa_load(s2pfa** out, s2p_st* st, s2p_qcache* qc,
                      s2p_gemm_mode mode, cudaStream_t stream) {
    if (!out || !st) return S2P_ERR_INVALID;
    *out = NULL;
    s2pfa* f = (s2pfa*)calloc(1, sizeof(s2pfa));
    if (!f) return S2P_ERR_OOM;
    f->mode  = mode;
    f->max_b = S2P_MAX_SESSIONS;

    s2p_status rc = S2P_OK;
    char name[128];
    for (int l = 0; l < FA_LAYERS && rc == S2P_OK; l++) {
        s2pfa_layer* L = &f->layers[l];
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.attention.wqkv.weight", l);
        if (!s2p_qcache_try_load(qc, name, FA_DIM, FA_QKVW, &L->wqkv, stream))
            rc = s2p_linear_from_st(&L->wqkv, st, name, FA_DIM, FA_QKVW,
                                    stream);
        if (rc != S2P_OK) break;
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.attention.wo.weight", l);
        if (!s2p_qcache_try_load(qc, name, FA_QW, FA_DIM, &L->wo, stream))
            rc = s2p_linear_from_st(&L->wo, st, name, FA_QW, FA_DIM, stream);
        if (rc != S2P_OK) break;
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.feed_forward.w13.fused", l);
        if (!s2p_qcache_try_load(qc, name, FA_DIM, 2 * FA_FFN, &L->w13,
                                 stream)) {
            snprintf(name, sizeof name,
                     "audio_decoder.layers.%d.feed_forward.w1.weight", l);
            char name3[128];
            snprintf(name3, sizeof name3,
                     "audio_decoder.layers.%d.feed_forward.w3.weight", l);
            rc = fa_load_fused_w13(&L->w13, st, name, name3, stream);
        }
        if (rc != S2P_OK) break;
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.feed_forward.w2.weight", l);
        if (!s2p_qcache_try_load(qc, name, FA_FFN, FA_DIM, &L->w2, stream))
            rc = s2p_linear_from_st(&L->w2, st, name, FA_FFN, FA_DIM, stream);
        if (rc != S2P_OK) break;
        int64_t dsh[1] = { FA_DIM };
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.attention_norm.weight", l);
        rc = s2p_st_load_device(st, name, S2P_DT_BF16, 1, dsh, &L->attn_norm,
                                stream);
        if (rc != S2P_OK) break;
        snprintf(name, sizeof name,
                 "audio_decoder.layers.%d.ffn_norm.weight", l);
        rc = s2p_st_load_device(st, name, S2P_DT_BF16, 1, dsh, &L->ffn_norm,
                                stream);
    }
    if (rc == S2P_OK) {
        int64_t dsh[1] = { FA_DIM };
        rc = s2p_st_load_device(st, "audio_decoder.norm.weight", S2P_DT_BF16,
                                1, dsh, &f->norm, stream);
    }
    if (rc == S2P_OK)
        rc = s2p_linear_from_st(&f->output, st, "audio_decoder.output.weight",
                                FA_DIM, FA_CB, stream);
    if (rc == S2P_OK) {
        int64_t sh[2] = { (int64_t)FA_NCB * FA_CB, FA_DIM };
        rc = s2p_st_load_device(st, "audio_decoder.codebook_embeddings.weight",
                                S2P_DT_BF16, 2, sh, &f->cb_emb, stream);
    }
    if (rc == S2P_OK) {
        int64_t sh[2] = { FA_CB, FA_DIM };
        rc = s2p_st_load_device(st, "audio_decoder.embeddings.weight",
                                S2P_DT_BF16, 2, sh, &f->emb, stream);
    }

    if (rc == S2P_OK && mode == S2P_GEMM_FP8) {
        for (int l = 0; l < FA_LAYERS && rc == S2P_OK; l++) {
            s2pfa_layer* L = &f->layers[l];
            rc = s2p_linear_prepare_fp8(&L->wqkv, stream);
            if (rc == S2P_OK) rc = s2p_linear_prepare_fp8(&L->wo, stream);
            if (rc == S2P_OK) rc = s2p_linear_prepare_fp8(&L->w13, stream);
            if (rc == S2P_OK) rc = s2p_linear_prepare_fp8(&L->w2, stream);
        }
        if (rc == S2P_OK) rc = s2p_linear_prepare_fp8(&f->output, stream);
    }
    if (rc == S2P_OK && mode == S2P_GEMM_INT8) {
        /* S2P_QSITE_FASTAR: the WHOLE fast-AR stays per-channel INT8 under
         * S2P_INT4=1. Narrowing the promotion was tried and measured OUT
         * (parity fast-AR argmax vs the 8/9 reference class): qkv+gate/up
         * at INT4-g32 -> 5/9; gate/up alone -> 3/9; everything -> 2/9.
         * This module decides 9 greedy 1024-way argmaxes per frame with
         * intra-frame feedback and only 4 layers of depth — every subset
         * of its tensors is 4-bit-sensitive. S2P_INT4_ALL=1 remains the
         * all-INT4 A/B switch. */
        for (int l = 0; l < FA_LAYERS && rc == S2P_OK; l++) {
            s2pfa_layer* L = &f->layers[l];
            rc = s2p_linear_prepare_int8_site(&L->wqkv, S2P_QSITE_FASTAR,
                                              stream);
            if (rc == S2P_OK)
                rc = s2p_linear_prepare_int8_site(&L->wo, S2P_QSITE_FASTAR,
                                                  stream);
            if (rc == S2P_OK)
                rc = s2p_linear_prepare_int8_site(&L->w13, S2P_QSITE_FASTAR,
                                                  stream);
            if (rc == S2P_OK)
                rc = s2p_linear_prepare_int8_site(&L->w2, S2P_QSITE_FASTAR,
                                                  stream);
        }
        if (rc == S2P_OK)
            rc = s2p_linear_prepare_int8_site(&f->output, S2P_QSITE_FASTAR,
                                              stream);
        for (int l = 0; l < FA_LAYERS && rc == S2P_OK; l++) {
            s2pfa_layer* L = &f->layers[l];
            snprintf(name, sizeof name,
                     "audio_decoder.layers.%d.attention.wqkv.weight", l);
            rc = s2p_qcache_put_linear(qc, name, &L->wqkv, stream);
            snprintf(name, sizeof name,
                     "audio_decoder.layers.%d.attention.wo.weight", l);
            if (rc == S2P_OK)
                rc = s2p_qcache_put_linear(qc, name, &L->wo, stream);
            snprintf(name, sizeof name,
                     "audio_decoder.layers.%d.feed_forward.w13.fused", l);
            if (rc == S2P_OK)
                rc = s2p_qcache_put_linear(qc, name, &L->w13, stream);
            snprintf(name, sizeof name,
                     "audio_decoder.layers.%d.feed_forward.w2.weight", l);
            if (rc == S2P_OK)
                rc = s2p_qcache_put_linear(qc, name, &L->w2, stream);
        }
        if (rc == S2P_OK)
            rc = s2p_qcache_put_linear(qc, "audio_decoder.output.weight",
                                       &f->output, stream);
    }

    /* decode workspace */
    if (rc == S2P_OK) {
        int B = f->max_b;
        rc = fa_alloc_bf16(&f->x, (size_t)B * FA_DIM);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->t, (size_t)B * FA_DIM);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->qkv, (size_t)B * FA_QKVW);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->q, (size_t)B * FA_QW);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->k, (size_t)B * FA_KVW);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->v, (size_t)B * FA_KVW);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->attn, (size_t)B * FA_QW);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->proj, (size_t)B * FA_DIM);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->gu, (size_t)B * 2 * FA_FFN);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->hff, (size_t)B * FA_FFN);
        if (rc == S2P_OK) rc = fa_alloc_bf16(&f->logits, (size_t)B * FA_CB);
        if (rc == S2P_OK) {
            f->kv_slab_bytes = (size_t)FA_LAYERS * B * 2 * FA_KVSZ *
                               sizeof(__nv_bfloat16);
            if (cudaMalloc((void**)&f->kv_slab, f->kv_slab_bytes) !=
                cudaSuccess)
                rc = S2P_ERR_OOM;
        }
        if (rc == S2P_OK &&
            cudaMalloc((void**)&f->ids64, B * sizeof(int64_t)) != cudaSuccess)
            rc = S2P_ERR_OOM;
        if (rc == S2P_OK &&
            cudaMalloc((void**)&f->sem_dev, B * sizeof(int32_t)) !=
                cudaSuccess)
            rc = S2P_ERR_OOM;
        if (rc == S2P_OK &&
            cudaMalloc((void**)&f->stage,
                       (size_t)(FA_NCB - 1) * B * sizeof(int32_t)) !=
                cudaSuccess)
            rc = S2P_ERR_OOM;
        if (rc == S2P_OK &&
            cudaMallocHost((void**)&f->stage_host,
                           (size_t)(FA_NCB - 1) * B * sizeof(int32_t)) !=
                cudaSuccess)
            rc = S2P_ERR_OOM;
        if (rc == S2P_OK &&
            cudaMallocHost((void**)&f->sem_host, B * sizeof(int32_t)) !=
                cudaSuccess)
            rc = S2P_ERR_OOM;
    }

    if (rc == S2P_OK) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "[s2pro] fastar load sync: %s\n",
                    cudaGetErrorString(e));
            rc = S2P_ERR_CUDA;
        }
    }
    if (rc != S2P_OK) {
        s2pfa_free(f);
        return rc;
    }
    *out = f;
    return S2P_OK;
}

void s2pfa_free(s2pfa* f) {
    if (!f) return;
    for (int l = 0; l < FA_LAYERS; l++) {
        s2pfa_layer* L = &f->layers[l];
        s2p_linear_free(&L->wqkv);
        s2p_linear_free(&L->wo);
        s2p_linear_free(&L->w13);
        s2p_linear_free(&L->w2);
        s2p_tensor_free(&L->attn_norm);
        s2p_tensor_free(&L->ffn_norm);
    }
    s2p_tensor_free(&f->norm);
    s2p_linear_free(&f->output);
    s2p_tensor_free(&f->cb_emb);
    s2p_tensor_free(&f->emb);
    cudaFree(f->x); cudaFree(f->t); cudaFree(f->qkv);
    cudaFree(f->q); cudaFree(f->k); cudaFree(f->v);
    cudaFree(f->attn); cudaFree(f->proj); cudaFree(f->gu);
    cudaFree(f->hff); cudaFree(f->logits);
    cudaFree(f->ids64); cudaFree(f->sem_dev); cudaFree(f->stage);
    cudaFree(f->kv_slab); cudaFree(f->vq_dev);
    cudaFreeHost(f->stage_host); cudaFreeHost(f->sem_host);
    free(f);
}

/* ------------------------------------------------------- vq embedding sum */

s2p_status s2pfa_vq_embed_add(s2pfa* f, const int32_t* codes, int T,
                              __nv_bfloat16* rows, cudaStream_t stream) {
    if (!f || !codes || !rows || T < 0) return S2P_ERR_INVALID;
    if (T == 0) return S2P_OK;
    size_t need = (size_t)FA_NCB * T;
    if (need > f->vq_cap) {
        cudaFree(f->vq_dev);
        f->vq_dev = NULL;
        f->vq_cap = 0;
        S2P_CUDA_TRY(cudaMalloc((void**)&f->vq_dev, need * sizeof(int32_t)));
        f->vq_cap = need;
    }
    S2P_CUDA_TRY(cudaMemcpyAsync(f->vq_dev, codes, need * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
    k_vq_embed_add<<<T, 256, 0, stream>>>(
        (const __nv_bfloat16*)f->cb_emb.data, f->vq_dev, T, rows);
    S2P_CUDA_TRY(cudaGetLastError());
    return S2P_OK;
}

s2p_status s2pfa_vq_embed_add_dev(s2pfa* f, const int32_t* codes_dev, int T,
                                  __nv_bfloat16* rows, cudaStream_t stream) {
    if (!f || !codes_dev || !rows || T < 0) return S2P_ERR_INVALID;
    if (T == 0) return S2P_OK;
    k_vq_embed_add<<<T, 256, 0, stream>>>(
        (const __nv_bfloat16*)f->cb_emb.data, codes_dev, T, rows);
    S2P_CUDA_TRY(cudaGetLastError());
    return S2P_OK;
}

/* ------------------------------------------------------------ decode loop */

/* Fused RoPE + KV-append + causal GQA attention for one fast-AR step
 * (S2P_FA_FUSED=1, EXPERIMENTAL, default OFF — a measured negative
 * result kept in the record): one block per (row, kv_head) replaces the
 * per-row s2pk_rope / s2pk_kv_append / s2pk_attention launch triple; the
 * <= 10-deep cache lives in shared memory and the softmax is the
 * reference kernel's single-tile body.
 *
 * Measured 2026-08-03 (all-INT4 + KV8, 60-frame smoke): decode
 * 24.4 -> 24.3 ms/frame — no meaningful win, because CUDA-graph replay
 * already amortizes the launch overhead the fusion removes; the fast-AR
 * cost is GEMV weight bandwidth, which fusion cannot touch. Parity:
 * argmax-equivalent on every first-frame residual step (cos delta
 * ~1e-5 vs the unfused kernels — a not-yet-located rounding-order
 * difference), so the fused path also fails the bit-exactness gate and
 * is NOT the default. Kept for future work: the same shape would carry
 * a norm+GEMV megakernel, which is where the remaining margin lives. */
static __device__ inline float fa_b2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
static __device__ inline __nv_bfloat16 fa_f2b(float v) {
    return __float2bfloat16(v);
}

static __global__ void k_fa_step(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v,
                                 __nv_bfloat16* kbase, __nv_bfloat16* vbase,
                                 size_t bstride, __nv_bfloat16* out,
                                 int cb_idx, float rope_base) {
    const int b = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x; /* == head_dim 128 */
    const int len = cb_idx + 1;  /* <= 10 */

    __shared__ __nv_bfloat16 kh[10][128 + 4]; /* +4: bank-conflict pad */
    __shared__ __nv_bfloat16 vh[10][128 + 4];
    __shared__ float sq[128];
    __shared__ float sp[128];
    __shared__ float red[128];
    __shared__ float s_tile[2];

    __nv_bfloat16* kc = kbase + (size_t)b * bstride +
                        ((size_t)kvh * FA_SEQ + cb_idx) * FA_HD;
    __nv_bfloat16* vc = vbase + (size_t)b * bstride +
                        ((size_t)kvh * FA_SEQ + cb_idx) * FA_HD;

    /* prior history rows from the global cache (written by earlier steps) */
    for (int j = 0; j < cb_idx; j++) {
        const __nv_bfloat16* kr = kc - (size_t)(cb_idx - j) * FA_HD;
        const __nv_bfloat16* vr = vc - (size_t)(cb_idx - j) * FA_HD;
        kh[j][tid] = kr[tid];
        vh[j][tid] = vr[tid];
    }

    /* current k: RoPE exactly like k_rope, then bf16 store (cache + smem) */
    {
        const int i = tid >> 1;
        const float ex = __fdiv_rn((float)(2 * i), (float)FA_HD);
        const float inv = __fdiv_rn(1.0f, powf(rope_base, ex));
        const float ang = __fmul_rn((float)cb_idx, inv);
        const float c = fa_b2f(fa_f2b(cosf(ang)));
        const float s = fa_b2f(fa_f2b(sinf(ang)));
        const size_t o = (size_t)b * (FA_KVH * FA_HD) + (size_t)kvh * FA_HD +
                         (size_t)(2 * i);
        const float x0 = fa_b2f(k[o]);
        const float x1 = fa_b2f(k[o + 1]);
        const float rv = (tid & 1)
                             ? __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s))
                             : __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
        const __nv_bfloat16 rb = fa_f2b(rv);
        kc[tid] = rb;
        kh[cb_idx][tid] = rb;
        const __nv_bfloat16 vb16 =
            v[(size_t)b * (FA_KVH * FA_HD) + (size_t)kvh * FA_HD + tid];
        vc[tid] = vb16;
        vh[cb_idx][tid] = vb16;
    }
    __syncthreads();

    const float scale = rsqrtf((float)FA_HD);
    for (int g = 0; g < FA_QH / FA_KVH; g++) {
        const int h = kvh * (FA_QH / FA_KVH) + g;
        /* q RoPE in registers (k_rope math), bf16-rounded like the
         * in-place reference write, then the k_attention load+scale */
        {
            const int i = tid >> 1;
            const float ex = __fdiv_rn((float)(2 * i), (float)FA_HD);
            const float inv = __fdiv_rn(1.0f, powf(rope_base, ex));
            const float ang = __fmul_rn((float)cb_idx, inv);
            const float c = fa_b2f(fa_f2b(cosf(ang)));
            const float s = fa_b2f(fa_f2b(sinf(ang)));
            const size_t o = (size_t)b * (FA_QH * FA_HD) + (size_t)h * FA_HD +
                             (size_t)(2 * i);
            const float x0 = fa_b2f(q[o]);
            const float x1 = fa_b2f(q[o + 1]);
            const float rv =
                (tid & 1) ? __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s))
                          : __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
            sq[tid] = fa_b2f(fa_f2b(rv)) * scale;
        }
        __syncthreads();

        /* single-tile online-softmax body, verbatim k_attention order */
        float s = -INFINITY;
        if (tid < len) {
            const __nv_bfloat16* kr = kh[tid];
            float d = 0.f;
            for (int e = 0; e < FA_HD; e++)
                d = fmaf(sq[e], fa_b2f(kr[e]), d);
            s = d;
        }
        red[tid] = s;
        __syncthreads();
        for (int off = 128 >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        if (tid == 0) s_tile[0] = red[0];
        __syncthreads();
        const float m_new = fmaxf(-INFINITY, s_tile[0]);
        const float p = (s == -INFINITY) ? 0.f : expf(s - m_new);
        sp[tid] = p;
        red[tid] = p;
        __syncthreads();
        for (int off = 128 >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) s_tile[1] = red[0];
        __syncthreads();
        const float l = s_tile[1];
        float acc = 0.f;
        for (int j2 = 0; j2 < len; j2++)
            acc = fmaf(sp[j2], fa_b2f(vh[j2][tid]), acc);
        out[(size_t)b * (FA_QH * FA_HD) + (size_t)h * FA_HD + tid] =
            fa_f2b(acc / l);
        __syncthreads(); /* sq/sp/red reuse for the next q head */
    }
}

static int fa_fused(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_FA_FUSED");
        v = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0; /* default OFF */
    }
    return v;
}

static inline __nv_bfloat16* fa_kcache(s2pfa* f, int l, int b) {
    return f->kv_slab + ((size_t)(l * f->max_b + b) * 2 + 0) * FA_KVSZ;
}
static inline __nv_bfloat16* fa_vcache(s2pfa* f, int l, int b) {
    return f->kv_slab + ((size_t)(l * f->max_b + b) * 2 + 1) * FA_KVSZ;
}

/* One 4-layer forward of x [B, 2560] at KV position cb_idx. Matches
 * TransformerBlock.forward_kvcached: h = x + attn(attention_norm(x)),
 * out = h + ff(ffn_norm(h)); attention has NO qk-norm, RoPE at pos cb_idx
 * (bf16-truncated table, interleaved pairs — kernels.h contract), append
 * k/v at cache position cb_idx, causal GQA attend over cache[:cb_idx+1]. */
static s2p_status fa_run_layers(s2pfa* f, int B, int cb_idx,
                                cudaStream_t stream) {
    for (int l = 0; l < FA_LAYERS; l++) {
        s2pfa_layer* L = &f->layers[l];
        S2P_CUDA_TRY(s2pk_rms_norm(f->x,
                                   (const __nv_bfloat16*)L->attn_norm.data,
                                   f->t, B, FA_DIM, S2P_NORM_EPS, stream));
        S2P_TRY(s2p_linear_forward(&L->wqkv, f->t, f->qkv, B, f->mode,
                                   stream));
        {
            int64_t n = (int64_t)B * FA_QKVW;
            int blocks = (int)((n + 255) / 256);
            k_qkv_split<<<blocks, 256, 0, stream>>>(f->qkv, f->q, f->k, f->v,
                                                    B);
            S2P_CUDA_TRY(cudaGetLastError());
        }
        /* NO qk-norm (attention_qk_norm=false — PORTING pitfall 8). */
        if (fa_fused()) {
            /* one launch replaces the rope/append/attention triple per row */
            dim3 g((unsigned)B, FA_KVH);
            k_fa_step<<<g, FA_HD, 0, stream>>>(
                f->q, f->k, f->v, fa_kcache(f, l, 0), fa_vcache(f, l, 0),
                (size_t)2 * FA_KVSZ, f->attn, cb_idx, S2P_ROPE_BASE);
            S2P_CUDA_TRY(cudaGetLastError());
        } else {
            for (int b = 0; b < B; b++) {
                S2P_CUDA_TRY(s2pk_rope(f->q + (size_t)b * FA_QW,
                                       f->k + (size_t)b * FA_KVW, 1, FA_QH,
                                       FA_KVH, FA_HD, cb_idx, S2P_ROPE_BASE,
                                       stream));
                S2P_CUDA_TRY(s2pk_kv_append(
                    f->k + (size_t)b * FA_KVW, f->v + (size_t)b * FA_KVW,
                    fa_kcache(f, l, b), fa_vcache(f, l, b), 1, FA_KVH, FA_HD,
                    cb_idx, FA_SEQ, stream));
                S2P_CUDA_TRY(s2pk_attention(
                    f->q + (size_t)b * FA_QW, fa_kcache(f, l, b),
                    fa_vcache(f, l, b), f->attn + (size_t)b * FA_QW, 1, FA_QH,
                    FA_KVH, FA_HD, cb_idx, FA_SEQ, stream));
            }
        }
        S2P_TRY(s2p_linear_forward(&L->wo, f->attn, f->proj, B, f->mode,
                                   stream));
        S2P_CUDA_TRY(s2pk_add(f->x, f->proj, (int64_t)B * FA_DIM, stream));
        S2P_CUDA_TRY(s2pk_rms_norm(f->x,
                                   (const __nv_bfloat16*)L->ffn_norm.data,
                                   f->t, B, FA_DIM, S2P_NORM_EPS, stream));
        S2P_TRY(s2p_linear_forward(&L->w13, f->t, f->gu, B, f->mode, stream));
        S2P_CUDA_TRY(s2pk_silu_mul(f->gu, f->hff, B, FA_FFN, stream));
        S2P_TRY(s2p_linear_forward(&L->w2, f->hff, f->proj, B, f->mode,
                                   stream));
        S2P_CUDA_TRY(s2pk_add(f->x, f->proj, (int64_t)B * FA_DIM, stream));
    }
    return S2P_OK;
}

s2p_status s2pfa_decode_frame_batch_dev(s2pfa* f, const __nv_bfloat16* hidden,
                                        const int32_t* sem_dev, int B,
                                        const int32_t** stage_dev,
                                        cudaStream_t stream) {
    if (!f || !hidden || !sem_dev || !stage_dev) return S2P_ERR_INVALID;
    if (B < 1 || B > f->max_b) return S2P_ERR_INVALID;
    *stage_dev = f->stage;

    /* reset_caches(): the reference zeroes the KV cache EVERY frame. */
    S2P_CUDA_TRY(cudaMemsetAsync(f->kv_slab, 0, f->kv_slab_bytes, stream));

    /* Prime pass, KV pos 0: project_in = Identity on the FINAL-NORMED slow
     * hidden (PORTING pitfall 5). The reference computes output(norm(x)) here
     * and discards it — we skip the dead head GEMM (consumed values equal). */
    S2P_CUDA_TRY(cudaMemcpyAsync(f->x, hidden,
                                 (size_t)B * FA_DIM * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToDevice, stream));
    S2P_TRY(fa_run_layers(f, B, 0, stream));

    /* Seed step 1 with embeddings(sem_id) — the 4096-row table, NOT
     * codebook_embeddings (PORTING pitfall 7). */
    {
        int blocks = (B + 63) / 64;
        k_widen_ids<<<blocks, 64, 0, stream>>>(sem_dev, f->ids64, B);
        S2P_CUDA_TRY(cudaGetLastError());
    }
    S2P_CUDA_TRY(s2pk_embed((const __nv_bfloat16*)f->emb.data, f->ids64, f->x,
                            B, FA_DIM, stream));

    /* 9 residual steps: greedy argmax, deterministic (PORTING §5). */
    for (int cb = 1; cb < FA_NCB; cb++) {
        S2P_TRY(fa_run_layers(f, B, cb, stream));
        S2P_CUDA_TRY(s2pk_rms_norm(f->x, (const __nv_bfloat16*)f->norm.data,
                                   f->t, B, FA_DIM, S2P_NORM_EPS, stream));
        S2P_TRY(s2p_linear_forward(&f->output, f->t, f->logits, B, f->mode,
                                   stream));
        if (s2psl_dump_dir() != NULL && B == 1) {
            /* parity: per-residual-step logits of the FIRST frame only
             * (fixture fast_ar_step0_logits [9,4096]) */
            static int dump_frame = 0;
            if (cb == 1) dump_frame++;
            if (dump_frame == 1) {
                char nm[48];
                snprintf(nm, sizeof(nm), "fast_ar_step0_cb%d", cb);
                s2psl_dump_vec_bf16(nm, f->logits, FA_CB, stream);
            }
        }
        int32_t* dst = f->stage + (size_t)(cb - 1) * B;
        S2P_CUDA_TRY(s2pk_argmax(f->logits, dst, B, FA_CB, stream));
        if (cb < FA_NCB - 1) {
            int blocks = (B + 63) / 64;
            k_widen_ids<<<blocks, 64, 0, stream>>>(dst, f->ids64, B);
            S2P_CUDA_TRY(cudaGetLastError());
            S2P_CUDA_TRY(s2pk_embed((const __nv_bfloat16*)f->emb.data,
                                    f->ids64, f->x, B, FA_DIM, stream));
        }
    }

    return S2P_OK;
}

s2p_status s2pfa_decode_frame_batch(s2pfa* f, const __nv_bfloat16* hidden,
                                    const int32_t* sem_ids, int B,
                                    int32_t* out_codes, cudaStream_t stream) {
    if (!f || !hidden || !sem_ids || !out_codes) return S2P_ERR_INVALID;
    if (B < 1 || B > f->max_b) return S2P_ERR_INVALID;

    memcpy(f->sem_host, sem_ids, (size_t)B * sizeof(int32_t));
    S2P_CUDA_TRY(cudaMemcpyAsync(f->sem_dev, f->sem_host,
                                 (size_t)B * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream));
    const int32_t* stage = NULL;
    S2P_TRY(s2pfa_decode_frame_batch_dev(f, hidden, f->sem_dev, B, &stage,
                                         stream));

    S2P_CUDA_TRY(cudaMemcpyAsync(f->stage_host, stage,
                                 (size_t)(FA_NCB - 1) * B * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream));
    S2P_CUDA_TRY(cudaStreamSynchronize(stream));

    for (int b = 0; b < B; b++)
        for (int cb = 1; cb < FA_NCB; cb++)
            out_codes[(size_t)b * FA_NCB + cb] =
                f->stage_host[(size_t)(cb - 1) * B + b];
    return S2P_OK;
}
