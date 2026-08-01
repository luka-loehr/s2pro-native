/* s2pro-native — slow-AR model load / free (model.h implementation, part 1).
 *
 * Checkpoint tensor names (verified against model.safetensors.index.json):
 *   text_model.model.embeddings.weight                     [155776, 2560]
 *   text_model.model.layers.{i}.attention.wqkv.weight      [6144, 2560]
 *   text_model.model.layers.{i}.attention.wo.weight        [2560, 4096]
 *   text_model.model.layers.{i}.attention.q_norm.weight    [128]
 *   text_model.model.layers.{i}.attention.k_norm.weight    [128]
 *   text_model.model.layers.{i}.attention_norm.weight      [2560]
 *   text_model.model.layers.{i}.ffn_norm.weight            [2560]
 *   text_model.model.layers.{i}.feed_forward.w1.weight     [9728, 2560]
 *   text_model.model.layers.{i}.feed_forward.w3.weight     [9728, 2560]
 *   text_model.model.layers.{i}.feed_forward.w2.weight     [2560, 9728]
 *   text_model.model.norm.weight                           [2560]
 *
 * w1/w3 are fused at load into one gate_up [19456, 2560] linear (w1 rows
 * first) so one GEMM feeds s2pk_silu_mul's [gate || up] layout. wqkv is
 * already fused in the checkpoint (concat order q,k,v). The embedding table
 * doubles as the tied lm_head via s2p_gemm_bf16.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slowar_internal.h"

#define S2P_GU_ROWS (2 * S2P_FFN_DIM) /* 19456 */

static s2p_status load_norm(s2p_st* st, const char* fmt, int layer, int width,
                            s2p_tensor* out, cudaStream_t stream) {
    char name[160];
    snprintf(name, sizeof(name), fmt, layer);
    int64_t shape[1] = {width};
    return s2p_st_load_device(st, name, S2P_DT_BF16, 1, shape, out, stream);
}

static s2p_status load_linear(s2p_st* st, const char* fmt, int layer, int in_f,
                              int out_f, s2p_gemm_mode mode, s2p_linear* lin,
                              cudaStream_t stream) {
    char name[160];
    snprintf(name, sizeof(name), fmt, layer);
    S2P_TRY(s2p_linear_from_st(lin, st, name, in_f, out_f, stream));
    if (mode == S2P_GEMM_FP8) S2P_TRY(s2p_linear_prepare_fp8(lin, stream));
    if (mode == S2P_GEMM_INT8) S2P_TRY(s2p_linear_prepare_int8(lin, stream));
    return S2P_OK;
}

/* Fuse feed_forward.w1 + w3 into a single [19456, 2560] linear. */
static s2p_status load_gate_up(s2p_st* st, int layer, s2p_gemm_mode mode,
                               s2p_linear* lin, cudaStream_t stream) {
    char name[160];
    s2p_st_view w1, w3;
    snprintf(name, sizeof(name),
             "text_model.model.layers.%d.feed_forward.w1.weight", layer);
    S2P_TRY(s2p_st_find(st, name, &w1));
    snprintf(name, sizeof(name),
             "text_model.model.layers.%d.feed_forward.w3.weight", layer);
    S2P_TRY(s2p_st_find(st, name, &w3));
    if (w1.dtype != S2P_DT_BF16 || w3.dtype != S2P_DT_BF16 || w1.ndim != 2 ||
        w3.ndim != 2 || w1.shape[0] != S2P_FFN_DIM || w1.shape[1] != S2P_DIM ||
        w3.shape[0] != S2P_FFN_DIM || w3.shape[1] != S2P_DIM)
        return S2P_ERR_FORMAT;

    memset(lin, 0, sizeof(*lin));
    lin->in_features = S2P_DIM;
    lin->out_features = S2P_GU_ROWS;
    int64_t shape[2] = {S2P_GU_ROWS, S2P_DIM};
    S2P_TRY(s2p_tensor_device_alloc(&lin->w_bf16, S2P_DT_BF16, 2, shape));
    const size_t half = (size_t)S2P_FFN_DIM * S2P_DIM * sizeof(uint16_t);
    S2P_CUDA_TRY(cudaMemcpyAsync(lin->w_bf16.data, w1.data, half,
                                 cudaMemcpyHostToDevice, stream));
    S2P_CUDA_TRY(cudaMemcpyAsync((char*)lin->w_bf16.data + half, w3.data, half,
                                 cudaMemcpyHostToDevice, stream));
    if (mode == S2P_GEMM_FP8) S2P_TRY(s2p_linear_prepare_fp8(lin, stream));
    if (mode == S2P_GEMM_INT8) S2P_TRY(s2p_linear_prepare_int8(lin, stream));
    return S2P_OK;
}

static s2p_status alloc_2d(s2p_tensor* t, int64_t r, int64_t c, s2p_dtype dt) {
    int64_t shape[2] = {r, c};
    return s2p_tensor_device_alloc(t, dt, 2, shape);
}

static s2p_status alloc_scratch(s2p_model* m) {
    const int64_t ctx = m->ctx_len;
    const int64_t ms = m->max_sessions;
    S2P_TRY(alloc_2d(&m->sx, ctx, S2P_DIM, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->snorm, ctx, S2P_DIM, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sqkv, ctx, S2P_QKV_WIDTH, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sq, ctx, S2P_Q_WIDTH, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sk, ctx, S2P_KV_WIDTH, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sv, ctx, S2P_KV_WIDTH, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sattn, ctx, S2P_Q_WIDTH, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sproj, ctx, S2P_DIM, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sgu, ctx, S2P_GU_ROWS, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->sffn, ctx, S2P_FFN_DIM, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->shidden, ms, S2P_DIM, S2P_DT_BF16));
    S2P_TRY(alloc_2d(&m->slogits, ms, S2P_TEXT_VOCAB, S2P_DT_BF16));
    {
        int64_t s1[1] = {ctx};
        S2P_TRY(s2p_tensor_device_alloc(&m->sids, S2P_DT_I64, 1, s1));
    }
    {
        int64_t s1[1] = {(int64_t)S2P_NUM_CODEBOOKS * ctx};
        S2P_TRY(s2p_tensor_device_alloc(&m->svq, S2P_DT_I32, 1, s1));
    }
    /* per-frame upload block (natural alignment by descending size) */
    const size_t B = (size_t)ms;
    s2p_upload_layout* up = &m->up;
    up->off_ptrs = 0;
    up->off_ids = up->off_ptrs + (size_t)S2P_SLOW_LAYERS * 2 * B * sizeof(void*);
    up->off_codes = up->off_ids + B * sizeof(int64_t);
    up->off_pos =
        up->off_codes + (size_t)S2P_NUM_CODEBOOKS * B * sizeof(int32_t);
    up->off_mask = up->off_pos + B * sizeof(int32_t);
    up->total = (up->off_mask + B + 7) & ~(size_t)7;
    S2P_CUDA_TRY(cudaMalloc(&m->d_up, up->total));
    S2P_CUDA_TRY(cudaMallocHost(&m->h_up, up->total));
    S2P_CUDA_TRY(cudaMallocHost((void**)&m->h_sem,
                                B * S2PS_N_CAND * sizeof(uint16_t)));
    S2P_CUDA_TRY(cudaMallocHost((void**)&m->h_frame,
                                B * S2P_NUM_CODEBOOKS * sizeof(int32_t)));
    S2P_CUDA_TRY(cudaMallocHost((void**)&m->h_semid, B * sizeof(int32_t)));
    return S2P_OK;
}

s2p_status s2p_model_load(const char* model_dir, const s2p_model_opts* opts,
                          s2p_model** out) {
    if (model_dir == NULL || out == NULL) return S2P_ERR_INVALID;
    *out = NULL;

    s2p_model* m = calloc(1, sizeof(*m));
    if (m == NULL) return S2P_ERR_OOM;
    m->mode = opts ? opts->gemm_mode : S2P_GEMM_BF16;
    m->ctx_len = opts && opts->ctx_len > 0 ? opts->ctx_len : S2P_CTX_LEN_DEFAULT;
    m->max_sessions =
        opts && opts->max_sessions > 0 ? opts->max_sessions : S2P_MAX_SESSIONS;
    if (m->max_sessions > S2P_SLOWAR_MAX_BATCH)
        m->max_sessions = S2P_SLOWAR_MAX_BATCH;
    if (m->ctx_len > S2P_MAX_SEQ) {
        free(m);
        return S2P_ERR_INVALID;
    }
    if (m->mode == S2P_GEMM_FP8 && !s2p_fso_available()) {
        fprintf(stderr,
                "[s2pro] FP8 gemm requested but fish-scales-ops unavailable; "
                "falling back to BF16\n");
        m->mode = S2P_GEMM_BF16;
    }

    s2p_status rc = S2P_OK;
    s2p_st* st = NULL;
    /* cuBLAS/FP8 context must cover prefill M = ctx_len. If the host app
     * already initialized it with a smaller max_m this re-init is expected to
     * be idempotent-or-grow (core contract). */
    rc = s2p_gemm_init(m->ctx_len);
    if (rc != S2P_OK) goto fail;

    if (cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking) !=
        cudaSuccess) {
        rc = S2P_ERR_CUDA;
        goto fail;
    }

    rc = s2p_st_open_dir(model_dir, &st);
    if (rc != S2P_OK) goto fail;

    {
        int64_t eshape[2] = {S2P_TEXT_VOCAB, S2P_DIM};
        rc = s2p_st_load_device(st, "text_model.model.embeddings.weight",
                                S2P_DT_BF16, 2, eshape, &m->embed, m->stream);
        if (rc != S2P_OK) goto fail;
    }
    if (m->mode == S2P_GEMM_INT8) {
        /* tied-head sidecar; the bf16 table stays for embedding lookups */
        int64_t qshape[2] = {S2P_TEXT_VOCAB, S2P_DIM};
        int64_t sshape[1] = {S2P_TEXT_VOCAB};
        rc = s2p_tensor_device_alloc(&m->embed_i8, S2P_DT_I8, 2, qshape);
        if (rc == S2P_OK)
            rc = s2p_tensor_device_alloc(&m->embed_scale, S2P_DT_F32, 1,
                                         sshape);
        if (rc == S2P_OK)
            rc = s2p_int8_quant(m->embed_i8.data,
                                (float*)m->embed_scale.data, m->embed.data,
                                S2P_TEXT_VOCAB, S2P_DIM, m->stream);
        if (rc != S2P_OK) goto fail;
    }
    for (int l = 0; l < S2P_SLOW_LAYERS; l++) {
        s2p_slow_layer* ly = &m->layers[l];
        rc = load_linear(st, "text_model.model.layers.%d.attention.wqkv.weight",
                         l, S2P_DIM, S2P_QKV_WIDTH, m->mode, &ly->wqkv,
                         m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_linear(st, "text_model.model.layers.%d.attention.wo.weight",
                         l, S2P_Q_WIDTH, S2P_DIM, m->mode, &ly->wo, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_gate_up(st, l, m->mode, &ly->gate_up, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_linear(st,
                         "text_model.model.layers.%d.feed_forward.w2.weight",
                         l, S2P_FFN_DIM, S2P_DIM, m->mode, &ly->w2, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_norm(st, "text_model.model.layers.%d.attention_norm.weight",
                       l, S2P_DIM, &ly->attn_norm, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_norm(st, "text_model.model.layers.%d.ffn_norm.weight", l,
                       S2P_DIM, &ly->ffn_norm, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_norm(st, "text_model.model.layers.%d.attention.q_norm.weight",
                       l, S2P_HEAD_DIM, &ly->q_norm, m->stream);
        if (rc != S2P_OK) goto fail;
        rc = load_norm(st, "text_model.model.layers.%d.attention.k_norm.weight",
                       l, S2P_HEAD_DIM, &ly->k_norm, m->stream);
        if (rc != S2P_OK) goto fail;
    }
    {
        int64_t nshape[1] = {S2P_DIM};
        rc = s2p_st_load_device(st, "text_model.model.norm.weight",
                                S2P_DT_BF16, 1, nshape, &m->final_norm,
                                m->stream);
        if (rc != S2P_OK) goto fail;
    }

    rc = s2pfa_load(&m->fastar, st, m->mode, m->stream);
    if (rc != S2P_OK) goto fail;

    rc = alloc_scratch(m);
    if (rc != S2P_OK) goto fail;

    /* all uploads must land before the mmap goes away */
    if (cudaStreamSynchronize(m->stream) != cudaSuccess) {
        rc = S2P_ERR_CUDA;
        goto fail;
    }
    s2p_st_close(st);
    *out = m;
    return S2P_OK;

fail:
    if (st != NULL) {
        cudaStreamSynchronize(m->stream);
        s2p_st_close(st);
    }
    s2p_model_free(m);
    return rc;
}

void s2p_model_free(s2p_model* m) {
    if (m == NULL) return;
    if (m->stream != NULL) cudaStreamSynchronize(m->stream);
    if (m->fastar != NULL) s2pfa_free(m->fastar);
    s2p_tensor_free(&m->embed);
    s2p_tensor_free(&m->embed_i8);
    s2p_tensor_free(&m->embed_scale);
    for (int l = 0; l < S2P_SLOW_LAYERS; l++) {
        s2p_slow_layer* ly = &m->layers[l];
        s2p_linear_free(&ly->wqkv);
        s2p_linear_free(&ly->wo);
        s2p_linear_free(&ly->gate_up);
        s2p_linear_free(&ly->w2);
        s2p_tensor_free(&ly->attn_norm);
        s2p_tensor_free(&ly->ffn_norm);
        s2p_tensor_free(&ly->q_norm);
        s2p_tensor_free(&ly->k_norm);
    }
    s2p_tensor_free(&m->final_norm);
    s2p_tensor_free(&m->sx);
    s2p_tensor_free(&m->snorm);
    s2p_tensor_free(&m->sqkv);
    s2p_tensor_free(&m->sq);
    s2p_tensor_free(&m->sk);
    s2p_tensor_free(&m->sv);
    s2p_tensor_free(&m->sattn);
    s2p_tensor_free(&m->sproj);
    s2p_tensor_free(&m->sgu);
    s2p_tensor_free(&m->sffn);
    s2p_tensor_free(&m->shidden);
    s2p_tensor_free(&m->slogits);
    s2p_tensor_free(&m->sids);
    s2p_tensor_free(&m->svq);
    if (m->d_up != NULL) cudaFree(m->d_up);
    if (m->h_up != NULL) cudaFreeHost(m->h_up);
    if (m->h_sem != NULL) cudaFreeHost(m->h_sem);
    if (m->h_frame != NULL) cudaFreeHost(m->h_frame);
    if (m->h_semid != NULL) cudaFreeHost(m->h_semid);
    if (m->stream != NULL) cudaStreamDestroy(m->stream);
    /* s2p_gemm_shutdown is process-scoped and may be shared with the serve
     * layer; the owner of process teardown calls it. */
    free(m);
}
