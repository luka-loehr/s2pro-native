/* s2pro-native — slow-AR sessions: prefill + lockstep batched frame decode
 * (model.h implementation, part 2).
 *
 * Frame pipeline (docs/PORTING.md §4-6):
 *   prefill: embed ids -> VQ-inject refs (all 10 codebooks summed onto the
 *   masked semantic positions, then / sqrt(11)) -> 36 layers M=L -> KV
 *   populated -> final-normed LAST hidden stashed (frame 0 samples from it).
 *
 *   frame:   sessions needing a decode step embed their fed-back semantic
 *   token, add the PREVIOUS frame's 10-codebook VQ sum, * 1/sqrt(11), run 36
 *   layers at M=B against per-session KV, final-norm; freshly prefilled
 *   sessions contribute their stashed hidden. One tied lm-head GEMM ->
 *   logits [B,155776]; host samples the semantic token (sampling.c); the
 *   fast-AR argmax-fills the 9 residuals from the SAME final-normed hidden.
 *   out_codes row = [sem_id, resid1..resid9]. The EOS frame emits no codes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "slowar_internal.h"

#define BF(t) ((__nv_bfloat16*)(t).data)

static const double S2P_SQRT11 = 3.3166247903554; /* sqrt(10 + 1) */

static uint64_t now_entropy(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int is_semantic(int64_t tok) {
    return tok >= S2P_TOK_SEMANTIC_START && tok <= S2P_TOK_SEMANTIC_END;
}

/* ---------------------------------------------------------------- sessions */

s2p_status s2p_session_create(s2p_model* m, const s2p_sampling_cfg* cfg,
                              s2p_session** out) {
    if (m == NULL || out == NULL) return S2P_ERR_INVALID;
    *out = NULL;
    if (m->n_sessions >= m->max_sessions) return S2P_ERR_FULL;

    s2p_session* s = calloc(1, sizeof(*s));
    if (s == NULL) return S2P_ERR_OOM;
    s->m = m;
    s->state = S2P_SESS_NEW;

    int64_t kv_shape[4] = {S2P_SLOW_LAYERS, 2 * S2P_SLOW_KV_HEADS, m->ctx_len,
                           S2P_HEAD_DIM};
    s2p_status rc = s2p_tensor_device_alloc(
        &s->kv, m->kv8 ? S2P_DT_I8 : S2P_DT_BF16, 4, kv_shape);
    if (rc != S2P_OK) {
        free(s);
        return rc;
    }
    if (m->kv8) {
        int64_t sc_shape[4] = {S2P_SLOW_LAYERS, 2 * S2P_SLOW_KV_HEADS,
                               m->ctx_len, S2P_HEAD_DIM / 32};
        rc = s2p_tensor_device_alloc(&s->kvs, S2P_DT_F16, 4, sc_shape);
        if (rc != S2P_OK) {
            s2p_tensor_free(&s->kv);
            free(s);
            return rc;
        }
    }
    int64_t h_shape[1] = {S2P_DIM};
    rc = s2p_tensor_device_alloc(&s->pending_hidden, S2P_DT_BF16, 1, h_shape);
    if (rc != S2P_OK) {
        s2p_tensor_free(&s->kv);
        free(s);
        return rc;
    }
    s2ps_sampler_init(&s->sampler, cfg, now_entropy() ^ (uint64_t)(uintptr_t)s);

    /* mirror the sampler into a device-resident state for s2pk_sample */
    {
        s2ps_dev_state hst;
        memset(&hst, 0, sizeof(hst));
        memcpy(hst.prev, s->sampler.prev, sizeof(hst.prev));
        hst.count = s->sampler.count;
        memcpy(hst.rng, s->sampler.rng, sizeof(hst.rng));
        hst.temperature = s->sampler.temperature;
        hst.top_p = s->sampler.top_p;
        hst.rep_penalty = s->sampler.rep_penalty;
        hst.window = s->sampler.window;
        hst.seeded = s->sampler.seeded;
        hst.seed31 = s->sampler.seed31;
        if (cudaMalloc(&s->dsamp, sizeof(hst)) != cudaSuccess ||
            cudaMemcpy(s->dsamp, &hst, sizeof(hst), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            if (s->dsamp) cudaFree(s->dsamp);
            s2p_tensor_free(&s->kv);
            s2p_tensor_free(&s->pending_hidden);
            free(s);
            return S2P_ERR_CUDA;
        }
    }
    m->n_sessions++;
    *out = s;
    return S2P_OK;
}

void s2p_session_destroy(s2p_session* s) {
    if (s == NULL) return;
    if (s->m != NULL) {
        cudaStreamSynchronize(s->m->stream);
        s->m->n_sessions--;
    }
    s2p_tensor_free(&s->kv);
    s2p_tensor_free(&s->kvs);
    s2p_tensor_free(&s->pending_hidden);
    if (s->dsamp) cudaFree(s->dsamp);
    free(s);
}

/* ------------------------------------------------------- layer application */

/* One transformer layer over `rows` rows of m->sx.
 * Single-session path (batch == 0): contract kernels, scalar pos0, this
 * session's caches. Batch path (batch != 0): per-row pos/caches from the
 * uploaded pointer table. */
typedef struct {
    const int32_t* pos_dev;              /* [rows] */
    void* const* kptr;                   /* [layers*rows] plane pointers
                                          * (bf16, or i8 when m->kv8) */
    void* const* vptr;
    void* const* ksptr;                  /* m->kv8: f16 scale planes */
    void* const* vsptr;
    int max_len;                         /* max over rows of pos+1 */
} s2p_batch_refs;

/* S2P_ATTN_LEGACY=1 keeps the single-pass decode attention (A/B against
 * the split-K flash-decode kernel). */
static int attn_legacy(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_ATTN_LEGACY");
        v = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return v;
}

/* S2P_NO_GRAPHS=1 disables CUDA-graph capture of the decode tick (A/B). */
static int graphs_disabled(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_NO_GRAPHS");
        v = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return v;
}

static s2p_status run_layer(s2p_model* m, int l, int rows, int pos0,
                            const s2p_session* single,
                            const s2p_batch_refs* batch) {
    const s2p_slow_layer* ly = &m->layers[l];
    cudaStream_t st = m->stream;

    S2P_CUDA_TRY(s2pk_rms_norm(BF(m->sx), BF(ly->attn_norm), BF(m->snorm),
                               rows, S2P_DIM, S2P_NORM_EPS, st));
    S2P_TRY(s2p_linear_forward(&ly->wqkv, m->snorm.data, m->sqkv.data, rows,
                               m->mode, st));
    S2P_CUDA_TRY(s2pk_qkv_split(BF(m->sqkv), BF(m->sq), BF(m->sk), BF(m->sv),
                                rows, S2P_Q_WIDTH, S2P_KV_WIDTH, st));
    /* per-head QK-RMSNorm BEFORE RoPE (backbone only; PORTING pitfall 8) */
    S2P_CUDA_TRY(s2pk_qk_norm(BF(m->sq), BF(ly->q_norm), rows,
                              S2P_SLOW_Q_HEADS, S2P_HEAD_DIM, S2P_NORM_EPS,
                              st));
    S2P_CUDA_TRY(s2pk_qk_norm(BF(m->sk), BF(ly->k_norm), rows,
                              S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, S2P_NORM_EPS,
                              st));
    if (batch != NULL) {
        void* const* kp = batch->kptr + (size_t)l * rows;
        void* const* vp = batch->vptr + (size_t)l * rows;
        S2P_CUDA_TRY(s2pk_rope_pos(BF(m->sq), BF(m->sk), rows,
                                   S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS,
                                   S2P_HEAD_DIM, batch->pos_dev, S2P_ROPE_BASE,
                                   st));
        if (m->kv8) {
            void* const* ksp = batch->ksptr + (size_t)l * rows;
            void* const* vsp = batch->vsptr + (size_t)l * rows;
            S2P_CUDA_TRY(s2pk_kv_append8_ptrs(
                BF(m->sk), BF(m->sv), (int8_t* const*)kp, (int8_t* const*)vp,
                ksp, vsp, rows, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                batch->pos_dev, m->ctx_len, st));
            /* S2P_ATTN_LEGACY has no INT8 variant: split-K serves both */
            S2P_CUDA_TRY(s2pk_attention8_decode(
                BF(m->sq), (const int8_t* const*)kp, (const int8_t* const*)vp,
                (const void* const*)ksp, (const void* const*)vsp,
                BF(m->sattn), rows, S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS,
                S2P_HEAD_DIM, batch->pos_dev, m->ctx_len, batch->max_len,
                (float*)m->sattn_part.data, st));
        } else if (!attn_legacy()) {
            S2P_CUDA_TRY(s2pk_kv_append_ptrs(
                BF(m->sk), BF(m->sv), (__nv_bfloat16* const*)kp,
                (__nv_bfloat16* const*)vp, rows, S2P_SLOW_KV_HEADS,
                S2P_HEAD_DIM, batch->pos_dev, m->ctx_len, st));
            S2P_CUDA_TRY(s2pk_attention_decode(
                BF(m->sq), (const __nv_bfloat16* const*)kp,
                (const __nv_bfloat16* const*)vp, BF(m->sattn), rows,
                S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                batch->pos_dev, m->ctx_len, batch->max_len,
                (float*)m->sattn_part.data, st));
        } else {
            S2P_CUDA_TRY(s2pk_kv_append_ptrs(
                BF(m->sk), BF(m->sv), (__nv_bfloat16* const*)kp,
                (__nv_bfloat16* const*)vp, rows, S2P_SLOW_KV_HEADS,
                S2P_HEAD_DIM, batch->pos_dev, m->ctx_len, st));
            S2P_CUDA_TRY(s2pk_attention_ptrs(
                BF(m->sq), (const __nv_bfloat16* const*)kp,
                (const __nv_bfloat16* const*)vp, BF(m->sattn), rows,
                S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                batch->pos_dev, m->ctx_len, st));
        }
    } else {
        S2P_CUDA_TRY(s2pk_rope(BF(m->sq), BF(m->sk), rows, S2P_SLOW_Q_HEADS,
                               S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, pos0,
                               S2P_ROPE_BASE, st));
        if (m->kv8) {
            int8_t* kc = s2p_kv8_k(single, l);
            int8_t* vc = s2p_kv8_v(single, l);
            void* ks = s2p_kv8_ks(single, l);
            void* vs = s2p_kv8_vs(single, l);
            S2P_CUDA_TRY(s2pk_kv_append8(BF(m->sk), BF(m->sv), kc, vc, ks, vs,
                                         rows, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                                         pos0, m->ctx_len, st));
            S2P_CUDA_TRY(s2pk_attention8(BF(m->sq), kc, vc, ks, vs,
                                         BF(m->sattn), rows, S2P_SLOW_Q_HEADS,
                                         S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, pos0,
                                         m->ctx_len, st));
        } else {
            __nv_bfloat16* kc = s2p_kv_k(single, l);
            __nv_bfloat16* vc = s2p_kv_v(single, l);
            S2P_CUDA_TRY(s2pk_kv_append(BF(m->sk), BF(m->sv), kc, vc, rows,
                                        S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, pos0,
                                        m->ctx_len, st));
            S2P_CUDA_TRY(s2pk_attention(BF(m->sq), kc, vc, BF(m->sattn), rows,
                                        S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS,
                                        S2P_HEAD_DIM, pos0, m->ctx_len, st));
        }
    }
    S2P_TRY(s2p_linear_forward(&ly->wo, m->sattn.data, m->sproj.data, rows,
                               m->mode, st));
    S2P_CUDA_TRY(
        s2pk_add(BF(m->sx), BF(m->sproj), (int64_t)rows * S2P_DIM, st));

    S2P_CUDA_TRY(s2pk_rms_norm(BF(m->sx), BF(ly->ffn_norm), BF(m->snorm), rows,
                               S2P_DIM, S2P_NORM_EPS, st));
    S2P_TRY(s2p_linear_forward(&ly->gate_up, m->snorm.data, m->sgu.data, rows,
                               m->mode, st));
    S2P_CUDA_TRY(s2pk_silu_mul(BF(m->sgu), BF(m->sffn), rows, S2P_FFN_DIM, st));
    S2P_TRY(s2p_linear_forward(&ly->w2, m->sffn.data, m->sproj.data, rows,
                               m->mode, st));
    S2P_CUDA_TRY(
        s2pk_add(BF(m->sx), BF(m->sproj), (int64_t)rows * S2P_DIM, st));
    return S2P_OK;
}

/* ----------------------------------------------------------------- prefill */

s2p_status s2p_session_prefill(s2p_session* s, const int64_t* ids,
                               const uint8_t* vq_mask, int n_ids,
                               const s2p_vq_part* parts, int n_parts) {
    if (s == NULL || ids == NULL || n_ids <= 0) return S2P_ERR_INVALID;
    if (n_parts > 0 && (parts == NULL || vq_mask == NULL))
        return S2P_ERR_INVALID;
    s2p_model* m = s->m;
    if (s->state != S2P_SESS_NEW) return S2P_ERR_STATE;
    /* pos0 > 0 when s2p_session_kv_load seeded a cached prompt prefix; the
     * remaining ids then prefill at that offset. */
    const int pos0 = s->kv_len;
    /* reference clamps max_new_tokens to ctx-1-prompt: need >= 1 decode slot */
    if (pos0 + n_ids > m->ctx_len - 1) return S2P_ERR_INVALID;

    for (int i = 0; i < n_ids; i++)
        if (ids[i] < 0 || ids[i] >= S2P_TEXT_VOCAB) return S2P_ERR_INVALID;

    int64_t total_mask = 0;
    if (vq_mask != NULL)
        for (int i = 0; i < n_ids; i++) total_mask += vq_mask[i] ? 1 : 0;
    int64_t total_t = 0;
    for (int p = 0; p < n_parts; p++) {
        if (parts[p].codes == NULL || parts[p].T <= 0) return S2P_ERR_INVALID;
        for (int64_t j = 0; j < (int64_t)S2P_NUM_CODEBOOKS * parts[p].T; j++)
            if (parts[p].codes[j] < 0 || parts[p].codes[j] >= S2P_CB_SIZE)
                return S2P_ERR_INVALID;
        total_t += parts[p].T;
    }
    if (total_mask != total_t) return S2P_ERR_INVALID;

    cudaStream_t st = m->stream;

    /* uploads (synchronous: host buffers are caller-owned pageable memory) */
    S2P_CUDA_TRY(cudaMemcpy(m->sids.data, ids, (size_t)n_ids * sizeof(int64_t),
                            cudaMemcpyHostToDevice));
    /* map parts to contiguous masked runs; stage codes at svq offsets */
    int part_start[64];
    size_t part_off[64];
    if (n_parts > 64) return S2P_ERR_INVALID;
    {
        int cur = 0;
        size_t off = 0;
        for (int p = 0; p < n_parts; p++) {
            while (cur < n_ids && !vq_mask[cur]) cur++;
            const int T = parts[p].T;
            if (cur + T > n_ids) return S2P_ERR_INVALID;
            for (int t = 0; t < T; t++)
                if (!vq_mask[cur + t]) return S2P_ERR_INVALID; /* gap in run */
            part_start[p] = cur;
            part_off[p] = off;
            S2P_CUDA_TRY(cudaMemcpy(
                (int32_t*)m->svq.data + off, parts[p].codes,
                (size_t)S2P_NUM_CODEBOOKS * T * sizeof(int32_t),
                cudaMemcpyHostToDevice));
            off += (size_t)S2P_NUM_CODEBOOKS * T;
            cur += T;
        }
    }

    /* input embeddings + interleaved VQ injection (PORTING pitfall 7) */
    if (m->embed_i8_only)
        S2P_CUDA_TRY(s2pk_embed_i8((const int8_t*)m->embed_i8.data,
                                   (const float*)m->embed_scale.data,
                                   (const int64_t*)m->sids.data, BF(m->sx),
                                   n_ids, S2P_DIM, st));
    else
        S2P_CUDA_TRY(s2pk_embed(BF(m->embed), (const int64_t*)m->sids.data,
                                BF(m->sx), n_ids, S2P_DIM, st));
    for (int p = 0; p < n_parts; p++) {
        const int T = parts[p].T;
        __nv_bfloat16* rows0 = BF(m->sx) + (size_t)part_start[p] * S2P_DIM;
        S2P_TRY(s2pfa_vq_embed_add(m->fastar,
                                   (const int32_t*)m->svq.data + part_off[p],
                                   T, rows0, st));
        /* embed_text_dim DIVIDES by sqrt(11) (decode path multiplies by the
         * reciprocal — the bf16 roundings differ, replicate each). */
        S2P_CUDA_TRY(s2pk_scale_div(rows0, (int64_t)T * S2P_DIM,
                                    (float)S2P_SQRT11, st));
    }

    for (int l = 0; l < S2P_SLOW_LAYERS; l++) {
        S2P_TRY(run_layer(m, l, n_ids, pos0, s, NULL));
        if (s2psl_dump_dir() != NULL) { /* parity: residual after layer l,
                                         * last prompt position */
            char nm[32];
            snprintf(nm, sizeof(nm), "backbone_layer%02d", l);
            s2psl_dump_vec_bf16(nm, BF(m->sx) + (size_t)(n_ids - 1) * S2P_DIM,
                                S2P_DIM, st);
        }
    }

    /* final-normed LAST hidden -> frame-0 sampling state (pitfall 5: the
     * fast-AR must be primed on THIS, not the pre-norm residual). */
    S2P_CUDA_TRY(s2pk_rms_norm(BF(m->sx) + (size_t)(n_ids - 1) * S2P_DIM,
                               BF(m->final_norm), BF(s->pending_hidden), 1,
                               S2P_DIM, S2P_NORM_EPS, st));
    s2psl_dump_vec_bf16("backbone_final_norm", s->pending_hidden.data, S2P_DIM,
                        st);
    S2P_CUDA_TRY(cudaStreamSynchronize(st));

    s->kv_len = pos0 + n_ids;
    s->state = S2P_SESS_PREFILLED;
    return S2P_OK;
}

/* -------------------------------------------------- KV prefix cache ----- */

/* The session KV is [36][2*KVH][ctx][128] bf16 — 36*16 contiguous planes of
 * [ctx][128]. A prefix blob keeps the same plane order at n_tokens depth,
 * so save/load are single strided 2D copies. */
#define S2P_KV_PLANES (S2P_SLOW_LAYERS * 2 * S2P_SLOW_KV_HEADS)

/* Payload elem size: bf16 (2 B) or i8 (1 B under m->kv8); INT8 blobs carry
 * the payload planes first, then the f16 scale planes (4 per position). */
static size_t kv_elem_bytes(const s2p_model* m) {
    return m->kv8 ? sizeof(int8_t) : sizeof(uint16_t);
}
static size_t kv_scale_width(const s2p_model* m, int n_tokens) {
    return m->kv8 ? (size_t)n_tokens * (S2P_HEAD_DIM / 32) * 2 : 0;
}

size_t s2p_session_kv_bytes(const s2p_model* m, int n_tokens) {
    if (m == NULL || n_tokens <= 0) return 0;
    return (size_t)S2P_KV_PLANES * n_tokens * S2P_HEAD_DIM * kv_elem_bytes(m) +
           (size_t)S2P_KV_PLANES * kv_scale_width(m, n_tokens);
}

s2p_status s2p_session_kv_save(s2p_session* s, int n_tokens, void* blob_dev) {
    if (s == NULL || blob_dev == NULL || n_tokens <= 0) return S2P_ERR_INVALID;
    if (n_tokens > s->kv_len) return S2P_ERR_INVALID;
    s2p_model* m = s->m;
    const size_t width = (size_t)n_tokens * S2P_HEAD_DIM * kv_elem_bytes(m);
    const size_t pitch = (size_t)m->ctx_len * S2P_HEAD_DIM * kv_elem_bytes(m);
    S2P_CUDA_TRY(cudaMemcpy2DAsync(blob_dev, width, s->kv.data, pitch, width,
                                   S2P_KV_PLANES, cudaMemcpyDeviceToDevice,
                                   m->stream));
    if (m->kv8) {
        char* sblob = (char*)blob_dev + (size_t)S2P_KV_PLANES * width;
        const size_t swidth = kv_scale_width(m, n_tokens);
        const size_t spitch = kv_scale_width(m, m->ctx_len);
        S2P_CUDA_TRY(cudaMemcpy2DAsync(sblob, swidth, s->kvs.data, spitch,
                                       swidth, S2P_KV_PLANES,
                                       cudaMemcpyDeviceToDevice, m->stream));
    }
    S2P_CUDA_TRY(cudaStreamSynchronize(m->stream));
    return S2P_OK;
}

s2p_status s2p_session_kv_load(s2p_session* s, const void* blob_dev,
                               int n_tokens) {
    if (s == NULL || blob_dev == NULL || n_tokens <= 0) return S2P_ERR_INVALID;
    s2p_model* m = s->m;
    if (s->state != S2P_SESS_NEW || s->kv_len != 0) return S2P_ERR_STATE;
    if (n_tokens > m->ctx_len - 1) return S2P_ERR_INVALID;
    const size_t width = (size_t)n_tokens * S2P_HEAD_DIM * kv_elem_bytes(m);
    const size_t pitch = (size_t)m->ctx_len * S2P_HEAD_DIM * kv_elem_bytes(m);
    S2P_CUDA_TRY(cudaMemcpy2DAsync(s->kv.data, pitch, blob_dev, width, width,
                                   S2P_KV_PLANES, cudaMemcpyDeviceToDevice,
                                   m->stream));
    if (m->kv8) {
        const char* sblob =
            (const char*)blob_dev + (size_t)S2P_KV_PLANES * width;
        const size_t swidth = kv_scale_width(m, n_tokens);
        const size_t spitch = kv_scale_width(m, m->ctx_len);
        S2P_CUDA_TRY(cudaMemcpy2DAsync(s->kvs.data, spitch, sblob, swidth,
                                       swidth, S2P_KV_PLANES,
                                       cudaMemcpyDeviceToDevice, m->stream));
    }
    S2P_CUDA_TRY(cudaStreamSynchronize(m->stream));
    s->kv_len = n_tokens;
    return S2P_OK;
}

/* ------------------------------------------------------ lockstep decode */

typedef struct {
    int a, t;
    size_t off;
} vq_run;

/* Enqueue one whole decode tick (upload -> backbone -> head -> sample ->
 * fast-AR -> result download) on m->stream. Pure device work: no syncs, no
 * host reads — the sequence is CUDA-graph capturable when the batch shape
 * is steady. `act` is only touched for the eager stashed-hidden copies
 * (bd < nact, never under capture). */
static s2p_status decode_tick_enqueue(s2p_model* m, s2p_session* const* act,
                                      int bd, int nact, int nruns,
                                      const vq_run* runs, int max_len) {
    cudaStream_t st = m->stream;
    const size_t B = (size_t)m->max_sessions;
    char* d = (char*)m->d_up;

    if (bd > 0) {
        S2P_CUDA_TRY(cudaMemcpyAsync(m->d_up, m->h_up, m->up.total,
                                     cudaMemcpyHostToDevice, st));
        s2p_batch_refs refs;
        refs.pos_dev = (const int32_t*)(d + m->up.off_pos);
        refs.kptr = (void* const*)(d + m->up.off_ptrs);
        refs.vptr = refs.kptr + (size_t)S2P_SLOW_LAYERS * B;
        refs.ksptr = refs.kptr + (size_t)2 * S2P_SLOW_LAYERS * B;
        refs.vsptr = refs.kptr + (size_t)3 * S2P_SLOW_LAYERS * B;
        refs.max_len = max_len;

        /* embed fed-back token + previous-frame VQ sum, * 1/sqrt(11) */
        if (m->embed_i8_only)
            S2P_CUDA_TRY(s2pk_embed_i8((const int8_t*)m->embed_i8.data,
                                       (const float*)m->embed_scale.data,
                                       (const int64_t*)(d + m->up.off_ids),
                                       BF(m->sx), bd, S2P_DIM, st));
        else
            S2P_CUDA_TRY(s2pk_embed(BF(m->embed),
                                    (const int64_t*)(d + m->up.off_ids),
                                    BF(m->sx), bd, S2P_DIM, st));
        for (int r = 0; r < nruns; r++)
            S2P_TRY(s2pfa_vq_embed_add_dev(
                m->fastar,
                (const int32_t*)(d + m->up.off_codes) + runs[r].off,
                runs[r].t, BF(m->sx) + (size_t)runs[r].a * S2P_DIM, st));
        S2P_CUDA_TRY(s2pk_scale_rows_masked(
            BF(m->sx), (const uint8_t*)(d + m->up.off_mask), bd, S2P_DIM,
            (float)(1.0 / S2P_SQRT11), st));

        for (int l = 0; l < S2P_SLOW_LAYERS; l++)
            S2P_TRY(run_layer(m, l, bd, 0, NULL, &refs));

        S2P_CUDA_TRY(s2pk_rms_norm(BF(m->sx), BF(m->final_norm),
                                   BF(m->shidden), bd, S2P_DIM, S2P_NORM_EPS,
                                   st));
    }

    /* stashed prefill hiddens complete the batch (eager path only) */
    for (int b = bd; b < nact; b++)
        S2P_CUDA_TRY(cudaMemcpyAsync(BF(m->shidden) + (size_t)b * S2P_DIM,
                                     act[b]->pending_hidden.data,
                                     (size_t)S2P_DIM * sizeof(uint16_t),
                                     cudaMemcpyDeviceToDevice, st));

    /* tied lm-head: logits[b] = hidden[b] @ embed^T. INT8 mode reads the
     * per-row-quantized sidecar; batches beyond the GEMV width use the kept
     * bf16 table. */
    if (m->mode == S2P_GEMM_INT8 && m->embed_i8.data != NULL &&
        nact <= S2P_INT8_GEMV_MAX_M)
        S2P_TRY(s2p_int8_gemv(m->slogits.data, m->shidden.data,
                              m->embed_i8.data,
                              (const float*)m->embed_scale.data, nact,
                              S2P_TEXT_VOCAB, S2P_DIM, st));
    else if (m->embed_i8_only)
        /* bf16 table dropped: serve oversize batches from the sidecar in
         * GEMV-width chunks (M uniform per call) */
        for (int m0 = 0; m0 < nact; m0 += S2P_INT8_GEMV_MAX_M) {
            const int mm = nact - m0 > S2P_INT8_GEMV_MAX_M
                               ? S2P_INT8_GEMV_MAX_M
                               : nact - m0;
            S2P_TRY(s2p_int8_gemv(
                (char*)m->slogits.data +
                    (size_t)m0 * S2P_TEXT_VOCAB * sizeof(uint16_t),
                (const char*)m->shidden.data +
                    (size_t)m0 * S2P_DIM * sizeof(uint16_t),
                m->embed_i8.data, (const float*)m->embed_scale.data, mm,
                S2P_TEXT_VOCAB, S2P_DIM, st));
        }
    else
        S2P_TRY(s2p_gemm_bf16(m->shidden.data, m->embed.data, m->slogits.data,
                              nact, S2P_TEXT_VOCAB, S2P_DIM, st));
    if (s2psl_dump_dir() != NULL && nact == 1) {
        /* parity: full logits row of the first two sampling steps (fixture
         * names prefill_logits / step1_logits); single-session runs only.
         * Never active under capture (steady gating excludes dump runs). */
        static int dump_step = 0;
        if (dump_step < 2)
            s2psl_dump_vec_bf16(dump_step == 0 ? "prefill_logits"
                                               : "step1_logits",
                                m->slogits.data, S2P_TEXT_VOCAB, st);
        dump_step++;
    }

    /* device sampling + fast-AR + packed result download */
    int64_t* d_tok = (int64_t*)m->d_out;
    int32_t* d_sem = (int32_t*)((char*)m->d_out + m->out_off_sem);
    int32_t* d_codes = (int32_t*)((char*)m->d_out + m->out_off_codes);
    uint8_t* d_eos = (uint8_t*)((char*)m->d_out + m->out_off_eos);
    S2P_CUDA_TRY(cudaMemcpyAsync(m->d_sampptr, m->h_sampptr,
                                 (size_t)nact * sizeof(void*),
                                 cudaMemcpyHostToDevice, st));
    S2P_CUDA_TRY(s2pk_sample((const __nv_bfloat16*)m->slogits.data,
                             S2P_TEXT_VOCAB,
                             (s2ps_dev_state* const*)m->d_sampptr, nact,
                             d_tok, d_sem, d_codes, d_eos, st));
    const int32_t* stage = NULL;
    S2P_TRY(s2pfa_decode_frame_batch_dev(m->fastar,
                                         (const __nv_bfloat16*)m->shidden.data,
                                         d_sem, nact, &stage, st));
    S2P_CUDA_TRY(s2pk_pack_frame(stage, nact, d_codes, st));
    S2P_CUDA_TRY(cudaMemcpyAsync(m->h_out, m->d_out, m->out_bytes,
                                 cudaMemcpyDeviceToHost, st));
    return S2P_OK;
}


s2p_status s2p_model_batch_next_frame(s2p_model* m, s2p_session** sess, int n,
                                      int32_t* out_codes, int* eos) {
    if (m == NULL || sess == NULL || out_codes == NULL || eos == NULL ||
        n < 0 || n > m->max_sessions)
        return S2P_ERR_INVALID;
    for (int i = 0; i < n; i++) {
        if (sess[i] == NULL || sess[i]->m != m) return S2P_ERR_INVALID;
        if (sess[i]->state == S2P_SESS_NEW) return S2P_ERR_STATE;
    }

    cudaStream_t st = m->stream;
    memset(out_codes, 0, (size_t)n * S2P_NUM_CODEBOOKS * sizeof(int32_t));

    /* active rows: decode-step sessions first (their final norm lands
     * directly in shidden[0..Bd)), then freshly prefilled ones. */
    s2p_session* act[S2P_SLOWAR_MAX_BATCH];
    int act_ix[S2P_SLOWAR_MAX_BATCH];
    int nact = 0;
    for (int i = 0; i < n; i++) {
        s2p_session* s = sess[i];
        eos[i] = s->state == S2P_SESS_FINISHED;
        if (s->state == S2P_SESS_DECODING) {
            if (s->kv_len >= m->ctx_len) { /* context exhausted: forced stop */
                s->state = S2P_SESS_FINISHED;
                eos[i] = 1;
                continue;
            }
            act[nact] = s;
            act_ix[nact++] = i;
        }
    }
    const int bd = nact;
    for (int i = 0; i < n; i++)
        if (sess[i]->state == S2P_SESS_PREFILLED) {
            act[nact] = sess[i];
            act_ix[nact++] = i;
        }
    if (nact == 0) return S2P_OK;

    /* ---- host staging for decode rows [0, bd) ---- */
    vq_run runs[S2P_SLOWAR_MAX_BATCH];
    int nruns = 0;
    if (bd > 0) {
        const size_t B = (size_t)m->max_sessions;
        char* h = (char*)m->h_up;
        void** hptr = (void**)(h + m->up.off_ptrs);
        int64_t* hids = (int64_t*)(h + m->up.off_ids);
        int32_t* hcodes = (int32_t*)(h + m->up.off_codes);
        int32_t* hpos = (int32_t*)(h + m->up.off_pos);
        uint8_t* hmask = (uint8_t*)(h + m->up.off_mask);

        for (int b = 0; b < bd; b++) {
            s2p_session* s = act[b];
            hids[b] = s->prev_token;
            hpos[b] = s->kv_len;
            hmask[b] = (uint8_t)is_semantic(s->prev_token);
            for (int l = 0; l < S2P_SLOW_LAYERS; l++) {
                if (m->kv8) {
                    hptr[(size_t)l * bd + b] = s2p_kv8_k(s, l);
                    hptr[(size_t)S2P_SLOW_LAYERS * B + (size_t)l * bd + b] =
                        s2p_kv8_v(s, l);
                    hptr[(size_t)2 * S2P_SLOW_LAYERS * B + (size_t)l * bd +
                         b] = s2p_kv8_ks(s, l);
                    hptr[(size_t)3 * S2P_SLOW_LAYERS * B + (size_t)l * bd +
                         b] = s2p_kv8_vs(s, l);
                } else {
                    hptr[(size_t)l * bd + b] = s2p_kv_k(s, l);
                    hptr[(size_t)S2P_SLOW_LAYERS * B + (size_t)l * bd + b] =
                        s2p_kv_v(s, l);
                }
            }
        }
        /* previous-frame codes, cb-major per contiguous masked run so each
         * run block matches s2pfa_vq_embed_add's [10*T] layout */
        {
            size_t off = 0;
            int b = 0;
            while (b < bd) {
                if (!hmask[b]) {
                    b++;
                    continue;
                }
                int a = b;
                while (b < bd && hmask[b]) b++;
                const int T = b - a;
                for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
                    for (int t = 0; t < T; t++)
                        hcodes[off + (size_t)cb * T + t] =
                            act[a + t]->prev_codes[cb];
                runs[nruns].a = a;
                runs[nruns].t = T;
                runs[nruns].off = off;
                nruns++;
                off += (size_t)S2P_NUM_CODEBOOKS * T;
            }
        }
    }
    for (int b = 0; b < nact; b++) m->h_sampptr[b] = act[b]->dsamp;

    /* ---- enqueue the tick: captured CUDA graph in the steady state ----
     * Steady = every active row is decoding (bd == nact) with the usual
     * one-contiguous-VQ-run shape and no dump hooks: then the kernel
     * sequence is invariant and all per-frame variation lives in pinned
     * buffer contents, so one captured graph replays every frame. */
    int max_len = 0;
    if (bd > 0) {
        const int32_t* hpos = (const int32_t*)((char*)m->h_up + m->up.off_pos);
        for (int b = 0; b < bd; b++)
            if (hpos[b] + 1 > max_len) max_len = hpos[b] + 1;
    }
    const int steady = bd == nact && bd >= 1 && bd <= 4 &&
                       s2psl_dump_dir() == NULL && nruns == 1 &&
                       runs[0].a == 0 && runs[0].t == bd &&
                       !graphs_disabled();
    if (steady) {
        if (!m->gready[bd]) {
            S2P_CUDA_TRY(cudaStreamBeginCapture(
                st, cudaStreamCaptureModeThreadLocal));
            s2p_status rc = decode_tick_enqueue(m, act, bd, nact, nruns, runs,
                                                m->ctx_len);
            cudaGraph_t g = NULL;
            cudaError_t ce = cudaStreamEndCapture(st, &g);
            if (rc != S2P_OK) return rc;
            if (ce != cudaSuccess) {
                fprintf(stderr, "[s2pro] graph capture failed: %s\n",
                        cudaGetErrorString(ce));
                return S2P_ERR_CUDA;
            }
            ce = cudaGraphInstantiate(&m->gexec[bd], g, 0);
            cudaGraphDestroy(g);
            if (ce != cudaSuccess) {
                fprintf(stderr, "[s2pro] graph instantiate failed: %s\n",
                        cudaGetErrorString(ce));
                return S2P_ERR_CUDA;
            }
            m->gready[bd] = 1;
        }
        S2P_CUDA_TRY(cudaGraphLaunch(m->gexec[bd], st));
    } else {
        S2P_TRY(decode_tick_enqueue(m, act, bd, nact, nruns, runs, max_len));
    }
    S2P_CUDA_TRY(cudaStreamSynchronize(st));
    for (int b = 0; b < bd; b++) act[b]->kv_len++;

    const int64_t* toks = (const int64_t*)m->h_out;
    const int32_t* codes = (const int32_t*)((char*)m->h_out + m->out_off_codes);
    const uint8_t* eflag = (const uint8_t*)((char*)m->h_out + m->out_off_eos);
    const int32_t* sems = (const int32_t*)((char*)m->h_out + m->out_off_sem);

    /* S2P_DUMP_FRAMES=path: append per-frame fast-AR training records for
     * the INT4 QAT self-distillation — [2560 bf16 final-normed hidden]
     * [i32 sem][10 x i32 codes] per non-EOS frame (5164 B). The teacher
     * trajectory is recomputed in torch from (hidden, sem); codes ride
     * along for verification. Worker-thread only; sync above makes
     * shidden/h_out coherent. */
    {
        static FILE* df = NULL;
        static int checked = 0;
        if (!checked) {
            checked = 1;
            const char* p = getenv("S2P_DUMP_FRAMES");
            if (p && p[0]) df = fopen(p, "ab");
        }
        if (df) {
            static uint16_t hrow[S2P_DIM];
            for (int b = 0; b < nact; b++) {
                if (eflag[b]) continue;
                if (cudaMemcpy(hrow,
                               (const uint16_t*)m->shidden.data +
                                   (size_t)b * S2P_DIM,
                               sizeof(hrow),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    continue;
                fwrite(hrow, sizeof(hrow), 1, df);
                fwrite(&sems[b], sizeof(int32_t), 1, df);
                fwrite(codes + (size_t)b * S2P_NUM_CODEBOOKS,
                       sizeof(int32_t), S2P_NUM_CODEBOOKS, df);
            }
            fflush(df);
        }
    }

    /* harvest */
    for (int b = 0; b < nact; b++) {
        s2p_session* s = act[b];
        const int i = act_ix[b];
        if (eflag[b]) {
            s->state = S2P_SESS_FINISHED;
            eos[i] = 1; /* EOS frame produces no codes */
            continue;
        }
        const int32_t* frame = codes + (size_t)b * S2P_NUM_CODEBOOKS;
        memcpy(out_codes + (size_t)i * S2P_NUM_CODEBOOKS, frame,
               S2P_NUM_CODEBOOKS * sizeof(int32_t));
        s->prev_token = toks[b];
        memcpy(s->prev_codes, frame, sizeof(s->prev_codes));
        s->state = S2P_SESS_DECODING;
        eos[i] = 0;
    }
    return S2P_OK;
}

s2p_status s2p_session_next_frame(s2p_session* s,
                                  int32_t out_codes[S2P_NUM_CODEBOOKS],
                                  int* is_eos) {
    if (s == NULL || out_codes == NULL || is_eos == NULL)
        return S2P_ERR_INVALID;
    return s2p_model_batch_next_frame(s->m, &s, 1, out_codes, is_eos);
}
