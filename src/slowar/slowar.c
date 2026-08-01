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
    s2p_status rc = s2p_tensor_device_alloc(&s->kv, S2P_DT_BF16, 4, kv_shape);
    if (rc != S2P_OK) {
        free(s);
        return rc;
    }
    int64_t h_shape[1] = {S2P_DIM};
    rc = s2p_tensor_device_alloc(&s->pending_hidden, S2P_DT_BF16, 1, h_shape);
    if (rc != S2P_OK) {
        s2p_tensor_free(&s->kv);
        free(s);
        return rc;
    }
    s2ps_sampler_init(&s->sampler, cfg, now_entropy() ^ (uint64_t)(uintptr_t)s);
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
    s2p_tensor_free(&s->pending_hidden);
    free(s);
}

/* ------------------------------------------------------- layer application */

/* One transformer layer over `rows` rows of m->sx.
 * Single-session path (batch == 0): contract kernels, scalar pos0, this
 * session's caches. Batch path (batch != 0): per-row pos/caches from the
 * uploaded pointer table. */
typedef struct {
    const int32_t* pos_dev;              /* [rows] */
    __nv_bfloat16* const* kptr;          /* [layers*rows] */
    __nv_bfloat16* const* vptr;
} s2p_batch_refs;

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
        __nv_bfloat16* const* kp = batch->kptr + (size_t)l * rows;
        __nv_bfloat16* const* vp = batch->vptr + (size_t)l * rows;
        S2P_CUDA_TRY(s2pk_rope_pos(BF(m->sq), BF(m->sk), rows,
                                   S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS,
                                   S2P_HEAD_DIM, batch->pos_dev, S2P_ROPE_BASE,
                                   st));
        S2P_CUDA_TRY(s2pk_kv_append_ptrs(BF(m->sk), BF(m->sv), kp, vp, rows,
                                         S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                                         batch->pos_dev, m->ctx_len, st));
        S2P_CUDA_TRY(s2pk_attention_ptrs(
            BF(m->sq), (const __nv_bfloat16* const*)kp,
            (const __nv_bfloat16* const*)vp, BF(m->sattn), rows,
            S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, batch->pos_dev,
            m->ctx_len, st));
    } else {
        __nv_bfloat16* kc = s2p_kv_k(single, l);
        __nv_bfloat16* vc = s2p_kv_v(single, l);
        S2P_CUDA_TRY(s2pk_rope(BF(m->sq), BF(m->sk), rows, S2P_SLOW_Q_HEADS,
                               S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, pos0,
                               S2P_ROPE_BASE, st));
        S2P_CUDA_TRY(s2pk_kv_append(BF(m->sk), BF(m->sv), kc, vc, rows,
                                    S2P_SLOW_KV_HEADS, S2P_HEAD_DIM, pos0,
                                    m->ctx_len, st));
        S2P_CUDA_TRY(s2pk_attention(BF(m->sq), kc, vc, BF(m->sattn), rows,
                                    S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS,
                                    S2P_HEAD_DIM, pos0, m->ctx_len, st));
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
    /* reference clamps max_new_tokens to ctx-1-prompt: need >= 1 decode slot */
    if (n_ids > m->ctx_len - 1) return S2P_ERR_INVALID;

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

    for (int l = 0; l < S2P_SLOW_LAYERS; l++)
        S2P_TRY(run_layer(m, l, n_ids, 0, s, NULL));

    /* final-normed LAST hidden -> frame-0 sampling state (pitfall 5: the
     * fast-AR must be primed on THIS, not the pre-norm residual). */
    S2P_CUDA_TRY(s2pk_rms_norm(BF(m->sx) + (size_t)(n_ids - 1) * S2P_DIM,
                               BF(m->final_norm), BF(s->pending_hidden), 1,
                               S2P_DIM, S2P_NORM_EPS, st));
    S2P_CUDA_TRY(cudaStreamSynchronize(st));

    s->kv_len = n_ids;
    s->state = S2P_SESS_PREFILLED;
    return S2P_OK;
}

/* ------------------------------------------------------ lockstep decode */

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

    /* ---- decode forward for rows [0, bd) ---- */
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
                hptr[(size_t)l * bd + b] = s2p_kv_k(s, l);
                hptr[(size_t)S2P_SLOW_LAYERS * B + (size_t)l * bd + b] =
                    s2p_kv_v(s, l);
            }
        }
        /* previous-frame codes, cb-major per contiguous masked run so each
         * run block matches s2pfa_vq_embed_add's [10*T] layout */
        typedef struct {
            int a, t;
            size_t off;
        } vq_run;
        vq_run runs[S2P_SLOWAR_MAX_BATCH];
        int nruns = 0;
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
        S2P_CUDA_TRY(cudaMemcpyAsync(m->d_up, m->h_up, m->up.total,
                                     cudaMemcpyHostToDevice, st));
        char* d = (char*)m->d_up;
        s2p_batch_refs refs;
        refs.pos_dev = (const int32_t*)(d + m->up.off_pos);
        refs.kptr = (__nv_bfloat16* const*)(d + m->up.off_ptrs);
        refs.vptr = refs.kptr + (size_t)S2P_SLOW_LAYERS * B;

        /* embed fed-back token + previous-frame VQ sum, * 1/sqrt(11) */
        S2P_CUDA_TRY(s2pk_embed(BF(m->embed),
                                (const int64_t*)(d + m->up.off_ids), BF(m->sx),
                                bd, S2P_DIM, st));
        for (int r = 0; r < nruns; r++)
            S2P_TRY(s2pfa_vq_embed_add(
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
        for (int b = 0; b < bd; b++) act[b]->kv_len++;
    }

    /* stashed prefill hiddens complete the batch */
    for (int b = bd; b < nact; b++)
        S2P_CUDA_TRY(cudaMemcpyAsync(BF(m->shidden) + (size_t)b * S2P_DIM,
                                     act[b]->pending_hidden.data,
                                     (size_t)S2P_DIM * sizeof(uint16_t),
                                     cudaMemcpyDeviceToDevice, st));

    /* tied lm-head: logits[b] = hidden[b] @ embed^T (always BF16 cuBLAS) */
    S2P_TRY(s2p_gemm_bf16(m->shidden.data, m->embed.data, m->slogits.data,
                          nact, S2P_TEXT_VOCAB, S2P_DIM, st));

    /* download the 4097 finite-after-bias candidate logits per row */
    for (int b = 0; b < nact; b++) {
        const uint16_t* lrow =
            (const uint16_t*)m->slogits.data + (size_t)b * S2P_TEXT_VOCAB;
        S2P_CUDA_TRY(cudaMemcpyAsync(m->h_sem + (size_t)b * S2PS_N_CAND,
                                     lrow + S2P_TOK_SEMANTIC_START,
                                     (size_t)S2PS_N_SEM * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost, st));
        S2P_CUDA_TRY(cudaMemcpyAsync(
            m->h_sem + (size_t)b * S2PS_N_CAND + S2PS_CAND_EOS,
            lrow + S2P_TOK_EOS, sizeof(uint16_t), cudaMemcpyDeviceToHost, st));
    }
    S2P_CUDA_TRY(cudaStreamSynchronize(st));

    /* host sampling (exact two-softmax order) */
    int64_t toks[S2P_SLOWAR_MAX_BATCH];
    int eflag[S2P_SLOWAR_MAX_BATCH];
    for (int b = 0; b < nact; b++) {
        int32_t sem_id = 0;
        toks[b] = s2ps_sample(&act[b]->sampler,
                              m->h_sem + (size_t)b * S2PS_N_CAND,
                              m->h_sem[(size_t)b * S2PS_N_CAND + S2PS_CAND_EOS],
                              &sem_id, &eflag[b]);
        m->h_semid[b] = sem_id; /* 0 when EOS, per reference */
    }

    /* fast-AR residual cascade for the whole batch (EOS rows run with
     * sem_id 0 and are dropped afterwards, matching the reference). */
    S2P_CUDA_TRY(cudaMemcpyAsync(m->ssemid.data, m->h_semid,
                                 (size_t)nact * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, st));
    S2P_CUDA_TRY(s2pk_i32_scatter_stride((int32_t*)m->sframe.data,
                                         (const int32_t*)m->ssemid.data, nact,
                                         S2P_NUM_CODEBOOKS, st));
    S2P_TRY(s2pfa_decode_frame_batch(m->fastar,
                                     (const __nv_bfloat16*)m->shidden.data,
                                     (const int32_t*)m->ssemid.data, nact,
                                     (int32_t*)m->sframe.data, st));
    S2P_CUDA_TRY(cudaMemcpyAsync(
        m->h_frame, m->sframe.data,
        (size_t)nact * S2P_NUM_CODEBOOKS * sizeof(int32_t),
        cudaMemcpyDeviceToHost, st));
    S2P_CUDA_TRY(cudaStreamSynchronize(st));

    /* harvest */
    for (int b = 0; b < nact; b++) {
        s2p_session* s = act[b];
        const int i = act_ix[b];
        if (eflag[b]) {
            s->state = S2P_SESS_FINISHED;
            eos[i] = 1; /* EOS frame produces no codes */
            continue;
        }
        const int32_t* frame = m->h_frame + (size_t)b * S2P_NUM_CODEBOOKS;
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
