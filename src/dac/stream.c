/* s2pro-native — streaming vocoder front end.
 *
 * Default path: the bit-exact incremental engine (stream_inc.c) — every
 * pushed frame emits its 2048 samples immediately, and the streamed PCM
 * equals s2p_dac_decode of the full sequence bit for bit. No windows, no
 * overlap re-decode, no crossfade, no held tail.
 *
 * S2P_STREAM_REFERENCE=1 selects the ported reference scheme instead
 * (fishaudio_s2_pro/streaming_vocoder.py: build_stream_vocoder_chunk /
 * flush_stream_vocoder_chunk / _apply_stream_crossfade /
 * trim_retained_stream_codes):
 *   stream_stride           = 10  frames  (first decode threshold)
 *   stream_followup_stride  = 90  frames  (subsequent thresholds: total+90)
 *   stream_overlap_tokens   = 20  frames  (re-decoded for causal context)
 *   stream_crossfade_samples= 512 samples (linspace fade; tail hold)
 * Window decode: window_start = max(code_start, emitted - overlap); the
 * decoded overlap prefix (overlap*2048 samples) is dropped, the remainder is
 * crossfaded with the held tail; the last 512 samples are withheld as the new
 * tail (emitted in full on finish). Retained codes are trimmed to the last
 * `overlap` frames after each emit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "s2pro/dac.h"
#include "dac_internal.h"

#define STREAM_STRIDE      10
#define STREAM_FOLLOWUP    90
#define STREAM_OVERLAP     20
#define STREAM_CROSSFADE   512

static int stream_use_reference(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_STREAM_REFERENCE");
        v = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return v;
}

#define STREAM_BATCH_MAX 4

/* S2P_STREAM_BATCH (1..4, default 4): frames per incremental-engine push.
 * Bigger pushes amortize kernel launches; the FIRST push always flushes at
 * one frame so TTFA is unaffected. */
static int stream_batch(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("S2P_STREAM_BATCH");
        v = e ? atoi(e) : STREAM_BATCH_MAX;
        if (v < 1) v = 1;
        if (v > STREAM_BATCH_MAX) v = STREAM_BATCH_MAX;
    }
    return v;
}

struct s2p_dac_stream {
    s2p_dac* d;
    s2pd_inc* inc;        /* non-NULL => incremental path */
    int32_t  bat[STREAM_BATCH_MAX][S2P_NUM_CODEBOOKS]; /* pending frames */
    int      bat_n;
    int      first_flushed; /* first push goes out at one frame (TTFA) */
    float*   pend_chunk;  /* reference-mode push_async stash */
    int64_t  pend_n;
    int32_t* frames;      /* frame-major [count][10] */
    int      cap, count;
    int      code_start;  /* absolute frame index of frames[0] */
    int      total;       /* total frames pushed */
    int      last;        /* frames already vocoded (last_vocode_tokens) */
    int      next;        /* next vocode threshold (0 = use first stride) */
    float*   tail;        /* pending crossfade tail */
    int64_t  tail_len;
    int      finished;
};

s2p_status s2p_dac_stream_create(s2p_dac* d, s2p_dac_stream** out) {
    if (!d || !out) return S2P_ERR_INVALID;
    s2p_dac_stream* s = (s2p_dac_stream*)calloc(1, sizeof(*s));
    if (!s) return S2P_ERR_OOM;
    s->d = d;
    if (!stream_use_reference()) {
        s2p_status rc = s2pd_inc_create(d, &s->inc);
        if (rc != S2P_OK) {
            free(s);
            return rc;
        }
    }
    *out = s;
    return S2P_OK;
}

void s2p_dac_stream_destroy(s2p_dac_stream* s) {
    if (!s) return;
    s2pd_inc_destroy(s->inc);
    free(s->pend_chunk);
    free(s->frames);
    free(s->tail);
    free(s);
}

/* Pipelined pair. Incremental: true enqueue/sync split. Reference: the push
 * is inherently synchronous, so push_async runs it and stashes the chunk. */
s2p_status s2p_dac_stream_push_async(s2p_dac_stream* s,
                                     const int32_t
                                         frame_codes[S2P_NUM_CODEBOOKS],
                                     cudaStream_t stream) {
    if (!s || !frame_codes) return S2P_ERR_INVALID;
    if (s->finished) return S2P_ERR_STATE;
    if (s->inc) {
        memcpy(s->bat[s->bat_n], frame_codes,
               S2P_NUM_CODEBOOKS * sizeof(int32_t));
        s->bat_n++;
        int target = s->first_flushed ? stream_batch() : 1;
        if (s->bat_n < target) return S2P_OK;
        s2p_status rc =
            s2pd_inc_push_async(s->inc, &s->bat[0][0], s->bat_n, stream);
        if (rc == S2P_OK) {
            s->bat_n = 0;
            s->first_flushed = 1;
        }
        return rc;
    }
    if (s->pend_chunk) return S2P_ERR_STATE;
    return s2p_dac_stream_push(s, frame_codes, &s->pend_chunk, &s->pend_n,
                               stream);
}

s2p_status s2p_dac_stream_collect(s2p_dac_stream* s, float** pcm_chunk,
                                  int64_t* n_out, cudaStream_t stream) {
    if (!s || !pcm_chunk || !n_out) return S2P_ERR_INVALID;
    *pcm_chunk = NULL;
    *n_out = 0;
    if (s->inc) {
        float* pcm = (float*)malloc((size_t)STREAM_BATCH_MAX *
                                    S2P_FRAME_SAMPLES * sizeof(float));
        if (!pcm) return S2P_ERR_OOM;
        int64_t n = 0;
        s2p_status rc = s2pd_inc_collect(s->inc, pcm, &n, stream);
        if (rc != S2P_OK || n == 0) {
            free(pcm);
            return rc;
        }
        *pcm_chunk = pcm;
        *n_out = n;
        return S2P_OK;
    }
    *pcm_chunk = s->pend_chunk;
    *n_out = s->pend_n;
    s->pend_chunk = NULL;
    s->pend_n = 0;
    return S2P_OK;
}

static void trim_codes(s2p_dac_stream* s, int keep_from) {
    if (keep_from <= s->code_start) return;
    int drop = keep_from - s->code_start;
    if (drop > s->count) drop = s->count;
    memmove(s->frames, s->frames + (size_t)drop * S2P_NUM_CODEBOOKS,
            (size_t)(s->count - drop) * S2P_NUM_CODEBOOKS * sizeof(int32_t));
    s->count -= drop;
    s->code_start = keep_from;
}

/* _build_stream_vocoder_chunk. On S2P_OK, *chunk may still be NULL (nothing
 * emitted). Chunk buffers are malloc'd; caller owns. */
static s2p_status build_chunk(s2p_dac_stream* s, int is_final, float** chunk,
                              int64_t* n_out, cudaStream_t stream) {
    *chunk = NULL;
    *n_out = 0;
    if (s->count == 0) return S2P_OK;

    if (s->total <= s->last) {
        if (!is_final) return S2P_OK;
        if (s->tail_len == 0) return S2P_OK;
        *chunk = s->tail;                       /* hand tail ownership over */
        *n_out = s->tail_len;
        s->tail = NULL;
        s->tail_len = 0;
        return S2P_OK;
    }

    int window_start = s->last - STREAM_OVERLAP;
    if (window_start < s->code_start) window_start = s->code_start;
    int offset = window_start - s->code_start;
    int W = s->count - offset;                  /* = total - window_start */
    if (W <= 0) return S2P_OK;

    /* frame-major storage -> codebook-major [10,W] for s2p_dac_decode */
    int32_t* cbmaj = (int32_t*)malloc((size_t)S2P_NUM_CODEBOOKS * W * sizeof(int32_t));
    if (!cbmaj) return S2P_ERR_OOM;
    for (int i = 0; i < W; i++)
        for (int q = 0; q < S2P_NUM_CODEBOOKS; q++)
            cbmaj[(size_t)q * W + i] =
                s->frames[(size_t)(offset + i) * S2P_NUM_CODEBOOKS + q];

    float* audio = NULL;
    int64_t alen = 0;
    s2p_status st = s2p_dac_decode(s->d, cbmaj, W, &audio, &alen, stream);
    free(cbmaj);
    if (st != S2P_OK) return st;

    int overlap_tokens = s->last - window_start;
    int64_t osamp = (int64_t)overlap_tokens * S2P_FRAME_SAMPLES;
    if (alen <= osamp) {          /* reference: early None, NO state updates */
        free(audio);
        return S2P_OK;
    }
    const float* delta = audio + osamp;
    int64_t dlen = alen - osamp;

    /* crossfade with pending tail */
    int64_t cf = STREAM_CROSSFADE;
    if (s->tail_len < cf) cf = s->tail_len;
    if (dlen < cf) cf = dlen;
    int64_t mlen = s->tail_len + dlen - cf;
    float* merged = (float*)malloc((size_t)mlen * sizeof(float));
    if (!merged) { free(audio); return S2P_ERR_OOM; }
    int64_t head = s->tail_len - cf;
    if (head > 0) memcpy(merged, s->tail, (size_t)head * sizeof(float));
    for (int64_t i = 0; i < cf; i++) {          /* torch.linspace(0,1,cf) */
        float f = cf > 1 ? (float)i / (float)(cf - 1) : 0.0f;
        merged[head + i] = s->tail[head + i] * (1.0f - f) + delta[i] * f;
    }
    memcpy(merged + s->tail_len, delta + cf,
           (size_t)(dlen - cf) * sizeof(float));
    free(audio);
    free(s->tail);
    s->tail = NULL;
    s->tail_len = 0;

    if (is_final) {
        s->last = s->total;
        *chunk = merged;
        *n_out = mlen;
        return S2P_OK;
    }

    int64_t hold = STREAM_CROSSFADE;
    if (mlen < hold) hold = mlen;
    if (hold > 0) {
        s->tail = (float*)malloc((size_t)hold * sizeof(float));
        if (!s->tail) { free(merged); return S2P_ERR_OOM; }
        memcpy(s->tail, merged + (mlen - hold), (size_t)hold * sizeof(float));
        s->tail_len = hold;
    }
    int64_t emit = mlen - hold;

    s->last = s->total;
    {
        int keep_from = s->total - STREAM_OVERLAP;
        if (keep_from < 0) keep_from = 0;
        trim_codes(s, keep_from);
    }

    if (emit == 0) {                            /* everything withheld */
        free(merged);
        return S2P_OK;
    }
    *chunk = merged;                            /* over-allocated is fine */
    *n_out = emit;
    return S2P_OK;
}

s2p_status s2p_dac_stream_push(s2p_dac_stream* s,
                               const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                               float** pcm_chunk, int64_t* n_out,
                               cudaStream_t stream) {
    if (!s || !frame_codes || !pcm_chunk || !n_out) return S2P_ERR_INVALID;
    *pcm_chunk = NULL;
    *n_out = 0;
    if (s->finished) return S2P_ERR_STATE;

    if (s->inc) {                       /* through the batcher, then sync */
        S2P_TRY(s2p_dac_stream_push_async(s, frame_codes, stream));
        return s2p_dac_stream_collect(s, pcm_chunk, n_out, stream);
    }

    if (s->count == s->cap) {
        int cap = s->cap ? s->cap * 2 : 256;
        int32_t* nf = (int32_t*)realloc(
            s->frames, (size_t)cap * S2P_NUM_CODEBOOKS * sizeof(int32_t));
        if (!nf) return S2P_ERR_OOM;
        s->frames = nf;
        s->cap = cap;
    }
    memcpy(s->frames + (size_t)s->count * S2P_NUM_CODEBOOKS, frame_codes,
           S2P_NUM_CODEBOOKS * sizeof(int32_t));
    s->count++;
    s->total++;

    int next = s->next ? s->next : STREAM_STRIDE;
    if (s->total < next) {
        s->next = next;
        return S2P_OK;
    }
    s2p_status st = build_chunk(s, 0, pcm_chunk, n_out, stream);
    s->next = s->total + STREAM_FOLLOWUP;       /* set even when chunk NULL */
    return st;
}

s2p_status s2p_dac_stream_finish(s2p_dac_stream* s, float** pcm_chunk,
                                 int64_t* n_out, cudaStream_t stream) {
    if (!s || !pcm_chunk || !n_out) return S2P_ERR_INVALID;
    *pcm_chunk = NULL;
    *n_out = 0;
    if (s->finished) return S2P_ERR_STATE;
    s->finished = 1;

    if (s->inc) {
        /* flush the remainder batch, then hand out whatever is in flight */
        if (s->bat_n > 0) {
            /* the caller collected before finishing, so nothing is pending */
            s2p_status rc =
                s2pd_inc_push_async(s->inc, &s->bat[0][0], s->bat_n, stream);
            if (rc != S2P_OK) return rc;
            s->bat_n = 0;
        }
        return s2p_dac_stream_collect(s, pcm_chunk, n_out, stream);
    }

    int has_codes = s->count > 0;
    int has_tail = s->tail_len > 0;
    if (!has_codes && !has_tail) return S2P_OK;
    if (!has_codes) {                           /* tail only */
        *pcm_chunk = s->tail;
        *n_out = s->tail_len;
        s->tail = NULL;
        s->tail_len = 0;
        return S2P_OK;
    }
    if (s->total <= s->last && !has_tail) return S2P_OK;
    return build_chunk(s, 1, pcm_chunk, n_out, stream);
}
