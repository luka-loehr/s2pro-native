/* s2pro-native — request scheduler. Implements include/s2pro/scheduler.h.
 *
 * One pthread worker owns the GPU decode loop. Requests are queued
 * (mutex+cond ring); up to max_sessions run concurrently, advanced in
 * LOCKSTEP: one s2p_model_batch_next_frame per tick across all active
 * sessions. A freshly prefilled session gets its FIRST frame via a
 * single-session step before joining the lockstep rendezvous, so TTFA never
 * pays the batch-alignment penalty. Each session streams its frames into a
 * DAC stream (crossfaded windows); emitted float chunks are converted to
 * S16LE and delivered through the request callback on the worker thread.
 *
 * Cancellation contract: after s2p_sched_cancel() returns, the request's
 * callback is never invoked again (cancel waits out an in-flight callback).
 * A callback returning nonzero means "client gone": the session is torn down
 * and no further callback (not even final) is made.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <cuda_runtime.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/model.h"
#include "s2pro/dac.h"
#include "s2pro/tokenizer.h"
#include "s2pro/wav.h"
#include "s2pro/scheduler.h"

#define SCHED_QUEUE_DEPTH_DEFAULT 32
/* Hard per-request frame cap (~ 3.2 min of audio); the model additionally
 * EOSes on its own context bound. */
#define SCHED_MAX_FRAMES 4096

/* Deep-copied request record; owned by the scheduler once submitted. */
typedef struct {
    int32_t* codes; /* [10*T] codebook-major */
    int      T;
} sched_ref;

typedef struct sched_req {
    uint64_t         id;
    char*            text;
    char*            ref_text;
    sched_ref*       refs;
    int              n_refs;
    s2p_sampling_cfg sampling;
    s2p_audio_cb     cb;
    void*            user;
    int              cancelled; /* set under s->mu */
    int              in_cb;     /* callback in flight (worker), under s->mu */
} sched_req;

typedef struct {
    sched_req*      req;     /* NULL => slot free */
    s2p_session*    sess;
    s2p_dac_stream* dstream;
    uint64_t        frames;
    int             fresh;   /* needs its priority first frame */
    int             done;    /* retire at next lock acquisition */
} sched_active;

struct s2p_sched {
    s2p_model*        model;
    s2p_dac*          dac;
    s2p_tok*          tok;
    const s2p_config* cfg;
    int               max_sessions;
    int               queue_depth;

    pthread_mutex_t mu;
    pthread_cond_t  cv;
    sched_req**     queue;   /* ring buffer of pending requests */
    int             q_head, q_count;
    sched_active*   active;  /* [max_sessions] */
    int             n_active;
    uint64_t        next_id;
    s2p_sched_stats stats;

    pthread_t worker;
    int       running;
    int       stop;

    cudaStream_t custream; /* DAC stream work */
};

/* ---------------------------------------------------------------- helpers */

static void req_free(sched_req* r) {
    if (!r) return;
    free(r->text);
    free(r->ref_text);
    for (int i = 0; i < r->n_refs; i++) free(r->refs[i].codes);
    free(r->refs);
    free(r);
}

static char* xstrdup(const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char* d = (char*)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

/* Invoke the callback outside the lock, honoring the cancel contract.
 * Returns: 0 delivered, 1 client-gone (cb returned nonzero), 2 cancelled. */
static int deliver(s2p_sched* s, sched_req* r, const int16_t* pcm, int64_t n,
                   int final) {
    pthread_mutex_lock(&s->mu);
    if (r->cancelled) {
        pthread_mutex_unlock(&s->mu);
        return 2;
    }
    r->in_cb = 1;
    pthread_mutex_unlock(&s->mu);
    int rc = r->cb(r->user, pcm, n, final);
    pthread_mutex_lock(&s->mu);
    r->in_cb = 0;
    pthread_cond_broadcast(&s->cv); /* wake a cancel() waiting on in_cb */
    int cancelled = r->cancelled;
    pthread_mutex_unlock(&s->mu);
    if (cancelled) return 2;
    return rc != 0 ? 1 : 0;
}

/* Convert a float chunk to S16LE and deliver. Frees pcm_f. Same returns as
 * deliver(); a NULL/empty chunk with final=0 is a no-op. */
static int deliver_chunk(s2p_sched* s, sched_req* r, float* pcm_f, int64_t n,
                         int final) {
    int rc = 0;
    if (n > 0 && pcm_f) {
        int16_t* pcm_s = (int16_t*)malloc((size_t)n * sizeof(int16_t));
        if (!pcm_s) {
            free(pcm_f);
            return 1;
        }
        s2p_f32_to_s16(pcm_f, pcm_s, n);
        rc = deliver(s, r, pcm_s, n, final);
        free(pcm_s);
    } else if (final) {
        rc = deliver(s, r, NULL, 0, 1);
    }
    free(pcm_f);
    return rc;
}

/* Tear down one active slot. done_ok: count as a completed request. */
static void retire(s2p_sched* s, sched_active* a, int done_ok) {
    if (a->sess) s2p_session_destroy(a->sess);
    if (a->dstream) s2p_dac_stream_destroy(a->dstream);
    pthread_mutex_lock(&s->mu);
    /* If a cancel() is still waiting on in_cb it holds a pointer to req, but
     * deliver() cleared in_cb before we got here, so freeing is safe. */
    req_free(a->req);
    memset(a, 0, sizeof(*a));
    s->n_active--;
    s->stats.active_sessions = s->n_active;
    if (done_ok) s->stats.requests_done++;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

/* Finish the DAC stream and deliver the final chunk. */
static void finish_stream(s2p_sched* s, sched_active* a) {
    float*  tail = NULL;
    int64_t n = 0;
    s2p_status rc = s2p_dac_stream_finish(a->dstream, &tail, &n, s->custream);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] sched: dac stream finish failed (%d) req %llu\n",
                (int)rc, (unsigned long long)a->req->id);
        tail = NULL;
        n = 0;
    }
    (void)deliver_chunk(s, a->req, tail, n, 1);
    retire(s, a, 1);
}

/* Handle one decoded frame for an active session. Retires on EOS / error /
 * client-gone / frame cap. */
static void handle_frame(s2p_sched* s, sched_active* a,
                         const int32_t codes[S2P_NUM_CODEBOOKS], int is_eos) {
    if (is_eos) {
        finish_stream(s, a);
        return;
    }
    pthread_mutex_lock(&s->mu);
    s->stats.frames_decoded++;
    int cancelled = a->req->cancelled;
    pthread_mutex_unlock(&s->mu);
    if (cancelled) {
        retire(s, a, 0);
        return;
    }
    a->frames++;
    float*  chunk = NULL;
    int64_t n = 0;
    s2p_status rc = s2p_dac_stream_push(a->dstream, codes, &chunk, &n,
                                        s->custream);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] sched: dac push failed (%d) req %llu\n",
                (int)rc, (unsigned long long)a->req->id);
        (void)deliver(s, a->req, NULL, 0, 1); /* close the client stream */
        retire(s, a, 0);
        return;
    }
    int drc = deliver_chunk(s, a->req, chunk, n, 0);
    if (drc != 0) { /* client gone or cancelled */
        retire(s, a, 0);
        return;
    }
    if (a->frames >= SCHED_MAX_FRAMES) {
        fprintf(stderr, "[s2pro] sched: req %llu hit frame cap %d\n",
                (unsigned long long)a->req->id, SCHED_MAX_FRAMES);
        finish_stream(s, a);
    }
}

/* Prompt-build + prefill + DAC stream creation for a request popped from the
 * queue. On failure the client gets a final callback and the req is freed.
 * Returns 1 if the session went active. Called WITHOUT the lock. */
static int start_request(s2p_sched* s, sched_req* r, sched_active* slot) {
    s2p_request_text req_text;
    memset(&req_text, 0, sizeof(req_text));
    req_text.text = r->text;
    req_text.ref_text = r->ref_text;
    s2p_vq_part* ref_parts = NULL;
    if (r->n_refs > 0) {
        ref_parts = (s2p_vq_part*)calloc((size_t)r->n_refs, sizeof(*ref_parts));
        if (!ref_parts) goto fail_cb;
        for (int i = 0; i < r->n_refs; i++) {
            ref_parts[i].codes = r->refs[i].codes;
            ref_parts[i].T = r->refs[i].T;
        }
        req_text.refs = ref_parts;
        req_text.n_refs = r->n_refs;
    }

    int64_t*     ids = NULL;
    uint8_t*     vq_mask = NULL;
    s2p_vq_part* parts = NULL;
    int          n_ids = 0, n_parts = 0;
    s2p_status rc = s2p_prompt_build(s->tok, s->cfg, &req_text, &ids, &vq_mask,
                                     &n_ids, &parts, &n_parts);
    free(ref_parts);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] sched: prompt build failed (%d) req %llu\n",
                (int)rc, (unsigned long long)r->id);
        goto fail_cb;
    }

    s2p_session* sess = NULL;
    rc = s2p_session_create(s->model, &r->sampling, &sess);
    if (rc == S2P_OK) {
        rc = s2p_session_prefill(sess, ids, vq_mask, n_ids, parts, n_parts);
        if (rc != S2P_OK) {
            s2p_session_destroy(sess);
            sess = NULL;
        }
    }
    free(ids);
    free(vq_mask);
    free(parts);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] sched: prefill failed (%d) req %llu\n",
                (int)rc, (unsigned long long)r->id);
        goto fail_cb;
    }

    s2p_dac_stream* dstream = NULL;
    rc = s2p_dac_stream_create(s->dac, &dstream);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] sched: dac stream create failed (%d)\n",
                (int)rc);
        s2p_session_destroy(sess);
        goto fail_cb;
    }

    slot->req = r;
    slot->sess = sess;
    slot->dstream = dstream;
    slot->frames = 0;
    slot->fresh = 1;
    slot->done = 0;
    return 1;

fail_cb:
    (void)deliver(s, r, NULL, 0, 1);
    pthread_mutex_lock(&s->mu);
    req_free(r);
    pthread_mutex_unlock(&s->mu);
    return 0;
}

/* ------------------------------------------------------------ worker loop */

static void* worker_main(void* arg) {
    s2p_sched* s = (s2p_sched*)arg;
    s2p_session** batch_sess =
        (s2p_session**)calloc((size_t)s->max_sessions, sizeof(*batch_sess));
    sched_active** batch_slot =
        (sched_active**)calloc((size_t)s->max_sessions, sizeof(*batch_slot));
    int32_t* batch_codes = (int32_t*)calloc(
        (size_t)s->max_sessions * S2P_NUM_CODEBOOKS, sizeof(int32_t));
    int* batch_eos = (int*)calloc((size_t)s->max_sessions, sizeof(int));
    if (!batch_sess || !batch_slot || !batch_codes || !batch_eos) {
        fprintf(stderr, "[s2pro] sched: worker alloc failed\n");
        free(batch_sess); free(batch_slot); free(batch_codes); free(batch_eos);
        return NULL;
    }

    pthread_mutex_lock(&s->mu);
    while (!s->stop) {
        /* Admit queued requests into free slots. */
        while (s->q_count > 0 && s->n_active < s->max_sessions) {
            sched_req* r = s->queue[s->q_head];
            s->q_head = (s->q_head + 1) % s->queue_depth;
            s->q_count--;
            s->stats.queued = s->q_count;
            if (r->cancelled) {
                req_free(r);
                continue;
            }
            sched_active* slot = NULL;
            for (int i = 0; i < s->max_sessions; i++)
                if (!s->active[i].req) { slot = &s->active[i]; break; }
            if (!slot) { /* invariant guard; cannot happen */
                fprintf(stderr, "[s2pro] sched: no free slot despite "
                        "n_active=%d\n", s->n_active);
                req_free(r);
                continue;
            }
            pthread_mutex_unlock(&s->mu);
            int started = start_request(s, r, slot);
            pthread_mutex_lock(&s->mu);
            if (started) {
                s->n_active++;
                s->stats.active_sessions = s->n_active;
            }
        }

        if (s->n_active == 0) {
            if (s->q_count == 0 && !s->stop)
                pthread_cond_wait(&s->cv, &s->mu);
            continue;
        }
        pthread_mutex_unlock(&s->mu);

        /* Priority first frame: freshly prefilled sessions step alone so
         * their TTFA is not gated on the lockstep rendezvous. */
        for (int i = 0; i < s->max_sessions; i++) {
            sched_active* a = &s->active[i];
            if (!a->req || !a->fresh) continue;
            a->fresh = 0;
            int32_t codes[S2P_NUM_CODEBOOKS];
            int     eos = 0;
            s2p_status rc = s2p_session_next_frame(a->sess, codes, &eos);
            if (rc != S2P_OK) {
                fprintf(stderr, "[s2pro] sched: first frame failed (%d) req "
                        "%llu\n", (int)rc, (unsigned long long)a->req->id);
                (void)deliver(s, a->req, NULL, 0, 1);
                retire(s, a, 0);
                continue;
            }
            handle_frame(s, a, codes, eos);
        }

        /* Lockstep: advance every remaining active session one frame. */
        int n = 0;
        for (int i = 0; i < s->max_sessions; i++) {
            sched_active* a = &s->active[i];
            if (!a->req || a->fresh) continue;
            batch_slot[n] = a;
            batch_sess[n] = a->sess;
            n++;
        }
        if (n > 0) {
            s2p_status rc = s2p_model_batch_next_frame(s->model, batch_sess, n,
                                                       batch_codes, batch_eos);
            if (rc != S2P_OK) {
                fprintf(stderr, "[s2pro] sched: batch decode failed (%d), "
                        "failing %d sessions\n", (int)rc, n);
                for (int i = 0; i < n; i++) {
                    (void)deliver(s, batch_slot[i]->req, NULL, 0, 1);
                    retire(s, batch_slot[i], 0);
                }
            } else {
                for (int i = 0; i < n; i++)
                    handle_frame(s, batch_slot[i],
                                 &batch_codes[i * S2P_NUM_CODEBOOKS],
                                 batch_eos[i]);
            }
        }
        pthread_mutex_lock(&s->mu);
    }

    /* Shutdown: close out every remaining stream so clients unblock. */
    pthread_mutex_unlock(&s->mu);
    for (int i = 0; i < s->max_sessions; i++) {
        sched_active* a = &s->active[i];
        if (!a->req) continue;
        (void)deliver(s, a->req, NULL, 0, 1);
        retire(s, a, 0);
    }
    pthread_mutex_lock(&s->mu);
    while (s->q_count > 0) {
        sched_req* r = s->queue[s->q_head];
        s->q_head = (s->q_head + 1) % s->queue_depth;
        s->q_count--;
        pthread_mutex_unlock(&s->mu);
        if (!r->cancelled) (void)deliver(s, r, NULL, 0, 1);
        pthread_mutex_lock(&s->mu);
        req_free(r);
    }
    s->stats.queued = 0;
    pthread_mutex_unlock(&s->mu);

    free(batch_sess);
    free(batch_slot);
    free(batch_codes);
    free(batch_eos);
    return NULL;
}

/* ------------------------------------------------------------- public API */

s2p_status s2p_sched_create(s2p_model* m, s2p_dac* d, s2p_tok* t,
                            const s2p_config* cfg, const s2p_sched_opts* opts,
                            s2p_sched** out) {
    if (!m || !d || !t || !out) return S2P_ERR_INVALID;
    s2p_sched* s = (s2p_sched*)calloc(1, sizeof(*s));
    if (!s) return S2P_ERR_OOM;
    s->model = m;
    s->dac = d;
    s->tok = t;
    s->cfg = cfg;
    s->max_sessions = (opts && opts->max_sessions > 0) ? opts->max_sessions
                                                       : S2P_MAX_SESSIONS;
    s->queue_depth = (opts && opts->queue_depth > 0) ? opts->queue_depth
                                                     : SCHED_QUEUE_DEPTH_DEFAULT;
    s->next_id = 1;
    s->queue = (sched_req**)calloc((size_t)s->queue_depth, sizeof(*s->queue));
    s->active =
        (sched_active*)calloc((size_t)s->max_sessions, sizeof(*s->active));
    if (!s->queue || !s->active) {
        free(s->queue);
        free(s->active);
        free(s);
        return S2P_ERR_OOM;
    }
    if (pthread_mutex_init(&s->mu, NULL) != 0 ||
        pthread_cond_init(&s->cv, NULL) != 0) {
        free(s->queue);
        free(s->active);
        free(s);
        return S2P_ERR_INTERNAL;
    }
    cudaError_t ce = cudaStreamCreateWithFlags(&s->custream,
                                               cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] sched: stream create: %s\n",
                cudaGetErrorString(ce));
        pthread_mutex_destroy(&s->mu);
        pthread_cond_destroy(&s->cv);
        free(s->queue);
        free(s->active);
        free(s);
        return S2P_ERR_CUDA;
    }
    *out = s;
    return S2P_OK;
}

s2p_status s2p_sched_submit(s2p_sched* s, const s2p_request_text* req,
                            const s2p_sampling_cfg* sampling, s2p_audio_cb cb,
                            void* user, uint64_t* req_id) {
    if (!s || !req || !req->text || !cb) return S2P_ERR_INVALID;
    if (req->n_refs < 0 || (req->n_refs > 0 && !req->refs))
        return S2P_ERR_INVALID;

    sched_req* r = (sched_req*)calloc(1, sizeof(*r));
    if (!r) return S2P_ERR_OOM;
    r->text = xstrdup(req->text);
    r->ref_text = xstrdup(req->ref_text);
    r->sampling = sampling ? *sampling : s2p_sampling_defaults();
    r->cb = cb;
    r->user = user;
    if (!r->text || (req->ref_text && !r->ref_text)) {
        req_free(r);
        return S2P_ERR_OOM;
    }
    if (req->n_refs > 0) {
        r->refs = (sched_ref*)calloc((size_t)req->n_refs, sizeof(sched_ref));
        if (!r->refs) {
            req_free(r);
            return S2P_ERR_OOM;
        }
        for (int i = 0; i < req->n_refs; i++) {
            const s2p_vq_part* p = &req->refs[i];
            if (!p->codes || p->T <= 0) {
                req_free(r);
                return S2P_ERR_INVALID;
            }
            size_t nb = (size_t)S2P_NUM_CODEBOOKS * (size_t)p->T *
                        sizeof(int32_t);
            r->refs[i].codes = (int32_t*)malloc(nb);
            if (!r->refs[i].codes) {
                req_free(r);
                return S2P_ERR_OOM;
            }
            memcpy(r->refs[i].codes, p->codes, nb);
            r->refs[i].T = p->T;
            r->n_refs = i + 1;
        }
    }

    pthread_mutex_lock(&s->mu);
    if (s->stop || !s->running) {
        pthread_mutex_unlock(&s->mu);
        req_free(r);
        return S2P_ERR_STATE;
    }
    if (s->q_count >= s->queue_depth) {
        pthread_mutex_unlock(&s->mu);
        req_free(r);
        return S2P_ERR_FULL;
    }
    r->id = s->next_id++;
    s->queue[(s->q_head + s->q_count) % s->queue_depth] = r;
    s->q_count++;
    s->stats.queued = s->q_count;
    if (req_id) *req_id = r->id;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return S2P_OK;
}

s2p_status s2p_sched_cancel(s2p_sched* s, uint64_t req_id) {
    if (!s || req_id == 0) return S2P_ERR_INVALID;
    pthread_mutex_lock(&s->mu);
    /* Queued? Mark; the worker frees it on pop (removal from the middle of
     * the ring is not worth the churn). */
    for (int i = 0; i < s->q_count; i++) {
        sched_req* r = s->queue[(s->q_head + i) % s->queue_depth];
        if (r->id == req_id) {
            r->cancelled = 1;
            pthread_mutex_unlock(&s->mu);
            return S2P_OK;
        }
    }
    /* Active? Mark cancelled and wait out any in-flight callback so the
     * caller may free cb resources the moment we return. */
    for (int i = 0; i < s->max_sessions; i++) {
        sched_req* r = s->active[i].req;
        if (r && r->id == req_id) {
            r->cancelled = 1;
            /* Slot check FIRST: once the worker retires the slot, r is freed
             * and must not be dereferenced. Holding mu, req==r => r alive. */
            while (s->active[i].req == r && r->in_cb)
                pthread_cond_wait(&s->cv, &s->mu);
            pthread_mutex_unlock(&s->mu);
            return S2P_OK;
        }
    }
    pthread_mutex_unlock(&s->mu);
    return S2P_ERR_INVALID; /* unknown or already finished: nothing to do */
}

s2p_status s2p_sched_start(s2p_sched* s) {
    if (!s) return S2P_ERR_INVALID;
    pthread_mutex_lock(&s->mu);
    if (s->running) {
        pthread_mutex_unlock(&s->mu);
        return S2P_ERR_STATE;
    }
    s->stop = 0;
    s->running = 1;
    pthread_mutex_unlock(&s->mu);
    if (pthread_create(&s->worker, NULL, worker_main, s) != 0) {
        pthread_mutex_lock(&s->mu);
        s->running = 0;
        pthread_mutex_unlock(&s->mu);
        return S2P_ERR_INTERNAL;
    }
    return S2P_OK;
}

s2p_status s2p_sched_stop(s2p_sched* s) {
    if (!s) return S2P_ERR_INVALID;
    pthread_mutex_lock(&s->mu);
    if (!s->running) {
        pthread_mutex_unlock(&s->mu);
        return S2P_OK;
    }
    s->stop = 1;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    pthread_join(s->worker, NULL);
    pthread_mutex_lock(&s->mu);
    s->running = 0;
    pthread_mutex_unlock(&s->mu);
    return S2P_OK;
}

void s2p_sched_destroy(s2p_sched* s) {
    if (!s) return;
    (void)s2p_sched_stop(s);
    /* Worker exit path drained the queue and retired all sessions. */
    cudaStreamDestroy(s->custream);
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    free(s->queue);
    free(s->active);
    free(s);
}

void s2p_sched_get_stats(s2p_sched* s, s2p_sched_stats* out) {
    if (!s || !out) return;
    pthread_mutex_lock(&s->mu);
    *out = s->stats;
    pthread_mutex_unlock(&s->mu);
}
