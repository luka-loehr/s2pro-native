/* s2pro-native — request scheduler: session table, lockstep batching,
 * streaming PCM delivery. Contract header: frozen.
 *
 * Model: one engine, up to S2P_MAX_SESSIONS concurrent generations advanced
 * in lockstep (one batched decode per frame across all active sessions, same
 * pattern as qwen3-tts-native ABI v3). A session's first frame may bypass the
 * lockstep rendezvous so TTFA does not pay a batching penalty.
 */
#pragma once

#include <stdint.h>
#include "s2pro/status.h"
#include "s2pro/model.h"
#include "s2pro/dac.h"
#include "s2pro/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_sched s2p_sched;

typedef struct {
    int max_sessions;   /* 0 -> S2P_MAX_SESSIONS */
    int queue_depth;    /* pending requests beyond active; 0 -> 32 */
} s2p_sched_opts;

/* Called from the scheduler thread with progressive PCM (host, S16LE mono
 * 44.1 kHz). final=1 on the last chunk (n may be 0). Return nonzero to
 * cancel the request (client gone). */
typedef int (*s2p_audio_cb)(void* user, const int16_t* pcm, int64_t n,
                            int final);

s2p_status s2p_sched_create(s2p_model* m, s2p_dac* d, s2p_tok* t,
                            const s2p_config* cfg, const s2p_sched_opts* opts,
                            s2p_sched** out);
s2p_status s2p_sched_submit(s2p_sched* s, const s2p_request_text* req,
                            const s2p_sampling_cfg* sampling, s2p_audio_cb cb,
                            void* user, uint64_t* req_id);
s2p_status s2p_sched_cancel(s2p_sched* s, uint64_t req_id);
/* Start/stop the scheduler thread. */
s2p_status s2p_sched_start(s2p_sched* s);
s2p_status s2p_sched_stop(s2p_sched* s);
void       s2p_sched_destroy(s2p_sched* s);

/* Introspection for /healthz. */
typedef struct {
    int      active_sessions;
    int      queued;
    uint64_t frames_decoded;
    uint64_t requests_done;
} s2p_sched_stats;
void s2p_sched_get_stats(s2p_sched* s, s2p_sched_stats* out);

#ifdef __cplusplus
}
#endif
