/* s2pro-native — HTTP server (POSIX sockets, no dependencies).
 * Contract header: frozen.
 *
 * Endpoints:
 *   GET  /healthz            -> 200 JSON stats (no auth)
 *   POST /v1/tts             -> chunked audio stream
 *        body JSON: {"text": "...", "format": "wav"|"pcm",
 *                    "temperature"?, "top_p"?, "seed"?}
 *        auth: Authorization: Bearer <token> when opts.auth_token set.
 * wav format streams a WAV header with unknown length then S16LE frames;
 * pcm streams raw S16LE 44.1 kHz mono.
 */
#pragma once

#include "s2pro/status.h"
#include "s2pro/scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int         port;         /* 0 -> 8010 */
    const char* bind_addr;    /* NULL -> 127.0.0.1 */
    const char* auth_token;   /* NULL -> no auth */
    int         max_conns;    /* 0 -> 64 */
} s2p_server_opts;

/* Blocks until SIGINT/SIGTERM. */
s2p_status s2p_server_run(s2p_sched* sched, const s2p_server_opts* opts);

#ifdef __cplusplus
}
#endif
