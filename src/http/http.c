/* s2pro-native — dependency-free HTTP/1.1 server. Implements
 * include/s2pro/server.h.
 *
 * POSIX sockets + poll(). Routes:
 *   GET  /healthz    -> 200 JSON scheduler stats (no auth)
 *   GET  /v1/voices  -> 200 JSON list of the named-voice registry
 *   POST /v1/tts     -> chunked Transfer-Encoding audio stream
 *                     body: {"text": "...", "format": "wav"|"pcm",
 *                            "temperature"?, "top_p"?, "seed"?,
 *                            "stream"?: true|false (default true),
 *                            "chunk_length"?: bytes (default 300; 0 = off;
 *                             long-form sentence chunking, voiced requests
 *                             only — see submit_chunk),
 *                            "chunk_sentences"?: max sentences per chunk
 *                             (default 2; 0 = byte limit only),
 *                            "chunk_gap_ms"?: normalized inter-chunk pause
 *                             (default 1000; 0 = raw concatenation),
 *                            "voice"?: "<registry name>",
 *                            "reference_audio_b64"?: "<wav, 44.1k mono s16,
 *                             max 15 s>", "reference_text"?: "<transcript>"}
 *                     reference_audio_b64 (on-the-fly clone) wins over
 *                     voice; neither -> zero-shot (unpinned voice).
 *                     Authorization: Bearer <token> when configured.
 *
 * Streaming model: the poll loop (main thread) parses requests and submits
 * to the scheduler; the scheduler worker thread then writes chunked audio
 * DIRECTLY to the connection fd from the audio callback. While streaming,
 * the poll loop only watches the fd for hangup (client gone -> cancel).
 * The callback signals completion via an atomic flag; only the poll loop
 * ever closes fds, and only after s2p_sched_cancel() guarantees the worker
 * is out of the callback — this makes fd reuse races impossible.
 * Every response is Connection: close (no keep-alive).
 *
 * "stream": false buffers the whole take instead and answers with exact
 * Content-Length and exact RIFF sizes. A chunked stream cannot carry a
 * correct WAV length — the 44 header bytes are on the wire before frame 1
 * is even sampled — so streamed WAVs advertise saturated sizes, which
 * strict players (Apple's, notably) refuse to seek/finish. Buffered mode
 * trades time-to-first-audio for a well-formed downloadable file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <stdatomic.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/json.h"
#include "s2pro/wav.h"
#include "s2pro/scheduler.h"
#include "s2pro/voices.h"
#include "s2pro/server.h"

/* Long-form chunking: text above this many bytes is split at sentence
 * boundaries and generated as chained requests (prosody stays at the
 * fresh-context quality the first ~25 s of a take have; see
 * src/text/chunker.c). A chunk additionally closes after
 * HTTP_CHUNK_SENTENCES_DEFAULT sentences (project owner's listening
 * verdict on the first long-form pass: prefer ~2-sentence chunks).
 * Request fields "chunk_length" / "chunk_sentences" override (0 disables /
 * 0 = byte limit only); envs S2P_CHUNK_BYTES / S2P_CHUNK_SENTENCES
 * override the defaults. Applies only when a voice reference pins the
 * voice — zero-shot chunks would each draw a new voice. */
#define HTTP_CHUNK_BYTES_DEFAULT     300
#define HTTP_CHUNK_SENTENCES_DEFAULT 2

/* Inter-chunk gap normalization: the model's own sentence pauses inside a
 * take run ~1.0-1.3 s (the project owner's "perfect" range), but the
 * trailing/leading silence around a chunk join is whatever the two takes
 * happened to emit — mostly under 1 s and variable. The filter holds back
 * a tail window per chunk, trims boundary silence on BOTH sides of a join,
 * and inserts exactly chunk_gap_ms of silence instead (request field
 * "chunk_gap_ms", env S2P_CHUNK_GAP_MS; 0 restores raw concatenation).
 * Take-level edges (start of chunk 1, end of the last chunk) are never
 * trimmed. */
#define HTTP_CHUNK_GAP_MS_DEFAULT 1000
#define GAP_HOLD_SAMPLES (3 * 44100)     /* boundary tail window held back */
#define GAP_LEAD_MAX     (44100 * 2)     /* max leading silence trimmed */
#define GAP_SIL_PEAK     400             /* |s16| below this = silence */
#define GAP_FRAME        441             /* 10 ms scan granularity */

#define HTTP_MAX_HEAD   (16 * 1024)
#define HTTP_MAX_BODY   (4 * 1024 * 1024) /* fits a 15 s wav as base64 JSON */
#define HTTP_CLONE_MAX_SAMPLES (15 * 44100) /* inline-clone cap: 15 s */
#define HTTP_IDLE_SECS  30
#define HTTP_SEND_SECS  30    /* SO_SNDTIMEO: stalled client => cb error */

typedef enum {
    C_FREE = 0,
    C_READ_HEAD,
    C_READ_BODY,
    C_STREAMING,
} conn_state;

typedef struct {
    int        fd;
    conn_state st;
    char*      buf;
    size_t     len, cap;
    size_t     head_end;     /* offset just past \r\n\r\n */
    long       content_len;
    time_t     last_activity;
    /* streaming request state */
    uint64_t    req_id;
    int         wav;          /* 1 wav framing, 0 raw pcm */
    int         sent_wav_hdr;
    atomic_int  finished;     /* set by the scheduler cb on final chunk */
    /* buffered mode ("stream": false): PCM accumulates here (scheduler
     * thread only) and one exact-length response goes out on final. */
    int         buffered;
    char*       acc;
    size_t      acc_len, acc_cap;
    /* long-form chunk chain: text split at sentence boundaries, one
     * scheduler request per chunk, all audio in ONE response. The audio cb
     * (scheduler thread) sets chunk_advance on a non-last chunk's final
     * instead of finished; the poll loop then submits the next chunk. */
    char**      chunks;
    int         n_chunks, cur_chunk;
    atomic_int  chunk_advance;
    int         gap_ms;             /* inter-chunk gap; 0 = raw concat */
    int16_t*    gap_hold;           /* boundary tail window (scheduler thr) */
    size_t      gap_hold_n;
    int         gap_lead_skip;      /* trimming next chunk's leading silence */
    size_t      gap_lead_skipped;
    const s2p_voice* lf_voice;      /* registry voice (stable) or NULL */
    float*      lf_clone_pcm;       /* inline clone, re-encoded per chunk */
    int64_t     lf_clone_n;
    char*       lf_clone_text;
    s2p_sampling_cfg lf_sampling;   /* base; per-chunk seed derived */
} conn;

typedef struct {
    s2p_sched*        sched;
    const char*       auth_token;
    const s2p_voices* voices; /* may be NULL / empty */
    int               listen_fd;
    conn*             conns;
    int               max_conns;
} http_srv;

/* Strict base64 decoder (padding required, whitespace tolerated). Returns
 * malloc'd buffer or NULL. */
static uint8_t* b64_decode(const char* s, size_t len, size_t* out_len) {
    static const int8_t T[256] = {
        ['A'] = 1,  ['B'] = 2,  ['C'] = 3,  ['D'] = 4,  ['E'] = 5,
        ['F'] = 6,  ['G'] = 7,  ['H'] = 8,  ['I'] = 9,  ['J'] = 10,
        ['K'] = 11, ['L'] = 12, ['M'] = 13, ['N'] = 14, ['O'] = 15,
        ['P'] = 16, ['Q'] = 17, ['R'] = 18, ['S'] = 19, ['T'] = 20,
        ['U'] = 21, ['V'] = 22, ['W'] = 23, ['X'] = 24, ['Y'] = 25,
        ['Z'] = 26, ['a'] = 27, ['b'] = 28, ['c'] = 29, ['d'] = 30,
        ['e'] = 31, ['f'] = 32, ['g'] = 33, ['h'] = 34, ['i'] = 35,
        ['j'] = 36, ['k'] = 37, ['l'] = 38, ['m'] = 39, ['n'] = 40,
        ['o'] = 41, ['p'] = 42, ['q'] = 43, ['r'] = 44, ['s'] = 45,
        ['t'] = 46, ['u'] = 47, ['v'] = 48, ['w'] = 49, ['x'] = 50,
        ['y'] = 51, ['z'] = 52, ['0'] = 53, ['1'] = 54, ['2'] = 55,
        ['3'] = 56, ['4'] = 57, ['5'] = 58, ['6'] = 59, ['7'] = 60,
        ['8'] = 61, ['9'] = 62, ['+'] = 63, ['/'] = 64,
    };
    uint8_t* out = (uint8_t*)malloc(len / 4 * 3 + 3);
    if (!out) return NULL;
    size_t o = 0;
    uint32_t acc = 0;
    int nbits = 0, pad = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '\n' || ch == '\r' || ch == ' ' || ch == '\t') continue;
        if (ch == '=') {
            pad++;
            continue;
        }
        int v = T[ch] - 1;
        if (v < 0 || pad > 0) {
            free(out);
            return NULL;
        }
        acc = (acc << 6) | (uint32_t)v;
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            out[o++] = (uint8_t)(acc >> nbits);
        }
    }
    *out_len = o;
    return out;
}

static volatile sig_atomic_t g_http_stop = 0;

static void http_sig_handler(int sig) {
    (void)sig;
    g_http_stop = 1;
}

/* ------------------------------------------------------------- socket I/O */

static int send_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static int send_chunk(int fd, const void* data, size_t n) {
    char hdr[32];
    int hn = snprintf(hdr, sizeof(hdr), "%zx\r\n", n);
    if (send_all(fd, hdr, (size_t)hn) != 0) return -1;
    if (n > 0 && send_all(fd, data, n) != 0) return -1;
    return send_all(fd, "\r\n", 2);
}

static void send_simple(int fd, int code, const char* reason,
                        const char* content_type, const char* body) {
    char head[512];
    size_t blen = body ? strlen(body) : 0;
    int hn = snprintf(head, sizeof(head),
                      "HTTP/1.1 %d %s\r\n"
                      "Content-Type: %s\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n",
                      code, reason, content_type, blen);
    if (hn > 0 && send_all(fd, head, (size_t)hn) == 0 && blen > 0)
        (void)send_all(fd, body, blen);
}

/* ------------------------------------------------------- audio callback */

/* Runs on the SCHEDULER thread. Writes chunked body bytes to the socket.
 * Nonzero return => cancel (client gone / write failed / timed out). */
/* Append audio to the response (accumulator or wire). Returns nonzero on
 * failure (client gone / oom; error handling done, caller just cancels). */
static int conn_emit(conn* c, const int16_t* pcm, size_t n) {
    if (c->buffered) {
        if (n == 0) return 0;
        size_t add = n * sizeof(int16_t);
        if (c->acc_len + add > c->acc_cap) {
            size_t nc = c->acc_cap ? c->acc_cap * 2 : (1u << 20);
            while (nc < c->acc_len + add) nc *= 2;
            char* nb = (char*)realloc(c->acc, nc);
            if (!nb) {
                send_simple(c->fd, 500, "Internal Server Error",
                            "application/json", "{\"error\":\"oom\"}");
                atomic_store(&c->finished, 1); /* poll loop reaps */
                return 1;
            }
            c->acc = nb;
            c->acc_cap = nc;
        }
        memcpy(c->acc + c->acc_len, pcm, add);
        c->acc_len += add;
        return 0;
    }
    if (c->wav && !c->sent_wav_hdr) {
        /* WAV header with unknown length: pass the max so players stream. */
        uint8_t hdr[44];
        size_t hn = s2p_wav_header(hdr, 0xFFFFFFFFu, S2P_SAMPLE_RATE);
        if (send_chunk(c->fd, hdr, hn) != 0) return 1;
        c->sent_wav_hdr = 1;
    }
    if (n > 0 && send_chunk(c->fd, pcm, n * sizeof(int16_t)) != 0) return 1;
    return 0;
}

/* Silent samples at the head / tail of a buffer (10 ms peak scan). */
static size_t sil_lead(const int16_t* p, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t f = n - i < GAP_FRAME ? n - i : GAP_FRAME;
        for (size_t k = 0; k < f; k++)
            if (p[i + k] > GAP_SIL_PEAK || p[i + k] < -GAP_SIL_PEAK)
                return i + k;
        i += f;
    }
    return n;
}

static size_t sil_tail(const int16_t* p, size_t n) {
    size_t i = n;
    while (i > 0) {
        size_t f = i < GAP_FRAME ? i : GAP_FRAME;
        for (size_t k = 0; k < f; k++)
            if (p[i - 1 - k] > GAP_SIL_PEAK || p[i - 1 - k] < -GAP_SIL_PEAK)
                return n - (i - k);
        i -= f;
    }
    return n;
}

/* Feed chunk audio through the boundary filter: trim a later chunk's
 * leading silence, hold back the tail window, forward the rest. */
static int gap_feed(conn* c, const int16_t* pcm, size_t n) {
    if (!c->gap_hold) return conn_emit(c, pcm, n); /* filter disabled */
    if (c->gap_lead_skip) {
        size_t lead = sil_lead(pcm, n);
        size_t budget = GAP_LEAD_MAX - c->gap_lead_skipped;
        if (lead > budget) lead = budget;
        c->gap_lead_skipped += lead;
        pcm += lead;
        n -= lead;
        if (n > 0 || c->gap_lead_skipped >= GAP_LEAD_MAX)
            c->gap_lead_skip = 0;
        if (n == 0) return 0;
    }
    while (n > 0) {
        size_t room = GAP_HOLD_SAMPLES - c->gap_hold_n;
        if (room == 0) {
            /* forward the oldest half, keep the newest half held back */
            size_t fwd = GAP_HOLD_SAMPLES / 2;
            if (conn_emit(c, c->gap_hold, fwd) != 0) return 1;
            memmove(c->gap_hold, c->gap_hold + fwd,
                    (c->gap_hold_n - fwd) * sizeof(int16_t));
            c->gap_hold_n -= fwd;
            room = GAP_HOLD_SAMPLES - c->gap_hold_n;
        }
        size_t take = n < room ? n : room;
        memcpy(c->gap_hold + c->gap_hold_n, pcm, take * sizeof(int16_t));
        c->gap_hold_n += take;
        pcm += take;
        n -= take;
    }
    return 0;
}

/* Non-last chunk finished: trim the trailing silence out of the held-back
 * tail, insert exactly gap_ms of silence, arm leading-trim for the next
 * chunk. */
static int gap_boundary(conn* c) {
    if (!c->gap_hold) return 0;
    size_t tail = sil_tail(c->gap_hold, c->gap_hold_n);
    if (conn_emit(c, c->gap_hold, c->gap_hold_n - tail) != 0) return 1;
    c->gap_hold_n = 0;
    static const int16_t zeros[4410] = {0};
    size_t gap = (size_t)c->gap_ms * 44100 / 1000;
    while (gap > 0) {
        size_t t = gap < 4410 ? gap : 4410;
        if (conn_emit(c, zeros, t) != 0) return 1;
        gap -= t;
    }
    c->gap_lead_skip = 1;
    c->gap_lead_skipped = 0;
    return 0;
}

static int tts_audio_cb(void* user, const int16_t* pcm, int64_t n, int final) {
    conn* c = (conn*)user;
    const int chained = c->n_chunks > 1;
    if (n > 0 && pcm) {
        int rc = chained ? gap_feed(c, pcm, (size_t)n)
                         : conn_emit(c, pcm, (size_t)n);
        if (rc != 0) return 1;
    }
    if (final && chained && c->cur_chunk + 1 < c->n_chunks) {
        if (gap_boundary(c) != 0) return 1;
        /* non-last chunk done: hand back to the poll loop for the next
         * submit. LAST touch of c by this thread until resubmitted. */
        atomic_store(&c->chunk_advance, 1);
        return 0;
    }
    if (final && chained && c->gap_hold) {
        /* last chunk: flush the held-back tail untrimmed */
        if (conn_emit(c, c->gap_hold, c->gap_hold_n) != 0) return 1;
        c->gap_hold_n = 0;
    }
    if (c->buffered) {
        if (final) {
            char head[256];
            size_t body = c->acc_len + (c->wav ? 44u : 0u);
            int hn = snprintf(head, sizeof(head),
                              "HTTP/1.1 200 OK\r\n"
                              "Content-Type: %s\r\n"
                              "Content-Length: %zu\r\n"
                              "Cache-Control: no-store\r\n"
                              "Connection: close\r\n\r\n",
                              c->wav ? "audio/wav" : "application/octet-stream",
                              body);
            int ok = hn > 0 && send_all(c->fd, head, (size_t)hn) == 0;
            if (ok && c->wav) {
                uint8_t hdr[44];
                size_t wn = s2p_wav_header(hdr, (uint32_t)c->acc_len,
                                           S2P_SAMPLE_RATE);
                ok = send_all(c->fd, hdr, wn) == 0;
            }
            if (ok && c->acc_len > 0)
                (void)send_all(c->fd, c->acc, c->acc_len);
            atomic_store(&c->finished, 1); /* LAST touch of c by this thread */
        }
        return 0;
    }
    if (final) {
        (void)conn_emit(c, NULL, 0); /* header even for an empty stream */
        (void)send_all(c->fd, "0\r\n\r\n", 5); /* terminal chunk */
        atomic_store(&c->finished, 1);         /* LAST touch of c by this thread */
    }
    return 0;
}

/* Submit chunk c->cur_chunk of a long-form chain (poll-loop thread only).
 * Reproducibility: an explicit seed varies per chunk (seed + k*P) so chunks
 * do not share sampler trajectories; seed 0 stays 0 (fresh RNG). */
static s2p_status submit_chunk(http_srv* s, conn* c) {
    s2p_request_text req;
    memset(&req, 0, sizeof(req));
    req.text = c->chunks[c->cur_chunk];
    if (c->lf_clone_pcm) {
        req.ref_pcm = c->lf_clone_pcm;
        req.ref_pcm_n = c->lf_clone_n;
        req.ref_text = c->lf_clone_text;
    } else if (c->lf_voice) {
        req.refs = &c->lf_voice->part;
        req.n_refs = 1;
        req.ref_text = c->lf_voice->transcript;
    }
    s2p_sampling_cfg sampling = c->lf_sampling;
    if (sampling.seed != 0) {
        sampling.seed += 1000003ull * (uint64_t)c->cur_chunk;
        if (sampling.seed == 0) sampling.seed = 1;
    }
    return s2p_sched_submit(s->sched, &req, &sampling, tts_audio_cb, c,
                            &c->req_id);
}

/* ------------------------------------------------------------- conn mgmt */

static void conn_reset(conn* c) {
    if (c->fd >= 0) close(c->fd);
    free(c->buf);
    free(c->acc);
    s2p_text_chunks_free(c->chunks, c->n_chunks);
    free(c->lf_clone_pcm);
    free(c->lf_clone_text);
    free(c->gap_hold);
    memset(c, 0, sizeof(*c));
    c->fd = -1;
    c->st = C_FREE;
}

static conn* conn_alloc(http_srv* s, int fd) {
    for (int i = 0; i < s->max_conns; i++) {
        if (s->conns[i].st == C_FREE) {
            conn* c = &s->conns[i];
            memset(c, 0, sizeof(*c));
            c->fd = fd;
            c->st = C_READ_HEAD;
            c->content_len = -1;
            c->last_activity = time(NULL);
            atomic_store(&c->finished, 0);
            return c;
        }
    }
    return NULL;
}

static int conn_append(conn* c, const char* data, size_t n) {
    if (c->len + n + 1 > c->cap) {
        size_t nc = c->cap ? c->cap * 2 : 4096;
        while (nc < c->len + n + 1) nc *= 2;
        if (nc > HTTP_MAX_HEAD + HTTP_MAX_BODY + 4096) return -1;
        char* nb = (char*)realloc(c->buf, nc);
        if (!nb) return -1;
        c->buf = nb;
        c->cap = nc;
    }
    memcpy(c->buf + c->len, data, n);
    c->len += n;
    c->buf[c->len] = '\0';
    return 0;
}

/* ---------------------------------------------------------- HTTP parsing */

/* Case-insensitive header lookup inside [buf, buf+head_end). Returns a
 * pointer to the value (not NUL-terminated) and its length, or NULL. */
static const char* find_header(const conn* c, const char* name, size_t* vlen) {
    size_t nlen = strlen(name);
    const char* p = c->buf;
    const char* end = c->buf + c->head_end;
    /* skip request line */
    const char* nl = memchr(p, '\n', (size_t)(end - p));
    if (!nl) return NULL;
    p = nl + 1;
    while (p < end) {
        nl = memchr(p, '\n', (size_t)(end - p));
        size_t line = nl ? (size_t)(nl - p) : (size_t)(end - p);
        if (line > nlen && p[nlen] == ':' && strncasecmp(p, name, nlen) == 0) {
            const char* v = p + nlen + 1;
            const char* ve = p + line;
            while (v < ve && (*v == ' ' || *v == '\t')) v++;
            while (ve > v && (ve[-1] == '\r' || ve[-1] == ' ')) ve--;
            *vlen = (size_t)(ve - v);
            return v;
        }
        if (!nl) break;
        p = nl + 1;
    }
    return NULL;
}

static int auth_ok(http_srv* s, conn* c) {
    if (!s->auth_token || !s->auth_token[0]) return 1;
    size_t vlen = 0;
    const char* v = find_header(c, "Authorization", &vlen);
    if (!v || vlen < 8 || strncasecmp(v, "Bearer ", 7) != 0) return 0;
    v += 7;
    vlen -= 7;
    size_t tlen = strlen(s->auth_token);
    if (vlen != tlen) return 0;
    /* constant-time compare */
    unsigned char diff = 0;
    for (size_t i = 0; i < tlen; i++)
        diff |= (unsigned char)(v[i] ^ s->auth_token[i]);
    return diff == 0;
}

static void route_healthz(http_srv* s, conn* c) {
    s2p_sched_stats st;
    memset(&st, 0, sizeof(st));
    s2p_sched_get_stats(s->sched, &st);
    char body[256];
    snprintf(body, sizeof(body),
             "{\"status\":\"ok\",\"active_sessions\":%d,\"queued\":%d,"
             "\"frames_decoded\":%llu,\"requests_done\":%llu}",
             st.active_sessions, st.queued,
             (unsigned long long)st.frames_decoded,
             (unsigned long long)st.requests_done);
    send_simple(c->fd, 200, "OK", "application/json", body);
    conn_reset(c);
}

static void route_voices(http_srv* s, conn* c) {
    char body[4096];
    size_t o = 0;
    o += (size_t)snprintf(body + o, sizeof(body) - o, "{\"voices\":[");
    int n = s2p_voices_count(s->voices);
    for (int i = 0; i < n && o < sizeof(body) - 128; i++) {
        const s2p_voice* v = s2p_voices_at(s->voices, i);
        o += (size_t)snprintf(body + o, sizeof(body) - o,
                              "%s{\"name\":\"%s\",\"duration_s\":%.1f,"
                              "\"reference_frames\":%d}",
                              i ? "," : "", v->name, v->duration_s, v->part.T);
    }
    snprintf(body + o, sizeof(body) - o, "]}");
    send_simple(c->fd, 200, "OK", "application/json", body);
    conn_reset(c);
}

static void route_tts(http_srv* s, conn* c) {
    if (!auth_ok(s, c)) {
        send_simple(c->fd, 401, "Unauthorized", "application/json",
                    "{\"error\":\"unauthorized\"}");
        conn_reset(c);
        return;
    }
    const char* body = c->buf + c->head_end;
    size_t blen = c->len - c->head_end;

    s2p_json* j = NULL;
    if (s2p_json_parse(body, blen, &j) != S2P_OK) {
        send_simple(c->fd, 400, "Bad Request", "application/json",
                    "{\"error\":\"invalid json body\"}");
        conn_reset(c);
        return;
    }
    const s2p_jval* root = s2p_json_root(j);
    const s2p_jval* jtext = s2p_jobj_get(root, "text");
    if (!jtext || !s2p_jis_str(jtext)) {
        s2p_json_free(j);
        send_simple(c->fd, 400, "Bad Request", "application/json",
                    "{\"error\":\"missing required string field: text\"}");
        conn_reset(c);
        return;
    }
    size_t tlen = 0;
    const char* text = s2p_jstr(jtext, &tlen);
    char* text_c = (char*)malloc(tlen + 1);
    if (!text_c) {
        s2p_json_free(j);
        send_simple(c->fd, 500, "Internal Server Error", "application/json",
                    "{\"error\":\"oom\"}");
        conn_reset(c);
        return;
    }
    memcpy(text_c, text, tlen);
    text_c[tlen] = '\0';

    int wav = 1;
    const s2p_jval* jfmt = s2p_jobj_get(root, "format");
    if (jfmt && s2p_jis_str(jfmt)) {
        size_t fl = 0;
        const char* f = s2p_jstr(jfmt, &fl);
        if (fl == 3 && strncmp(f, "pcm", 3) == 0) wav = 0;
        else if (fl == 3 && strncmp(f, "wav", 3) == 0) wav = 1;
        else {
            free(text_c);
            s2p_json_free(j);
            send_simple(c->fd, 400, "Bad Request", "application/json",
                        "{\"error\":\"format must be \\\"wav\\\" or "
                        "\\\"pcm\\\"\"}");
            conn_reset(c);
            return;
        }
    }

    s2p_sampling_cfg sampling = s2p_sampling_defaults();
    const s2p_jval* v;
    if ((v = s2p_jobj_get(root, "temperature")) != NULL && !s2p_jis_null(v))
        sampling.temperature = (float)s2p_jnum(v);
    if ((v = s2p_jobj_get(root, "top_p")) != NULL && !s2p_jis_null(v))
        sampling.top_p = (float)s2p_jnum(v);
    if ((v = s2p_jobj_get(root, "seed")) != NULL && !s2p_jis_null(v))
        sampling.seed = (uint64_t)s2p_jint(v);

    int buffered = 0;
    if ((v = s2p_jobj_get(root, "stream")) != NULL && !s2p_jis_null(v))
        buffered = !s2p_jbool(v);

    int chunk_target = HTTP_CHUNK_BYTES_DEFAULT;
    int chunk_sentences = HTTP_CHUNK_SENTENCES_DEFAULT;
    {
        const char* e = getenv("S2P_CHUNK_BYTES");
        if (e && e[0]) chunk_target = atoi(e);
        e = getenv("S2P_CHUNK_SENTENCES");
        if (e && e[0]) chunk_sentences = atoi(e);
    }
    if ((v = s2p_jobj_get(root, "chunk_length")) != NULL && !s2p_jis_null(v))
        chunk_target = (int)s2p_jint(v);
    if ((v = s2p_jobj_get(root, "chunk_sentences")) != NULL &&
        !s2p_jis_null(v))
        chunk_sentences = (int)s2p_jint(v);
    int chunk_gap_ms = HTTP_CHUNK_GAP_MS_DEFAULT;
    {
        const char* e = getenv("S2P_CHUNK_GAP_MS");
        if (e && e[0]) chunk_gap_ms = atoi(e);
    }
    if ((v = s2p_jobj_get(root, "chunk_gap_ms")) != NULL && !s2p_jis_null(v))
        chunk_gap_ms = (int)s2p_jint(v);
    if (chunk_gap_ms < 0) chunk_gap_ms = 0;
    if (chunk_gap_ms > 10000) chunk_gap_ms = 10000;

    /* voice selection / on-the-fly cloning (all errors here are pre-header,
     * so clients still get proper JSON error responses) */
    const s2p_voice* named = NULL;
    float*           clone_pcm = NULL;
    int64_t          clone_n = 0;
    char*            clone_text_c = NULL;
#define TTS_FAIL(msg)                                                          \
    do {                                                                       \
        free(text_c);                                                          \
        free(clone_pcm);                                                       \
        free(clone_text_c);                                                    \
        s2p_json_free(j);                                                      \
        send_simple(c->fd, 400, "Bad Request", "application/json", (msg));    \
        conn_reset(c);                                                         \
        return;                                                                \
    } while (0)

    if ((v = s2p_jobj_get(root, "reference_audio_b64")) != NULL &&
        !s2p_jis_null(v)) {
        if (!s2p_jis_str(v)) TTS_FAIL("{\"error\":\"reference_audio_b64 must "
                                      "be a base64 string\"}");
        size_t b64l = 0;
        const char* b64 = s2p_jstr(v, &b64l);
        size_t   wl = 0;
        uint8_t* wbuf = b64_decode(b64, b64l, &wl);
        if (!wbuf) TTS_FAIL("{\"error\":\"invalid base64\"}");
        s2p_status prc = s2p_wav_parse_f32(wbuf, wl, &clone_pcm, &clone_n);
        free(wbuf);
        if (prc != S2P_OK)
            TTS_FAIL("{\"error\":\"reference audio must be a RIFF wav, "
                     "16-bit PCM mono 44100 Hz\"}");
        if (clone_n > HTTP_CLONE_MAX_SAMPLES)
            TTS_FAIL("{\"error\":\"reference audio longer than 15 s\"}");
        const s2p_jval* jt = s2p_jobj_get(root, "reference_text");
        if (!jt || !s2p_jis_str(jt))
            TTS_FAIL("{\"error\":\"reference_text (exact transcript) is "
                     "required with reference_audio_b64\"}");
        size_t rl = 0;
        const char* rt = s2p_jstr(jt, &rl);
        clone_text_c = (char*)malloc(rl + 1);
        if (!clone_text_c) TTS_FAIL("{\"error\":\"oom\"}");
        memcpy(clone_text_c, rt, rl);
        clone_text_c[rl] = '\0';
    } else if ((v = s2p_jobj_get(root, "voice")) != NULL && !s2p_jis_null(v)) {
        if (!s2p_jis_str(v))
            TTS_FAIL("{\"error\":\"voice must be a string\"}");
        size_t nl = 0;
        const char* nm = s2p_jstr(v, &nl);
        char name[128];
        if (nl >= sizeof(name)) TTS_FAIL("{\"error\":\"unknown voice\"}");
        memcpy(name, nm, nl);
        name[nl] = '\0';
        named = s2p_voices_find(s->voices, name);
        if (!named)
            TTS_FAIL("{\"error\":\"unknown voice (see GET /v1/voices)\"}");
    }
#undef TTS_FAIL
    s2p_json_free(j);

    /* Long-form chunking: only when a reference pins the voice (zero-shot
     * chunks would each draw a new voice), and only when the split actually
     * yields more than one chunk. */
    if ((named != NULL || clone_pcm != NULL) && chunk_target >= 32) {
        char** chunks = NULL;
        int    n = 0;
        if (s2p_text_chunks(text_c, chunk_target, chunk_sentences, &chunks,
                            &n) == S2P_OK) {
            if (n > 1) {
                c->chunks = chunks;
                c->n_chunks = n;
                c->cur_chunk = 0;
                c->lf_voice = named;
                c->lf_clone_pcm = clone_pcm;   /* ownership -> conn */
                c->lf_clone_n = clone_n;
                c->lf_clone_text = clone_text_c;
                clone_pcm = NULL;
                clone_text_c = NULL;
                c->gap_ms = chunk_gap_ms;
                c->gap_hold_n = 0;
                c->gap_lead_skip = 0;
                if (chunk_gap_ms > 0) {
                    c->gap_hold = (int16_t*)malloc(GAP_HOLD_SAMPLES *
                                                   sizeof(int16_t));
                    /* alloc failure: filter off, raw concatenation */
                }
            } else {
                s2p_text_chunks_free(chunks, n);
            }
        }
    }

    s2p_request_text req;
    memset(&req, 0, sizeof(req));
    req.text = text_c;
    if (clone_pcm != NULL) {
        req.ref_pcm = clone_pcm;
        req.ref_pcm_n = clone_n;
        req.ref_text = clone_text_c;
    } else if (named != NULL) {
        req.refs = &named->part;
        req.n_refs = 1;
        req.ref_text = named->transcript;
    }

    c->wav = wav;
    c->sent_wav_hdr = 0;
    c->buffered = buffered;
    atomic_store(&c->finished, 0);
    atomic_store(&c->chunk_advance, 0);

    if (!buffered) {
        /* Response headers BEFORE submit so the cb can stream immediately.
         * If generation later fails, the chunked stream just ends early. */
        char head[256];
        int hn = snprintf(head, sizeof(head),
                          "HTTP/1.1 200 OK\r\n"
                          "Content-Type: %s\r\n"
                          "Transfer-Encoding: chunked\r\n"
                          "Cache-Control: no-store\r\n"
                          "Connection: close\r\n\r\n",
                          wav ? "audio/wav" : "application/octet-stream");
        if (hn <= 0 || send_all(c->fd, head, (size_t)hn) != 0) {
            free(text_c);
            free(clone_pcm);
            free(clone_text_c);
            conn_reset(c);
            return;
        }
    } /* buffered: nothing on the wire until the final callback */

    s2p_status rc;
    if (c->n_chunks > 1) {
        c->lf_sampling = sampling;
        rc = submit_chunk(s, c);
    } else {
        uint64_t req_id = 0;
        rc = s2p_sched_submit(s->sched, &req, &sampling, tts_audio_cb, c,
                              &req_id);
        c->req_id = req_id;
    }
    free(text_c); /* scheduler deep-copies everything */
    free(clone_pcm);
    free(clone_text_c);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] http: submit failed (%d)\n", (int)rc);
        if (buffered)
            send_simple(c->fd, 503, "Service Unavailable", "application/json",
                        "{\"error\":\"submit failed\"}");
        else /* headers already went out; end the stream with zero audio */
            (void)send_all(c->fd, "0\r\n\r\n", 5);
        conn_reset(c);
        return;
    }
    c->st = C_STREAMING;
}

static void dispatch(http_srv* s, conn* c) {
    /* Request line: METHOD SP PATH SP HTTP/1.x */
    char method[8] = {0}, path[256] = {0};
    if (sscanf(c->buf, "%7s %255s", method, path) != 2) {
        send_simple(c->fd, 400, "Bad Request", "text/plain", "bad request\n");
        conn_reset(c);
        return;
    }
    char* q = strchr(path, '?');
    if (q) *q = '\0';

    if (strcmp(method, "GET") == 0 && strcmp(path, "/healthz") == 0) {
        route_healthz(s, c);
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/voices") == 0) {
        route_voices(s, c);
    } else if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/tts") == 0) {
        route_tts(s, c);
    } else if (strcmp(path, "/healthz") == 0 || strcmp(path, "/v1/tts") == 0 ||
               strcmp(path, "/v1/voices") == 0) {
        send_simple(c->fd, 405, "Method Not Allowed", "text/plain",
                    "method not allowed\n");
        conn_reset(c);
    } else {
        send_simple(c->fd, 404, "Not Found", "text/plain", "not found\n");
        conn_reset(c);
    }
}

/* Feed newly read bytes through the header/body state machine. */
static void advance_conn(http_srv* s, conn* c) {
    if (c->st == C_READ_HEAD) {
        char* he = strstr(c->buf, "\r\n\r\n");
        if (!he) {
            if (c->len > HTTP_MAX_HEAD) {
                send_simple(c->fd, 431, "Request Header Fields Too Large",
                            "text/plain", "headers too large\n");
                conn_reset(c);
            }
            return;
        }
        c->head_end = (size_t)(he - c->buf) + 4;
        size_t vlen = 0;
        const char* v = find_header(c, "Content-Length", &vlen);
        c->content_len = 0;
        if (v) {
            char tmp[24] = {0};
            if (vlen >= sizeof(tmp)) vlen = sizeof(tmp) - 1;
            memcpy(tmp, v, vlen);
            c->content_len = strtol(tmp, NULL, 10);
        }
        if (c->content_len < 0 || c->content_len > HTTP_MAX_BODY) {
            send_simple(c->fd, 413, "Payload Too Large", "text/plain",
                        "body too large\n");
            conn_reset(c);
            return;
        }
        /* curl sends Expect: 100-continue for larger bodies. */
        v = find_header(c, "Expect", &vlen);
        if (v && vlen >= 12 && strncasecmp(v, "100-continue", 12) == 0)
            (void)send_all(c->fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
        c->st = C_READ_BODY;
    }
    if (c->st == C_READ_BODY) {
        if (c->len - c->head_end < (size_t)c->content_len) return;
        dispatch(s, c);
    }
}

/* ------------------------------------------------------------- main loop */

s2p_status s2p_server_run(s2p_sched* sched, const s2p_server_opts* opts) {
    if (!sched) return S2P_ERR_INVALID;
    int port = (opts && opts->port > 0) ? opts->port : 8010;
    const char* bind_addr =
        (opts && opts->bind_addr) ? opts->bind_addr : "127.0.0.1";
    int max_conns = (opts && opts->max_conns > 0) ? opts->max_conns : 64;

    http_srv srv;
    memset(&srv, 0, sizeof(srv));
    srv.sched = sched;
    srv.voices = opts ? opts->voices : NULL;
    srv.auth_token = opts ? opts->auth_token : NULL;
    srv.max_conns = max_conns;
    srv.conns = (conn*)calloc((size_t)max_conns, sizeof(conn));
    if (!srv.conns) return S2P_ERR_OOM;
    for (int i = 0; i < max_conns; i++) srv.conns[i].fd = -1;

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = http_sig_handler; /* no SA_RESTART: poll must EINTR */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    srv.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv.listen_fd < 0) {
        perror("[s2pro] http: socket");
        free(srv.conns);
        return S2P_ERR_IO;
    }
    int one = 1;
    setsockopt(srv.listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, bind_addr, &addr.sin_addr) != 1) {
        fprintf(stderr, "[s2pro] http: bad bind address '%s'\n", bind_addr);
        close(srv.listen_fd);
        free(srv.conns);
        return S2P_ERR_INVALID;
    }
    if (bind(srv.listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(srv.listen_fd, 64) < 0) {
        perror("[s2pro] http: bind/listen");
        close(srv.listen_fd);
        free(srv.conns);
        return S2P_ERR_IO;
    }
    fprintf(stderr, "[s2pro] http: listening on %s:%d%s\n", bind_addr, port,
            (srv.auth_token && srv.auth_token[0]) ? " (bearer auth)" : "");

    struct pollfd* pfds =
        (struct pollfd*)calloc((size_t)max_conns + 1, sizeof(struct pollfd));
    if (!pfds) {
        close(srv.listen_fd);
        free(srv.conns);
        return S2P_ERR_OOM;
    }

    char rbuf[16384];
    while (!g_http_stop) {
        int np = 0;
        pfds[np].fd = srv.listen_fd;
        pfds[np].events = POLLIN;
        np++;
        for (int i = 0; i < max_conns; i++) {
            conn* c = &srv.conns[i];
            if (c->st == C_FREE) continue;
            if (c->st == C_STREAMING && atomic_load(&c->finished)) {
                conn_reset(c); /* final chunk delivered; close */
                continue;
            }
            if (c->st == C_STREAMING && atomic_load(&c->chunk_advance)) {
                /* previous long-form chunk finished; submit the next */
                atomic_store(&c->chunk_advance, 0);
                c->cur_chunk++;
                s2p_status crc = submit_chunk(&srv, c);
                if (crc != S2P_OK) {
                    fprintf(stderr,
                            "[s2pro] http: chunk %d/%d submit failed (%d)\n",
                            c->cur_chunk + 1, c->n_chunks, (int)crc);
                    if (c->buffered)
                        send_simple(c->fd, 503, "Service Unavailable",
                                    "application/json",
                                    "{\"error\":\"submit failed\"}");
                    else
                        (void)send_all(c->fd, "0\r\n\r\n", 5);
                    conn_reset(c);
                    continue;
                }
            }
            pfds[np].fd = c->fd;
            /* Streaming conns are watched for hangup only (POLLIN fires on
             * FIN too); the scheduler thread owns the writes. */
            pfds[np].events = POLLIN;
            pfds[np].revents = 0;
            np++;
        }

        int pr = poll(pfds, (nfds_t)np, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            perror("[s2pro] http: poll");
            break;
        }

        /* New connections. */
        if (pfds[0].revents & POLLIN) {
            int fd = accept(srv.listen_fd, NULL, NULL);
            if (fd >= 0) {
                conn* c = conn_alloc(&srv, fd);
                if (!c) {
                    send_simple(fd, 503, "Service Unavailable", "text/plain",
                                "too many connections\n");
                    close(fd);
                } else {
                    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one,
                               sizeof(one));
                    struct timeval tv = { HTTP_SEND_SECS, 0 };
                    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
                }
            }
        }

        time_t now = time(NULL);
        for (int p = 1; p < np; p++) {
            /* Map fd back to its conn (slots may have moved states). */
            conn* c = NULL;
            for (int i = 0; i < max_conns; i++)
                if (srv.conns[i].st != C_FREE && srv.conns[i].fd == pfds[p].fd) {
                    c = &srv.conns[i];
                    break;
                }
            if (!c) continue;

            if (c->st == C_STREAMING) {
                if (pfds[p].revents & (POLLIN | POLLERR | POLLHUP)) {
                    char tmp[512];
                    ssize_t r = recv(c->fd, tmp, sizeof(tmp), MSG_DONTWAIT);
                    if (r == 0 || (r < 0 && errno != EAGAIN &&
                                   errno != EWOULDBLOCK && errno != EINTR)) {
                        /* Client gone. cancel() guarantees the worker is out
                         * of the callback before we close the fd. */
                        (void)s2p_sched_cancel(srv.sched, c->req_id);
                        conn_reset(c);
                    }
                    /* r > 0: stray request bytes on a streaming conn — drop. */
                }
                continue;
            }

            if (pfds[p].revents & (POLLERR | POLLHUP)) {
                conn_reset(c);
                continue;
            }
            if (pfds[p].revents & POLLIN) {
                ssize_t r = recv(c->fd, rbuf, sizeof(rbuf), 0);
                if (r <= 0) {
                    if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
                    conn_reset(c);
                    continue;
                }
                c->last_activity = now;
                if (conn_append(c, rbuf, (size_t)r) != 0) {
                    send_simple(c->fd, 413, "Payload Too Large", "text/plain",
                                "request too large\n");
                    conn_reset(c);
                    continue;
                }
                advance_conn(&srv, c);
            } else if (c->st != C_FREE &&
                       now - c->last_activity > HTTP_IDLE_SECS) {
                send_simple(c->fd, 408, "Request Timeout", "text/plain",
                            "timeout\n");
                conn_reset(c);
            }
        }
    }

    fprintf(stderr, "[s2pro] http: shutting down\n");
    close(srv.listen_fd);
    for (int i = 0; i < max_conns; i++) {
        conn* c = &srv.conns[i];
        if (c->st == C_STREAMING)
            (void)s2p_sched_cancel(srv.sched, c->req_id);
        if (c->st != C_FREE) conn_reset(c);
    }
    free(pfds);
    free(srv.conns);
    return S2P_OK;
}
