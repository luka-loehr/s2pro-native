/* s2pro-native — sentence-aware text chunker for long-form synthesis.
 *
 * Long single-shot AR generation flattens prosodically: measured on 104 s
 * takes, punctuation pauses per 10 s bucket decay from ~0.5-1.0 s early to
 * ~0-0.2 s after ~40 s at EVERY weight precision (INT8 == INT4), i.e. it is
 * a property of the growing self-generated context, not of quantization.
 * The reference stack serves long text in chunks for the same reason. Each
 * chunk restarts generation close to the (energetic) voice reference, which
 * is exactly the horizon where takes are prosodically at their best.
 *
 * Splitting rules: sentences end at . ! ? … ؟ 。 ！ ？ followed by
 * whitespace/end (trailing closing quotes/brackets stay with the sentence);
 * a blank line is always a boundary. Sentences pack greedily into chunks of
 * at most `target_bytes`; a chunk always holds at least one sentence, and a
 * single oversize sentence is split at the last space before the limit
 * (never inside a UTF-8 sequence). ASCII '.' between digits (3.14) or
 * inside [bracket] control tags does not end a sentence.
 */
#include <stdlib.h>
#include <string.h>

#include "s2pro/status.h"
#include "s2pro/tokenizer.h"

static int is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Byte length of the UTF-8 sequence starting at p (1 on malformed). */
static int u8_len(const unsigned char* p) {
    if (p[0] < 0x80) return 1;
    if ((p[0] & 0xE0) == 0xC0) return 2;
    if ((p[0] & 0xF0) == 0xE0) return 3;
    if ((p[0] & 0xF8) == 0xF0) return 4;
    return 1;
}

/* Does the sequence at p end a sentence? Returns its byte length if so. */
static int end_punct(const unsigned char* p) {
    if (p[0] == '.' || p[0] == '!' || p[0] == '?') return 1;
    if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0xA6) return 3; /* … */
    if (p[0] == 0xD8 && p[1] == 0x9F) return 2;                 /* ؟ */
    if (p[0] == 0xE3 && p[1] == 0x80 && p[2] == 0x82) return 3; /* 。 */
    if (p[0] == 0xEF && p[1] == 0xBC && (p[2] == 0x81 || p[2] == 0x9F))
        return 3;                                               /* ！ ？ */
    return 0;
}

typedef struct {
    char** v;
    int    n, cap;
} strvec;

static int push(strvec* s, const char* start, size_t len) {
    /* trim surrounding whitespace */
    while (len > 0 && is_space((unsigned char)start[0])) { start++; len--; }
    while (len > 0 && is_space((unsigned char)start[len - 1])) len--;
    if (len == 0) return 0;
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 8;
        char** nv = (char**)realloc(s->v, (size_t)nc * sizeof(char*));
        if (!nv) return -1;
        s->v = nv;
        s->cap = nc;
    }
    char* d = (char*)malloc(len + 1);
    if (!d) return -1;
    memcpy(d, start, len);
    d[len] = '\0';
    s->v[s->n++] = d;
    return 0;
}

/* Split into sentences (each including its terminator + trailing quotes). */
static s2p_status sentences(const char* text, strvec* out) {
    const unsigned char* p = (const unsigned char*)text;
    size_t n = strlen(text);
    size_t start = 0, i = 0;
    int in_bracket = 0;
    while (i < n) {
        const unsigned char* c = p + i;
        if (*c == '[') in_bracket = 1;
        if (*c == ']') in_bracket = 0;
        /* blank line = hard boundary */
        if (*c == '\n') {
            size_t j = i + 1;
            while (j < n && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r')) j++;
            if (j < n && p[j] == '\n') {
                if (push(out, text + start, i - start) != 0)
                    return S2P_ERR_OOM;
                start = j + 1;
                i = j + 1;
                in_bracket = 0;
                continue;
            }
        }
        int el = in_bracket ? 0 : end_punct(c);
        if (el > 0) {
            /* '.' between digits: 3.14 — not a boundary */
            if (*c == '.' && i > 0 && i + 1 < n && p[i - 1] >= '0' &&
                p[i - 1] <= '9' && p[i + 1] >= '0' && p[i + 1] <= '9') {
                i += 1;
                continue;
            }
            size_t j = i + el;
            /* keep runs of terminators (?! ...) and closing marks */
            while (j < n) {
                int e2 = end_punct(p + j);
                if (e2 > 0) { j += (size_t)e2; continue; }
                if (p[j] == '"' || p[j] == '\'' || p[j] == ')' || p[j] == ']')
                    { j++; continue; }
                if (p[j] == 0xC2 && j + 1 < n && p[j + 1] == 0xBB)
                    { j += 2; continue; } /* » */
                if (p[j] == 0xE2 && j + 2 < n && p[j + 1] == 0x80 &&
                    (p[j + 2] == 0x9C || p[j + 2] == 0x9D))
                    { j += 3; continue; } /* “ ” */
                break;
            }
            /* boundary only when followed by whitespace or end */
            if (j >= n || is_space(p[j])) {
                if (push(out, text + start, j - start) != 0)
                    return S2P_ERR_OOM;
                start = j;
                i = j;
                continue;
            }
            i = j;
            continue;
        }
        i += (size_t)u8_len(c);
    }
    if (start < n && push(out, text + start, n - start) != 0)
        return S2P_ERR_OOM;
    return S2P_OK;
}

/* Hard-split an oversize sentence: prefer the last comma before
 * target_bytes (the v1.5.1 reference splitter's fallback hierarchy:
 * sentence -> comma -> whitespace -> hard cut), then the last space, then
 * a UTF-8 boundary. */
static s2p_status push_hard(strvec* out, const char* s, size_t len,
                            size_t target) {
    while (len > target) {
        size_t cut = 0;
        for (size_t k = 1; k <= target; k++) { /* last comma, kept left */
            if (s[k - 1] == ',') cut = k;
            else if ((unsigned char)s[k - 1] == 0xEF && k + 1 < len &&
                     (unsigned char)s[k] == 0xBC &&
                     (unsigned char)s[k + 1] == 0x8C)
                cut = k + 2; /* ， */
        }
        if (cut == 0) {
            cut = target;
            while (cut > 0 && !is_space((unsigned char)s[cut])) cut--;
        }
        if (cut == 0) {
            cut = target; /* no comma/space: cut at a UTF-8 boundary */
            while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) cut--;
            if (cut == 0) cut = len; /* degenerate; emit as-is */
        }
        if (push(out, s, cut) != 0) return S2P_ERR_OOM;
        s += cut;
        len -= cut;
        while (len > 0 && is_space((unsigned char)s[0])) { s++; len--; }
    }
    if (len > 0 && push(out, s, len) != 0) return S2P_ERR_OOM;
    return S2P_OK;
}

s2p_status s2p_text_chunks(const char* utf8, int target_bytes,
                           int max_sentences, char*** out, int* out_n) {
    if (!utf8 || !out || !out_n || target_bytes < 32) return S2P_ERR_INVALID;
    *out = NULL;
    *out_n = 0;

    strvec sen = {0}, chunks = {0};
    char* cur = NULL; /* before the first goto: done frees it */
    size_t cur_len = 0, cur_cap = 0;
    int cur_sent = 0;
    s2p_status rc = sentences(utf8, &sen);
    if (rc != S2P_OK) goto done;

    /* greedy pack: a chunk closes at target_bytes or (when max_sentences
     * > 0) after that many sentences, whichever comes first */
    for (int i = 0; i < sen.n; i++) {
        size_t sl = strlen(sen.v[i]);
        if (cur_len > 0 &&
            (cur_len + 1 + sl > (size_t)target_bytes ||
             (max_sentences > 0 && cur_sent >= max_sentences))) {
            rc = push(&chunks, cur, cur_len) == 0 ? S2P_OK : S2P_ERR_OOM;
            cur_len = 0;
            cur_sent = 0;
            if (rc != S2P_OK) goto done;
        }
        if (sl > (size_t)target_bytes && cur_len == 0) {
            rc = push_hard(&chunks, sen.v[i], sl, (size_t)target_bytes);
            if (rc != S2P_OK) goto done;
            continue;
        }
        if (cur_len + 1 + sl + 1 > cur_cap) {
            size_t nc = cur_cap ? cur_cap * 2 : 512;
            while (nc < cur_len + 1 + sl + 1) nc *= 2;
            char* nb = (char*)realloc(cur, nc);
            if (!nb) { rc = S2P_ERR_OOM; goto done; }
            cur = nb;
            cur_cap = nc;
        }
        if (cur_len > 0) cur[cur_len++] = ' ';
        memcpy(cur + cur_len, sen.v[i], sl);
        cur_len += sl;
        cur_sent++;
    }
    if (cur_len > 0) {
        rc = push(&chunks, cur, cur_len) == 0 ? S2P_OK : S2P_ERR_OOM;
        if (rc != S2P_OK) goto done;
    }

    *out = chunks.v;
    *out_n = chunks.n;
    chunks.v = NULL;
    chunks.n = 0;
    rc = S2P_OK;

done:
    free(cur);
    for (int i = 0; i < sen.n; i++) free(sen.v[i]);
    free(sen.v);
    for (int i = 0; i < chunks.n; i++) free(chunks.v[i]);
    free(chunks.v);
    return rc;
}

void s2p_text_chunks_free(char** chunks, int n) {
    if (!chunks) return;
    for (int i = 0; i < n; i++) free(chunks[i]);
    free(chunks);
}
