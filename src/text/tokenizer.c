/* s2pro-native — hand-written HF byte-level BPE tokenizer (tokenizer.json).
 *
 * Replicates the exact Qwen3/Fish pipeline per docs/PORTING.md §3:
 *   added-token atomic match (leftmost-longest, on RAW text — all added
 *   tokens are normalized:false) → NFC normalize per remaining segment →
 *   Split on the isolated regex
 *     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
 *     ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *   → ByteLevel(add_prefix_space=false, use_regex=false) → BPE
 *   (151387 merges, no byte_fallback, no ignore_merges). No automatic
 *   BOS/EOS; ByteLevel decode.
 *
 * Implementation notes:
 *   - Vocab keys / merge operands are stored as RAW BYTES (the GPT-2
 *     byte-to-unicode map is inverted at load), so BPE runs directly on the
 *     UTF-8 bytes of each pretoken; initial symbols are single bytes.
 *   - Merges are keyed by (left_id, right_id) — every operand and result is
 *     itself a vocab entry (guaranteed by the HF BPE format).
 *   - The merge loop is the leftmost-min-rank scan (equivalent to the HF
 *     queue algorithm); pretokens are short so the scan is cheap.
 *   - NFC is implemented in full: canonical decomposition (generated full-NFD
 *     expansion table + algorithmic Hangul), canonical reorder (ccc
 *     stable-sort), canonical composition (generated pair table + Hangul
 *     LV/LVT). Fuzz-verified bit-exact against HF tokenizers on random
 *     codepoint soup including raw combining-mark clusters.
 *   - Unicode class tables (\p{L}, \p{N}, White_Space, ccc, NFD, comp)
 *     are generated into unicode_tables.h (UCD 16).
 *
 * Selftest: compile with -DS2P_TOK_SELFTEST (links prompt.c + core json):
 *   gcc -DS2P_TOK_SELFTEST -O2 -std=c11 -Iinclude \
 *       src/text/tokenizer.c src/text/prompt.c src/core/json.c -o tok_test
 *   ./tok_test model    # prints ids per fixed string, diffable vs HF
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s2pro/tokenizer.h"
#include "s2pro/json.h"
#include "unicode_tables.h"

/* ------------------------------------------------------------ containers */

typedef struct { /* bytes -> id (open addressing, FNV-1a) */
    int32_t  id;   /* -1 = empty */
    uint32_t off;  /* into arena */
    uint32_t len;
} tk_bent;

typedef struct { /* (left_id<<32|right_id) -> rank, merged id */
    uint64_t key;  /* UINT64_MAX = empty */
    int32_t  rank;
    int32_t  merged;
} tk_ment;

struct s2p_tok {
    uint8_t* arena;
    size_t   arena_len, arena_cap;
    /* id -> raw bytes (BPE tokens decoded, added tokens literal) */
    uint32_t* id_off;
    uint32_t* id_len;
    /* hashes */
    tk_bent* vocab;  uint32_t vocab_cap;
    tk_ment* merges; uint32_t merge_cap;
    tk_bent* added;  uint32_t added_cap;
    int32_t  byte2id[256];
    int      added_min, added_max; /* content byte lengths */
};

static uint64_t tk_fnv(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static s2p_status tk_arena_push(s2p_tok* t, const uint8_t* p, size_t n,
                                uint32_t* off) {
    if (t->arena_len + n > t->arena_cap) {
        size_t cap = t->arena_cap ? t->arena_cap * 2 : (1u << 20);
        while (cap < t->arena_len + n) cap *= 2;
        uint8_t* a = (uint8_t*)realloc(t->arena, cap);
        if (!a) return S2P_ERR_OOM;
        t->arena = a;
        t->arena_cap = cap;
    }
    memcpy(t->arena + t->arena_len, p, n);
    *off = (uint32_t)t->arena_len;
    t->arena_len += n;
    return S2P_OK;
}

static void tk_bput(s2p_tok* t, tk_bent* tab, uint32_t cap, uint32_t off,
                    uint32_t len, int32_t id) {
    uint32_t i = (uint32_t)(tk_fnv(t->arena + off, len) & (cap - 1));
    while (tab[i].id >= 0) i = (i + 1) & (cap - 1);
    tab[i].id = id; tab[i].off = off; tab[i].len = len;
}

static int32_t tk_bget(const s2p_tok* t, const tk_bent* tab, uint32_t cap,
                       const uint8_t* p, size_t n) {
    uint32_t i = (uint32_t)(tk_fnv(p, n) & (cap - 1));
    while (tab[i].id >= 0) {
        if (tab[i].len == n && memcmp(t->arena + tab[i].off, p, n) == 0)
            return tab[i].id;
        i = (i + 1) & (cap - 1);
    }
    return -1;
}

static void tk_mput(tk_ment* tab, uint32_t cap, uint64_t key, int32_t rank,
                    int32_t merged) {
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (cap - 1);
    while (tab[i].key != UINT64_MAX) {
        if (tab[i].key == key) return; /* keep first (lowest) rank */
        i = (i + 1) & (cap - 1);
    }
    tab[i].key = key; tab[i].rank = rank; tab[i].merged = merged;
}

static const tk_ment* tk_mget(const s2p_tok* t, uint64_t key) {
    uint32_t cap = t->merge_cap;
    uint32_t i = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & (cap - 1);
    while (t->merges[i].key != UINT64_MAX) {
        if (t->merges[i].key == key) return &t->merges[i];
        i = (i + 1) & (cap - 1);
    }
    return NULL;
}

/* --------------------------------------------- GPT-2 byte-level alphabet */

/* bytes '!'..'~', 0xA1..0xAC, 0xAE..0xFF map to themselves; the other 68
 * map to 0x100+k in order. Inverse table covers cp < 0x144. */
#define TK_U2B_MAX 0x144
static void tk_build_u2b(uint16_t u2b[TK_U2B_MAX]) {
    for (int i = 0; i < TK_U2B_MAX; i++) u2b[i] = 0xFFFF;
    int k = 0;
    for (int b = 0; b < 256; b++) {
        int self = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                   (b >= 0xAE && b <= 0xFF);
        uint32_t cp = self ? (uint32_t)b : (uint32_t)(0x100 + k++);
        u2b[cp] = (uint16_t)b;
    }
}

/* Decode one UTF-8 cp; invalid input yields marker 0x110000+byte, len 1. */
static uint32_t tk_u8_next(const uint8_t* s, size_t n, size_t i, int* len) {
    uint8_t b = s[i];
    if (b < 0x80) { *len = 1; return b; }
    int need; uint32_t cp;
    if ((b & 0xE0) == 0xC0) { need = 1; cp = b & 0x1F; }
    else if ((b & 0xF0) == 0xE0) { need = 2; cp = b & 0x0F; }
    else if ((b & 0xF8) == 0xF0) { need = 3; cp = b & 0x07; }
    else { *len = 1; return 0x110000u + b; }
    if (i + (size_t)need >= n) { *len = 1; return 0x110000u + b; }
    for (int k = 1; k <= need; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *len = 1; return 0x110000u + b; }
        cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    /* reject overlong/surrogate/out-of-range */
    if ((need == 1 && cp < 0x80) || (need == 2 && cp < 0x800) ||
        (need == 3 && cp < 0x10000) || cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF)) {
        *len = 1; return 0x110000u + b;
    }
    *len = need + 1;
    return cp;
}

static int tk_u8_emit(uint32_t cp, uint8_t out[4]) {
    if (cp >= 0x110000u) { out[0] = (uint8_t)(cp - 0x110000u); return 1; }
    if (cp < 0x80) { out[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (uint8_t)(0xC0 | (cp >> 6));
        out[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (uint8_t)(0xE0 | (cp >> 12));
        out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (uint8_t)(0xF0 | (cp >> 18));
    out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (uint8_t)(0x80 | (cp & 0x3F));
    return 4;
}

/* Byte-level token string (UTF-8 of mapped cps) -> raw bytes, in place OK. */
static int tk_bytelevel_decode(const uint16_t* u2b, const uint8_t* s,
                               size_t n, uint8_t* out, size_t* out_n) {
    size_t o = 0;
    for (size_t i = 0; i < n;) {
        int len;
        uint32_t cp = tk_u8_next(s, n, i, &len);
        if (cp >= TK_U2B_MAX || u2b[cp] == 0xFFFF) return -1;
        out[o++] = (uint8_t)u2b[cp];
        i += (size_t)len;
    }
    *out_n = o;
    return 0;
}

/* ------------------------------------------------ unicode classification */

static int uc_in(const uint32_t (*tab)[2], int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < tab[mid][0]) hi = mid - 1;
        else if (cp > tab[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}

static int uc_L(uint32_t cp) {
    if (cp >= 0x110000u) return 0;
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return uc_in(s2p_uc_letter, S2P_UC_LETTER_N, cp);
}
static int uc_N(uint32_t cp) {
    if (cp >= 0x110000u) return 0;
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return uc_in(s2p_uc_number, S2P_UC_NUMBER_N, cp);
}
static int uc_ws(uint32_t cp) {
    if (cp >= 0x110000u) return 0;
    return uc_in(s2p_uc_ws, S2P_UC_WS_N, cp);
}
static int uc_ccc(uint32_t cp) {
    if (cp >= 0x110000u) return 0;
    int lo = 0, hi = S2P_UC_CCC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < s2p_uc_ccc[mid][0]) hi = mid - 1;
        else if (cp > s2p_uc_ccc[mid][1]) lo = mid + 1;
        else return (int)s2p_uc_ccc[mid][2];
    }
    return 0;
}

/* canonical composition: table pairs + algorithmic Hangul LV/LVT */
static uint32_t uc_compose(uint32_t a, uint32_t b) {
    if (a >= 0x1100 && a <= 0x1112 && b >= 0x1161 && b <= 0x1175)
        return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    if (a >= 0xAC00 && a <= 0xD7A3 && (a - 0xAC00) % 28 == 0 &&
        b >= 0x11A8 && b <= 0x11C2)
        return a + (b - 0x11A7);
    int lo = 0, hi = S2P_UC_COMP_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (a < s2p_uc_comp[mid][0] ||
            (a == s2p_uc_comp[mid][0] && b < s2p_uc_comp[mid][1]))
            hi = mid - 1;
        else if (a > s2p_uc_comp[mid][0] ||
                 (a == s2p_uc_comp[mid][0] && b > s2p_uc_comp[mid][1]))
            lo = mid + 1;
        else return s2p_uc_comp[mid][2];
    }
    return 0;
}

/* Canonical decomposition of one cp into dst (cap 4); returns count. */
static int uc_decompose(uint32_t c, uint32_t* dst) {
    if (c >= 0xAC00 && c <= 0xD7A3) { /* Hangul: algorithmic */
        uint32_t s = c - 0xAC00;
        int k = 0;
        dst[k++] = 0x1100 + s / (21 * 28);
        dst[k++] = 0x1161 + (s % (21 * 28)) / 28;
        if (s % 28) dst[k++] = 0x11A7 + s % 28;
        return k;
    }
    int lo = 0, hi = S2P_UC_DECOMP_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < s2p_uc_decomp[mid][0]) hi = mid - 1;
        else if (c > s2p_uc_decomp[mid][0]) lo = mid + 1;
        else {
            int len = (int)s2p_uc_decomp[mid][2];
            const uint32_t* p = s2p_uc_decomp_pool + s2p_uc_decomp[mid][1];
            for (int k = 0; k < len; k++) dst[k] = p[k];
            return len;
        }
    }
    dst[0] = c;
    return 1;
}

/* Full NFC (decompose + reorder + compose): src[n] -> dst (cap >= 2n+4),
 * returns output length. dst != src. */
static int tk_nfc(const uint32_t* src, int n, uint32_t* cp) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (src[i] >= 0x110000u) { cp[m++] = src[i]; continue; } /* marker */
        m += uc_decompose(src[i], cp + m);
    }
    n = m;
    /* canonical ordering: stable sort runs of ccc>0 by ccc */
    for (int i = 1; i < n; i++) {
        int ci = uc_ccc(cp[i]);
        if (ci == 0) continue;
        int j = i;
        while (j > 0) {
            int cj = uc_ccc(cp[j - 1]);
            if (cj == 0 || cj <= ci) break;
            uint32_t tmp = cp[j]; cp[j] = cp[j - 1]; cp[j - 1] = tmp;
            j--;
        }
    }
    /* canonical composition */
    int out = 0, last_starter = -1;
    for (int i = 0; i < n; i++) {
        uint32_t c = cp[i];
        int cc = uc_ccc(c);
        if (last_starter >= 0) {
            int prevcc = (out - 1 > last_starter) ? uc_ccc(cp[out - 1]) : 0;
            int blocked = (out - 1 > last_starter) && prevcc >= cc;
            if (!blocked) {
                uint32_t m = uc_compose(cp[last_starter], c);
                if (m) { cp[last_starter] = m; continue; }
            }
        }
        cp[out] = c;
        if (cc == 0) last_starter = out;
        out++;
    }
    return out;
}

/* ------------------------------------------------- pretokenizer (regex) */

/* Leftmost-first alternation with greedy backtracking, exactly the isolated
 * Split pattern (see file header). Returns match length in codepoints. */
static int tk_rx_match(const uint32_t* cp, int n, int i) {
    uint32_t c = cp[i];
    /* 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (c == '\'' && i + 1 < n) {
        uint32_t a = cp[i + 1] | ((cp[i + 1] >= 'A' && cp[i + 1] <= 'Z') ? 0x20 : 0);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (i + 2 < n) {
            uint32_t b2 = cp[i + 2] |
                          ((cp[i + 2] >= 'A' && cp[i + 2] <= 'Z') ? 0x20 : 0);
            if ((a == 'r' || a == 'v') && b2 == 'e') return 3;
            if (a == 'l' && b2 == 'l') return 3;
        }
    }
    /* 2: [^\r\n\p{L}\p{N}]?\p{L}+  (optional prefix taken greedily) */
    if (c != '\r' && c != '\n' && !uc_L(c) && !uc_N(c) && i + 1 < n &&
        uc_L(cp[i + 1])) {
        int j = i + 2;
        while (j < n && uc_L(cp[j])) j++;
        return j - i;
    }
    if (uc_L(c)) {
        int j = i + 1;
        while (j < n && uc_L(cp[j])) j++;
        return j - i;
    }
    /* 3: \p{N} */
    if (uc_N(c)) return 1;
    /* 4:  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        int j = i;
        if (cp[j] == ' ' && j + 1 < n && !uc_ws(cp[j + 1]) &&
            !uc_L(cp[j + 1]) && !uc_N(cp[j + 1]))
            j++;
        if (j < n && !uc_ws(cp[j]) && !uc_L(cp[j]) && !uc_N(cp[j])) {
            j++;
            while (j < n && !uc_ws(cp[j]) && !uc_L(cp[j]) && !uc_N(cp[j])) j++;
            while (j < n && (cp[j] == '\r' || cp[j] == '\n')) j++;
            return j - i;
        }
    }
    if (uc_ws(c)) {
        int w = i;
        while (w < n && uc_ws(cp[w])) w++;
        /* 5: \s*[\r\n]+ — greedy \s* backtracks to the LAST CR/LF in the
         * run, then [\r\n]+ takes the (single) newline there. */
        for (int k = w - 1; k >= i; k--)
            if (cp[k] == '\r' || cp[k] == '\n') return k + 1 - i;
        /* 6: \s+(?!\S) — full run at EOS, else run minus final char. */
        if (w >= n) return w - i;
        if (w - i >= 2) return w - i - 1;
        /* 7: \s+ */
        return w - i;
    }
    return 1; /* unreachable: every cp classifies above; keep as gap piece */
}

/* --------------------------------------------------------------- encoder */

typedef struct { int64_t* v; int n, cap; } tk_i64v;

static s2p_status tk_push_id(tk_i64v* o, int64_t id) {
    if (o->n == o->cap) {
        int cap = o->cap ? o->cap * 2 : 64;
        int64_t* v = (int64_t*)realloc(o->v, (size_t)cap * sizeof(int64_t));
        if (!v) return S2P_ERR_OOM;
        o->v = v; o->cap = cap;
    }
    o->v[o->n++] = id;
    return S2P_OK;
}

/* BPE over the raw bytes of one pretoken: leftmost-min-rank merge loop. */
static s2p_status tk_bpe_word(const s2p_tok* t, const uint8_t* w, int n,
                              tk_i64v* out) {
    if (n <= 0) return S2P_OK;
    int32_t stack_ids[128];
    int32_t* ids = stack_ids;
    if (n > 128) {
        ids = (int32_t*)malloc((size_t)n * sizeof(int32_t));
        if (!ids) return S2P_ERR_OOM;
    }
    for (int i = 0; i < n; i++) ids[i] = t->byte2id[w[i]];
    int cnt = n;
    for (;;) {
        int32_t best_rank = INT32_MAX, best_i = -1, best_m = -1;
        for (int i = 0; i + 1 < cnt; i++) {
            uint64_t key = ((uint64_t)(uint32_t)ids[i] << 32) |
                           (uint32_t)ids[i + 1];
            const tk_ment* m = tk_mget(t, key);
            if (m && m->rank < best_rank) {
                best_rank = m->rank; best_i = i; best_m = m->merged;
            }
        }
        if (best_i < 0) break;
        ids[best_i] = best_m;
        memmove(ids + best_i + 1, ids + best_i + 2,
                (size_t)(cnt - best_i - 2) * sizeof(int32_t));
        cnt--;
    }
    s2p_status rc = S2P_OK;
    for (int i = 0; i < cnt && rc == S2P_OK; i++)
        rc = tk_push_id(out, ids[i]);
    if (ids != stack_ids) free(ids);
    return rc;
}

/* normalize + pretokenize + BPE one raw segment (between added tokens) */
static s2p_status tk_encode_segment(const s2p_tok* t, const uint8_t* s,
                                    size_t n, tk_i64v* out) {
    if (n == 0) return S2P_OK;
    /* raw cps <= n; NFD expansion factor <= 2 per input BYTE (worst case:
     * 4-cp expansions come from >= 2-byte chars), hence the 2n+8 cap. */
    uint32_t* raw = (uint32_t*)malloc((n + 1) * sizeof(uint32_t));
    uint32_t* cp = (uint32_t*)malloc((2 * n + 8) * sizeof(uint32_t));
    uint8_t*  bytes = (uint8_t*)malloc((2 * n + 8) * 4);
    int*      off = (int*)malloc((2 * n + 9) * sizeof(int));
    if (!raw || !cp || !bytes || !off) {
        free(raw); free(cp); free(bytes); free(off);
        return S2P_ERR_OOM;
    }
    int nraw = 0;
    for (size_t i = 0; i < n;) {
        int len;
        raw[nraw++] = tk_u8_next(s, n, i, &len);
        i += (size_t)len;
    }
    int ncp = tk_nfc(raw, nraw, cp);
    free(raw);
    int nb = 0;
    for (int i = 0; i < ncp; i++) {
        off[i] = nb;
        nb += tk_u8_emit(cp[i], bytes + nb);
    }
    off[ncp] = nb;

    s2p_status rc = S2P_OK;
    int i = 0;
    while (i < ncp && rc == S2P_OK) {
        int len = tk_rx_match(cp, ncp, i);
        if (len < 1) len = 1;
        rc = tk_bpe_word(t, bytes + off[i], off[i + len] - off[i], out);
        i += len;
    }
    free(cp); free(bytes); free(off);
    return rc;
}

s2p_status s2p_tok_encode(s2p_tok* t, const char* utf8, int64_t** ids,
                          int* n) {
    if (!t || !utf8 || !ids || !n) return S2P_ERR_INVALID;
    const uint8_t* s = (const uint8_t*)utf8;
    size_t len = strlen(utf8);
    tk_i64v out = {0};
    s2p_status rc = S2P_OK;
    size_t seg = 0;
    for (size_t i = 0; i < len && rc == S2P_OK;) {
        if (s[i] == '<') { /* all added tokens start with '<' */
            size_t maxl = (size_t)t->added_max;
            if (maxl > len - i) maxl = len - i;
            int32_t hit = -1; size_t hl = 0;
            for (size_t l = maxl; l >= (size_t)t->added_min && hit < 0; l--) {
                hit = tk_bget(t, t->added, t->added_cap, s + i, l);
                if (hit >= 0) hl = l;
                if (l == (size_t)t->added_min) break;
            }
            if (hit >= 0) {
                rc = tk_encode_segment(t, s + seg, i - seg, &out);
                if (rc == S2P_OK) rc = tk_push_id(&out, hit);
                i += hl;
                seg = i;
                continue;
            }
        }
        i++;
    }
    if (rc == S2P_OK) rc = tk_encode_segment(t, s + seg, len - seg, &out);
    if (rc != S2P_OK) { free(out.v); return rc; }
    if (!out.v) { /* empty input: return a valid (freeable) pointer */
        out.v = (int64_t*)malloc(sizeof(int64_t));
        if (!out.v) return S2P_ERR_OOM;
    }
    *ids = out.v;
    *n = out.n;
    return S2P_OK;
}

s2p_status s2p_tok_decode(s2p_tok* t, const int64_t* ids, int n, char** utf8) {
    if (!t || !ids || !utf8 || n < 0) return S2P_ERR_INVALID;
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        int64_t id = ids[i];
        if (id >= 0 && id < S2P_TEXT_VOCAB) total += t->id_len[id];
    }
    char* s = (char*)malloc(total + 1);
    if (!s) return S2P_ERR_OOM;
    size_t o = 0;
    for (int i = 0; i < n; i++) {
        int64_t id = ids[i];
        if (id < 0 || id >= S2P_TEXT_VOCAB || t->id_len[id] == 0) continue;
        memcpy(s + o, t->arena + t->id_off[id], t->id_len[id]);
        o += t->id_len[id];
    }
    s[o] = '\0';
    *utf8 = s;
    return S2P_OK;
}

/* ------------------------------------------------------------------ load */

static s2p_status tk_read_file(const char* path, char** buf, size_t* len) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return S2P_ERR_IO;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return S2P_ERR_IO; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return S2P_ERR_IO; }
    if (fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return S2P_ERR_IO; }
    char* b = (char*)malloc((size_t)sz + 1);
    if (!b) { fclose(fp); return S2P_ERR_OOM; }
    if (sz > 0 && fread(b, 1, (size_t)sz, fp) != (size_t)sz) {
        free(b); fclose(fp);
        return S2P_ERR_IO;
    }
    fclose(fp);
    b[sz] = '\0';
    *buf = b;
    *len = (size_t)sz;
    return S2P_OK;
}

void s2p_tok_free(s2p_tok* t) {
    if (!t) return;
    free(t->arena);
    free(t->id_off); free(t->id_len);
    free(t->vocab); free(t->merges); free(t->added);
    free(t);
}

/* Register one vocab entry (raw bytes already in scratch). */
static s2p_status tk_add_vocab(s2p_tok* t, const uint8_t* raw, size_t rn,
                               int32_t id) {
    if (id < 0 || id >= S2P_TEXT_VOCAB || rn > UINT32_MAX)
        return S2P_ERR_FORMAT;
    uint32_t off;
    S2P_TRY(tk_arena_push(t, raw, rn, &off));
    tk_bput(t, t->vocab, t->vocab_cap, off, (uint32_t)rn, id);
    t->id_off[id] = off;
    t->id_len[id] = (uint32_t)rn;
    if (rn == 1) t->byte2id[raw[0]] = id;
    return S2P_OK;
}

s2p_status s2p_tok_load(const char* model_dir, s2p_tok** out) {
    if (!model_dir || !out) return S2P_ERR_INVALID;
    *out = NULL;

    char path[1024];
    snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
    char* buf = NULL;
    size_t blen = 0;
    S2P_TRY(tk_read_file(path, &buf, &blen));

    s2p_json* j = NULL;
    s2p_status rc = s2p_json_parse(buf, blen, &j);
    if (rc != S2P_OK) { free(buf); return rc; }

    s2p_tok* t = (s2p_tok*)calloc(1, sizeof(s2p_tok));
    if (!t) { s2p_json_free(j); free(buf); return S2P_ERR_OOM; }
    t->vocab_cap = 1u << 18;
    t->merge_cap = 1u << 18;
    t->added_cap = 1u << 13;
    t->vocab  = (tk_bent*)malloc(t->vocab_cap * sizeof(tk_bent));
    t->merges = (tk_ment*)malloc(t->merge_cap * sizeof(tk_ment));
    t->added  = (tk_bent*)malloc(t->added_cap * sizeof(tk_bent));
    t->id_off = (uint32_t*)calloc(S2P_TEXT_VOCAB, sizeof(uint32_t));
    t->id_len = (uint32_t*)calloc(S2P_TEXT_VOCAB, sizeof(uint32_t));
    uint8_t* scratch = (uint8_t*)malloc(4096);
    if (!t->vocab || !t->merges || !t->added || !t->id_off || !t->id_len ||
        !scratch) {
        rc = S2P_ERR_OOM;
        goto done;
    }
    for (uint32_t i = 0; i < t->vocab_cap; i++) t->vocab[i].id = -1;
    for (uint32_t i = 0; i < t->merge_cap; i++) t->merges[i].key = UINT64_MAX;
    for (uint32_t i = 0; i < t->added_cap; i++) t->added[i].id = -1;
    for (int i = 0; i < 256; i++) t->byte2id[i] = -1;
    t->added_min = 0x7FFFFFFF;

    {
        uint16_t u2b[TK_U2B_MAX];
        tk_build_u2b(u2b);

        const s2p_jval* root = s2p_json_root(j);
        const s2p_jval* model = s2p_jobj_get(root, "model");
        const s2p_jval* vocab = model ? s2p_jobj_get(model, "vocab") : NULL;
        const s2p_jval* merges = model ? s2p_jobj_get(model, "merges") : NULL;
        if (!vocab || !merges) { rc = S2P_ERR_FORMAT; goto done; }

        int nv = s2p_jobj_len(vocab);
        for (int i = 0; i < nv; i++) {
            const char* key;
            const s2p_jval* val;
            rc = s2p_jobj_at(vocab, i, &key, &val);
            if (rc != S2P_OK) goto done;
            size_t kn = strlen(key), rn;
            if (kn > 4096 ||
                tk_bytelevel_decode(u2b, (const uint8_t*)key, kn, scratch,
                                    &rn) != 0) {
                rc = S2P_ERR_FORMAT;
                goto done;
            }
            rc = tk_add_vocab(t, scratch, rn, (int32_t)s2p_jint(val));
            if (rc != S2P_OK) goto done;
        }
        for (int b = 0; b < 256; b++)
            if (t->byte2id[b] < 0) { rc = S2P_ERR_FORMAT; goto done; }

        /* merges: ["a","b"] pairs (current format) or "a b" strings */
        int nm = s2p_jarr_len(merges);
        for (int r = 0; r < nm; r++) {
            const s2p_jval* mv = s2p_jarr_at(merges, r);
            const char *as = NULL, *bs = NULL;
            size_t al = 0, bl = 0;
            char splitbuf[4096];
            if (s2p_jis_str(mv)) {
                size_t ml;
                const char* m = s2p_jstr(mv, &ml);
                if (!m || ml >= sizeof splitbuf) { rc = S2P_ERR_FORMAT; goto done; }
                memcpy(splitbuf, m, ml);
                splitbuf[ml] = '\0';
                char* sp = strchr(splitbuf, ' ');
                if (!sp) { rc = S2P_ERR_FORMAT; goto done; }
                *sp = '\0';
                as = splitbuf; al = (size_t)(sp - splitbuf);
                bs = sp + 1;   bl = ml - al - 1;
            } else {
                if (s2p_jarr_len(mv) != 2) { rc = S2P_ERR_FORMAT; goto done; }
                as = s2p_jstr(s2p_jarr_at(mv, 0), &al);
                bs = s2p_jstr(s2p_jarr_at(mv, 1), &bl);
                if (!as || !bs) { rc = S2P_ERR_FORMAT; goto done; }
            }
            size_t an, bn;
            if (al + bl > 2048 ||
                tk_bytelevel_decode(u2b, (const uint8_t*)as, al, scratch,
                                    &an) != 0 ||
                tk_bytelevel_decode(u2b, (const uint8_t*)bs, bl,
                                    scratch + an, &bn) != 0) {
                rc = S2P_ERR_FORMAT;
                goto done;
            }
            int32_t ia = tk_bget(t, t->vocab, t->vocab_cap, scratch, an);
            int32_t ib = tk_bget(t, t->vocab, t->vocab_cap, scratch + an, bn);
            int32_t im = tk_bget(t, t->vocab, t->vocab_cap, scratch, an + bn);
            if (ia < 0 || ib < 0 || im < 0) { rc = S2P_ERR_FORMAT; goto done; }
            tk_mput(t->merges, t->merge_cap,
                    ((uint64_t)(uint32_t)ia << 32) | (uint32_t)ib, r, im);
        }

        /* added tokens: atomic matches + decode strings (all normalized:false,
         * lstrip/rstrip/single_word:false in this tokenizer.json) */
        const s2p_jval* added = s2p_jobj_get(root, "added_tokens");
        int na = added ? s2p_jarr_len(added) : 0;
        for (int i = 0; i < na; i++) {
            const s2p_jval* av = s2p_jarr_at(added, i);
            const s2p_jval* cv = s2p_jobj_get(av, "content");
            const s2p_jval* iv = s2p_jobj_get(av, "id");
            if (!cv || !iv) { rc = S2P_ERR_FORMAT; goto done; }
            size_t cn;
            const char* cs = s2p_jstr(cv, &cn);
            int32_t id = (int32_t)s2p_jint(iv);
            if (!cs || cn == 0 || id < 0 || id >= S2P_TEXT_VOCAB) {
                rc = S2P_ERR_FORMAT;
                goto done;
            }
            uint32_t off;
            rc = tk_arena_push(t, (const uint8_t*)cs, cn, &off);
            if (rc != S2P_OK) goto done;
            tk_bput(t, t->added, t->added_cap, off, (uint32_t)cn, id);
            t->id_off[id] = off;
            t->id_len[id] = (uint32_t)cn;
            if ((int)cn < t->added_min) t->added_min = (int)cn;
            if ((int)cn > t->added_max) t->added_max = (int)cn;
        }
        if (na == 0) { rc = S2P_ERR_FORMAT; goto done; }
    }

done:
    free(scratch);
    s2p_json_free(j);
    free(buf);
    if (rc != S2P_OK) {
        s2p_tok_free(t);
        return rc;
    }
    *out = t;
    return S2P_OK;
}

/* ---------------------------------------------------------------- selftest */
#ifdef S2P_TOK_SELFTEST
/* Encodes fixed strings and prints ids (one line each), then a full prompt
 * build. Diff against HF: AutoTokenizer.from_pretrained(dir).encode(s). */
int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "model";
    s2p_tok* t = NULL;
    s2p_status rc = s2p_tok_load(dir, &t);
    if (rc != S2P_OK) {
        fprintf(stderr, "load failed: %d\n", rc);
        return 1;
    }
    static const char* tests[] = {
        "Hello, world!",
        "Die Universit\xc3\xa4t Karlsruhe ist wundersch\xc3\xb6n, oder?",
        "<|im_start|>system\n",
        "convert the provided text to speech reference to the following:\n\nText:\n",
        "\n\nSpeech:\n",
        "<|speaker:0|>Hello there, this is a longer voice test 123.",
        "<|im_start|>assistant\n<|voice|>",
        "  multiple   spaces\n\nand 12345 numbers...",
        "don't we've I'll they'd it's O'Brien",
        "<|semantic:0|><|semantic:123|><|semantic:4095|><|im_end|>",
        "Mixed <|voice|> inline special and text",
        "tabs\tand\r\nCRLF line endings",
    };
    int nt = (int)(sizeof tests / sizeof tests[0]);
    for (int i = 0; i < nt; i++) {
        int64_t* ids = NULL;
        int n = 0;
        rc = s2p_tok_encode(t, tests[i], &ids, &n);
        if (rc != S2P_OK) {
            fprintf(stderr, "encode %d failed: %d\n", i, rc);
            return 1;
        }
        printf("ENC %02d:", i);
        for (int k = 0; k < n; k++) printf(" %lld", (long long)ids[k]);
        printf("\n");
        char* back = NULL;
        if (s2p_tok_decode(t, ids, n, &back) == S2P_OK) {
            printf("DEC %02d: %s\n", i, back);
            free(back);
        }
        free(ids);
    }
    /* prompt build with one fake reference (cb-major [10*4]) */
    {
        int32_t codes[40];
        for (int cb = 0; cb < 10; cb++)
            for (int k = 0; k < 4; k++)
                codes[cb * 4 + k] = cb == 0 ? 100 + k : (cb * 7 + k) % 1024;
        s2p_vq_part ref = { codes, 4 };
        s2p_request_text req = { "Hello world, this is a test.",
                                 "Reference transcript.", &ref, 1 };
        int64_t* ids = NULL;
        uint8_t* mask = NULL;
        s2p_vq_part* parts = NULL;
        int n = 0, np = 0;
        rc = s2p_prompt_build(t, NULL, &req, &ids, &mask, &n, &parts, &np);
        if (rc != S2P_OK) {
            fprintf(stderr, "prompt build failed: %d\n", rc);
            return 1;
        }
        printf("PROMPT ids:");
        for (int k = 0; k < n; k++) printf(" %lld", (long long)ids[k]);
        printf("\nPROMPT mask:");
        for (int k = 0; k < n; k++) printf(" %d", mask[k]);
        printf("\nPROMPT parts: %d (T=%d)\n", np, np ? parts[0].T : 0);
        free(ids);
        free(mask);
        if (parts) {
            free((void*)parts[0].codes);
            free(parts);
        }
    }
    s2p_tok_free(t);
    return 0;
}
#endif
