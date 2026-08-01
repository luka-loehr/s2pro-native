/* s2pro-native — strict JSON parser (include/s2pro/json.h).
 *
 * Design, per contract: single pass over the input, iterative with explicit
 * value/key/container stacks (no recursion — nesting depth is bounded only by
 * a sanity cap), and one growable chunked arena for every node, string and
 * child list (no per-node malloc; chunking keeps pointers stable across
 * growth). Strings are copied into the arena unescaped and NUL-terminated,
 * with full escape handling including \uXXXX (+ surrogate pairs) -> UTF-8.
 * Numbers: exact int64 fast path; fraction/exponent/overflow falls back to
 * strtod (assumes the process stays in the default "C" locale — nothing in
 * s2pro calls setlocale).
 *
 * Sized for the 12 MB tokenizer.json: the first arena chunk is ~len/2 and
 * child lists are materialized once per container close, so the whole parse
 * is O(bytes) with two memcpys worst-case per string/child-list.
 */
#include "s2pro/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- value model */

enum { J_NULL = 0, J_FALSE, J_TRUE, J_INT, J_NUM, J_STR, J_ARR, J_OBJ };

typedef struct s2p_jkv s2p_jkv;

struct s2p_jval {
    uint8_t  type;
    uint32_t n; /* J_STR: byte length; J_ARR: elements; J_OBJ: pairs */
    union {
        const char*            s; /* arena, NUL-terminated */
        int64_t                i;
        double                 d;
        const s2p_jval* const* a; /* arena array of n pointers */
        const s2p_jkv*         o; /* arena array of n pairs */
    } u;
};

struct s2p_jkv {
    const char*     key;  /* arena, NUL-terminated */
    uint32_t        klen;
    const s2p_jval* val;
};

typedef struct chunk {
    struct chunk* next;
    size_t        cap;
    size_t        used;
    /* payload follows the header */
} chunk;

struct s2p_json {
    chunk*          chunks; /* newest first */
    const s2p_jval* root;
};

/* ------------------------------------------------------------------- arena */

#define ARENA_ALIGN     8u
#define ARENA_CHUNK_MAX ((size_t)32u << 20)
#define JSON_MAX_DEPTH  (1u << 20)

static void* arena_alloc(s2p_json* j, size_t* next_cap, size_t n)
{
    chunk* c = j->chunks;
    n = (n + (ARENA_ALIGN - 1)) & ~(size_t)(ARENA_ALIGN - 1);
    if (!c || c->cap - c->used < n) {
        size_t cap = *next_cap;
        chunk* nc;
        if (cap < n) cap = n;
        nc = (chunk*)malloc(sizeof(chunk) + cap);
        if (!nc) return NULL;
        nc->next = j->chunks;
        nc->cap  = cap;
        nc->used = 0;
        j->chunks = nc;
        c = nc;
        if (*next_cap < ARENA_CHUNK_MAX) *next_cap *= 2;
    }
    {
        void* p = (char*)(c + 1) + c->used;
        c->used += n;
        return p;
    }
}

/* ------------------------------------------------------------ parser state */

typedef struct { const char* s; uint32_t len; } kent;              /* pending key */
typedef struct { s2p_jval* v; size_t vbase; size_t kbase; } cent;  /* open container */

typedef struct {
    const char* p;
    const char* end;
    const char* begin;
    s2p_json*   j;
    size_t      next_cap; /* next arena chunk size */
    s2p_jval**  vs; size_t vn, vcap;
    kent*       ks; size_t kn, kcap;
    cent*       cs; size_t cn, ccap;
} P;

static s2p_status jfail(const P* ps, const char* msg)
{
    fprintf(stderr, "[s2pro] json parse error at byte %zu: %s\n",
            (size_t)(ps->p - ps->begin), msg);
    return S2P_ERR_FORMAT;
}

static int push_v(P* ps, s2p_jval* v)
{
    if (ps->vn == ps->vcap) {
        size_t nc = ps->vcap ? ps->vcap * 2 : 256;
        void* np = realloc(ps->vs, nc * sizeof(*ps->vs));
        if (!np) return 0;
        ps->vs = (s2p_jval**)np;
        ps->vcap = nc;
    }
    ps->vs[ps->vn++] = v;
    return 1;
}

static int push_k(P* ps, const char* s, uint32_t len)
{
    if (ps->kn == ps->kcap) {
        size_t nc = ps->kcap ? ps->kcap * 2 : 128;
        void* np = realloc(ps->ks, nc * sizeof(*ps->ks));
        if (!np) return 0;
        ps->ks = (kent*)np;
        ps->kcap = nc;
    }
    ps->ks[ps->kn].s = s;
    ps->ks[ps->kn].len = len;
    ps->kn++;
    return 1;
}

static int push_c(P* ps, s2p_jval* v)
{
    if (ps->cn >= JSON_MAX_DEPTH) return -1;
    if (ps->cn == ps->ccap) {
        size_t nc = ps->ccap ? ps->ccap * 2 : 64;
        void* np = realloc(ps->cs, nc * sizeof(*ps->cs));
        if (!np) return 0;
        ps->cs = (cent*)np;
        ps->ccap = nc;
    }
    ps->cs[ps->cn].v = v;
    ps->cs[ps->cn].vbase = ps->vn;
    ps->cs[ps->cn].kbase = ps->kn;
    ps->cn++;
    return 1;
}

static s2p_jval* new_val(P* ps, uint8_t type)
{
    s2p_jval* v = (s2p_jval*)arena_alloc(ps->j, &ps->next_cap, sizeof(*v));
    if (!v) return NULL;
    v->type = type;
    v->n = 0;
    v->u.i = 0;
    return v;
}

static void skip_ws(P* ps)
{
    const char* p = ps->p;
    while (p < ps->end &&
           (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        p++;
    ps->p = p;
}

/* ----------------------------------------------------------------- strings */

static int hex4(const char* s, uint32_t* out)
{
    uint32_t v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9')      v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static char* utf8_put(char* w, uint32_t cp)
{
    if (cp < 0x80) {
        *w++ = (char)cp;
    } else if (cp < 0x800) {
        *w++ = (char)(0xC0 | (cp >> 6));
        *w++ = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *w++ = (char)(0xE0 | (cp >> 12));
        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *w++ = (char)(0xF0 | (cp >> 18));
        *w++ = (char)(0x80 | ((cp >> 12) & 0x3F));
        *w++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (char)(0x80 | (cp & 0x3F));
    }
    return w;
}

/* ps->p at the opening quote. On success ps->p is one past the closing quote
 * and *out_s / *out_len hold the arena-copied, NUL-terminated string. */
static s2p_status parse_string(P* ps, const char** out_s, uint32_t* out_len)
{
    const char* start = ps->p + 1;
    const char* s = start;
    const char* q;       /* closing quote */
    int esc = 0;
    char* dst;
    size_t raw;

    /* pass 1: locate the closing quote, reject raw control chars */
    for (;;) {
        unsigned char c;
        if (s >= ps->end) return jfail(ps, "unterminated string");
        c = (unsigned char)*s;
        if (c == '"') break;
        if (c == '\\') {
            s += 2;
            if (s > ps->end) return jfail(ps, "unterminated escape");
            esc = 1;
            continue;
        }
        if (c < 0x20) return jfail(ps, "raw control char in string");
        s++;
    }
    q = s;
    raw = (size_t)(q - start);
    if (raw > 0xFFFFFFFFu) return jfail(ps, "string too long");

    dst = (char*)arena_alloc(ps->j, &ps->next_cap, raw + 1);
    if (!dst) return S2P_ERR_OOM;

    if (!esc) {
        memcpy(dst, start, raw);
        dst[raw] = '\0';
        *out_s = dst;
        *out_len = (uint32_t)raw;
        ps->p = q + 1;
        return S2P_OK;
    }

    /* pass 2: decode escapes (output never exceeds raw bytes) */
    {
        char* w = dst;
        s = start;
        while (s < q) {
            unsigned char c = (unsigned char)*s;
            if (c != '\\') {
                *w++ = (char)c;
                s++;
                continue;
            }
            s++; /* at escape char; pass 1 guarantees s < q+1 */
            switch (*s++) {
            case '"':  *w++ = '"';  break;
            case '\\': *w++ = '\\'; break;
            case '/':  *w++ = '/';  break;
            case 'b':  *w++ = '\b'; break;
            case 'f':  *w++ = '\f'; break;
            case 'n':  *w++ = '\n'; break;
            case 'r':  *w++ = '\r'; break;
            case 't':  *w++ = '\t'; break;
            case 'u': {
                uint32_t cp;
                if (q - s < 4 || !hex4(s, &cp))
                    return jfail(ps, "bad \\u escape");
                s += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    uint32_t lo;
                    if (q - s < 6 || s[0] != '\\' || s[1] != 'u' ||
                        !hex4(s + 2, &lo) || lo < 0xDC00 || lo > 0xDFFF)
                        return jfail(ps, "lone high surrogate");
                    s += 6;
                    cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    return jfail(ps, "lone low surrogate");
                }
                w = utf8_put(w, cp);
                break;
            }
            default:
                return jfail(ps, "invalid escape");
            }
        }
        *w = '\0';
        *out_s = dst;
        *out_len = (uint32_t)(w - dst);
    }
    ps->p = q + 1;
    return S2P_OK;
}

/* ----------------------------------------------------------------- numbers */

static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* ps->p at '-' or a digit. Writes the parsed value into *v. */
static s2p_status parse_number(P* ps, s2p_jval* v)
{
    const char* s = ps->p;
    const char* q = s;
    int neg = 0, isint = 1, overflow = 0;
    uint64_t mag = 0;

    if (*q == '-') { neg = 1; q++; }
    if (q >= ps->end || !is_digit(*q)) return jfail(ps, "bad number");
    if (*q == '0') {
        q++;
        if (q < ps->end && is_digit(*q)) return jfail(ps, "leading zero");
    } else {
        while (q < ps->end && is_digit(*q)) {
            unsigned d = (unsigned)(*q - '0');
            if (mag > (UINT64_MAX - d) / 10u) overflow = 1;
            else mag = mag * 10u + d;
            q++;
        }
    }
    if (q < ps->end && *q == '.') {
        q++;
        if (q >= ps->end || !is_digit(*q)) return jfail(ps, "bad fraction");
        while (q < ps->end && is_digit(*q)) q++;
        isint = 0;
    }
    if (q < ps->end && (*q == 'e' || *q == 'E')) {
        q++;
        if (q < ps->end && (*q == '+' || *q == '-')) q++;
        if (q >= ps->end || !is_digit(*q)) return jfail(ps, "bad exponent");
        while (q < ps->end && is_digit(*q)) q++;
        isint = 0;
    }

    if (isint && !overflow &&
        (neg ? mag <= (uint64_t)INT64_MAX + 1u : mag <= (uint64_t)INT64_MAX)) {
        v->type = J_INT;
        if (neg && mag == (uint64_t)INT64_MAX + 1u) v->u.i = INT64_MIN;
        else v->u.i = neg ? -(int64_t)mag : (int64_t)mag;
        ps->p = q;
        return S2P_OK;
    }

    /* double path: copy the token (input may not be NUL-terminated) */
    {
        char tmp[512];
        char* buf = tmp;
        size_t tl = (size_t)(q - s);
        char* endp = NULL;
        if (tl + 1 > sizeof(tmp)) {
            buf = (char*)arena_alloc(ps->j, &ps->next_cap, tl + 1);
            if (!buf) return S2P_ERR_OOM;
        }
        memcpy(buf, s, tl);
        buf[tl] = '\0';
        v->type = J_NUM;
        v->u.d = strtod(buf, &endp);
        if (endp != buf + tl) return jfail(ps, "bad number");
    }
    ps->p = q;
    return S2P_OK;
}

/* -------------------------------------------------------------- main parse */

s2p_status s2p_json_parse(const char* buf, size_t len, s2p_json** out)
{
    P ps;
    s2p_json* j;
    s2p_status st = S2P_ERR_FORMAT;
    int r;

    if (!buf || !out) return S2P_ERR_INVALID;
    *out = NULL;

    j = (s2p_json*)calloc(1, sizeof(*j));
    if (!j) return S2P_ERR_OOM;

    memset(&ps, 0, sizeof(ps));
    ps.begin = buf;
    ps.p = buf;
    ps.end = buf + len;
    ps.j = j;
    ps.next_cap = len / 2 + 4096;
    if (ps.next_cap > ARENA_CHUNK_MAX) ps.next_cap = ARENA_CHUNK_MAX;

    /* tolerate a UTF-8 BOM */
    if (len >= 3 && (unsigned char)buf[0] == 0xEF &&
        (unsigned char)buf[1] == 0xBB && (unsigned char)buf[2] == 0xBF)
        ps.p += 3;

want_value:
    skip_ws(&ps);
    if (ps.p >= ps.end) { st = jfail(&ps, "unexpected end of input"); goto out; }
    switch (*ps.p) {
    case '{': case '[': {
        int is_obj = (*ps.p == '{');
        s2p_jval* v = new_val(&ps, is_obj ? (uint8_t)J_OBJ : (uint8_t)J_ARR);
        if (!v) { st = S2P_ERR_OOM; goto out; }
        r = push_c(&ps, v);
        if (r == 0) { st = S2P_ERR_OOM; goto out; }
        if (r < 0)  { st = jfail(&ps, "nesting too deep"); goto out; }
        ps.p++;
        skip_ws(&ps);
        if (ps.p < ps.end && *ps.p == (is_obj ? '}' : ']')) {
            ps.p++;
            goto close_container;
        }
        if (is_obj) goto want_key;
        goto want_value;
    }
    case '"': {
        const char* s;
        uint32_t slen;
        s2p_jval* v;
        st = parse_string(&ps, &s, &slen);
        if (st != S2P_OK) goto out;
        v = new_val(&ps, J_STR);
        if (!v) { st = S2P_ERR_OOM; goto out; }
        v->u.s = s;
        v->n = slen;
        if (!push_v(&ps, v)) { st = S2P_ERR_OOM; goto out; }
        goto have_value;
    }
    case 't': case 'f': case 'n': {
        s2p_jval* v = NULL;
        if (ps.end - ps.p >= 4 && !memcmp(ps.p, "true", 4)) {
            v = new_val(&ps, J_TRUE);  ps.p += 4;
        } else if (ps.end - ps.p >= 5 && !memcmp(ps.p, "false", 5)) {
            v = new_val(&ps, J_FALSE); ps.p += 5;
        } else if (ps.end - ps.p >= 4 && !memcmp(ps.p, "null", 4)) {
            v = new_val(&ps, J_NULL);  ps.p += 4;
        } else {
            st = jfail(&ps, "bad literal");
            goto out;
        }
        if (!v) { st = S2P_ERR_OOM; goto out; }
        if (!push_v(&ps, v)) { st = S2P_ERR_OOM; goto out; }
        goto have_value;
    }
    default: {
        s2p_jval* v;
        if (*ps.p != '-' && !is_digit(*ps.p)) {
            st = jfail(&ps, "unexpected character");
            goto out;
        }
        v = new_val(&ps, J_INT);
        if (!v) { st = S2P_ERR_OOM; goto out; }
        st = parse_number(&ps, v);
        if (st != S2P_OK) goto out;
        if (!push_v(&ps, v)) { st = S2P_ERR_OOM; goto out; }
        goto have_value;
    }
    }

want_key:
    skip_ws(&ps);
    if (ps.p >= ps.end || *ps.p != '"') {
        st = jfail(&ps, "expected object key");
        goto out;
    }
    {
        const char* s;
        uint32_t slen;
        st = parse_string(&ps, &s, &slen);
        if (st != S2P_OK) goto out;
        if (!push_k(&ps, s, slen)) { st = S2P_ERR_OOM; goto out; }
    }
    skip_ws(&ps);
    if (ps.p >= ps.end || *ps.p != ':') {
        st = jfail(&ps, "expected ':'");
        goto out;
    }
    ps.p++;
    goto want_value;

close_container:
    {
        cent top = ps.cs[--ps.cn];
        size_t nv = ps.vn - top.vbase;
        if (nv > (size_t)INT32_MAX) {
            st = jfail(&ps, "container too large");
            goto out;
        }
        if (top.v->type == J_OBJ) {
            size_t nk = ps.kn - top.kbase;
            s2p_jkv* pairs = NULL;
            size_t i;
            if (nk != nv) { st = S2P_ERR_INTERNAL; goto out; }
            if (nv) {
                pairs = (s2p_jkv*)arena_alloc(ps.j, &ps.next_cap,
                                              nv * sizeof(*pairs));
                if (!pairs) { st = S2P_ERR_OOM; goto out; }
                for (i = 0; i < nv; i++) {
                    pairs[i].key  = ps.ks[top.kbase + i].s;
                    pairs[i].klen = ps.ks[top.kbase + i].len;
                    pairs[i].val  = ps.vs[top.vbase + i];
                }
            }
            top.v->n = (uint32_t)nv;
            top.v->u.o = pairs;
            ps.kn = top.kbase;
        } else {
            s2p_jval** arr = NULL;
            if (nv) {
                arr = (s2p_jval**)arena_alloc(ps.j, &ps.next_cap,
                                              nv * sizeof(*arr));
                if (!arr) { st = S2P_ERR_OOM; goto out; }
                memcpy(arr, ps.vs + top.vbase, nv * sizeof(*arr));
            }
            top.v->n = (uint32_t)nv;
            top.v->u.a = (const s2p_jval* const*)arr;
        }
        ps.vn = top.vbase;
        if (!push_v(&ps, top.v)) { st = S2P_ERR_OOM; goto out; }
    }
    /* fall through */

have_value:
    if (ps.cn == 0) goto done;
    skip_ws(&ps);
    if (ps.p >= ps.end) {
        st = jfail(&ps, "unterminated container");
        goto out;
    }
    {
        int in_obj = (ps.cs[ps.cn - 1].v->type == J_OBJ);
        char c = *ps.p;
        if (c == ',') {
            ps.p++;
            if (in_obj) goto want_key;
            goto want_value;
        }
        if (in_obj && c == '}') { ps.p++; goto close_container; }
        if (!in_obj && c == ']') { ps.p++; goto close_container; }
        st = jfail(&ps, in_obj ? "expected ',' or '}'" : "expected ',' or ']'");
        goto out;
    }

done:
    skip_ws(&ps);
    if (ps.p != ps.end) {
        st = jfail(&ps, "trailing garbage after value");
        goto out;
    }
    if (ps.vn != 1) { st = S2P_ERR_INTERNAL; goto out; }
    j->root = ps.vs[0];
    free(ps.vs);
    free(ps.ks);
    free(ps.cs);
    *out = j;
    return S2P_OK;

out:
    free(ps.vs);
    free(ps.ks);
    free(ps.cs);
    s2p_json_free(j);
    return st;
}

void s2p_json_free(s2p_json* j)
{
    chunk* c;
    if (!j) return;
    c = j->chunks;
    while (c) {
        chunk* n = c->next;
        free(c);
        c = n;
    }
    free(j);
}

/* --------------------------------------------------------------- accessors */

const s2p_jval* s2p_json_root(const s2p_json* j)
{
    return j ? j->root : NULL;
}

const s2p_jval* s2p_jobj_get(const s2p_jval* v, const char* key)
{
    size_t kl;
    uint32_t i;
    if (!v || v->type != J_OBJ || !key) return NULL;
    kl = strlen(key);
    if (kl > 0xFFFFFFFFu) return NULL;
    for (i = 0; i < v->n; i++) {
        if (v->u.o[i].klen == (uint32_t)kl &&
            !memcmp(v->u.o[i].key, key, kl))
            return v->u.o[i].val;
    }
    return NULL;
}

int s2p_jobj_len(const s2p_jval* v)
{
    return (v && v->type == J_OBJ) ? (int)v->n : 0;
}

s2p_status s2p_jobj_at(const s2p_jval* v, int i, const char** key,
                       const s2p_jval** val)
{
    if (!v || v->type != J_OBJ || i < 0 || (uint32_t)i >= v->n)
        return S2P_ERR_INVALID;
    if (key) *key = v->u.o[i].key;
    if (val) *val = v->u.o[i].val;
    return S2P_OK;
}

int s2p_jarr_len(const s2p_jval* v)
{
    return (v && v->type == J_ARR) ? (int)v->n : 0;
}

const s2p_jval* s2p_jarr_at(const s2p_jval* v, int i)
{
    if (!v || v->type != J_ARR || i < 0 || (uint32_t)i >= v->n) return NULL;
    return v->u.a[i];
}

int s2p_jis_str(const s2p_jval* v)
{
    return v && v->type == J_STR;
}

const char* s2p_jstr(const s2p_jval* v, size_t* len)
{
    if (!v || v->type != J_STR) {
        if (len) *len = 0;
        return NULL;
    }
    if (len) *len = v->n;
    return v->u.s;
}

double s2p_jnum(const s2p_jval* v)
{
    if (!v) return 0.0;
    if (v->type == J_NUM) return v->u.d;
    if (v->type == J_INT) return (double)v->u.i;
    return 0.0;
}

int64_t s2p_jint(const s2p_jval* v)
{
    if (!v) return 0;
    if (v->type == J_INT) return v->u.i;
    if (v->type == J_NUM) return (int64_t)v->u.d;
    return 0;
}

int s2p_jbool(const s2p_jval* v)
{
    return v && v->type == J_TRUE;
}

int s2p_jis_null(const s2p_jval* v)
{
    return v && v->type == J_NULL;
}

/* ---------------------------------------------------------------- selftest */
/* Build: gcc -O2 -std=c11 -Wall -Iinclude -DS2P_CORE_SELFTEST \
 *            src/core/json.c -o build/json-selftest
 * Run:   ./build/json-selftest [config.json] [tokenizer.json]              */
#ifdef S2P_CORE_SELFTEST

#include <time.h>

static int st_fails = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "SELFTEST FAIL: %s\n", what);                      \
            st_fails++;                                                        \
        }                                                                      \
    } while (0)

static const s2p_jval* walk2(const s2p_json* j, const char* a, const char* b)
{
    const s2p_jval* v = s2p_jobj_get(s2p_json_root(j), a);
    return b ? s2p_jobj_get(v, b) : v;
}

static void check_i64(const s2p_json* j, const char* a, const char* b,
                      int64_t want, const char* what)
{
    const s2p_jval* v = walk2(j, a, b);
    int64_t got = v ? (s2p_jint(v) ? s2p_jint(v) : s2p_jbool(v)) : -999999;
    if (got != want) {
        fprintf(stderr, "SELFTEST FAIL: %s = %lld, want %lld\n", what,
                (long long)got, (long long)want);
        st_fails++;
    }
}

static char* read_all(const char* path, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    long sz;
    char* buf;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    buf = (char*)malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    *out_len = (size_t)sz;
    return buf;
}

int main(int argc, char** argv)
{
    /* -- unit: values, escapes, int64 edges -- */
    {
        static const char t1[] =
            "[1,-2,3.5,\"a\\u00e9\\uD83D\\uDE00b\",true,false,null,{},"
            "{\"k\":[]},\"\\n\\t\\\\\\\"\\/\",9223372036854775807,"
            "-9223372036854775808,9223372036854775808,1e-06,-0]";
        s2p_json* j = NULL;
        const s2p_jval* r;
        size_t sl = 0;
        const char* s;
        CHECK(s2p_json_parse(t1, sizeof(t1) - 1, &j) == S2P_OK, "t1 parse");
        r = s2p_json_root(j);
        CHECK(s2p_jarr_len(r) == 15, "t1 len");
        CHECK(s2p_jint(s2p_jarr_at(r, 0)) == 1, "t1[0]");
        CHECK(s2p_jint(s2p_jarr_at(r, 1)) == -2, "t1[1]");
        CHECK(s2p_jnum(s2p_jarr_at(r, 2)) == 3.5, "t1[2]");
        s = s2p_jstr(s2p_jarr_at(r, 3), &sl);
        CHECK(s && sl == 8 && !memcmp(s, "a\xC3\xA9\xF0\x9F\x98\x80" "b", 8),
              "t1[3] utf8");
        CHECK(s2p_jbool(s2p_jarr_at(r, 4)) == 1, "t1[4]");
        CHECK(s2p_jbool(s2p_jarr_at(r, 5)) == 0, "t1[5]");
        CHECK(s2p_jis_null(s2p_jarr_at(r, 6)), "t1[6]");
        CHECK(s2p_jobj_len(s2p_jarr_at(r, 7)) == 0, "t1[7]");
        CHECK(s2p_jarr_len(s2p_jobj_get(s2p_jarr_at(r, 8), "k")) == 0,
              "t1[8].k");
        s = s2p_jstr(s2p_jarr_at(r, 9), &sl);
        CHECK(s && sl == 5 && !memcmp(s, "\n\t\\\"/", 5), "t1[9] escapes");
        CHECK(s2p_jint(s2p_jarr_at(r, 10)) == INT64_MAX, "t1[10] i64max");
        CHECK(s2p_jint(s2p_jarr_at(r, 11)) == INT64_MIN, "t1[11] i64min");
        CHECK(s2p_jnum(s2p_jarr_at(r, 12)) > 9.2e18, "t1[12] overflow->dbl");
        CHECK(s2p_jnum(s2p_jarr_at(r, 13)) == 1e-06, "t1[13] 1e-06");
        CHECK(s2p_jint(s2p_jarr_at(r, 14)) == 0, "t1[14] -0");
        s2p_json_free(j);
    }

    /* -- unit: strict rejections -- */
    {
        static const char* bad[] = {
            "{,}", "[1,]", "01", "\"\\x\"", "tru", "{\"a\":1 \"b\":2}",
            "\"abc", "1e", "[1]x", "\"\\ud800\"", "\"\\udc00\"",
            "{\"a\"1}", "nulll", "+1", ".5", "1.", "[,1]", "{\"a\":}", "",
        };
        size_t i;
        for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            s2p_json* j = (s2p_json*)(void*)1;
            s2p_status rc = s2p_json_parse(bad[i], strlen(bad[i]), &j);
            if (rc == S2P_OK || j != NULL) {
                fprintf(stderr, "SELFTEST FAIL: accepted bad json #%zu: %s\n",
                        i, bad[i]);
                st_fails++;
                if (rc == S2P_OK) s2p_json_free(j);
            }
        }
    }

    /* -- config.json: verify against include/s2pro/config.h ground truth -- */
    {
        const char* path = argc > 1 ? argv[1] : "model/config.json";
        size_t len = 0;
        char* buf = read_all(path, &len);
        s2p_json* j = NULL;
        if (!buf) {
            fprintf(stderr, "SELFTEST FAIL: cannot read %s\n", path);
            return 1;
        }
        CHECK(s2p_json_parse(buf, len, &j) == S2P_OK, "config parse");
        if (j) {
            check_i64(j, "text_config", "n_layer", 36, "text.n_layer");
            check_i64(j, "text_config", "dim", 2560, "text.dim");
            check_i64(j, "text_config", "n_head", 32, "text.n_head");
            check_i64(j, "text_config", "n_local_heads", 8, "text.kv_heads");
            check_i64(j, "text_config", "head_dim", 128, "text.head_dim");
            check_i64(j, "text_config", "intermediate_size", 9728, "text.ffn");
            check_i64(j, "text_config", "vocab_size", 155776, "text.vocab");
            check_i64(j, "text_config", "max_seq_len", 32768, "text.max_seq");
            check_i64(j, "text_config", "rope_base", 1000000, "text.rope");
            check_i64(j, "text_config", "attention_qk_norm", 1, "text.qknorm");
            check_i64(j, "text_config", "tie_word_embeddings", 1, "text.tied");
            check_i64(j, "audio_decoder_config", "n_layer", 4, "fast.n_layer");
            check_i64(j, "audio_decoder_config", "num_codebooks", 10,
                      "fast.codebooks");
            check_i64(j, "audio_decoder_config", "vocab_size", 4096,
                      "fast.vocab");
            check_i64(j, "audio_decoder_config", "max_seq_len", 11,
                      "fast.max_seq");
            check_i64(j, "audio_decoder_config", "attention_qk_norm", 0,
                      "fast.qknorm");
            check_i64(j, "eos_token_id", NULL, 151645, "eos");
            check_i64(j, "pad_token_id", NULL, 151669, "pad");
            check_i64(j, "audio_pad_token_id", NULL, 151677, "audio_pad");
            check_i64(j, "semantic_start_token_id", NULL, 151678, "sem_start");
            check_i64(j, "semantic_end_token_id", NULL, 155773, "sem_end");
            {
                double eps = s2p_jnum(walk2(j, "text_config", "norm_eps"));
                CHECK(eps > 0.9e-6 && eps < 1.1e-6, "text.norm_eps ~1e-6");
            }
            s2p_json_free(j);
        }
        free(buf);
    }

    /* -- optional: tokenizer.json bulk parse + timing -- */
    if (argc > 2) {
        size_t len = 0;
        char* buf = read_all(argv[2], &len);
        if (!buf) {
            fprintf(stderr, "SELFTEST FAIL: cannot read %s\n", argv[2]);
            return 1;
        }
        {
            clock_t t0 = clock();
            s2p_json* j = NULL;
            s2p_status rc = s2p_json_parse(buf, len, &j);
            double ms = (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC;
            CHECK(rc == S2P_OK, "tokenizer parse");
            if (j) {
                const s2p_jval* model = s2p_jobj_get(s2p_json_root(j), "model");
                int nvocab = s2p_jobj_len(s2p_jobj_get(model, "vocab"));
                int nmerge = s2p_jarr_len(s2p_jobj_get(model, "merges"));
                printf("tokenizer.json: %zu bytes, %.1f ms, vocab=%d "
                       "merges=%d\n", len, ms, nvocab, nmerge);
                CHECK(nvocab > 150000, "tokenizer vocab size");
                CHECK(nmerge > 150000, "tokenizer merges size");
                s2p_json_free(j);
            }
        }
        free(buf);
    }

    if (st_fails) {
        fprintf(stderr, "json selftest: %d FAILURES\n", st_fails);
        return 1;
    }
    printf("json selftest OK\n");
    return 0;
}

#endif /* S2P_CORE_SELFTEST */
