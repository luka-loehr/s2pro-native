/* s2pro-native — strict JSON DOM parser. Contract header: frozen.
 *
 * Implemented by core (src/core/json.c); consumed by text (tokenizer.json)
 * and serve (HTTP payloads). Single pass, iterative (explicit stacks, no
 * recursion), arena-backed (no per-node malloc) — sized to chew the 12 MB
 * tokenizer.json quickly. RFC 8259 strict: rejects trailing commas, comments,
 * unquoted keys, raw control chars, invalid escapes and lone surrogates.
 *
 * All returned pointers (values, strings, keys) are owned by the s2p_json
 * handle and stay valid until s2p_json_free. Strings are unescaped UTF-8 and
 * NUL-terminated; the u0000 escape yields an embedded NUL — use the len out-param when
 * binary-exactness matters. Numbers keep an exact int64 fast path; anything
 * with a fraction/exponent (or overflowing int64) is stored as double.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_json s2p_json;
typedef struct s2p_jval s2p_jval;

s2p_status s2p_json_parse(const char* buf, size_t len, s2p_json** out);
void s2p_json_free(s2p_json* j);
const s2p_jval* s2p_json_root(const s2p_json* j);

/* Objects. get: NULL on miss / non-object (linear scan — iterate via
 * len/at for bulk consumption such as the tokenizer vocab). */
const s2p_jval* s2p_jobj_get(const s2p_jval* v, const char* key);
int s2p_jobj_len(const s2p_jval* v); /* pair count; 0 if not an object */
s2p_status s2p_jobj_at(const s2p_jval* v, int i, const char** key,
                       const s2p_jval** val);

/* Arrays. */
int s2p_jarr_len(const s2p_jval* v); /* element count; 0 if not an array */
const s2p_jval* s2p_jarr_at(const s2p_jval* v, int i);

/* Scalars. Type-mismatched access returns NULL / 0 — never traps. */
int s2p_jis_str(const s2p_jval* v);
const char* s2p_jstr(const s2p_jval* v, size_t* len); /* len may be NULL */
double s2p_jnum(const s2p_jval* v);   /* number (int widened); else 0.0 */
int64_t s2p_jint(const s2p_jval* v);  /* int, or truncated double; else 0 */
int s2p_jbool(const s2p_jval* v);     /* 1 for true; 0 otherwise */
int s2p_jis_null(const s2p_jval* v);

#ifdef __cplusplus
}
#endif
