/* s2pro-native — JSON parser selftest (public s2pro/json.h contract only).
 * Exercises exactly the shapes the HTTP layer feeds it. Exit 0 on pass.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "s2pro/status.h"
#include "s2pro/json.h"

static int failures = 0;

#define T(cond, name)                                                          \
    do {                                                                       \
        if (cond) {                                                            \
            fprintf(stderr, "ok   %s\n", (name));                              \
        } else {                                                               \
            fprintf(stderr, "FAIL %s\n", (name));                              \
            failures++;                                                        \
        }                                                                      \
    } while (0)

int main(void) {
    /* the /v1/tts body shape */
    const char* body =
        "{\"text\": \"Hallo \\\"Welt\\\" \\u00e4\\u00f6\\u00fc\", "
        "\"format\": \"wav\", \"temperature\": 0.8, \"top_p\": 0.8, "
        "\"seed\": 42, \"nested\": {\"a\": [1, 2.5, -3, true, false, null]}}";
    s2p_json* j = NULL;
    T(s2p_json_parse(body, strlen(body), &j) == S2P_OK && j, "parse tts body");
    if (!j) return 1;
    const s2p_jval* root = s2p_json_root(j);
    T(root != NULL, "root");

    const s2p_jval* text = s2p_jobj_get(root, "text");
    T(text && s2p_jis_str(text), "text is str");
    size_t tl = 0;
    const char* ts = s2p_jstr(text, &tl);
    T(ts && tl == strlen("Hallo \"Welt\" \xc3\xa4\xc3\xb6\xc3\xbc") &&
          memcmp(ts, "Hallo \"Welt\" \xc3\xa4\xc3\xb6\xc3\xbc", tl) == 0,
      "escapes + unicode decoded");

    const s2p_jval* temp = s2p_jobj_get(root, "temperature");
    T(temp && s2p_jnum(temp) > 0.79 && s2p_jnum(temp) < 0.81, "temperature");
    const s2p_jval* seed = s2p_jobj_get(root, "seed");
    T(seed && s2p_jint(seed) == 42, "seed int");
    T(s2p_jobj_get(root, "missing") == NULL, "missing key -> NULL");
    T(s2p_jobj_len(root) == 6, "obj len");

    const char* k = NULL;
    const s2p_jval* v = NULL;
    T(s2p_jobj_at(root, 0, &k, &v) == S2P_OK && k && strcmp(k, "text") == 0,
      "obj at 0");

    const s2p_jval* nested = s2p_jobj_get(root, "nested");
    const s2p_jval* arr = s2p_jobj_get(nested, "a");
    T(arr && s2p_jarr_len(arr) == 6, "array len");
    T(s2p_jint(s2p_jarr_at(arr, 0)) == 1, "arr[0] int");
    T(s2p_jnum(s2p_jarr_at(arr, 1)) == 2.5, "arr[1] num");
    T(s2p_jint(s2p_jarr_at(arr, 2)) == -3, "arr[2] negative");
    T(s2p_jbool(s2p_jarr_at(arr, 3)) == 1, "arr[3] true");
    T(s2p_jbool(s2p_jarr_at(arr, 4)) == 0, "arr[4] false");
    T(s2p_jis_null(s2p_jarr_at(arr, 5)), "arr[5] null");
    T(s2p_jarr_at(arr, 6) == NULL, "arr oob -> NULL");
    s2p_json_free(j);

    /* malformed inputs must fail cleanly, not crash */
    const char* bad[] = { "{",       "{\"a\":}",  "[1,2",   "\"unterminated",
                          "{\"a\" 1}", "nul",     "{}trail garbage",
                          "{\"a\":1e}" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        s2p_json* bj = NULL;
        s2p_status rc = s2p_json_parse(bad[i], strlen(bad[i]), &bj);
        T(rc != S2P_OK, "reject malformed");
        if (rc == S2P_OK) s2p_json_free(bj);
    }

    /* top-level scalars are valid JSON */
    s2p_json* sj = NULL;
    T(s2p_json_parse("155776", 6, &sj) == S2P_OK &&
          s2p_jint(s2p_json_root(sj)) == 155776,
      "top-level int");
    if (sj) s2p_json_free(sj);

    if (failures) {
        printf("selftest_json: %d FAILURES\n", failures);
        return 1;
    }
    printf("selftest_json: all passed\n");
    return 0;
}
