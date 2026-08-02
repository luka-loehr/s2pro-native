/* s2pro-native — tokenizer + prompt-builder selftest (public s2pro/tokenizer.h
 * contract only). Needs a model dir with tokenizer.json:
 *
 *   selftest_tok MODEL_DIR
 *
 * Checks encode/decode round-trips and the hand-written ChatML prompt shape
 * from docs/PORTING.md §3 (im_start open, <|voice|> tail, semantic-token
 * placement + vq_mask alignment for a reference clip). Exit 0 on pass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/tokenizer.h"

#define TOK_IM_START 151644

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

static int roundtrip(s2p_tok* tok, const char* text) {
    int64_t* ids = NULL;
    int      n = 0;
    if (s2p_tok_encode(tok, text, &ids, &n) != S2P_OK || n <= 0) return 0;
    char* back = NULL;
    if (s2p_tok_decode(tok, ids, n, &back) != S2P_OK) {
        free(ids);
        return 0;
    }
    int ok = strcmp(back, text) == 0;
    if (!ok)
        fprintf(stderr, "     roundtrip mismatch: '%s' -> '%s'\n", text, back);
    free(ids);
    free(back);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    s2p_tok* tok = NULL;
    if (s2p_tok_load(argv[1], &tok) != S2P_OK) {
        fprintf(stderr, "FAIL tokenizer load from %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "ok   tokenizer load\n");

    T(roundtrip(tok, "Hello, world!"), "roundtrip ascii");
    T(roundtrip(tok, "Die Straße führt über die Brücke — größer als 100 €."),
      "roundtrip german umlauts");
    T(roundtrip(tok, "Line one\nline two\n\n  indented, it's 42."),
      "roundtrip whitespace + contraction");
    T(roundtrip(tok, "日本語のテキストと emoji 🎉 mixed in"),
      "roundtrip multibyte");

    s2p_config* cfg = NULL;
    (void)s2p_config_load(argv[1], &cfg); /* optional for prompt building */

    /* -- plain prompt (no refs): must end with assistant open + <|voice|>. */
    s2p_request_text req;
    memset(&req, 0, sizeof(req));
    req.text = "Guten Tag, this is a test.";
    int64_t*     ids = NULL;
    uint8_t*     mask = NULL;
    s2p_vq_part* parts = NULL;
    int          n = 0, n_parts = 0;
    T(s2p_prompt_build(tok, cfg, &req, &ids, &mask, &n, &parts, &n_parts, NULL) ==
              S2P_OK &&
          n > 0,
      "prompt build (no refs)");
    if (n > 0) {
        T(ids[0] == TOK_IM_START, "prompt starts with <|im_start|>");
        T(ids[n - 1] == 151673, "prompt ends with <|voice|>");
        T(n_parts == 0, "no vq parts without refs");
        int any_mask = 0;
        for (int i = 0; i < n; i++) any_mask |= mask[i];
        T(!any_mask, "vq_mask all-0 without refs");
    }
    free(ids);
    free(mask);
    free(parts);

    /* -- prompt with one reference clip: cb0 codes become semantic ids at
     * masked positions; the full [10,T] rides along as one vq part. */
    enum { RT = 5 };
    int32_t ref_codes[S2P_NUM_CODEBOOKS * RT];
    for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
        for (int t = 0; t < RT; t++)
            ref_codes[cb * RT + t] = cb == 0 ? (100 + t) : (cb * 7 + t) % 1024;
    s2p_vq_part ref = { ref_codes, RT };
    req.ref_text = "Referenztext for the clip.";
    req.refs = &ref;
    req.n_refs = 1;
    ids = NULL; mask = NULL; parts = NULL; n = 0; n_parts = 0;
    T(s2p_prompt_build(tok, cfg, &req, &ids, &mask, &n, &parts, &n_parts, NULL) ==
              S2P_OK &&
          n > 0,
      "prompt build (1 ref)");
    if (n > 0) {
        T(n_parts == 1 && parts[0].T == RT, "one vq part, T preserved");
        int masked = 0, sem_ok = 1, run_start = -1;
        for (int i = 0; i < n; i++) {
            if (!mask[i]) continue;
            if (run_start < 0) run_start = i;
            if (ids[i] != S2P_SEMANTIC_ID(100 + masked)) sem_ok = 0;
            masked++;
        }
        T(masked == RT, "vq_mask marks exactly T positions");
        T(sem_ok, "masked ids are <|semantic:cb0|>");
        if (run_start >= 0) {
            int contiguous = 1;
            for (int i = 0; i < RT; i++)
                if (!mask[run_start + i]) contiguous = 0;
            T(contiguous, "semantic block contiguous");
        }
    }
    free(ids);
    free(mask);
    free(parts);

    if (cfg) s2p_config_free(cfg);
    s2p_tok_free(tok);

    if (failures) {
        printf("selftest_tok: %d FAILURES\n", failures);
        return 1;
    }
    printf("selftest_tok: all passed\n");
    return 0;
}
