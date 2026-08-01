/* s2pro-native — hand-written ChatML prompt builder (docs/PORTING.md §3).
 *
 * Replicates S2ProTokenizerAdapter.build_prompt EXACTLY — per-segment
 * tokenization (BPE merges never cross a segment boundary), the literal
 * strings below including every newline, and the two injection channels of a
 * reference clip: codebook-0 codes become real <|semantic:i|> ids in
 * input_ids with vq_mask=1, while ALL 10 codebooks ride along as ONE
 * time-concatenated vq part for prefill embedding injection. Multiple refs
 * share exactly ONE "\n\nSpeech:\n" header and ONE semantic block.
 *
 * Layout (refs present):
 *   <|im_start|>system\n
 *   convert the provided text to speech reference to the following:\n\nText:\n
 *   <|speaker:0|>{ref_text}            (only if ref_text non-empty)
 *   \n\nSpeech:\n
 *   <|semantic:c|>... (cb0 of all refs, time-concatenated)   [vq_mask=1]
 *   <|im_end|>\n
 * then always:
 *   <|im_start|>user\n
 *   <|speaker:0|>{text}
 *   <|im_end|>\n
 *   <|im_start|>assistant\n<|voice|>   (generation starts right after)
 *
 * Speaker is fixed to 0 (the reference default; the request struct carries no
 * speaker field). "<|speaker:0|>" is ORDINARY TEXT — byte-level BPE'd, never
 * a special token (PORTING §3).
 *
 * Ownership: *ids, *vq_mask, *parts and parts[0].codes are malloc'd; caller
 * frees all four (codes via free((void*)parts[i].codes)). cfg may be NULL —
 * every constant needed here is compile-time (config.h).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "s2pro/tokenizer.h"

typedef struct {
    int64_t* ids;
    uint8_t* mask;
    int      n, cap;
} pb_buf;

static s2p_status pb_reserve(pb_buf* b, int extra) {
    if (b->n + extra <= b->cap) return S2P_OK;
    int cap = b->cap ? b->cap : 256;
    while (cap < b->n + extra) cap *= 2;
    int64_t* ids = (int64_t*)realloc(b->ids, (size_t)cap * sizeof(int64_t));
    if (!ids) return S2P_ERR_OOM;
    b->ids = ids;
    uint8_t* mask = (uint8_t*)realloc(b->mask, (size_t)cap);
    if (!mask) return S2P_ERR_OOM;
    b->mask = mask;
    b->cap = cap;
    return S2P_OK;
}

/* append_text: tokenize the literal string independently, mask all-zero */
static s2p_status pb_text(s2p_tok* t, pb_buf* b, const char* s) {
    int64_t* ids = NULL;
    int n = 0;
    S2P_TRY(s2p_tok_encode(t, s, &ids, &n));
    s2p_status rc = pb_reserve(b, n);
    if (rc == S2P_OK) {
        memcpy(b->ids + b->n, ids, (size_t)n * sizeof(int64_t));
        memset(b->mask + b->n, 0, (size_t)n);
        b->n += n;
    }
    free(ids);
    return rc;
}

/* "<|speaker:0|>" + body as ONE segment (merges may cross the boundary
 * between the tag and the body, exactly like the reference f-string). */
static s2p_status pb_text_speaker(s2p_tok* t, pb_buf* b, const char* body) {
    static const char tag[] = "<|speaker:0|>";
    size_t n = sizeof(tag) - 1 + strlen(body) + 1;
    char* s = (char*)malloc(n);
    if (!s) return S2P_ERR_OOM;
    memcpy(s, tag, sizeof(tag) - 1);
    strcpy(s + sizeof(tag) - 1, body);
    s2p_status rc = pb_text(t, b, s);
    free(s);
    return rc;
}

s2p_status s2p_prompt_build(s2p_tok* t, const s2p_config* cfg,
                            const s2p_request_text* req, int64_t** ids,
                            uint8_t** vq_mask, int* n_ids,
                            s2p_vq_part** parts, int* n_parts) {
    (void)cfg;
    if (!t || !req || !req->text || !ids || !vq_mask || !n_ids || !parts ||
        !n_parts)
        return S2P_ERR_INVALID;
    if (req->n_refs < 0 || (req->n_refs > 0 && !req->refs))
        return S2P_ERR_INVALID;

    pb_buf b = {0};
    s2p_vq_part* out_parts = NULL;
    int32_t* cat = NULL;
    int np = 0;
    s2p_status rc = S2P_OK;

    /* system block: one block for ALL refs combined */
    if (req->n_refs > 0) {
        int64_t total_T = 0;
        for (int i = 0; i < req->n_refs; i++) {
            if (req->refs[i].T < 0 || (req->refs[i].T > 0 && !req->refs[i].codes)) {
                rc = S2P_ERR_INVALID;
                goto fail;
            }
            total_T += req->refs[i].T;
        }
        if (total_T > INT32_MAX / S2P_NUM_CODEBOOKS) { rc = S2P_ERR_INVALID; goto fail; }

        rc = pb_text(t, &b, "<|im_start|>system\n");
        if (rc == S2P_OK)
            rc = pb_text(t, &b,
                         "convert the provided text to speech reference to "
                         "the following:\n\nText:\n");
        if (rc == S2P_OK && req->ref_text && req->ref_text[0])
            rc = pb_text_speaker(t, &b, req->ref_text);
        if (rc == S2P_OK) rc = pb_text(t, &b, "\n\nSpeech:\n");
        if (rc != S2P_OK) goto fail;

        if (total_T > 0) {
            /* append_vq(cat(all refs, dim=1)): time-concatenate cb-major */
            int T = (int)total_T;
            cat = (int32_t*)malloc((size_t)S2P_NUM_CODEBOOKS * T *
                                   sizeof(int32_t));
            if (!cat) { rc = S2P_ERR_OOM; goto fail; }
            int off = 0;
            for (int i = 0; i < req->n_refs; i++) {
                const s2p_vq_part* r = &req->refs[i];
                for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
                    memcpy(cat + (size_t)cb * T + off,
                           r->codes + (size_t)cb * r->T,
                           (size_t)r->T * sizeof(int32_t));
                off += r->T;
            }
            /* cb0 -> real semantic ids, vq_mask=1 across the block */
            rc = pb_reserve(&b, T);
            if (rc != S2P_OK) goto fail;
            for (int k = 0; k < T; k++) {
                int32_t c = cat[k]; /* row 0 */
                if (c < 0) c = 0;
                if (c >= S2P_CB_SIZE) c = S2P_CB_SIZE - 1;
                b.ids[b.n]  = S2P_SEMANTIC_ID(c);
                b.mask[b.n] = 1;
                b.n++;
            }
            out_parts = (s2p_vq_part*)malloc(sizeof(s2p_vq_part));
            if (!out_parts) { rc = S2P_ERR_OOM; goto fail; }
            out_parts[0].codes = cat;
            out_parts[0].T = T;
            cat = NULL; /* owned by out_parts now */
            np = 1;
        }
        rc = pb_text(t, &b, "<|im_end|>\n");
        if (rc != S2P_OK) goto fail;
    }

    /* user block */
    rc = pb_text(t, &b, "<|im_start|>user\n");
    if (rc == S2P_OK) rc = pb_text_speaker(t, &b, req->text);
    if (rc == S2P_OK) rc = pb_text(t, &b, "<|im_end|>\n");
    /* assistant open — no trailing newline; decode starts after <|voice|> */
    if (rc == S2P_OK) rc = pb_text(t, &b, "<|im_start|>assistant\n<|voice|>");
    if (rc != S2P_OK) goto fail;

    *ids = b.ids;
    *vq_mask = b.mask;
    *n_ids = b.n;
    *parts = out_parts;
    *n_parts = np;
    return S2P_OK;

fail:
    free(b.ids);
    free(b.mask);
    free(cat);
    if (out_parts) {
        free((void*)out_parts[0].codes);
        free(out_parts);
    }
    return rc;
}
