/* s2pro-native — BPE tokenizer + ChatML prompt builder. Contract header: frozen.
 *
 * Loads model/tokenizer.json (HF format, byte-level BPE). The prompt is built
 * BY HAND per docs/PORTING.md — NOT via the Jinja chat template.
 */
#pragma once

#include <stdint.h>
#include "s2pro/status.h"
#include "s2pro/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_tok s2p_tok;

s2p_status s2p_tok_load(const char* model_dir, s2p_tok** out);
/* ids/out buffers are malloc'd; caller frees. */
s2p_status s2p_tok_encode(s2p_tok* t, const char* utf8, int64_t** ids, int* n);
s2p_status s2p_tok_decode(s2p_tok* t, const int64_t* ids, int n, char** utf8);
void       s2p_tok_free(s2p_tok* t);

/* One reference clip: 10 codebooks x T frames, codebook-major [10*T]. */
typedef struct {
    const int32_t* codes; /* row cb, col t: codes[cb*T + t] */
    int            T;
} s2p_vq_part;

typedef struct {
    const char*        text;      /* required: text to speak */
    const char*        ref_text;  /* optional: transcript of reference audio */
    const s2p_vq_part* refs;      /* optional: encoded reference clips */
    int                n_refs;
    /* On-the-fly cloning (additive, 2026-08-01): raw 44.1 kHz mono float
     * PCM to be DAC-encoded by the SCHEDULER on its worker thread (keeps
     * all GPU work single-threaded). Used only when refs is empty; the
     * prompt builder itself ignores these fields. */
    const float*       ref_pcm;
    int64_t            ref_pcm_n;
} s2p_request_text;

/* Sentence-aware UTF-8 text chunking for long-form synthesis (additive,
 * 2026-08-02; src/text/chunker.c). Splits at sentence boundaries and packs
 * greedily up to target_bytes per chunk (>= 32); a lone oversize sentence
 * is split at the last space. Returns a malloc'd array of malloc'd strings;
 * free with s2p_text_chunks_free. */
s2p_status s2p_text_chunks(const char* utf8, int target_bytes, char*** out,
                           int* out_n);
void s2p_text_chunks_free(char** chunks, int n);

/* Build the full prompt: token ids, a mask marking VQ-injected positions
 * (1 where codebook-0 semantic tokens sit and vq_parts embeddings must be
 * added), and the vq parts array consumed by prefill. All outputs malloc'd. */
s2p_status s2p_prompt_build(s2p_tok* t, const s2p_config* cfg,
                            const s2p_request_text* req, int64_t** ids,
                            uint8_t** vq_mask, int* n_ids,
                            s2p_vq_part** parts, int* n_parts);

#ifdef __cplusplus
}
#endif
