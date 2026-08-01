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
} s2p_request_text;

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
