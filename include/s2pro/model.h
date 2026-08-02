/* s2pro-native — Dual-AR model: slow-AR backbone + fast-AR decoder.
 * Contract header: frozen.
 *
 * A session = one generation stream. Per frame: slow-AR samples ONE semantic
 * token (two-softmax order per docs/PORTING.md — numerically load-bearing),
 * fast-AR then argmax-emits the 9 residual codebooks conditioned on the
 * final-normed hidden state and the sampled semantic id.
 * out_codes layout per frame: [semantic, resid1..resid9].
 */
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_model   s2p_model;
typedef struct s2p_session s2p_session;

typedef struct {
    s2p_gemm_mode gemm_mode;
    int           ctx_len;       /* 0 -> S2P_CTX_LEN_DEFAULT */
    int           max_sessions;  /* 0 -> S2P_MAX_SESSIONS */
} s2p_model_opts;

typedef struct {
    float    temperature;   /* 0 -> greedy */
    float    top_p;
    float    repetition_penalty;
    int      repetition_window;
    uint64_t seed;
} s2p_sampling_cfg;

s2p_sampling_cfg s2p_sampling_defaults(void); /* reference defaults, PORTING §sampling */

s2p_status s2p_model_load(const char* model_dir, const s2p_model_opts* opts,
                          s2p_model** out);
void       s2p_model_free(s2p_model* m);

s2p_status s2p_session_create(s2p_model* m, const s2p_sampling_cfg* cfg,
                              s2p_session** out);
/* Prefill the full prompt. vq_mask marks positions whose input embedding is
 * the sum of the token embedding and the 10-codebook VQ embedding (PORTING
 * §VQ injection). parts are consumed in prompt order. */
s2p_status s2p_session_prefill(s2p_session* s, const int64_t* ids,
                               const uint8_t* vq_mask, int n_ids,
                               const s2p_vq_part* parts, int n_parts);

/* KV prefix caching (additive, 2026-08-02). A constant prompt prefix (the
 * per-voice system block incl. VQ injection) is prefetched once and reused:
 * save copies the first n_tokens of a prefilled session's KV into a caller
 * device blob; load seeds a NEW session's KV from the blob, after which
 * s2p_session_prefill continues at that offset with the remaining ids
 * (attention, RoPE and the KV append are offset-native). Blob layout is the
 * session KV's plane order at n_tokens depth; s2p_session_kv_bytes sizes it. */
size_t     s2p_session_kv_bytes(const s2p_model* m, int n_tokens);
s2p_status s2p_session_kv_save(s2p_session* s, int n_tokens, void* blob_dev);
s2p_status s2p_session_kv_load(s2p_session* s, const void* blob_dev,
                               int n_tokens);
/* One frame: 10 codes out. *is_eos set when generation finished (the frame
 * containing EOS produces no codes). */
s2p_status s2p_session_next_frame(s2p_session* s,
                                  int32_t out_codes[S2P_NUM_CODEBOOKS],
                                  int* is_eos);
void       s2p_session_destroy(s2p_session* s);

/* Lockstep batched decode: one call advances every active session one frame.
 * out_codes is [n][S2P_NUM_CODEBOOKS]; eos is [n]. Sessions must belong to m. */
s2p_status s2p_model_batch_next_frame(s2p_model* m, s2p_session** sess, int n,
                                      int32_t* out_codes, int* eos);

#ifdef __cplusplus
}
#endif
