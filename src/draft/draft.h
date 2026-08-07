/* s2pro-native — speculative-decoding draft model (docs/SPECULATIVE.md §7).
 *
 * One backbone-shaped block behind a fuse linear, predicting the next
 * final-normed hidden from [h_t ; e_{t+1}]. Weights come from the trained
 * draft.safetensors (bf16); the forward reuses the frozen kernel contract
 * (rms_norm, qk_norm, rope, kv_append, attention, silu_mul, gemm_bf16).
 * RoPE positions are PER-TAKE frame indices (the draft trained on take
 * sequences), so each session's draft state resets with the session.
 */
#pragma once

#include <cuda_runtime.h>

#include "s2pro/status.h"
#include "s2pro/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define S2P_DRAFT_DIM 2560
#define S2P_DRAFT_FFN 9728
#define S2P_DRAFT_CTX 4096

typedef struct s2p_draft_model {
    /* device bf16 weights, engine layout [out, in] */
    s2p_tensor fuse;                      /* [2560, 5120] */
    s2p_tensor wqkv;                      /* [6144, 2560] */
    s2p_tensor wo;                        /* [2560, 4096] */
    s2p_tensor w1, w3;                    /* [9728, 2560] */
    s2p_tensor w2;                        /* [2560, 9728] */
    s2p_tensor q_norm, k_norm;            /* [128] */
    s2p_tensor attn_norm, ffn_norm;       /* [2560] */
    s2p_tensor final_norm;                /* [2560] */
    /* scratch, single row */
    s2p_tensor xcat;                      /* [1, 5120] */
    s2p_tensor x, a;                      /* [1, 2560] */
    s2p_tensor qkv;                       /* [1, 6144] */
    s2p_tensor attn;                      /* [1, 4096] */
    s2p_tensor gu;                        /* [1, 2*9728] */
    s2p_tensor ffn;                       /* [1, 9728] */
} s2p_draft_model;

typedef struct s2p_draft_state {
    s2p_tensor kc, vc; /* bf16 [8, S2P_DRAFT_CTX, 128] each */
    int pos;           /* frames consumed (draft rope position) */
} s2p_draft_state;

/* Load draft.safetensors (the trainer's fixed tensor names). */
s2p_status s2p_draft_load(s2p_draft_model* m, const char* path);
void       s2p_draft_free(s2p_draft_model* m);

s2p_status s2p_draft_state_init(s2p_draft_state* s);
void       s2p_draft_state_free(s2p_draft_state* s);
static inline void s2p_draft_state_reset(s2p_draft_state* s) { s->pos = 0; }

/* One step: consume (h [2560 bf16], e [2560 bf16]) at the state's next
 * position, emit the predicted next final-normed hidden into out [2560
 * bf16]. All pointers device-resident; advances s->pos. */
s2p_status s2p_draft_step(s2p_draft_model* m, s2p_draft_state* s,
                          const void* h, const void* e, void* out,
                          cudaStream_t st);

#ifdef __cplusplus
}
#endif
