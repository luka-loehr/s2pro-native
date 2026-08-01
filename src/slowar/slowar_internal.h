/* s2pro-native — slow-AR internal structures (private to src/slowar).
 *
 * The slow-AR module owns: the 36-layer Qwen3 backbone weights, per-session
 * KV caches, prefill + lockstep batched decode, the exact two-softmax
 * semantic sampler (docs/PORTING.md §5), and the handoff to the fast-AR
 * (src/fastar/fastar.h) for VQ embedding sums and residual codebooks.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/tensor.h"
#include "s2pro/safetensors.h"
#include "s2pro/gemm.h"
#include "s2pro/model.h"

#include "slowar_kernels.h" /* also provides the C-mode __nv_bfloat16 shim */
#include "../fastar/fastar.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ----- sampling (sampling.c) --------------------------------------------- */

/* Reference constants (sglang_model.py). Not user-tunable. */
#define S2PS_TOP_K 30           /* _GRAPH_TOP_K, fixed decode width */
#define S2PS_RAS_TEMPERATURE 1.0f
#define S2PS_RAS_TOP_P 0.9f
#define S2PS_MAX_WINDOW 64      /* cap on cfg.repetition_window */
#define S2PS_DEF_WINDOW 16      /* rep_history_len default */
#define S2PS_N_SEM 4096         /* semantic candidates */
#define S2PS_N_CAND (S2PS_N_SEM + 1) /* + im_end -> exactly 4097 unmasked */
#define S2PS_CAND_EOS S2PS_N_SEM     /* candidate index of im_end */

typedef struct {
    /* effective request params */
    float temperature; /* 0 -> greedy (contract), else reference clamp 1e-5 */
    float top_p;
    float rep_penalty;
    int window; /* rep history length (reference: 16) */
    /* seeded path (multinomial_with_seed replication) */
    int seeded;
    uint32_t seed31; /* seed & 0x7FFFFFFF, per resolve_row_seed */
    /* history: raw vocab ids of emitted semantic tokens, oldest first */
    int64_t prev[S2PS_MAX_WINDOW];
    uint64_t count; /* UNCAPPED emitted count == step counter (pre-draw) */
    /* unseeded path RNG (xoshiro256**) */
    uint64_t rng[4];
} s2ps_sampler;

void s2ps_sampler_init(s2ps_sampler* sp, const s2p_sampling_cfg* cfg,
                       uint64_t entropy);

/* One semantic draw. sem_bf16: the 4096 bf16 logits at vocab ids
 * [S2P_TOK_SEMANTIC_START .. S2P_TOK_SEMANTIC_END]; eos_bf16: the bf16 logit
 * at S2P_TOK_EOS. Everything else is -inf after the semantic bias and can
 * never enter the top-30, so only these 4097 values participate — exactly
 * the reference candidate set. Returns the raw vocab id; on non-EOS pushes
 * it into the history and advances the step counter (EOS frames leave both
 * untouched, mirroring collect_s2pro_step_outputs). */
int64_t s2ps_sample(s2ps_sampler* sp, const uint16_t* sem_bf16,
                    uint16_t eos_bf16, int32_t* out_sem_id, int* out_eos);

/* ----- model / session (model.c, slowar.c) ------------------------------- */

/* Hard cap on concurrent sessions / lockstep batch rows (host-side staging
 * arrays are statically sized to this; s2p_model_load clamps opts). */
#define S2P_SLOWAR_MAX_BATCH 64

typedef struct {
    s2p_linear wqkv;    /* [6144, 2560] fused q,k,v */
    s2p_linear wo;      /* [2560, 4096] */
    s2p_linear gate_up; /* [19456, 2560] = w1 rows then w3 rows */
    s2p_linear w2;      /* [2560, 9728] */
    s2p_tensor attn_norm; /* [2560] */
    s2p_tensor ffn_norm;  /* [2560] */
    s2p_tensor q_norm;    /* [128] per-head RMSNorm */
    s2p_tensor k_norm;    /* [128] */
} s2p_slow_layer;

/* Per-frame device upload block layout (one pinned->device memcpy).
 * Order chosen for natural alignment: ptr table, ids, codes, pos, mask. */
typedef struct {
    size_t off_ptrs;  /* void* x 36*2*B: k ptrs [l*B+b], then v ptrs */
    size_t off_ids;   /* int64 x B  : previous semantic token per row */
    size_t off_codes; /* int32 x 10B: per-run cb-major prev-frame codes */
    size_t off_pos;   /* int32 x B  : per-row kv position */
    size_t off_mask;  /* uint8 x B  : row gets VQ combine */
    size_t total;
} s2p_upload_layout;

struct s2p_model {
    s2p_gemm_mode mode;
    int ctx_len;
    int max_sessions;
    int n_sessions;
    cudaStream_t stream;

    /* weights */
    s2p_tensor embed; /* [155776, 2560] bf16, tied lm_head; ALWAYS kept —
                       * embedding lookups and the M>8 head fallback read it */
    /* INT8 sidecar of the tied head (mode == S2P_GEMM_INT8 only): the head
     * is 0.8 GB of bf16 traffic per frame, ~1/5 of the whole decode read. */
    s2p_tensor embed_i8;    /* [155776, 2560] int8 */
    s2p_tensor embed_scale; /* [155776] f32 per-row */
    s2p_slow_layer layers[S2P_SLOW_LAYERS];
    s2p_tensor final_norm; /* [2560] */
    s2pfa* fastar;

    /* device scratch (rows = ctx_len for prefill, = max_sessions for decode) */
    s2p_tensor sx;      /* [ctx, 2560] residual stream */
    s2p_tensor snorm;   /* [ctx, 2560] */
    s2p_tensor sqkv;    /* [ctx, 6144] */
    s2p_tensor sq;      /* [ctx, 4096] */
    s2p_tensor sk;      /* [ctx, 1024] */
    s2p_tensor sv;      /* [ctx, 1024] */
    s2p_tensor sattn;   /* [ctx, 4096] */
    s2p_tensor sproj;   /* [ctx, 2560] */
    s2p_tensor sgu;     /* [ctx, 19456] */
    s2p_tensor sffn;    /* [ctx, 9728] */
    s2p_tensor shidden; /* [max_sessions, 2560] final-normed batch hidden */
    s2p_tensor slogits; /* [max_sessions, 155776] bf16 */
    s2p_tensor sattn_part; /* [max_sessions, 32, ctx/64, 130] f32 split-K
                            * flash-decode partials (s2pk_attention_decode) */
    /* device sampling: per-frame output block (fixed layout by max_sessions:
     * i64 tok[B] | i32 codes[B*10] | u8 eos[B]) + state-pointer table */
    void*   d_out;
    void*   h_out;         /* pinned mirror, ONE D2H per frame */
    size_t  out_bytes;
    size_t  out_off_sem, out_off_codes, out_off_eos;
    void**  d_sampptr;     /* [B] device: per-row s2ps_dev_state* */
    void**  h_sampptr;     /* pinned mirror */
    s2p_tensor sids;    /* [ctx] i64 prefill ids */
    s2p_tensor svq;     /* [10*ctx] i32 prefill VQ codes staging */

    /* per-frame upload block */
    s2p_upload_layout up;
    void* d_up;  /* device */
    void* h_up;  /* pinned host mirror */
    /* pinned download staging */
    uint16_t* h_sem;   /* [max_sessions * 4097] bf16 candidate logits */
    int32_t* h_frame;  /* [max_sessions * 10] frame codes (cb0 = sem_id,
                          cb1..9 written by s2pfa_decode_frame_batch) */
    int32_t* h_semid;  /* [max_sessions] sampled semantic ids (pinned) */
};

typedef enum {
    S2P_SESS_NEW = 0,   /* created, no prompt yet */
    S2P_SESS_PREFILLED, /* prompt in KV, pending_hidden holds frame-0 state */
    S2P_SESS_DECODING,  /* >= 1 frame emitted, feedback token pending */
    S2P_SESS_FINISHED,  /* EOS seen (or context exhausted) */
} s2p_sess_state;

struct s2p_session {
    s2p_model* m;
    s2p_sess_state state;
    int kv_len; /* tokens resident in the KV cache */
    s2p_tensor kv; /* [36][2*8][ctx][128] bf16: per layer k then v planes */
    s2p_tensor pending_hidden; /* [2560] final-normed prefill hidden */
    int64_t prev_token;    /* raw sampled vocab id fed back next frame */
    int32_t prev_codes[S2P_NUM_CODEBOOKS]; /* [sem_id, resid1..9] */
    void*   dsamp;         /* device s2ps_dev_state (kernels.cu sampler) */
    s2ps_sampler sampler;
};

/* ----- parity dump hooks (debug_dump.c) ----------------------------------
 * Active only when S2P_DUMP_DIR is set; every call is a no-op otherwise.
 * Device bf16 vectors are downloaded, widened to f32, and written as raw
 * little-endian f32 files "<dir>/<name>.f32" for tools/parity_compare.py. */
const char* s2psl_dump_dir(void);
void s2psl_dump_vec_bf16(const char* name, const void* dev_bf16, int64_t n,
                         cudaStream_t st);

/* layer l k/v cache base pointers inside a session's kv tensor */
static inline __nv_bfloat16* s2p_kv_k(const s2p_session* s, int layer) {
    return (__nv_bfloat16*)s->kv.data +
           (size_t)layer * 2 * S2P_SLOW_KV_HEADS * s->m->ctx_len * S2P_HEAD_DIM;
}
static inline __nv_bfloat16* s2p_kv_v(const s2p_session* s, int layer) {
    return s2p_kv_k(s, layer) +
           (size_t)S2P_SLOW_KV_HEADS * s->m->ctx_len * S2P_HEAD_DIM;
}

#ifdef __cplusplus
}
#endif
