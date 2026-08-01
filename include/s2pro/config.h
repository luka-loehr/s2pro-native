/* s2pro-native — model dimensions and special tokens. Contract header: frozen.
 *
 * All values verified against the S2-Pro checkpoint config.json and
 * docs/PORTING.md §2. Do NOT re-derive from configuration.py defaults.
 */
#pragma once

#include <stdint.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Slow-AR backbone (text_config, model_type fish_qwen3) ---- */
#define S2P_SLOW_LAYERS      36
#define S2P_DIM              2560
#define S2P_SLOW_Q_HEADS     32
#define S2P_SLOW_KV_HEADS    8      /* GQA 4:1 */
#define S2P_HEAD_DIM         128    /* explicit in ckpt; != dim/n_head */
#define S2P_Q_WIDTH          4096   /* 32*128 */
#define S2P_KV_WIDTH         1024   /* 8*128 */
#define S2P_QKV_WIDTH        6144   /* fused, concat order q,k,v */
#define S2P_FFN_DIM          9728   /* SwiGLU intermediate */
#define S2P_TEXT_VOCAB       155776 /* tied lm_head = embedding */
#define S2P_ROPE_BASE        1000000.0f
#define S2P_NORM_EPS         1e-6f
#define S2P_MAX_SEQ          32768
#define S2P_CTX_LEN_DEFAULT  4096
/* Slow-AR HAS per-head RMSNorm on Q and K (attention_qk_norm=true). */

/* ---- Fast-AR audio decoder (audio_decoder_config) ---- */
#define S2P_FAST_LAYERS      4
#define S2P_CB_SIZE          4096   /* codebook size == fast-AR vocab */
#define S2P_NUM_CODEBOOKS    10     /* 1 semantic + 9 residual */
#define S2P_FAST_SEQ         11     /* fast-AR KV depth = num_codebooks+1 */
/* Fast-AR has NO qk-norm and an UNTIED output head Linear(2560,4096). */
/* codebook_embeddings: Embedding(40960, 2560); embeddings: Embedding(4096, 2560) */

/* ---- Codec / framing ---- */
#define S2P_FRAME_SAMPLES    2048   /* one code frame = 2048 samples */
#define S2P_SAMPLE_RATE      44100  /* frame rate 44100/2048 ≈ 21.533 Hz */

/* ---- Special token ids (verified from checkpoint config.json) ---- */
#define S2P_TOK_EOS              151645
#define S2P_TOK_PAD              151669
#define S2P_TOK_AUDIO_PAD        151677
#define S2P_TOK_SEMANTIC_START   151678 /* <|semantic:0|> */
#define S2P_TOK_SEMANTIC_END     155773 /* <|semantic:4095|> */
#define S2P_SEMANTIC_ID(cb0)     (S2P_TOK_SEMANTIC_START + (cb0))

/* ---- Serving limits ---- */
#define S2P_MAX_SESSIONS     8

/* Runtime-loaded config (paths + anything only known at load time). */
typedef struct s2p_config s2p_config;

s2p_status s2p_config_load(const char* model_dir, s2p_config** out);
void       s2p_config_free(s2p_config* c);
/* Generic accessor for values parsed from config.json (returns 0 on miss). */
int64_t    s2p_config_i64(const s2p_config* c, const char* key);
const char* s2p_config_model_dir(const s2p_config* c);

#ifdef __cplusplus
}
#endif
