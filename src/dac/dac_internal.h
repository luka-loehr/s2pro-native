/* s2pro-native — DAC module private header (src/dac only; NOT contract).
 *
 * Layout conventions used throughout the module:
 *   - Conv-stage activations are channels-first f32: x[c*T + t]  ("[C,T]").
 *   - Transformer/ConvNeXt inner activations are channels-last f32:
 *     x[t*C + c] ("[T,C]"), produced by explicit transposes.
 *   - All math is float32. The only bf16 in the codec is the RoPE cos/sin
 *     table, which is computed in f32 and ROUNDED to bf16 before use
 *     (PORTING.md pitfall 1); we store the bf16-rounded values as f32.
 *   - Weights come pre-folded (weight-norm g*v/||v|| already materialized)
 *     from the converted codec.bin/codec.idx artifact; see dac_weights.c for
 *     the on-disk format.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/dac.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- codec.bin / codec.idx loader (dac_weights.c) ---------- */

typedef struct {
    char    name[160];
    int64_t off;        /* byte offset into codec.bin */
    int64_t nbytes;     /* byte length (= 4 * prod(dims)) */
    int     ndim;       /* 1..4 */
    int64_t d[4];       /* dims, row-major; unused trail = 1 */
    int64_t h_off;      /* f16 storage: byte offset into base_h, else -1 */
    int64_t f_off;      /* f32 keeper arena offset, else -1 */
} s2p_dacw_ent;

typedef struct {
    s2p_dacw_ent* ents;   /* sorted as listed in codec.idx */
    int           n_ents;
    float*        base;   /* device f32 blob; freed after f16 conversion */
    void*         base_h; /* f16 weights (conv/matmul w-class tensors) */
    float*        base_f; /* f32 keepers (biases, alphas, norms, codebooks) */
    int           f16;    /* conversion active (S2P_DAC_F32=1 disables) */
    int64_t       total_bytes;
} s2p_dacw;

s2p_status s2p_dacw_load(const char* model_dir, s2p_dacw* w);
void       s2p_dacw_free(s2p_dacw* w);
/* Device pointer for a tensor by exact name, NULL if absent. */
const float*        s2p_dacw_find(const s2p_dacw* w, const char* name);
const s2p_dacw_ent* s2p_dacw_ent_find(const s2p_dacw* w, const char* name);

/* ---------------- resolved weight views ---------------------------------- */

typedef struct { const float *w, *b; } s2pd_cw;              /* conv weight+bias */

typedef struct {                                             /* ResidualUnit */
    const float* alpha0;   /* Snake1d [1,C,1] */
    s2pd_cw      c7;       /* CausalConv C->C k7 dilation dil */
    const float* alpha1;
    s2pd_cw      c1;       /* CausalConv C->C k1 */
    int          dil;
} s2pd_ru;

typedef struct {                                             /* DecoderBlock */
    const float* alpha;    /* Snake1d on input dim */
    s2pd_cw      up;       /* CausalTransConv cin->cout k=2s stride s */
    s2pd_ru      ru[3];    /* dilations 1,3,9 at cout */
} s2pd_dblock;

typedef struct {                                             /* ConvNeXtBlock dim 1024 */
    s2pd_cw      dw;       /* depthwise causal conv k7, w [C,1,7] */
    const float *ln_w, *ln_b;      /* LayerNorm eps 1e-6 */
    const float *pw1_w, *pw1_b;    /* Linear C -> 4C */
    const float *pw2_w, *pw2_b;    /* Linear 4C -> C */
    const float *gamma;            /* layer scale [C] */
} s2pd_cnx;

typedef struct {                                             /* transformer layer */
    const float *wqkv;     /* [3072,1024] */
    const float *wo;       /* [1024,1024] */
    const float *attn_norm, *ffn_norm;      /* RMSNorm [1024], eps 1e-5 */
    const float *w1, *w3;  /* [3072,1024] */
    const float *w2;       /* [1024,3072] */
    const float *g_attn, *g_ffn;            /* LayerScale gamma [1024] */
} s2pd_tl;

typedef struct {                                             /* WindowLimitedTransformer */
    s2pd_tl      l[8];
    int          n_layers;
    const float* norm;     /* final RMSNorm [1024] */
    int          window;   /* causal attention window (128 pre/post, 512 encoder) */
} s2pd_tf;

typedef struct {                                             /* one VQ codebook */
    const float* cb;       /* codebook [n,8] (raw, un-normalized) */
    const float *in_w, *in_b;    /* in_proj 1x1 conv 1024->8  (encode only) */
    const float *out_w, *out_b;  /* out_proj 1x1 conv 8->1024 */
    int          n;        /* 4096 semantic / 1024 residual */
} s2pd_vq;

typedef struct {                                             /* EncoderBlock */
    s2pd_ru      ru[3];    /* at cin, dilations 1,3,9 */
    const float* alpha;    /* Snake1d at cin */
    s2pd_cw      down;     /* CausalConv cin->cout k=2s stride s */
} s2pd_eblock;

struct s2p_dac {
    s2p_dacw w;

    /* ---- decode path ---- */
    s2pd_cw      dec_conv0;      /* 1024->1536 k7 */
    s2pd_dblock  dec_b[4];       /* 1536->768 s8, 768->384 s8, 384->192 s4, 192->96 s2 */
    const float* dec_alpha;      /* model.5 Snake [96] */
    s2pd_cw      dec_convf;      /* 96->1 k7 (+tanh) */
    s2pd_cw      up_t[2];        /* transposed conv 1024->1024 k2 s2 */
    s2pd_cnx     up_cnx[2];
    s2pd_tf      post;           /* quantizer.post_module, 8L window 128 */
    s2pd_vq      sem;            /* semantic codebook, n=4096 */
    s2pd_vq      res[9];         /* residual codebooks, n=1024 */

    /* ---- encode path (weights OPTIONAL in codec.bin; see S2P_GAP in dac.c) */
    int          has_encoder;
    s2pd_cw      enc_conv0;      /* 1->64 k7 */
    s2pd_eblock  enc_b[4];       /* 64->128 s2, 128->256 s4, 256->512 s8, 512->1024 s8 */
    s2pd_tf      enc_tf;         /* encoder.block.4.block.5: 4L window 512 */
    const float* enc_alpha;      /* encoder.block.5 Snake [1024] */
    s2pd_cw      enc_convf;      /* 1024->1024 k3 */
    s2pd_cw      down_c[2];      /* CausalConv 1024->1024 k2 s2 */
    s2pd_cnx     down_cnx[2];
    s2pd_tf      pre;            /* quantizer.pre_module, 8L window 128 */

    /* ---- runtime state ---- */
    float*   rope;               /* device [rope_rows, 32, 2] f32 (bf16-rounded) */
    int      rope_rows;
    float*   ws;                 /* device workspace, 4 equal buffers */
    size_t   ws_bytes;           /* total allocation */
    size_t   ws_buf_floats;      /* floats per buffer (ws_bytes / 4 bufs / 4B) */
    int32_t* codes_dev;
    int      codes_cap;          /* frames */
};

/* Decoder-only entry: latent [1024, Tlat] host f32 (post-upsample, i.e. the
 * conv-decoder input; Tlat = 4*frames) -> PCM Tlat*512 samples. Lets smoke
 * tests bypass RVQ+post_module+upsample entirely (fixture codec_latent.npy). */
s2p_status s2p_dac_decode_latent(s2p_dac* d, const float* latent, int Tlat,
                                 float** pcm_out, int64_t* n_samples,
                                 cudaStream_t stream);

/* ---------------- incremental streaming decode (stream_inc.c) ------------ */
/* Bit-exact per-frame decode: every stateful op keeps a device history of
 * its own input (histories replace the causal zero left-pad exactly), the
 * post_module keeps a 127-row K/V tail per layer (attention window 128).
 * One pushed frame emits its 2048 samples immediately — no windows, no
 * crossfade, no re-decode. Output equals s2p_dac_decode of the full code
 * sequence bit for bit (validated by S2P_TEST_STREAM_WAV). */
typedef struct s2pd_inc s2pd_inc;
s2p_status s2pd_rope_ensure(s2p_dac* d, int rows_needed);
s2p_status s2pd_inc_create(s2p_dac* d, s2pd_inc** out);
void       s2pd_inc_destroy(s2pd_inc* s);
/* Decode ONE frame; writes 2048 samples to pcm_host. */
s2p_status s2pd_inc_push(s2pd_inc* s,
                         const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                         float* pcm_host, cudaStream_t stream);
/* Pipelined pair: push_async enqueues tn frames (frame-major [tn][10]; no
 * sync; one push in flight, tn <= 4), collect syncs and copies the samples
 * out (n_out 0 if none pending; pcm_host must hold tn*2048 floats). */
s2p_status s2pd_inc_push_async(s2pd_inc* s, const int32_t* frame_codes,
                               int tn, cudaStream_t stream);
s2p_status s2pd_inc_collect(s2pd_inc* s, float* pcm_host, int64_t* n_out,
                            cudaStream_t stream);

/* ---------------- CUDA kernel launchers ---------------------------------- */
/* All launchers enqueue on `st` and return cudaPeekAtLastError().           */

/* decoder.cu — shared primitives */
/* conv/matmul weights are void*: f32, or f16 when the load-time conversion
 * ran (s2pdk_weights_f16 selects the kernel instantiation globally). */
cudaError_t s2pdk_conv1d(const float* in, int cin, int tin,
                         const void* w, const float* b, int cout,
                         int k, int dil, int stride, int leftpad,
                         float* out, int tout, cudaStream_t st);
cudaError_t s2pdk_dwconv1d(const float* in, int c, int tin,
                           const void* w, const float* b, int k, int leftpad,
                           float* out, int tout, cudaStream_t st);
cudaError_t s2pdk_tconv1d(const float* in, int cin, int tin,
                          const void* w, const float* b, int cout,
                          int k, int stride, float* out, int tout,
                          cudaStream_t st);
void        s2pdk_weights_f16(int on);
cudaError_t s2pdk_conv1d_w32(const float* in, int cin, int tin,
                             const void* w, const float* b, int cout, int k,
                             int dil, int stride, int leftpad, float* out,
                             int tout, cudaStream_t st);
cudaError_t s2pdk_f32_to_f16(const float* src, void* dst, int64_t n,
                             cudaStream_t st);
cudaError_t s2pdk_snake(const float* in, float* out, const float* alpha,
                        int c, int t, cudaStream_t st);
cudaError_t s2pdk_tanh_ip(float* x, int64_t n, cudaStream_t st);
cudaError_t s2pdk_add_ip(float* x, const float* y, int64_t n, cudaStream_t st);
cudaError_t s2pdk_sub_ip(float* x, const float* y, int64_t n, cudaStream_t st);
cudaError_t s2pdk_transpose(const float* in, int rows, int cols, float* out,
                            cudaStream_t st);
cudaError_t s2pdk_rmsnorm(const float* in, const float* w, float eps,
                          int t, int c, float* out, cudaStream_t st);
cudaError_t s2pdk_layernorm_ip(float* x, const float* w, const float* b,
                               float eps, int t, int c, cudaStream_t st);
cudaError_t s2pdk_matmul(const float* a, int m, int k,
                         const void* w, int n, const float* bias,
                         float* out, cudaStream_t st);
cudaError_t s2pdk_rope_ip(float* x, int t, int heads, int hd, int row_stride,
                          const float* tab, cudaStream_t st);
cudaError_t s2pdk_rope_ip_off(float* x, int t0, int t, int heads, int hd,
                              int row_stride, const float* tab,
                              cudaStream_t st);
cudaError_t s2pdk_sdpa(const float* q, const float* k, const float* v,
                       int qkv_stride, int t, int heads, int hd, int window,
                       float* out, int out_stride, cudaStream_t st);
cudaError_t s2pdk_sdpa_inc(const float* q, int q_stride, const float* kv_k,
                           const float* kv_v, int kv_stride, int hl, int tn,
                           int heads, int hd, int window, float* out,
                           int out_stride, cudaStream_t st);
cudaError_t s2pdk_silu_mul_ip(float* a, const float* b, int64_t n,
                              cudaStream_t st);
cudaError_t s2pdk_gelu_ip(float* x, int64_t n, cudaStream_t st);
cudaError_t s2pdk_colscale_ip(float* x, const float* g, int t, int c,
                              cudaStream_t st);
cudaError_t s2pdk_scale_add_ip(float* x, const float* y, const float* g,
                               int t, int c, cudaStream_t st);

/* rvq.cu — from_indices */
typedef struct { const float *cb, *ow, *ob; int n; } s2pdk_rvq_tab;
typedef struct { s2pdk_rvq_tab t[S2P_NUM_CODEBOOKS]; } s2pdk_rvq_tabs;
cudaError_t s2pdk_rvq_from_indices(const int32_t* codes, int T,
                                   s2pdk_rvq_tabs tabs, float* z,
                                   cudaStream_t st);

/* encoder.cu — nearest-codebook search (encode path) */
cudaError_t s2pdk_vq_nearest(const float* ze, int T, const float* cb, int n,
                             int32_t* idx, cudaStream_t st);
cudaError_t s2pdk_vq_dequant(const float* ze, const int32_t* idx,
                             const float* cb, int T, float* zq8,
                             cudaStream_t st);

#ifdef __cplusplus
}
#endif
