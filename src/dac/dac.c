/* s2pro-native — modded-DAC / Firefly-GAN codec: load + decode + encode.
 *
 * Decode pipeline (DAC.from_indices == decoder(quantizer.decode(indices))):
 *   codes [10,T] --clamp--> RVQ from_indices sum -> z [1024,T]
 *   -> post_module (8L WindowLimitedTransformer, causal window 128)
 *   -> upsample x2 (CausalTransConv k2 s2 + ConvNeXtBlock) -> [1024, 4T]
 *   -> Decoder: CausalConv 1024->1536 k7; 4 DecoderBlocks
 *      (Snake -> CausalTransConv k=2s -> 3 ResidualUnits dil 1/3/9),
 *      1536->768 s8 ->384 s8 ->192 s4 ->96 s2; Snake -> conv 96->1 k7 -> tanh
 *   -> PCM [T*2048] @ 44.1 kHz.
 * No length reconciliation anywhere: the decode path is exact (PORTING.md §7).
 *
 * Precision: everything f32; the only bf16 is the RoPE cos/sin table, built
 * in f32 and rounded to bf16 (values stored as f32) per PORTING.md pitfall 1.
 * RMSNorm eps 1e-5 (codec), LayerNorm eps 1e-6 (ConvNeXt), exact-erf GELU.
 *
 * Thread-safety: one in-flight call per s2p_dac (shared workspace); the
 * scheduler serializes vocoder work per model instance.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "s2pro/dac.h"
#include "dac_internal.h"

/* ------------------------------------------------------------------ utils */

static float bf16_round(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t r = v.u + 0x7FFFu + ((v.u >> 16) & 1u);   /* round-nearest-even */
    v.u = r & 0xFFFF0000u;
    return v.f;
}

/* Resolve tensor by printf-name; verifies element count when elems > 0.
 * On miss: clears *ok; logs unless quiet. */
static const float* dacw_get(s2p_dac* d, int* ok, int quiet, int64_t elems,
                         const char* fmt, ...) {
    char name[192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    const s2p_dacw_ent* e = s2p_dacw_ent_find(&d->w, name);
    if (!e) {
        *ok = 0;
        if (!quiet) fprintf(stderr, "[s2pro] dac: missing tensor %s\n", name);
        return NULL;
    }
    if (elems > 0 && e->nbytes != elems * 4) {
        *ok = 0;
        fprintf(stderr, "[s2pro] dac: tensor %s has %lld bytes, want %lld\n",
                name, (long long)e->nbytes, (long long)(elems * 4));
        return NULL;
    }
    return (const float*)((const char*)d->w.base + e->off);
}

static s2p_status rope_ensure(s2p_dac* d, int rows_needed) {
    if (rows_needed <= d->rope_rows) return S2P_OK;
    int rows = d->rope_rows > 0 ? d->rope_rows : 8192;
    while (rows < rows_needed) rows *= 2;
    /* precompute_freqs_cis(., 64, 10000): f32 angles, f32 cos/sin, bf16 round */
    size_t bytes = (size_t)rows * 32 * 2 * sizeof(float);
    float* host = (float*)malloc(bytes);
    if (!host) return S2P_ERR_OOM;
    for (int t = 0; t < rows; t++) {
        for (int j = 0; j < 32; j++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * j) / 64.0f);
            float ang = (float)t * freq;
            host[((size_t)t * 32 + j) * 2 + 0] = bf16_round(cosf(ang));
            host[((size_t)t * 32 + j) * 2 + 1] = bf16_round(sinf(ang));
        }
    }
    if (d->rope) cudaFree(d->rope);
    d->rope = NULL;
    d->rope_rows = 0;
    cudaError_t ce = cudaMalloc((void**)&d->rope, bytes);
    if (ce != cudaSuccess) { free(host); return S2P_ERR_OOM; }
    ce = cudaMemcpy(d->rope, host, bytes, cudaMemcpyHostToDevice);
    free(host);
    if (ce != cudaSuccess) return S2P_ERR_CUDA;
    d->rope_rows = rows;
    return S2P_OK;
}

/* 4 equal device buffers of `floats` each */
static s2p_status ws_ensure(s2p_dac* d, size_t floats) {
    size_t bytes = 4u * floats * sizeof(float);
    if (bytes <= d->ws_bytes) { d->ws_buf_floats = d->ws_bytes / 4 / sizeof(float); return S2P_OK; }
    if (d->ws) cudaFree(d->ws);
    d->ws = NULL;
    d->ws_bytes = 0;
    cudaError_t ce = cudaMalloc((void**)&d->ws, bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] dac: workspace alloc %zu failed: %s\n", bytes,
                cudaGetErrorString(ce));
        return S2P_ERR_OOM;
    }
    d->ws_bytes = bytes;
    d->ws_buf_floats = floats;
    return S2P_OK;
}

static s2p_status codes_ensure(s2p_dac* d, int frames) {
    if (frames <= d->codes_cap) return S2P_OK;
    int cap = d->codes_cap > 0 ? d->codes_cap : 512;
    while (cap < frames) cap *= 2;
    if (d->codes_dev) cudaFree(d->codes_dev);
    d->codes_dev = NULL;
    d->codes_cap = 0;
    S2P_CUDA_TRY(cudaMalloc((void**)&d->codes_dev,
                            (size_t)cap * S2P_NUM_CODEBOOKS * sizeof(int32_t)));
    d->codes_cap = cap;
    return S2P_OK;
}

/* ------------------------------------------------------------- load ------ */

static void load_ru(s2p_dac* d, s2pd_ru* ru, int dim, int dil, int* ok,
                    int quiet, const char* fmt, ...) {
    char pfx[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(pfx, sizeof(pfx), fmt, ap);
    va_end(ap);
    ru->dil = dil;
    ru->alpha0 = dacw_get(d, ok, quiet, dim, "%s.block.0.alpha", pfx);
    ru->c7.w = dacw_get(d, ok, quiet, (int64_t)dim * dim * 7, "%s.block.1.conv.weight", pfx);
    ru->c7.b = dacw_get(d, ok, quiet, dim, "%s.block.1.conv.bias", pfx);
    ru->alpha1 = dacw_get(d, ok, quiet, dim, "%s.block.2.alpha", pfx);
    ru->c1.w = dacw_get(d, ok, quiet, (int64_t)dim * dim, "%s.block.3.conv.weight", pfx);
    ru->c1.b = dacw_get(d, ok, quiet, dim, "%s.block.3.conv.bias", pfx);
}

static void load_cnx(s2p_dac* d, s2pd_cnx* c, int* ok, int quiet,
                     const char* pfx) {
    c->dw.w = dacw_get(d, ok, quiet, 1024 * 7, "%s.dwconv.conv.weight", pfx);
    c->dw.b = dacw_get(d, ok, quiet, 1024, "%s.dwconv.conv.bias", pfx);
    c->ln_w = dacw_get(d, ok, quiet, 1024, "%s.norm.weight", pfx);
    c->ln_b = dacw_get(d, ok, quiet, 1024, "%s.norm.bias", pfx);
    c->pw1_w = dacw_get(d, ok, quiet, 4096 * 1024, "%s.pwconv1.weight", pfx);
    c->pw1_b = dacw_get(d, ok, quiet, 4096, "%s.pwconv1.bias", pfx);
    c->pw2_w = dacw_get(d, ok, quiet, 1024 * 4096, "%s.pwconv2.weight", pfx);
    c->pw2_b = dacw_get(d, ok, quiet, 1024, "%s.pwconv2.bias", pfx);
    c->gamma = dacw_get(d, ok, quiet, 1024, "%s.gamma", pfx);
}

static void load_tf(s2p_dac* d, s2pd_tf* tf, int n_layers, int window, int* ok,
                    int quiet, const char* pfx) {
    tf->n_layers = n_layers;
    tf->window = window;
    for (int i = 0; i < n_layers; i++) {
        s2pd_tl* l = &tf->l[i];
        l->wqkv = dacw_get(d, ok, quiet, 3072 * 1024, "%s.layers.%d.attention.wqkv.weight", pfx, i);
        l->wo = dacw_get(d, ok, quiet, 1024 * 1024, "%s.layers.%d.attention.wo.weight", pfx, i);
        l->attn_norm = dacw_get(d, ok, quiet, 1024, "%s.layers.%d.attention_norm.weight", pfx, i);
        l->ffn_norm = dacw_get(d, ok, quiet, 1024, "%s.layers.%d.ffn_norm.weight", pfx, i);
        l->g_attn = dacw_get(d, ok, quiet, 1024, "%s.layers.%d.attention_layer_scale.gamma", pfx, i);
        l->g_ffn = dacw_get(d, ok, quiet, 1024, "%s.layers.%d.ffn_layer_scale.gamma", pfx, i);
        l->w1 = dacw_get(d, ok, quiet, 3072 * 1024, "%s.layers.%d.feed_forward.w1.weight", pfx, i);
        l->w2 = dacw_get(d, ok, quiet, 1024 * 3072, "%s.layers.%d.feed_forward.w2.weight", pfx, i);
        l->w3 = dacw_get(d, ok, quiet, 3072 * 1024, "%s.layers.%d.feed_forward.w3.weight", pfx, i);
    }
    tf->norm = dacw_get(d, ok, quiet, 1024, "%s.norm.weight", pfx);
}

static void load_vq(s2p_dac* d, s2pd_vq* vq, int n, int* ok_dec, int* ok_enc,
                    const char* pfx) {
    vq->n = n;
    vq->cb = dacw_get(d, ok_dec, 0, (int64_t)n * 8, "%s.codebook.weight", pfx);
    vq->out_w = dacw_get(d, ok_dec, 0, 1024 * 8, "%s.out_proj.weight", pfx);
    vq->out_b = dacw_get(d, ok_dec, 0, 1024, "%s.out_proj.bias", pfx);
    vq->in_w = dacw_get(d, ok_enc, 1, 8 * 1024, "%s.in_proj.weight", pfx);
    vq->in_b = dacw_get(d, ok_enc, 1, 8, "%s.in_proj.bias", pfx);
}

/* decoder block dims: cin, cout, kernel(=2*stride), stride */
static const int DEC_CIN[4] = { 1536, 768, 384, 192 };
static const int DEC_COUT[4] = { 768, 384, 192, 96 };
static const int DEC_K[4] = { 16, 16, 8, 4 };
static const int DEC_S[4] = { 8, 8, 4, 2 };
static const int RU_DIL[3] = { 1, 3, 9 };

/* encoder block dims: cin(=RU dim), cout, kernel(=2*stride), stride */
static const int ENC_CIN[4] = { 64, 128, 256, 512 };
static const int ENC_COUT[4] = { 128, 256, 512, 1024 };
static const int ENC_K[4] = { 4, 8, 16, 16 };
static const int ENC_S[4] = { 2, 4, 8, 8 };

s2p_status s2p_dac_load(const char* model_dir, s2p_dac** out) {
    if (!model_dir || !out) return S2P_ERR_INVALID;
    *out = NULL;
    s2p_dac* d = (s2p_dac*)calloc(1, sizeof(*d));
    if (!d) return S2P_ERR_OOM;

    s2p_status st = s2p_dacw_load(model_dir, &d->w);
    if (st != S2P_OK) { free(d); return st; }

    int ok = 1;       /* decode-path tensors: mandatory */
    int ok_enc = 1;   /* encode-path tensors: optional (converter drops them) */

    /* ---- Decoder conv stack ---- */
    d->dec_conv0.w = dacw_get(d, &ok, 0, (int64_t)1536 * 1024 * 7, "decoder.model.0.conv.weight");
    d->dec_conv0.b = dacw_get(d, &ok, 0, 1536, "decoder.model.0.conv.bias");
    for (int i = 0; i < 4; i++) {
        s2pd_dblock* b = &d->dec_b[i];
        b->alpha = dacw_get(d, &ok, 0, DEC_CIN[i], "decoder.model.%d.block.0.alpha", i + 1);
        b->up.w = dacw_get(d, &ok, 0, (int64_t)DEC_CIN[i] * DEC_COUT[i] * DEC_K[i],
                       "decoder.model.%d.block.1.conv.weight", i + 1);
        b->up.b = dacw_get(d, &ok, 0, DEC_COUT[i], "decoder.model.%d.block.1.conv.bias", i + 1);
        for (int r = 0; r < 3; r++)
            load_ru(d, &b->ru[r], DEC_COUT[i], RU_DIL[r], &ok, 0,
                    "decoder.model.%d.block.%d", i + 1, r + 2);
    }
    d->dec_alpha = dacw_get(d, &ok, 0, 96, "decoder.model.5.alpha");
    d->dec_convf.w = dacw_get(d, &ok, 0, 96 * 7, "decoder.model.6.conv.weight");
    d->dec_convf.b = dacw_get(d, &ok, 0, 1, "decoder.model.6.conv.bias");

    /* ---- RVQ upsample + post transformer ---- */
    for (int i = 0; i < 2; i++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "quantizer.upsample.%d.1", i);
        d->up_t[i].w = dacw_get(d, &ok, 0, (int64_t)1024 * 1024 * 2, "quantizer.upsample.%d.0.conv.weight", i);
        d->up_t[i].b = dacw_get(d, &ok, 0, 1024, "quantizer.upsample.%d.0.conv.bias", i);
        load_cnx(d, &d->up_cnx[i], &ok, 0, pfx);
    }
    load_tf(d, &d->post, 8, 128, &ok, 0, "quantizer.post_module");

    /* ---- VQ codebooks ---- */
    load_vq(d, &d->sem, S2P_CB_SIZE, &ok, &ok_enc, "quantizer.semantic_quantizer.quantizers.0");
    for (int i = 0; i < 9; i++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "quantizer.quantizer.quantizers.%d", i);
        load_vq(d, &d->res[i], 1024, &ok, &ok_enc, pfx);
    }

    if (!ok) {
        s2p_dacw_free(&d->w);
        free(d);
        return S2P_ERR_FORMAT;
    }

    /* ---- Encoder (optional): conv stack + transformer + downsample + pre.
     * S2P_GAP: the shipped converter (fish-s2-native/tools/convert_codec.py)
     * emits DECODE-PATH tensors only — `encoder.*` (whole conv stack incl.
     * encoder.block.4.block.5 transformer) and `quantizer.pre_module.*` are
     * dropped from codec.bin/codec.idx, so voice-cloning encode cannot run
     * from the current artifact. The full encode compute path below and in
     * s2p_dac_encode is implemented and takes effect as soon as a converter
     * that keeps those tensors is used; until then s2p_dac_encode returns
     * S2P_ERR_UNSUPPORTED. Missing names (state-dict, post weight-norm fold):
     *   encoder.block.0.conv.{weight,bias}
     *   encoder.block.{1..4}.block.{0,1,2}.block.{0.alpha,1.conv.*,2.alpha,3.conv.*}
     *   encoder.block.{1..4}.block.3.alpha
     *   encoder.block.{1..4}.block.4.conv.{weight,bias}
     *   encoder.block.4.block.5.layers.{0..3}.{attention.wqkv.weight,
     *     attention.wo.weight,attention_norm.weight,ffn_norm.weight,
     *     attention_layer_scale.gamma,ffn_layer_scale.gamma,
     *     feed_forward.{w1,w2,w3}.weight}
     *   encoder.block.4.block.5.norm.weight
     *   encoder.block.5.alpha
     *   encoder.block.6.conv.{weight,bias}
     *   quantizer.pre_module.layers.{0..7}.<same layer fields> + .norm.weight
     * (quantizer.downsample.* and the VQ in_proj.* ARE present already.)   */
    d->enc_conv0.w = dacw_get(d, &ok_enc, 1, 64 * 7, "encoder.block.0.conv.weight");
    d->enc_conv0.b = dacw_get(d, &ok_enc, 1, 64, "encoder.block.0.conv.bias");
    for (int i = 0; i < 4; i++) {
        s2pd_eblock* b = &d->enc_b[i];
        for (int r = 0; r < 3; r++)
            load_ru(d, &b->ru[r], ENC_CIN[i], RU_DIL[r], &ok_enc, 1,
                    "encoder.block.%d.block.%d", i + 1, r);
        b->alpha = dacw_get(d, &ok_enc, 1, ENC_CIN[i], "encoder.block.%d.block.3.alpha", i + 1);
        b->down.w = dacw_get(d, &ok_enc, 1, (int64_t)ENC_COUT[i] * ENC_CIN[i] * ENC_K[i],
                         "encoder.block.%d.block.4.conv.weight", i + 1);
        b->down.b = dacw_get(d, &ok_enc, 1, ENC_COUT[i], "encoder.block.%d.block.4.conv.bias", i + 1);
    }
    load_tf(d, &d->enc_tf, 4, 512, &ok_enc, 1, "encoder.block.4.block.5");
    d->enc_alpha = dacw_get(d, &ok_enc, 1, 1024, "encoder.block.5.alpha");
    d->enc_convf.w = dacw_get(d, &ok_enc, 1, (int64_t)1024 * 1024 * 3, "encoder.block.6.conv.weight");
    d->enc_convf.b = dacw_get(d, &ok_enc, 1, 1024, "encoder.block.6.conv.bias");
    for (int i = 0; i < 2; i++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "quantizer.downsample.%d.1", i);
        d->down_c[i].w = dacw_get(d, &ok_enc, 1, (int64_t)1024 * 1024 * 2, "quantizer.downsample.%d.0.conv.weight", i);
        d->down_c[i].b = dacw_get(d, &ok_enc, 1, 1024, "quantizer.downsample.%d.0.conv.bias", i);
        load_cnx(d, &d->down_cnx[i], &ok_enc, 1, pfx);
    }
    load_tf(d, &d->pre, 8, 128, &ok_enc, 1, "quantizer.pre_module");
    d->has_encoder = ok_enc;
    if (!ok_enc)
        fprintf(stderr, "[s2pro] dac: encoder weights absent from codec.bin "
                        "(decode-only artifact); s2p_dac_encode disabled\n");

    st = rope_ensure(d, 8192);
    if (st != S2P_OK) {
        s2p_dacw_free(&d->w);
        free(d);
        return st;
    }
    *out = d;
    return S2P_OK;
}

void s2p_dac_free(s2p_dac* d) {
    if (!d) return;
    if (d->rope) cudaFree(d->rope);
    if (d->ws) cudaFree(d->ws);
    if (d->codes_dev) cudaFree(d->codes_dev);
    s2p_dacw_free(&d->w);
    free(d);
}

/* ----------------------------------------------------------- runners ----- */

/* ResidualUnit at dim C on x [C,T] in place; t1,t2 scratch >= C*T floats */
static s2p_status run_ru(const s2pd_ru* ru, float* x, int C, int T, float* t1,
                         float* t2, cudaStream_t st) {
    S2P_CUDA_TRY(s2pdk_snake(x, t1, ru->alpha0, C, T, st));
    S2P_CUDA_TRY(s2pdk_conv1d(t1, C, T, ru->c7.w, ru->c7.b, C, 7, ru->dil, 1,
                              6 * ru->dil, t2, T, st));
    S2P_CUDA_TRY(s2pdk_snake(t2, t2, ru->alpha1, C, T, st));
    S2P_CUDA_TRY(s2pdk_conv1d(t2, C, T, ru->c1.w, ru->c1.b, C, 1, 1, 1, 0, t1,
                              T, st));
    S2P_CUDA_TRY(s2pdk_add_ip(x, t1, (int64_t)C * T, st));
    return S2P_OK;
}

/* ConvNeXtBlock (dim 1024) on x [1024,T] in place.
 * tA >= 1024*T, tB >= 1024*T, tC >= 4096*T floats. */
static s2p_status run_cnx(const s2pd_cnx* c, float* x, int T, float* tA,
                          float* tB, float* tC, cudaStream_t st) {
    const int C = 1024;
    S2P_CUDA_TRY(s2pdk_dwconv1d(x, C, T, c->dw.w, c->dw.b, 7, 6, tA, T, st));
    S2P_CUDA_TRY(s2pdk_transpose(tA, C, T, tB, st));            /* [T,C] */
    S2P_CUDA_TRY(s2pdk_layernorm_ip(tB, c->ln_w, c->ln_b, 1e-6f, T, C, st));
    S2P_CUDA_TRY(s2pdk_matmul(tB, T, C, c->pw1_w, 4096, c->pw1_b, tC, st));
    S2P_CUDA_TRY(s2pdk_gelu_ip(tC, (int64_t)T * 4096, st));
    S2P_CUDA_TRY(s2pdk_matmul(tC, T, 4096, c->pw2_w, C, c->pw2_b, tA, st));
    S2P_CUDA_TRY(s2pdk_colscale_ip(tA, c->gamma, T, C, st));
    S2P_CUDA_TRY(s2pdk_transpose(tA, T, C, tB, st));            /* [C,T] */
    S2P_CUDA_TRY(s2pdk_add_ip(x, tB, (int64_t)C * T, st));
    return S2P_OK;
}

/* WindowLimitedTransformer on x [T,1024] in place (channels-last).
 * t1,t2,t3 scratch >= 3072*T floats each. */
static s2p_status run_tf(s2p_dac* d, const s2pd_tf* tf, float* x, int T,
                         float* t1, float* t2, float* t3, cudaStream_t st) {
    const int C = 1024, H = 16, HD = 64;
    for (int i = 0; i < tf->n_layers; i++) {
        const s2pd_tl* l = &tf->l[i];
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, l->attn_norm, 1e-5f, T, C, t1, st));
        S2P_CUDA_TRY(s2pdk_matmul(t1, T, C, l->wqkv, 3072, NULL, t2, st));
        S2P_CUDA_TRY(s2pdk_rope_ip(t2, T, H, HD, 3072, d->rope, st));
        S2P_CUDA_TRY(s2pdk_rope_ip(t2 + 1024, T, H, HD, 3072, d->rope, st));
        S2P_CUDA_TRY(s2pdk_sdpa(t2, t2 + 1024, t2 + 2048, 3072, T, H, HD,
                                tf->window, t1, C, st));
        S2P_CUDA_TRY(s2pdk_matmul(t1, T, C, l->wo, C, NULL, t3, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, t3, l->g_attn, T, C, st));
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, l->ffn_norm, 1e-5f, T, C, t1, st));
        S2P_CUDA_TRY(s2pdk_matmul(t1, T, C, l->w1, 3072, NULL, t2, st));
        S2P_CUDA_TRY(s2pdk_matmul(t1, T, C, l->w3, 3072, NULL, t3, st));
        S2P_CUDA_TRY(s2pdk_silu_mul_ip(t2, t3, (int64_t)T * 3072, st));
        S2P_CUDA_TRY(s2pdk_matmul(t2, T, 3072, l->w2, C, NULL, t1, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, t1, l->g_ffn, T, C, st));
    }
    S2P_CUDA_TRY(s2pdk_rmsnorm(x, tf->norm, 1e-5f, T, C, x, st));
    return S2P_OK;
}

/* Conv decoder: latent [1024,L] in bufs[cur] -> tanh PCM [1,512*L].
 * On success *out_buf points at the PCM buffer (one of bufs). */
static s2p_status run_decoder_stack(s2p_dac* d, float* bufs[4], int cur, int L,
                                    float** out_buf, cudaStream_t st) {
    int nxt = (cur + 1) & 3;
    S2P_CUDA_TRY(s2pdk_conv1d(bufs[cur], 1024, L, d->dec_conv0.w,
                              d->dec_conv0.b, 1536, 7, 1, 1, 6, bufs[nxt], L, st));
    cur = nxt;
    int Lc = L;
    for (int i = 0; i < 4; i++) {
        const s2pd_dblock* b = &d->dec_b[i];
        S2P_CUDA_TRY(s2pdk_snake(bufs[cur], bufs[cur], b->alpha, DEC_CIN[i], Lc, st));
        nxt = (cur + 1) & 3;
        int Lo = Lc * DEC_S[i];
        S2P_CUDA_TRY(s2pdk_tconv1d(bufs[cur], DEC_CIN[i], Lc, b->up.w, b->up.b,
                                   DEC_COUT[i], DEC_K[i], DEC_S[i], bufs[nxt],
                                   Lo, st));
        cur = nxt;
        Lc = Lo;
        for (int r = 0; r < 3; r++)
            S2P_TRY(run_ru(&b->ru[r], bufs[cur], DEC_COUT[i], Lc,
                           bufs[(cur + 1) & 3], bufs[(cur + 2) & 3], st));
    }
    S2P_CUDA_TRY(s2pdk_snake(bufs[cur], bufs[cur], d->dec_alpha, 96, Lc, st));
    nxt = (cur + 1) & 3;
    S2P_CUDA_TRY(s2pdk_conv1d(bufs[cur], 96, Lc, d->dec_convf.w, d->dec_convf.b,
                              1, 7, 1, 1, 6, bufs[nxt], Lc, st));
    S2P_CUDA_TRY(s2pdk_tanh_ip(bufs[nxt], Lc, st));
    *out_buf = bufs[nxt];
    return S2P_OK;
}

static void carve_bufs(s2p_dac* d, float* bufs[4]) {
    for (int i = 0; i < 4; i++) bufs[i] = d->ws + (size_t)i * d->ws_buf_floats;
}

/* ----------------------------------------------------------- decode ------ */

s2p_status s2p_dac_decode(s2p_dac* d, const int32_t* codes, int T,
                          float** pcm_out, int64_t* n_samples,
                          cudaStream_t stream) {
    if (!d || !codes || !pcm_out || !n_samples || T < 1) return S2P_ERR_INVALID;
    *pcm_out = NULL;
    *n_samples = 0;

    S2P_TRY(rope_ensure(d, T));
    S2P_TRY(ws_ensure(d, (size_t)196608 * T));
    S2P_TRY(codes_ensure(d, T));

    /* clamp a copy (reference clamps in place; min-clamp added for memory
     * safety — invalid negatives would index out of bounds) */
    int32_t* cl = (int32_t*)malloc((size_t)10 * T * sizeof(int32_t));
    if (!cl) return S2P_ERR_OOM;
    for (int t = 0; t < T; t++) {
        int32_t v = codes[t];
        cl[t] = v < 0 ? 0 : (v > S2P_CB_SIZE - 1 ? S2P_CB_SIZE - 1 : v);
    }
    for (int q = 1; q < 10; q++)
        for (int t = 0; t < T; t++) {
            int32_t v = codes[(size_t)q * T + t];
            cl[(size_t)q * T + t] = v < 0 ? 0 : (v > 1023 ? 1023 : v);
        }
    cudaError_t ce = cudaMemcpyAsync(d->codes_dev, cl,
                                     (size_t)10 * T * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream);
    if (ce != cudaSuccess) { free(cl); return S2P_ERR_CUDA; }
    ce = cudaStreamSynchronize(stream);   /* cl freed below */
    free(cl);
    if (ce != cudaSuccess) return S2P_ERR_CUDA;

    float* bufs[4];
    carve_bufs(d, bufs);

    /* RVQ from_indices -> z [1024,T] */
    s2pdk_rvq_tabs tabs;
    tabs.t[0].cb = d->sem.cb; tabs.t[0].ow = d->sem.out_w;
    tabs.t[0].ob = d->sem.out_b; tabs.t[0].n = d->sem.n;
    for (int i = 0; i < 9; i++) {
        tabs.t[i + 1].cb = d->res[i].cb;
        tabs.t[i + 1].ow = d->res[i].out_w;
        tabs.t[i + 1].ob = d->res[i].out_b;
        tabs.t[i + 1].n = d->res[i].n;
    }
    S2P_CUDA_TRY(s2pdk_rvq_from_indices(d->codes_dev, T, tabs, bufs[0], stream));

    /* post_module transformer (channels-last) */
    S2P_CUDA_TRY(s2pdk_transpose(bufs[0], 1024, T, bufs[1], stream));
    S2P_TRY(run_tf(d, &d->post, bufs[1], T, bufs[0], bufs[2], bufs[3], stream));
    S2P_CUDA_TRY(s2pdk_transpose(bufs[1], T, 1024, bufs[0], stream));

    /* upsample x2: CausalTransConv k2 s2 (crop k-s = 0) + ConvNeXt */
    S2P_CUDA_TRY(s2pdk_tconv1d(bufs[0], 1024, T, d->up_t[0].w, d->up_t[0].b,
                               1024, 2, 2, bufs[1], 2 * T, stream));
    S2P_TRY(run_cnx(&d->up_cnx[0], bufs[1], 2 * T, bufs[2], bufs[3], bufs[0], stream));
    S2P_CUDA_TRY(s2pdk_tconv1d(bufs[1], 1024, 2 * T, d->up_t[1].w, d->up_t[1].b,
                               1024, 2, 2, bufs[0], 4 * T, stream));
    S2P_TRY(run_cnx(&d->up_cnx[1], bufs[0], 4 * T, bufs[2], bufs[3], bufs[1], stream));

    /* conv decoder: [1024,4T] -> [1, 2048*T] */
    float* pcm_dev = NULL;
    S2P_TRY(run_decoder_stack(d, bufs, 0, 4 * T, &pcm_dev, stream));

    int64_t n = (int64_t)T * S2P_FRAME_SAMPLES;
    float* host = (float*)malloc((size_t)n * sizeof(float));
    if (!host) return S2P_ERR_OOM;
    ce = cudaMemcpyAsync(host, pcm_dev, (size_t)n * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] dac: decode D2H failed: %s\n",
                cudaGetErrorString(ce));
        free(host);
        return S2P_ERR_CUDA;
    }
    *pcm_out = host;
    *n_samples = n;
    return S2P_OK;
}

s2p_status s2p_dac_decode_latent(s2p_dac* d, const float* latent, int Tlat,
                                 float** pcm_out, int64_t* n_samples,
                                 cudaStream_t stream) {
    if (!d || !latent || !pcm_out || !n_samples || Tlat < 1)
        return S2P_ERR_INVALID;
    *pcm_out = NULL;
    *n_samples = 0;
    S2P_TRY(ws_ensure(d, (size_t)49152 * Tlat));
    float* bufs[4];
    carve_bufs(d, bufs);
    S2P_CUDA_TRY(cudaMemcpyAsync(bufs[0], latent,
                                 (size_t)1024 * Tlat * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    float* pcm_dev = NULL;
    S2P_TRY(run_decoder_stack(d, bufs, 0, Tlat, &pcm_dev, stream));
    int64_t n = (int64_t)Tlat * 512;
    float* host = (float*)malloc((size_t)n * sizeof(float));
    if (!host) return S2P_ERR_OOM;
    cudaError_t ce = cudaMemcpyAsync(host, pcm_dev, (size_t)n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) { free(host); return S2P_ERR_CUDA; }
    *pcm_out = host;
    *n_samples = n;
    return S2P_OK;
}

/* ----------------------------------------------------------- encode ------ */

/* One VectorQuantize inference step on `resid` [1024,T]:
 * writes codes row, subtracts out_proj(straight-through z_q) from resid.
 * scratch: ze8/zq8 >= 8*T, zq >= 1024*T floats. */
static s2p_status run_vq_step(const s2pd_vq* vq, float* resid, int T,
                              int32_t* codes_row, float* ze8, float* zq8,
                              float* zq, cudaStream_t st) {
    S2P_CUDA_TRY(s2pdk_conv1d(resid, 1024, T, vq->in_w, vq->in_b, 8, 1, 1, 1,
                              0, ze8, T, st));
    S2P_CUDA_TRY(s2pdk_vq_nearest(ze8, T, vq->cb, vq->n, codes_row, st));
    S2P_CUDA_TRY(s2pdk_vq_dequant(ze8, codes_row, vq->cb, T, zq8, st));
    S2P_CUDA_TRY(s2pdk_conv1d(zq8, 8, T, vq->out_w, vq->out_b, 1024, 1, 1, 1,
                              0, zq, T, st));
    S2P_CUDA_TRY(s2pdk_sub_ip(resid, zq, (int64_t)1024 * T, st));
    return S2P_OK;
}

s2p_status s2p_dac_encode(s2p_dac* d, const float* pcm, int64_t n_samples,
                          int32_t** codes_out, int* T_out, cudaStream_t stream) {
    if (!d || !pcm || !codes_out || !T_out || n_samples < 1)
        return S2P_ERR_INVALID;
    *codes_out = NULL;
    *T_out = 0;
    if (!d->has_encoder) {
        /* S2P_GAP: encoder.* + quantizer.pre_module.* absent from the
         * converted codec.bin (see s2p_dac_load for the exact tensor list);
         * full encode compute path below is implemented and activates once a
         * converter keeps those tensors. */
        fprintf(stderr, "[s2pro] dac: encode unavailable — codec.bin lacks "
                        "encoder/pre_module weights\n");
        return S2P_ERR_UNSUPPORTED;
    }

    int64_t L64 = ((n_samples + S2P_FRAME_SAMPLES - 1) / S2P_FRAME_SAMPLES) *
                  S2P_FRAME_SAMPLES;
    if (L64 > INT32_MAX / 4) return S2P_ERR_INVALID;   /* keep int indexing safe */
    int L = (int)L64;
    int F = L / S2P_FRAME_SAMPLES;      /* output frames */
    int Tenc = L / 512;                 /* encoder latent length (4F) */

    S2P_TRY(rope_ensure(d, Tenc));
    S2P_TRY(ws_ensure(d, (size_t)64 * L));
    S2P_TRY(codes_ensure(d, F));

    /* right-pad audio with zeros to a multiple of frame_length (DAC.encode) */
    float* padded = (float*)calloc((size_t)L, sizeof(float));
    if (!padded) return S2P_ERR_OOM;
    memcpy(padded, pcm, (size_t)n_samples * sizeof(float));
    float* bufs[4];
    carve_bufs(d, bufs);
    cudaError_t ce = cudaMemcpyAsync(bufs[0], padded, (size_t)L * sizeof(float),
                                     cudaMemcpyHostToDevice, stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(stream);
    free(padded);
    if (ce != cudaSuccess) return S2P_ERR_CUDA;

    /* encoder conv stack */
    int cur = 1;
    S2P_CUDA_TRY(s2pdk_conv1d(bufs[0], 1, L, d->enc_conv0.w, d->enc_conv0.b,
                              64, 7, 1, 1, 6, bufs[1], L, stream));
    int Lc = L;
    for (int i = 0; i < 4; i++) {
        const s2pd_eblock* b = &d->enc_b[i];
        for (int r = 0; r < 3; r++)
            S2P_TRY(run_ru(&b->ru[r], bufs[cur], ENC_CIN[i], Lc,
                           bufs[(cur + 1) & 3], bufs[(cur + 2) & 3], stream));
        S2P_CUDA_TRY(s2pdk_snake(bufs[cur], bufs[cur], b->alpha, ENC_CIN[i], Lc, stream));
        int nxt = (cur + 1) & 3;
        int Lo = Lc / ENC_S[i];
        /* causal strided conv: left pad = eff_k - stride = stride */
        S2P_CUDA_TRY(s2pdk_conv1d(bufs[cur], ENC_CIN[i], Lc, b->down.w,
                                  b->down.b, ENC_COUT[i], ENC_K[i], 1,
                                  ENC_S[i], ENC_S[i], bufs[nxt], Lo, stream));
        cur = nxt;
        Lc = Lo;
        if (i == 3) {   /* deepest stage transformer, 4L window 512 */
            int nb = (cur + 1) & 3;
            S2P_CUDA_TRY(s2pdk_transpose(bufs[cur], 1024, Lc, bufs[nb], stream));
            S2P_TRY(run_tf(d, &d->enc_tf, bufs[nb], Lc, bufs[cur],
                           bufs[(cur + 2) & 3], bufs[(cur + 3) & 3], stream));
            S2P_CUDA_TRY(s2pdk_transpose(bufs[nb], Lc, 1024, bufs[cur], stream));
        }
    }
    S2P_CUDA_TRY(s2pdk_snake(bufs[cur], bufs[cur], d->enc_alpha, 1024, Lc, stream));
    int nxt = (cur + 1) & 3;
    S2P_CUDA_TRY(s2pdk_conv1d(bufs[cur], 1024, Lc, d->enc_convf.w,
                              d->enc_convf.b, 1024, 3, 1, 1, 2, bufs[nxt], Lc,
                              stream));
    cur = nxt;                                     /* z [1024, Tenc] */

    /* quantizer.downsample: 2 x (CausalConv k2 s2, NO pad + ConvNeXt) */
    for (int i = 0; i < 2; i++) {
        nxt = (cur + 1) & 3;
        int Lo = Lc / 2;
        S2P_CUDA_TRY(s2pdk_conv1d(bufs[cur], 1024, Lc, d->down_c[i].w,
                                  d->down_c[i].b, 1024, 2, 1, 2, 0, bufs[nxt],
                                  Lo, stream));
        cur = nxt;
        Lc = Lo;
        S2P_TRY(run_cnx(&d->down_cnx[i], bufs[cur], Lc, bufs[(cur + 1) & 3],
                        bufs[(cur + 2) & 3], bufs[(cur + 3) & 3], stream));
    }

    /* pre_module transformer at frame rate */
    nxt = (cur + 1) & 3;
    S2P_CUDA_TRY(s2pdk_transpose(bufs[cur], 1024, F, bufs[nxt], stream));
    S2P_TRY(run_tf(d, &d->pre, bufs[nxt], F, bufs[cur], bufs[(cur + 2) & 3],
                   bufs[(cur + 3) & 3], stream));
    S2P_CUDA_TRY(s2pdk_transpose(bufs[nxt], F, 1024, bufs[cur], stream));

    /* semantic quantize, then 9-codebook residual cascade */
    float* z = bufs[cur];                              /* becomes residual */
    float* zq = bufs[(cur + 1) & 3];
    float* ze8 = bufs[(cur + 2) & 3];
    float* zq8 = ze8 + (size_t)8 * F;
    S2P_TRY(run_vq_step(&d->sem, z, F, d->codes_dev, ze8, zq8, zq, stream));
    for (int i = 0; i < 9; i++)
        S2P_TRY(run_vq_step(&d->res[i], z, F, d->codes_dev + (size_t)(i + 1) * F,
                            ze8, zq8, zq, stream));

    int32_t* host = (int32_t*)malloc((size_t)10 * F * sizeof(int32_t));
    if (!host) return S2P_ERR_OOM;
    ce = cudaMemcpyAsync(host, d->codes_dev, (size_t)10 * F * sizeof(int32_t),
                         cudaMemcpyDeviceToHost, stream);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) { free(host); return S2P_ERR_CUDA; }
    *codes_out = host;
    *T_out = F;
    return S2P_OK;
}
