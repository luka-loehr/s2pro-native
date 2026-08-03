/* s2pro-native — incremental streaming DAC: bit-exact per-frame decode.
 *
 * Replaces the reference's window/overlap/crossfade scheme with true
 * incremental state so a streamed decode equals s2p_dac_decode of the full
 * code sequence BIT FOR BIT:
 *
 *   - post_module (8L, causal attention window 128): per-layer K/V history
 *     of the last 127 rows; new rows are RoPE'd at their ABSOLUTE position
 *     and attend exactly the whole-buffer key set (s2pdk_sdpa_inc).
 *   - every causal conv keeps a device history of its own input, sized
 *     (k-1)*dilation columns and zero-initialized — the history IS the
 *     causal left pad, so the first frames read the same zeros the
 *     whole-buffer kernel pads with.
 *   - transposed convs (upsamplers) need one input column of context; the
 *     proven whole-buffer kernel runs over [context | new] and the context
 *     column's raw outputs (first `stride` columns) are dropped by a 2D
 *     copy.
 *   - everything else (Snake, RMSNorm/LayerNorm, GELU, matmuls, RVQ lookup,
 *     tanh) is per-token/per-column and needs no state.
 *
 * One pushed frame emits its 2048 samples immediately: no stride windows,
 * no overlap re-decode, no crossfade, no held-back tail. Per-stream state
 * is ~15 MB device memory; scratch is allocated once at create.
 *
 * Validation: S2P_TEST_STREAM_WAV diffs this path against s2p_dac_decode on
 * identical codes; the contract is max|diff| == 0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "s2pro/dac.h"
#include "dac_internal.h"

#define INC_WINDOW   128
#define INC_KVHIST   (INC_WINDOW - 1)          /* retained K/V rows */
#define INC_C        1024
#define INC_HEADS    16
#define INC_HD       64

/* Up to INC_MAX_TN frames decode per push (the stream front end batches;
 * bigger pushes amortize kernel-launch overhead and raise occupancy of the
 * small early-stage convs). Scratch sizing follows the largest ext buffer,
 * the block-2 RU dil-9 input [192, 54 + 1024*tn]; one size covers all five
 * rotating buffers at tn = 8 (~32 MB scratch per stream). */
#define INC_MAX_TN 8
#define INC_BUF_FLOATS 1593344

/* decoder block dims (mirror dac.c) */
static const int IDEC_CIN[4] = { 1536, 768, 384, 192 };
static const int IDEC_COUT[4] = { 768, 384, 192, 96 };
static const int IDEC_K[4] = { 16, 16, 8, 4 };
static const int IDEC_S[4] = { 8, 8, 4, 2 };
static const int IRU_DIL[3] = { 1, 3, 9 };

typedef struct {
    float* d;      /* device [C, len], zero-init */
    int    C, len;
} inc_hist;

struct s2pd_inc {
    s2p_dac* dac;
    int64_t  pos;            /* frames pushed so far (absolute position) */
    int      kv_len;         /* valid K/V history rows (min(127, pos)) */

    /* post_module state */
    float* kh[8];            /* [127, 1024] each */
    float* vh[8];

    /* conv input histories */
    inc_hist cnx_h[2];       /* ConvNeXt dwconv inputs [1024, 6] */
    inc_hist c0_h;           /* decoder conv0 input [1024, 6] */
    inc_hist up_h[4];        /* block tconv inputs [cin, 1] (post-snake) */
    inc_hist ru_h[4][3];     /* RU c7 inputs [cout, 6*dil] (post-snake) */
    inc_hist cf_h;           /* final conv input [96, 6] (post-snake) */

    /* scratch (one allocation, carved) */
    float*   scratch;
    float *bx, *by, *bt1, *bt2, *bext;          /* INC_BUF_FLOATS each */
    float *axk, *axv;                           /* [127+tn, 1024] each */
    float *aq, *af1, *af2;                      /* [tn, 3072] each */
    float *at1, *at3;                           /* [tn, 1024] each */
    float *brow;                                /* [tn, 1024] transformer x */
    int32_t* codes_dev;                         /* [10, tn] cb-major */
    float*   pcm_pin;                           /* pinned [tn*2048] */
    int      pending;                           /* frames in flight (0=idle) */
};

static s2p_status hist_init(inc_hist* h, int C, int len) {
    h->C = C;
    h->len = len;
    S2P_CUDA_TRY(cudaMalloc((void**)&h->d, (size_t)C * len * sizeof(float)));
    S2P_CUDA_TRY(cudaMemset(h->d, 0, (size_t)C * len * sizeof(float)));
    return S2P_OK;
}

s2p_status s2pd_inc_create(s2p_dac* d, s2pd_inc** out) {
    if (!d || !out) return S2P_ERR_INVALID;
    *out = NULL;
    s2pd_inc* s = (s2pd_inc*)calloc(1, sizeof(*s));
    if (!s) return S2P_ERR_OOM;
    s->dac = d;

    s2p_status rc = S2P_OK;
    for (int l = 0; l < 8 && rc == S2P_OK; l++) {
        size_t bytes = (size_t)INC_KVHIST * INC_C * sizeof(float);
        if (cudaMalloc((void**)&s->kh[l], bytes) != cudaSuccess ||
            cudaMalloc((void**)&s->vh[l], bytes) != cudaSuccess)
            rc = S2P_ERR_OOM;
    }
    for (int i = 0; i < 2 && rc == S2P_OK; i++)
        rc = hist_init(&s->cnx_h[i], INC_C, 6);
    if (rc == S2P_OK) rc = hist_init(&s->c0_h, INC_C, 6);
    for (int i = 0; i < 4 && rc == S2P_OK; i++) {
        rc = hist_init(&s->up_h[i], IDEC_CIN[i], 1);
        for (int r = 0; r < 3 && rc == S2P_OK; r++)
            rc = hist_init(&s->ru_h[i][r], IDEC_COUT[i], 6 * IRU_DIL[r]);
    }
    if (rc == S2P_OK) rc = hist_init(&s->cf_h, 96, 6);

    if (rc == S2P_OK) {
        size_t floats = 5u * INC_BUF_FLOATS +
                        2u * (INC_KVHIST + INC_MAX_TN) * INC_C +
                        3u * INC_MAX_TN * 3072 + 3u * INC_MAX_TN * 1024;
        if (cudaMalloc((void**)&s->scratch, floats * sizeof(float)) !=
            cudaSuccess)
            rc = S2P_ERR_OOM;
    }
    if (rc == S2P_OK &&
        cudaMalloc((void**)&s->codes_dev, (size_t)INC_MAX_TN *
                       S2P_NUM_CODEBOOKS * sizeof(int32_t)) != cudaSuccess)
        rc = S2P_ERR_OOM;
    if (rc == S2P_OK &&
        cudaMallocHost((void**)&s->pcm_pin,
                       (size_t)INC_MAX_TN * S2P_FRAME_SAMPLES *
                           sizeof(float)) != cudaSuccess)
        rc = S2P_ERR_OOM;
    if (rc != S2P_OK) {
        s2pd_inc_destroy(s);
        return rc;
    }
    float* p = s->scratch;
    s->bx = p;   p += INC_BUF_FLOATS;
    s->by = p;   p += INC_BUF_FLOATS;
    s->bt1 = p;  p += INC_BUF_FLOATS;
    s->bt2 = p;  p += INC_BUF_FLOATS;
    s->bext = p; p += INC_BUF_FLOATS;
    s->axk = p;  p += (size_t)(INC_KVHIST + INC_MAX_TN) * INC_C;
    s->axv = p;  p += (size_t)(INC_KVHIST + INC_MAX_TN) * INC_C;
    s->aq = p;   p += (size_t)INC_MAX_TN * 3072;
    s->af1 = p;  p += (size_t)INC_MAX_TN * 3072;
    s->af2 = p;  p += (size_t)INC_MAX_TN * 3072;
    s->at1 = p;  p += (size_t)INC_MAX_TN * 1024;
    s->at3 = p;  p += (size_t)INC_MAX_TN * 1024;
    s->brow = p;
    *out = s;
    return S2P_OK;
}

void s2pd_inc_destroy(s2pd_inc* s) {
    if (!s) return;
    for (int l = 0; l < 8; l++) {
        if (s->kh[l]) cudaFree(s->kh[l]);
        if (s->vh[l]) cudaFree(s->vh[l]);
    }
    for (int i = 0; i < 2; i++)
        if (s->cnx_h[i].d) cudaFree(s->cnx_h[i].d);
    if (s->c0_h.d) cudaFree(s->c0_h.d);
    for (int i = 0; i < 4; i++) {
        if (s->up_h[i].d) cudaFree(s->up_h[i].d);
        for (int r = 0; r < 3; r++)
            if (s->ru_h[i][r].d) cudaFree(s->ru_h[i][r].d);
    }
    if (s->cf_h.d) cudaFree(s->cf_h.d);
    if (s->scratch) cudaFree(s->scratch);
    if (s->codes_dev) cudaFree(s->codes_dev);
    if (s->pcm_pin) cudaFreeHost(s->pcm_pin);
    free(s);
}

/* ext = [hist | in_new] along T (channels-first). */
static s2p_status ext_build(const inc_hist* h, const float* in_new, int tn,
                            float* ext, cudaStream_t st) {
    size_t ep = (size_t)(h->len + tn) * sizeof(float);
    S2P_CUDA_TRY(cudaMemcpy2DAsync(ext, ep, h->d,
                                   (size_t)h->len * sizeof(float),
                                   (size_t)h->len * sizeof(float), h->C,
                                   cudaMemcpyDeviceToDevice, st));
    S2P_CUDA_TRY(cudaMemcpy2DAsync(ext + h->len, ep, in_new,
                                   (size_t)tn * sizeof(float),
                                   (size_t)tn * sizeof(float), h->C,
                                   cudaMemcpyDeviceToDevice, st));
    return S2P_OK;
}

/* hist = last h->len columns of ext (ext has h->len + tn columns). */
static s2p_status hist_update(inc_hist* h, const float* ext, int tn,
                              cudaStream_t st) {
    size_t ep = (size_t)(h->len + tn) * sizeof(float);
    S2P_CUDA_TRY(cudaMemcpy2DAsync(h->d, (size_t)h->len * sizeof(float),
                                   ext + tn, ep,
                                   (size_t)h->len * sizeof(float), h->C,
                                   cudaMemcpyDeviceToDevice, st));
    return S2P_OK;
}

/* Causal conv over [hist | new]: out [cout, tn]. hist->len == (k-1)*dil, so
 * leftpad 0 over ext reproduces the whole-buffer padding exactly. */
static s2p_status conv_hist(const s2pd_cw* cw, inc_hist* h, int cin, int cout,
                            int k, int dil, const float* in_new, int tn,
                            float* ext, float* out, cudaStream_t st) {
    S2P_TRY(ext_build(h, in_new, tn, ext, st));
    S2P_CUDA_TRY(s2pdk_conv1d(ext, cin, h->len + tn, cw->w, cw->b, cout, k,
                              dil, 1, 0, out, tn, st));
    return hist_update(h, ext, tn, st);
}

/* ---- ConvNeXt block (dim 1024) on x [1024, tn], with dwconv history ---- */
static s2p_status inc_cnx(s2pd_inc* s, const s2pd_cnx* c, inc_hist* h,
                          float* x, int tn, cudaStream_t st) {
    float *tA = s->bt1, *tB = s->bt2, *tC = s->bext + 32768; /* 4096*tn <= 16k */
    S2P_TRY(ext_build(h, x, tn, s->bext, st));
    S2P_CUDA_TRY(s2pdk_dwconv1d(s->bext, INC_C, h->len + tn, c->dw.w, c->dw.b,
                                7, 0, tA, tn, st));
    S2P_TRY(hist_update(h, s->bext, tn, st));
    S2P_CUDA_TRY(s2pdk_transpose(tA, INC_C, tn, tB, st));       /* [tn,C] */
    S2P_CUDA_TRY(s2pdk_layernorm_ip(tB, c->ln_w, c->ln_b, 1e-6f, tn, INC_C, st));
    S2P_CUDA_TRY(s2pdk_matmul(tB, tn, INC_C, c->pw1_w, 4096, c->pw1_b, tC, st));
    S2P_CUDA_TRY(s2pdk_gelu_ip(tC, (int64_t)tn * 4096, st));
    S2P_CUDA_TRY(s2pdk_matmul(tC, tn, 4096, c->pw2_w, INC_C, c->pw2_b, tA, st));
    S2P_CUDA_TRY(s2pdk_colscale_ip(tA, c->gamma, tn, INC_C, st));
    S2P_CUDA_TRY(s2pdk_transpose(tA, tn, INC_C, tB, st));       /* [C,tn] */
    S2P_CUDA_TRY(s2pdk_add_ip(x, tB, (int64_t)INC_C * tn, st));
    return S2P_OK;
}

/* ---- post_module, tn new rows x [tn, 1024] at absolute position pos ---- */
static s2p_status inc_post(s2pd_inc* s, float* x, int tn, cudaStream_t st) {
    s2p_dac* d = s->dac;
    const s2pd_tf* tf = &d->post;
    const int hl = s->kv_len;
    const int t0 = (int)s->pos;

    for (int l = 0; l < tf->n_layers; l++) {
        const s2pd_tl* ly = &tf->l[l];
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, ly->attn_norm, 1e-5f, tn, INC_C, s->at1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, tn, INC_C, ly->wqkv, 3072, NULL,
                                  s->aq, st));
        S2P_CUDA_TRY(s2pdk_rope_ip_off(s->aq, t0, tn, INC_HEADS, INC_HD, 3072,
                                       d->rope, st));
        S2P_CUDA_TRY(s2pdk_rope_ip_off(s->aq + 1024, t0, tn, INC_HEADS, INC_HD,
                                       3072, d->rope, st));
        /* kv_ext = [hist | new rows] */
        if (hl > 0) {
            S2P_CUDA_TRY(cudaMemcpyAsync(s->axk, s->kh[l],
                                         (size_t)hl * INC_C * sizeof(float),
                                         cudaMemcpyDeviceToDevice, st));
            S2P_CUDA_TRY(cudaMemcpyAsync(s->axv, s->vh[l],
                                         (size_t)hl * INC_C * sizeof(float),
                                         cudaMemcpyDeviceToDevice, st));
        }
        S2P_CUDA_TRY(cudaMemcpy2DAsync(
            s->axk + (size_t)hl * INC_C, INC_C * sizeof(float), s->aq + 1024,
            3072 * sizeof(float), INC_C * sizeof(float), tn,
            cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(cudaMemcpy2DAsync(
            s->axv + (size_t)hl * INC_C, INC_C * sizeof(float), s->aq + 2048,
            3072 * sizeof(float), INC_C * sizeof(float), tn,
            cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(s2pdk_sdpa_inc(s->aq, 3072, s->axk, s->axv, INC_C, hl,
                                    tn, INC_HEADS, INC_HD, tf->window, s->at1,
                                    INC_C, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, tn, INC_C, ly->wo, INC_C, NULL,
                                  s->at3, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, s->at3, ly->g_attn, tn, INC_C, st));
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, ly->ffn_norm, 1e-5f, tn, INC_C, s->at1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, tn, INC_C, ly->w1, 3072, NULL,
                                  s->af1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, tn, INC_C, ly->w3, 3072, NULL,
                                  s->af2, st));
        S2P_CUDA_TRY(s2pdk_silu_mul_ip(s->af1, s->af2, (int64_t)tn * 3072, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->af1, tn, 3072, ly->w2, INC_C, NULL,
                                  s->at1, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, s->at1, ly->g_ffn, tn, INC_C, st));

        /* retain last min(127, hl+tn) K/V rows: kv_ext tail */
        int keep = hl + tn < INC_KVHIST ? hl + tn : INC_KVHIST;
        int from = hl + tn - keep;
        S2P_CUDA_TRY(cudaMemcpyAsync(s->kh[l], s->axk + (size_t)from * INC_C,
                                     (size_t)keep * INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(cudaMemcpyAsync(s->vh[l], s->axv + (size_t)from * INC_C,
                                     (size_t)keep * INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
    }
    S2P_CUDA_TRY(s2pdk_rmsnorm(x, tf->norm, 1e-5f, tn, INC_C, x, st));
    return S2P_OK;
}

s2p_status s2pd_inc_push_async(s2pd_inc* s,
                               const int32_t* frame_codes, int tn,
                               cudaStream_t st) {
    if (!s || !frame_codes || tn < 1 || tn > INC_MAX_TN)
        return S2P_ERR_INVALID;
    if (s->pending) return S2P_ERR_STATE;
    s2p_dac* d = s->dac;
    if (s->pos + tn > (int64_t)INT32_MAX / 4) return S2P_ERR_INVALID;

    if ((int)s->pos + tn > d->rope_rows) {
        /* table growth reallocates d->rope; drain in-flight users first */
        S2P_CUDA_TRY(cudaStreamSynchronize(st));
    }
    S2P_TRY(s2pd_rope_ensure(d, (int)s->pos + tn));

    /* clamp exactly like s2p_dac_decode; frame_codes is frame-major
     * [tn][10], the staging buffer codebook-major [10][tn]. */
    int32_t cl[S2P_NUM_CODEBOOKS * INC_MAX_TN];
    for (int t = 0; t < tn; t++) {
        int32_t v = frame_codes[(size_t)t * S2P_NUM_CODEBOOKS];
        cl[t] = v < 0 ? 0 : (v > S2P_CB_SIZE - 1 ? S2P_CB_SIZE - 1 : v);
        for (int q = 1; q < S2P_NUM_CODEBOOKS; q++) {
            v = frame_codes[(size_t)t * S2P_NUM_CODEBOOKS + q];
            cl[(size_t)q * tn + t] = v < 0 ? 0 : (v > 1023 ? 1023 : v);
        }
    }
    S2P_CUDA_TRY(cudaMemcpyAsync(s->codes_dev, cl,
                                 (size_t)S2P_NUM_CODEBOOKS * tn *
                                     sizeof(int32_t),
                                 cudaMemcpyHostToDevice, st));

    s2pdk_rvq_tabs tabs;
    tabs.t[0].cb = d->sem.cb; tabs.t[0].ow = d->sem.out_w;
    tabs.t[0].ob = d->sem.out_b; tabs.t[0].n = d->sem.n;
    for (int i = 0; i < 9; i++) {
        tabs.t[i + 1].cb = d->res[i].cb;
        tabs.t[i + 1].ow = d->res[i].out_w;
        tabs.t[i + 1].ob = d->res[i].out_b;
        tabs.t[i + 1].n = d->res[i].n;
    }
    float* x = s->bx;
    /* RVQ emits [1024, tn] channels-first; the transformer runs on rows */
    S2P_CUDA_TRY(s2pdk_rvq_from_indices(s->codes_dev, tn, tabs, x, st));
    S2P_CUDA_TRY(s2pdk_transpose(x, INC_C, tn, s->brow, st));   /* [tn,C] */

    S2P_TRY(inc_post(s, s->brow, tn, st));

    S2P_CUDA_TRY(s2pdk_transpose(s->brow, tn, INC_C, x, st));   /* [C,tn] */

    /* upsample x2: tconv k2 s2 is column-local (out j reads only in[j/2]) */
    float* y = s->by;
    S2P_CUDA_TRY(s2pdk_tconv1d(x, INC_C, tn, d->up_t[0].w, d->up_t[0].b,
                               INC_C, 2, 2, y, 2 * tn, st));
    S2P_TRY(inc_cnx(s, &d->up_cnx[0], &s->cnx_h[0], y, 2 * tn, st));
    S2P_CUDA_TRY(s2pdk_tconv1d(y, INC_C, 2 * tn, d->up_t[1].w, d->up_t[1].b,
                               INC_C, 2, 2, x, 4 * tn, st));
    S2P_TRY(inc_cnx(s, &d->up_cnx[1], &s->cnx_h[1], x, 4 * tn, st));

    /* decoder conv0 (input history) */
    S2P_TRY(conv_hist(&d->dec_conv0, &s->c0_h, INC_C, 1536, 7, 1, x, 4 * tn,
                      s->bext, y, st));

    /* 4 decoder blocks */
    int Lc = 4 * tn;
    for (int i = 0; i < 4; i++) {
        /* y holds block input [cin, Lc] */
        S2P_CUDA_TRY(s2pdk_snake(y, y, d->dec_b[i].alpha, IDEC_CIN[i], Lc, st));
        S2P_TRY(ext_build(&s->up_h[i], y, Lc, s->bext, st));
        int Lo = Lc * IDEC_S[i];
        /* full tconv over [context col | new cols] with the proven kernel,
         * then drop the context column's raw outputs (first `stride` cols) */
        S2P_CUDA_TRY(s2pdk_tconv1d(s->bext, IDEC_CIN[i], 1 + Lc,
                                   d->dec_b[i].up.w, d->dec_b[i].up.b,
                                   IDEC_COUT[i], IDEC_K[i], IDEC_S[i], s->bt2,
                                   Lo + IDEC_S[i], st));
        S2P_CUDA_TRY(cudaMemcpy2DAsync(
            x, (size_t)Lo * sizeof(float), s->bt2 + IDEC_S[i],
            (size_t)(Lo + IDEC_S[i]) * sizeof(float),
            (size_t)Lo * sizeof(float), IDEC_COUT[i],
            cudaMemcpyDeviceToDevice, st));
        S2P_TRY(hist_update(&s->up_h[i], s->bext, Lc, st));
        Lc = Lo;
        for (int r = 0; r < 3; r++) {
            const s2pd_ru* ru = &d->dec_b[i].ru[r];
            S2P_CUDA_TRY(s2pdk_snake(x, s->bt1, ru->alpha0, IDEC_COUT[i], Lc, st));
            S2P_TRY(conv_hist(&ru->c7, &s->ru_h[i][r], IDEC_COUT[i],
                              IDEC_COUT[i], 7, IRU_DIL[r], s->bt1, Lc,
                              s->bext, s->bt2, st));
            S2P_CUDA_TRY(s2pdk_snake(s->bt2, s->bt2, ru->alpha1, IDEC_COUT[i],
                                     Lc, st));
            S2P_CUDA_TRY(s2pdk_conv1d(s->bt2, IDEC_COUT[i], Lc, ru->c1.w,
                                      ru->c1.b, IDEC_COUT[i], 1, 1, 1, 0,
                                      s->bt1, Lc, st));
            S2P_CUDA_TRY(s2pdk_add_ip(x, s->bt1, (int64_t)IDEC_COUT[i] * Lc, st));
        }
        { float* t = x; x = y; y = t; }   /* block output becomes next input */
    }
    /* y holds [96, 2048*tn] */
    S2P_CUDA_TRY(s2pdk_snake(y, y, d->dec_alpha, 96, Lc, st));
    S2P_TRY(conv_hist(&d->dec_convf, &s->cf_h, 96, 1, 7, 1, y, Lc, s->bext,
                      x, st));
    S2P_CUDA_TRY(s2pdk_tanh_ip(x, Lc, st));

    S2P_CUDA_TRY(cudaMemcpyAsync(s->pcm_pin, x,
                                 (size_t)tn * S2P_FRAME_SAMPLES *
                                     sizeof(float),
                                 cudaMemcpyDeviceToHost, st));

    s->pending = tn;
    s->pos += tn;
    s->kv_len = s->kv_len + tn < INC_KVHIST ? s->kv_len + tn : INC_KVHIST;
    return S2P_OK;
}

/* Sync `st` and hand out the in-flight frames' samples (0 if none). */
s2p_status s2pd_inc_collect(s2pd_inc* s, float* pcm_host, int64_t* n_out,
                            cudaStream_t st) {
    if (!s || !pcm_host || !n_out) return S2P_ERR_INVALID;
    *n_out = 0;
    if (!s->pending) return S2P_OK;
    S2P_CUDA_TRY(cudaStreamSynchronize(st));
    int64_t n = (int64_t)s->pending * S2P_FRAME_SAMPLES;
    memcpy(pcm_host, s->pcm_pin, (size_t)n * sizeof(float));
    s->pending = 0;
    *n_out = n;
    return S2P_OK;
}

s2p_status s2pd_inc_push(s2pd_inc* s,
                         const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                         float* pcm_host, cudaStream_t st) {
    if (!s || !frame_codes || !pcm_host) return S2P_ERR_INVALID;
    S2P_TRY(s2pd_inc_push_async(s, frame_codes, 1, st));
    int64_t n = 0;
    return s2pd_inc_collect(s, pcm_host, &n, st);
}

/* ------------------- cross-session batched push (tn == 1) -------------- */
/* Table slots into d->btab_dev: one per distinct per-session buffer role.
 * Buffer addresses are fixed per inc, so the tables are rebuilt only when
 * the session set changes contents (cheap: one pinned write + one H2D per
 * push). Weight kernels take (in-table, out-table); everything stateful
 * or element-wise loops per session. */
enum {
    TB_BX = 0, TB_BY, TB_BT1, TB_BT2, TB_BEXT, TB_B32K,
    TB_AQ, TB_AF1, TB_AF2, TB_AT1, TB_AT3, TB_BROW, TB_N
};

static s2p_status btab_ensure(s2p_dac* d, int nb) {
    if (d->btab_cap >= nb && d->btab_pin) return S2P_OK;
    if (d->btab_pin) cudaFreeHost(d->btab_pin);
    if (d->btab_dev) cudaFree(d->btab_dev);
    d->btab_pin = NULL;
    d->btab_dev = NULL;
    size_t bytes = (size_t)TB_N * nb * sizeof(void*);
    S2P_CUDA_TRY(cudaMallocHost((void**)&d->btab_pin, bytes));
    S2P_CUDA_TRY(cudaMalloc((void**)&d->btab_dev, bytes));
    d->btab_cap = nb;
    d->btab_nb = 0; /* fresh allocation: roster cache invalid */
    return S2P_OK;
}

#define TB(t) ((const float* const*)(d->btab_dev + (size_t)(t) * nb))
#define TBO(t) ((float* const*)(d->btab_dev + (size_t)(t) * nb))

s2p_status s2pd_inc_push_batch(s2pd_inc* const* ss, int nb, int tn,
                               const int32_t* frame_codes, cudaStream_t st) {
    if (!ss || !frame_codes || nb < 1 || tn < 1 || tn > INC_MAX_TN)
        return S2P_ERR_INVALID;
    if (nb == 1) {
        S2P_TRY(s2pd_inc_push_async(ss[0], frame_codes, tn, st));
        return S2P_OK;
    }
    s2p_dac* d = ss[0]->dac;
    int64_t max_pos = 0;
    for (int i = 0; i < nb; i++) {
        if (!ss[i] || ss[i]->dac != d) return S2P_ERR_INVALID;
        if (ss[i]->pending) return S2P_ERR_STATE;
        if (ss[i]->pos > max_pos) max_pos = ss[i]->pos;
    }
    if ((int)max_pos + tn > d->rope_rows)
        S2P_CUDA_TRY(cudaStreamSynchronize(st));
    S2P_TRY(s2pd_rope_ensure(d, (int)max_pos + tn));

    S2P_TRY(btab_ensure(d, nb));
    /* upload the tables only when the session roster changed (research
     * condition C3): contents are fixed per inc, so steady state pays
     * zero H2D here */
    int roster_same = (d->btab_nb == nb);
    for (int i = 0; roster_same && i < nb; i++)
        if (d->btab_pin[(size_t)TB_BX * nb + i] != ss[i]->bx)
            roster_same = 0;
    if (roster_same) goto tables_ready;
    d->btab_nb = nb;
    for (int i = 0; i < nb; i++) {
        s2pd_inc* c = ss[i];
        d->btab_pin[(size_t)TB_BX * nb + i] = c->bx;
        d->btab_pin[(size_t)TB_BY * nb + i] = c->by;
        d->btab_pin[(size_t)TB_BT1 * nb + i] = c->bt1;
        d->btab_pin[(size_t)TB_BT2 * nb + i] = c->bt2;
        d->btab_pin[(size_t)TB_BEXT * nb + i] = c->bext;
        d->btab_pin[(size_t)TB_B32K * nb + i] = c->bext + 32768;
        d->btab_pin[(size_t)TB_AQ * nb + i] = c->aq;
        d->btab_pin[(size_t)TB_AF1 * nb + i] = c->af1;
        d->btab_pin[(size_t)TB_AF2 * nb + i] = c->af2;
        d->btab_pin[(size_t)TB_AT1 * nb + i] = c->at1;
        d->btab_pin[(size_t)TB_AT3 * nb + i] = c->at3;
        d->btab_pin[(size_t)TB_BROW * nb + i] = c->brow;
    }
    S2P_CUDA_TRY(cudaMemcpyAsync(d->btab_dev, d->btab_pin,
                                 (size_t)TB_N * nb * sizeof(void*),
                                 cudaMemcpyHostToDevice, st));
tables_ready:;

    /* codes upload + RVQ + transpose, per session (lookups, no weights) */
    s2pdk_rvq_tabs tabs;
    tabs.t[0].cb = d->sem.cb; tabs.t[0].ow = d->sem.out_w;
    tabs.t[0].ob = d->sem.out_b; tabs.t[0].n = d->sem.n;
    for (int q = 0; q < 9; q++) {
        tabs.t[q + 1].cb = d->res[q].cb;
        tabs.t[q + 1].ow = d->res[q].out_w;
        tabs.t[q + 1].ob = d->res[q].out_b;
        tabs.t[q + 1].n = d->res[q].n;
    }
    for (int i = 0; i < nb; i++) {
        s2pd_inc* c = ss[i];
        int32_t cl[S2P_NUM_CODEBOOKS * INC_MAX_TN];
        const int32_t* fc =
            frame_codes + (size_t)i * tn * S2P_NUM_CODEBOOKS;
        for (int t = 0; t < tn; t++) {
            int32_t v = fc[(size_t)t * S2P_NUM_CODEBOOKS];
            cl[t] = v < 0 ? 0 : (v > S2P_CB_SIZE - 1 ? S2P_CB_SIZE - 1 : v);
            for (int q = 1; q < S2P_NUM_CODEBOOKS; q++) {
                v = fc[(size_t)t * S2P_NUM_CODEBOOKS + q];
                cl[(size_t)q * tn + t] = v < 0 ? 0 : (v > 1023 ? 1023 : v);
            }
        }
        S2P_CUDA_TRY(cudaMemcpyAsync(c->codes_dev, cl,
                                     (size_t)S2P_NUM_CODEBOOKS * tn *
                                         sizeof(int32_t),
                                     cudaMemcpyHostToDevice, st));
        S2P_CUDA_TRY(s2pdk_rvq_from_indices(c->codes_dev, tn, tabs, c->bx,
                                            st));
        S2P_CUDA_TRY(s2pdk_transpose(c->bx, INC_C, tn, c->brow, st));
    }

    /* post_module: matmuls batched, the rest per session (per-session
     * kv_len / absolute position differ and are activation-only) */
    const s2pd_tf* tf = &d->post;
    for (int l = 0; l < tf->n_layers; l++) {
        const s2pd_tl* ly = &tf->l[l];
        S2P_CUDA_TRY(s2pdk_rmsnorm_b(TB(TB_BROW), ly->attn_norm, 1e-5f,
                                     tn, INC_C, TBO(TB_AT1), nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_AT1), tn, INC_C, ly->wqkv, 3072,
                                    NULL, TBO(TB_AQ), nb, st));
        for (int i = 0; i < nb; i++) {
            s2pd_inc* c = ss[i];
            const int t0 = (int)c->pos;
            const int hl = c->kv_len;
            S2P_CUDA_TRY(s2pdk_rope_ip_off(c->aq, t0, tn, INC_HEADS, INC_HD,
                                           3072, d->rope, st));
            S2P_CUDA_TRY(s2pdk_rope_ip_off(c->aq + 1024, t0, tn, INC_HEADS,
                                           INC_HD, 3072, d->rope, st));
            if (hl > 0) {
                S2P_CUDA_TRY(cudaMemcpyAsync(
                    c->axk, c->kh[l], (size_t)hl * INC_C * sizeof(float),
                    cudaMemcpyDeviceToDevice, st));
                S2P_CUDA_TRY(cudaMemcpyAsync(
                    c->axv, c->vh[l], (size_t)hl * INC_C * sizeof(float),
                    cudaMemcpyDeviceToDevice, st));
            }
            S2P_CUDA_TRY(cudaMemcpy2DAsync(
                c->axk + (size_t)hl * INC_C, INC_C * sizeof(float),
                c->aq + 1024, 3072 * sizeof(float), INC_C * sizeof(float),
                tn, cudaMemcpyDeviceToDevice, st));
            S2P_CUDA_TRY(cudaMemcpy2DAsync(
                c->axv + (size_t)hl * INC_C, INC_C * sizeof(float),
                c->aq + 2048, 3072 * sizeof(float), INC_C * sizeof(float),
                tn, cudaMemcpyDeviceToDevice, st));
            S2P_CUDA_TRY(s2pdk_sdpa_inc(c->aq, 3072, c->axk, c->axv, INC_C,
                                        hl, tn, INC_HEADS, INC_HD, tf->window,
                                        c->at1, INC_C, st));
        }
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_AT1), tn, INC_C, ly->wo, INC_C,
                                    NULL, TBO(TB_AT3), nb, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip_b(TBO(TB_BROW), TB(TB_AT3),
                                          ly->g_attn, tn, INC_C, nb, st));
        S2P_CUDA_TRY(s2pdk_rmsnorm_b(TB(TB_BROW), ly->ffn_norm, 1e-5f, tn,
                                     INC_C, TBO(TB_AT1), nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_AT1), tn, INC_C, ly->w1, 3072,
                                    NULL, TBO(TB_AF1), nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_AT1), tn, INC_C, ly->w3, 3072,
                                    NULL, TBO(TB_AF2), nb, st));
        S2P_CUDA_TRY(s2pdk_silu_mul_ip_b(TBO(TB_AF1), TB(TB_AF2),
                                         (int64_t)tn * 3072, nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_AF1), tn, 3072, ly->w2, INC_C,
                                    NULL, TBO(TB_AT1), nb, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip_b(TBO(TB_BROW), TB(TB_AT1),
                                          ly->g_ffn, tn, INC_C, nb, st));
        for (int i = 0; i < nb; i++) {
            s2pd_inc* c = ss[i];
            const int hl = c->kv_len;
            int keep = hl + tn < INC_KVHIST ? hl + tn : INC_KVHIST;
            int from = hl + tn - keep;
            S2P_CUDA_TRY(cudaMemcpyAsync(
                c->kh[l], c->axk + (size_t)from * INC_C,
                (size_t)keep * INC_C * sizeof(float),
                cudaMemcpyDeviceToDevice, st));
            S2P_CUDA_TRY(cudaMemcpyAsync(
                c->vh[l], c->axv + (size_t)from * INC_C,
                (size_t)keep * INC_C * sizeof(float),
                cudaMemcpyDeviceToDevice, st));
        }
    }
    S2P_CUDA_TRY(s2pdk_rmsnorm_b(TB(TB_BROW), tf->norm, 1e-5f, tn, INC_C,
                                 TBO(TB_BROW), nb, st));
    S2P_CUDA_TRY(s2pdk_transpose_b(TB(TB_BROW), tn, INC_C, TBO(TB_BX), nb,
                                   st));

    /* upsample x2 (tconv k2 s2 column-local) + ConvNeXt */
    S2P_CUDA_TRY(s2pdk_tconv1d_b(TB(TB_BX), INC_C, tn, d->up_t[0].w,
                                 d->up_t[0].b, INC_C, 2, 2, TBO(TB_BY),
                                 2 * tn, nb, st));
    /* cnx on y [1024, 2tn] */
    {
        const s2pd_cnx* c0 = &d->up_cnx[0];
        int T = 2 * tn;
        for (int i = 0; i < nb; i++)
            S2P_TRY(ext_build(&ss[i]->cnx_h[0], ss[i]->by, T, ss[i]->bext,
                              st));
        S2P_CUDA_TRY(s2pdk_dwconv1d_b(TB(TB_BEXT), INC_C,
                                      ss[0]->cnx_h[0].len + T, c0->dw.w,
                                      c0->dw.b, 7, 0, TBO(TB_BT1), T, nb,
                                      st));
        for (int i = 0; i < nb; i++)
            S2P_TRY(hist_update(&ss[i]->cnx_h[0], ss[i]->bext, T, st));
        S2P_CUDA_TRY(s2pdk_transpose_b(TB(TB_BT1), INC_C, T, TBO(TB_BT2),
                                       nb, st));
        S2P_CUDA_TRY(s2pdk_layernorm_ip_b(TBO(TB_BT2), c0->ln_w, c0->ln_b,
                                          1e-6f, T, INC_C, nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_BT2), T, INC_C, c0->pw1_w, 4096,
                                    c0->pw1_b, TBO(TB_B32K), nb, st));
        S2P_CUDA_TRY(s2pdk_gelu_ip_b(TBO(TB_B32K), (int64_t)T * 4096, nb,
                                     st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_B32K), T, 4096, c0->pw2_w, INC_C,
                                    c0->pw2_b, TBO(TB_BT1), nb, st));
        S2P_CUDA_TRY(s2pdk_colscale_ip_b(TBO(TB_BT1), c0->gamma, T, INC_C,
                                         nb, st));
        S2P_CUDA_TRY(s2pdk_transpose_b(TB(TB_BT1), T, INC_C, TBO(TB_BT2),
                                       nb, st));
        S2P_CUDA_TRY(s2pdk_add_ip_b(TBO(TB_BY), TB(TB_BT2),
                                    (int64_t)INC_C * T, nb, st));
    }
    S2P_CUDA_TRY(s2pdk_tconv1d_b(TB(TB_BY), INC_C, 2 * tn, d->up_t[1].w,
                                 d->up_t[1].b, INC_C, 2, 2, TBO(TB_BX),
                                 4 * tn, nb, st));
    {
        const s2pd_cnx* c1 = &d->up_cnx[1];
        int T = 4 * tn;
        for (int i = 0; i < nb; i++)
            S2P_TRY(ext_build(&ss[i]->cnx_h[1], ss[i]->bx, T, ss[i]->bext,
                              st));
        S2P_CUDA_TRY(s2pdk_dwconv1d_b(TB(TB_BEXT), INC_C,
                                      ss[0]->cnx_h[1].len + T, c1->dw.w,
                                      c1->dw.b, 7, 0, TBO(TB_BT1), T, nb,
                                      st));
        for (int i = 0; i < nb; i++)
            S2P_TRY(hist_update(&ss[i]->cnx_h[1], ss[i]->bext, T, st));
        S2P_CUDA_TRY(s2pdk_transpose_b(TB(TB_BT1), INC_C, T, TBO(TB_BT2),
                                       nb, st));
        S2P_CUDA_TRY(s2pdk_layernorm_ip_b(TBO(TB_BT2), c1->ln_w, c1->ln_b,
                                          1e-6f, T, INC_C, nb, st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_BT2), T, INC_C, c1->pw1_w, 4096,
                                    c1->pw1_b, TBO(TB_B32K), nb, st));
        S2P_CUDA_TRY(s2pdk_gelu_ip_b(TBO(TB_B32K), (int64_t)T * 4096, nb,
                                     st));
        S2P_CUDA_TRY(s2pdk_matmul_b(TB(TB_B32K), T, 4096, c1->pw2_w, INC_C,
                                    c1->pw2_b, TBO(TB_BT1), nb, st));
        S2P_CUDA_TRY(s2pdk_colscale_ip_b(TBO(TB_BT1), c1->gamma, T, INC_C,
                                         nb, st));
        S2P_CUDA_TRY(s2pdk_transpose_b(TB(TB_BT1), T, INC_C, TBO(TB_BT2),
                                       nb, st));
        S2P_CUDA_TRY(s2pdk_add_ip_b(TBO(TB_BX), TB(TB_BT2),
                                    (int64_t)INC_C * T, nb, st));
    }

    /* decoder conv0 over [hist | new] */
    for (int i = 0; i < nb; i++)
        S2P_TRY(ext_build(&ss[i]->c0_h, ss[i]->bx, 4 * tn, ss[i]->bext, st));
    S2P_CUDA_TRY(s2pdk_conv1d_b(TB(TB_BEXT), INC_C, ss[0]->c0_h.len + 4 * tn,
                                d->dec_conv0.w, d->dec_conv0.b, 1536, 7, 1,
                                1, 0, TBO(TB_BY), 4 * tn, nb, st));
    for (int i = 0; i < nb; i++)
        S2P_TRY(hist_update(&ss[i]->c0_h, ss[i]->bext, 4 * tn, st));

    /* 4 decoder blocks; parity tracks the x/y swap (same for every
     * session) */
    int Lc = 4 * tn;
    int par = 0; /* 0: y = TB_BY holds input, x = TB_BX is scratch */
    for (int blk = 0; blk < 4; blk++) {
        {
            int t_y = par ? TB_BX : TB_BY;
            S2P_CUDA_TRY(s2pdk_snake_b(TB(t_y), TBO(t_y),
                                       d->dec_b[blk].alpha, IDEC_CIN[blk],
                                       Lc, nb, st));
        }
        for (int i = 0; i < nb; i++) {
            float* yin = par ? ss[i]->bx : ss[i]->by;
            S2P_TRY(ext_build(&ss[i]->up_h[blk], yin, Lc, ss[i]->bext, st));
        }
        int Lo = Lc * IDEC_S[blk];
        S2P_CUDA_TRY(s2pdk_tconv1d_b(TB(TB_BEXT), IDEC_CIN[blk], 1 + Lc,
                                     d->dec_b[blk].up.w, d->dec_b[blk].up.b,
                                     IDEC_COUT[blk], IDEC_K[blk],
                                     IDEC_S[blk], TBO(TB_BT2),
                                     Lo + IDEC_S[blk], nb, st));
        for (int i = 0; i < nb; i++) {
            float* xb = par ? ss[i]->by : ss[i]->bx;
            S2P_CUDA_TRY(cudaMemcpy2DAsync(
                xb, (size_t)Lo * sizeof(float), ss[i]->bt2 + IDEC_S[blk],
                (size_t)(Lo + IDEC_S[blk]) * sizeof(float),
                (size_t)Lo * sizeof(float), IDEC_COUT[blk],
                cudaMemcpyDeviceToDevice, st));
            S2P_TRY(hist_update(&ss[i]->up_h[blk], ss[i]->bext, Lc, st));
        }
        Lc = Lo;
        for (int r = 0; r < 3; r++) {
            const s2pd_ru* ru = &d->dec_b[blk].ru[r];
            {
                int t_x = par ? TB_BY : TB_BX;
                S2P_CUDA_TRY(s2pdk_snake_b(TB(t_x), TBO(TB_BT1), ru->alpha0,
                                           IDEC_COUT[blk], Lc, nb, st));
            }
            for (int i = 0; i < nb; i++)
                S2P_TRY(ext_build(&ss[i]->ru_h[blk][r], ss[i]->bt1, Lc,
                                  ss[i]->bext, st));
            S2P_CUDA_TRY(s2pdk_conv1d_b(
                TB(TB_BEXT), IDEC_COUT[blk],
                ss[0]->ru_h[blk][r].len + Lc, ru->c7.w, ru->c7.b,
                IDEC_COUT[blk], 7, IRU_DIL[r], 1, 0, TBO(TB_BT2), Lc, nb,
                st));
            for (int i = 0; i < nb; i++)
                S2P_TRY(hist_update(&ss[i]->ru_h[blk][r], ss[i]->bext, Lc,
                                    st));
            S2P_CUDA_TRY(s2pdk_snake_b(TB(TB_BT2), TBO(TB_BT2), ru->alpha1,
                                       IDEC_COUT[blk], Lc, nb, st));
            S2P_CUDA_TRY(s2pdk_conv1d_b(TB(TB_BT2), IDEC_COUT[blk], Lc,
                                        ru->c1.w, ru->c1.b, IDEC_COUT[blk],
                                        1, 1, 1, 0, TBO(TB_BT1), Lc, nb,
                                        st));
            {
                int t_x = par ? TB_BY : TB_BX;
                S2P_CUDA_TRY(s2pdk_add_ip_b(TBO(t_x), TB(TB_BT1),
                                            (int64_t)IDEC_COUT[blk] * Lc,
                                            nb, st));
            }
        }
        par ^= 1;
    }
    /* buffer roles traced against push_async's x/y swaps: block 3 wrote
     * its output into the by role, so `by` holds [96, 2048*tn] here —
     * exactly push_async's final `y`. */
    S2P_CUDA_TRY(s2pdk_snake_b(TB(TB_BY), TBO(TB_BY), d->dec_alpha, 96, Lc,
                               nb, st));
    for (int i = 0; i < nb; i++)
        S2P_TRY(ext_build(&ss[i]->cf_h, ss[i]->by, Lc, ss[i]->bext, st));
    S2P_CUDA_TRY(s2pdk_conv1d_b(TB(TB_BEXT), 96, ss[0]->cf_h.len + Lc,
                                d->dec_convf.w, d->dec_convf.b, 1, 7, 1, 1,
                                0, TBO(TB_BT1), Lc, nb, st));
    S2P_CUDA_TRY(s2pdk_tanh_ip_b(TBO(TB_BT1), Lc, nb, st));
    for (int i = 0; i < nb; i++) {
        s2pd_inc* c = ss[i];
        S2P_TRY(hist_update(&c->cf_h, c->bext, Lc, st));
        S2P_CUDA_TRY(cudaMemcpyAsync(c->pcm_pin, c->bt1,
                                     (size_t)tn * S2P_FRAME_SAMPLES *
                                         sizeof(float),
                                     cudaMemcpyDeviceToHost, st));
        c->pending = tn;
        c->pos += tn;
        c->kv_len = c->kv_len + tn < INC_KVHIST ? c->kv_len + tn
                                                : INC_KVHIST;
    }
    return S2P_OK;
}
