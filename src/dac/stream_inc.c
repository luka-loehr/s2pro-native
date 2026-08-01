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

/* conv scratch sizing: largest ext buffer is the block-2 RU dil-9 input,
 * [192, 54 + 1024] = 206,  976 floats; activations peak at [96, 2048] =
 * 196,608. One size covers all five rotating buffers. */
#define INC_BUF_FLOATS 212992

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
    float *axk, *axv;                           /* [128, 1024] each */
    float *aq, *af1, *af2;                      /* [3072] each */
    float *at1, *at3;                           /* [1024] each */
    int32_t* codes_dev;                         /* [10] */
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
        size_t floats = 5u * INC_BUF_FLOATS + 2u * INC_WINDOW * INC_C +
                        3u * 3072 + 2u * 1024;
        if (cudaMalloc((void**)&s->scratch, floats * sizeof(float)) !=
            cudaSuccess)
            rc = S2P_ERR_OOM;
    }
    if (rc == S2P_OK &&
        cudaMalloc((void**)&s->codes_dev,
                   S2P_NUM_CODEBOOKS * sizeof(int32_t)) != cudaSuccess)
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
    s->axk = p;  p += (size_t)INC_WINDOW * INC_C;
    s->axv = p;  p += (size_t)INC_WINDOW * INC_C;
    s->aq = p;   p += 3072;
    s->af1 = p;  p += 3072;
    s->af2 = p;  p += 3072;
    s->at1 = p;  p += 1024;
    s->at3 = p;
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

/* ---- post_module, one new row x [1, 1024] at absolute position pos ---- */
static s2p_status inc_post(s2pd_inc* s, float* x, cudaStream_t st) {
    s2p_dac* d = s->dac;
    const s2pd_tf* tf = &d->post;
    const int hl = s->kv_len;
    const int t0 = (int)s->pos;

    for (int l = 0; l < tf->n_layers; l++) {
        const s2pd_tl* ly = &tf->l[l];
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, ly->attn_norm, 1e-5f, 1, INC_C, s->at1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, 1, INC_C, ly->wqkv, 3072, NULL,
                                  s->aq, st));
        S2P_CUDA_TRY(s2pdk_rope_ip_off(s->aq, t0, 1, INC_HEADS, INC_HD, 3072,
                                       d->rope, st));
        S2P_CUDA_TRY(s2pdk_rope_ip_off(s->aq + 1024, t0, 1, INC_HEADS, INC_HD,
                                       3072, d->rope, st));
        /* kv_ext = [hist | new row] */
        if (hl > 0) {
            S2P_CUDA_TRY(cudaMemcpyAsync(s->axk, s->kh[l],
                                         (size_t)hl * INC_C * sizeof(float),
                                         cudaMemcpyDeviceToDevice, st));
            S2P_CUDA_TRY(cudaMemcpyAsync(s->axv, s->vh[l],
                                         (size_t)hl * INC_C * sizeof(float),
                                         cudaMemcpyDeviceToDevice, st));
        }
        S2P_CUDA_TRY(cudaMemcpyAsync(s->axk + (size_t)hl * INC_C,
                                     s->aq + 1024, INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(cudaMemcpyAsync(s->axv + (size_t)hl * INC_C,
                                     s->aq + 2048, INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(s2pdk_sdpa_inc(s->aq, 3072, s->axk, s->axv, INC_C, hl, 1,
                                    INC_HEADS, INC_HD, tf->window, s->at1,
                                    INC_C, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, 1, INC_C, ly->wo, INC_C, NULL,
                                  s->at3, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, s->at3, ly->g_attn, 1, INC_C, st));
        S2P_CUDA_TRY(s2pdk_rmsnorm(x, ly->ffn_norm, 1e-5f, 1, INC_C, s->at1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, 1, INC_C, ly->w1, 3072, NULL,
                                  s->af1, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->at1, 1, INC_C, ly->w3, 3072, NULL,
                                  s->af2, st));
        S2P_CUDA_TRY(s2pdk_silu_mul_ip(s->af1, s->af2, 3072, st));
        S2P_CUDA_TRY(s2pdk_matmul(s->af1, 1, 3072, ly->w2, INC_C, NULL,
                                  s->at1, st));
        S2P_CUDA_TRY(s2pdk_scale_add_ip(x, s->at1, ly->g_ffn, 1, INC_C, st));

        /* retain last min(127, hl+1) K/V rows: kv_ext tail */
        int keep = hl + 1 < INC_KVHIST ? hl + 1 : INC_KVHIST;
        int from = hl + 1 - keep;
        S2P_CUDA_TRY(cudaMemcpyAsync(s->kh[l], s->axk + (size_t)from * INC_C,
                                     (size_t)keep * INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
        S2P_CUDA_TRY(cudaMemcpyAsync(s->vh[l], s->axv + (size_t)from * INC_C,
                                     (size_t)keep * INC_C * sizeof(float),
                                     cudaMemcpyDeviceToDevice, st));
    }
    S2P_CUDA_TRY(s2pdk_rmsnorm(x, tf->norm, 1e-5f, 1, INC_C, x, st));
    return S2P_OK;
}

s2p_status s2pd_inc_push(s2pd_inc* s,
                         const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                         float* pcm_host, cudaStream_t st) {
    if (!s || !frame_codes || !pcm_host) return S2P_ERR_INVALID;
    s2p_dac* d = s->dac;
    if (s->pos + 1 > (int64_t)INT32_MAX / 4) return S2P_ERR_INVALID;

    S2P_TRY(s2pd_rope_ensure(d, (int)s->pos + 1));

    /* clamp exactly like s2p_dac_decode */
    int32_t cl[S2P_NUM_CODEBOOKS];
    {
        int32_t v = frame_codes[0];
        cl[0] = v < 0 ? 0 : (v > S2P_CB_SIZE - 1 ? S2P_CB_SIZE - 1 : v);
        for (int q = 1; q < S2P_NUM_CODEBOOKS; q++) {
            v = frame_codes[q];
            cl[q] = v < 0 ? 0 : (v > 1023 ? 1023 : v);
        }
    }
    S2P_CUDA_TRY(cudaMemcpyAsync(s->codes_dev, cl, sizeof(cl),
                                 cudaMemcpyHostToDevice, st));

    /* RVQ from_indices -> z [1024, 1]; a [1,1024] row and a [1024,1] column
     * are the same 1024 contiguous floats, so no transpose is needed. */
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
    S2P_CUDA_TRY(s2pdk_rvq_from_indices(s->codes_dev, 1, tabs, x, st));

    S2P_TRY(inc_post(s, x, st));

    /* upsample x2: tconv k2 s2 is column-local (out j reads only in[j/2]) */
    float* y = s->by;
    S2P_CUDA_TRY(s2pdk_tconv1d(x, INC_C, 1, d->up_t[0].w, d->up_t[0].b, INC_C,
                               2, 2, y, 2, st));
    S2P_TRY(inc_cnx(s, &d->up_cnx[0], &s->cnx_h[0], y, 2, st));
    S2P_CUDA_TRY(s2pdk_tconv1d(y, INC_C, 2, d->up_t[1].w, d->up_t[1].b, INC_C,
                               2, 2, x, 4, st));
    S2P_TRY(inc_cnx(s, &d->up_cnx[1], &s->cnx_h[1], x, 4, st));

    /* decoder conv0 (input history) */
    S2P_TRY(conv_hist(&d->dec_conv0, &s->c0_h, INC_C, 1536, 7, 1, x, 4,
                      s->bext, y, st));

    /* 4 decoder blocks */
    int Lc = 4;
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
    /* y holds [96, 2048] */
    S2P_CUDA_TRY(s2pdk_snake(y, y, d->dec_alpha, 96, Lc, st));
    S2P_TRY(conv_hist(&d->dec_convf, &s->cf_h, 96, 1, 7, 1, y, Lc, s->bext,
                      x, st));
    S2P_CUDA_TRY(s2pdk_tanh_ip(x, Lc, st));

    S2P_CUDA_TRY(cudaMemcpyAsync(pcm_host, x,
                                 (size_t)S2P_FRAME_SAMPLES * sizeof(float),
                                 cudaMemcpyDeviceToHost, st));
    S2P_CUDA_TRY(cudaStreamSynchronize(st));

    s->pos++;
    if (s->kv_len < INC_KVHIST) s->kv_len++;
    return S2P_OK;
}
