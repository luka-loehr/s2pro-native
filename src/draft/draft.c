/* s2pro-native — draft-model loader and forward (docs/SPECULATIVE.md §7). */
#include "draft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/kernels.h"

#define TRY(x)                                                              \
    do {                                                                    \
        s2p_status rc_ = (x);                                               \
        if (rc_ != S2P_OK) return rc_;                                      \
    } while (0)
#define CU(x)                                                               \
    do {                                                                    \
        cudaError_t ce_ = (x);                                              \
        if (ce_ != cudaSuccess) {                                           \
            fprintf(stderr, "[draft] CUDA: %s\n", cudaGetErrorString(ce_)); \
            return S2P_ERR_CUDA;                                            \
        }                                                                   \
    } while (0)

/* Minimal safetensors reader for the trainer's FIXED tensor names (the
 * file is produced by tools/draft_train.py; a general reader lives in
 * src/core and can replace this when the draft joins the server load
 * path). Header = u64 LE JSON length + JSON; we locate each name and
 * parse its data_offsets. */
static int st_find(const char* json, const char* name, size_t* off,
                   size_t* len) {
    char key[128];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char* p = strstr(json, key);
    if (!p) return -1;
    p = strstr(p, "\"data_offsets\"");
    if (!p) return -1;
    unsigned long long a = 0, b = 0;
    if (sscanf(p, "\"data_offsets\":[%llu,%llu]", &a, &b) != 2) return -1;
    *off = (size_t)a;
    *len = (size_t)(b - a);
    return 0;
}

static s2p_status load_one(s2p_tensor* t, const char* json,
                           const unsigned char* base, const char* name,
                           int ndim, const int64_t* shape) {
    size_t off = 0, len = 0;
    if (st_find(json, name, &off, &len) != 0) {
        fprintf(stderr, "[draft] missing tensor %s\n", name);
        return S2P_ERR_INVALID;
    }
    TRY(s2p_tensor_device_alloc(t, S2P_DT_BF16, ndim, shape));
    if (len != t->bytes) {
        fprintf(stderr, "[draft] %s: %zu bytes, want %zu\n", name, len,
                t->bytes);
        return S2P_ERR_INVALID;
    }
    return s2p_tensor_upload(t, base + off, len, 0);
}

s2p_status s2p_draft_load(s2p_draft_model* m, const char* path) {
    memset(m, 0, sizeof(*m));
    FILE* f = fopen(path, "rb");
    if (!f) return S2P_ERR_IO;
    unsigned char hl[8];
    if (fread(hl, 1, 8, f) != 8) { fclose(f); return S2P_ERR_IO; }
    size_t jlen = 0;
    for (int i = 7; i >= 0; i--) jlen = (jlen << 8) | hl[i];
    char* json = (char*)malloc(jlen + 1);
    if (!json || fread(json, 1, jlen, f) != jlen) {
        free(json); fclose(f); return S2P_ERR_IO;
    }
    json[jlen] = '\0';
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    size_t dlen = (size_t)fsz - 8 - jlen;
    unsigned char* data = (unsigned char*)malloc(dlen);
    fseek(f, 8 + (long)jlen, SEEK_SET);
    if (!data || fread(data, 1, dlen, f) != dlen) {
        free(json); free(data); fclose(f); return S2P_ERR_IO;
    }
    fclose(f);

    const int64_t D = S2P_DRAFT_DIM, F = S2P_DRAFT_FFN;
    s2p_status rc = S2P_OK;
    struct { s2p_tensor* t; const char* n; int nd; int64_t s[2]; } spec[] = {
        { &m->fuse, "fuse.weight", 2, { D, 2 * D } },
        { &m->wqkv, "wqkv.weight", 2, { 6144, D } },
        { &m->wo, "wo.weight", 2, { D, 4096 } },
        { &m->w1, "w1.weight", 2, { F, D } },
        { &m->w3, "w3.weight", 2, { F, D } },
        { &m->w2, "w2.weight", 2, { D, F } },
        { &m->q_norm, "q_norm", 1, { S2P_HEAD_DIM, 0 } },
        { &m->k_norm, "k_norm", 1, { S2P_HEAD_DIM, 0 } },
        { &m->attn_norm, "attn_norm", 1, { D, 0 } },
        { &m->ffn_norm, "ffn_norm", 1, { D, 0 } },
        { &m->final_norm, "final_norm", 1, { D, 0 } },
    };
    for (size_t i = 0; i < sizeof(spec) / sizeof(spec[0]) && rc == S2P_OK;
         i++)
        rc = load_one(spec[i].t, json, data, spec[i].n, spec[i].nd,
                      spec[i].s);
    free(json);
    free(data);
    if (rc != S2P_OK) return rc;

    struct { s2p_tensor* t; int64_t w; } scr[] = {
        { &m->xcat, 2 * D }, { &m->x, D },   { &m->a, D },
        { &m->qkv, 6144 },   { &m->attn, 4096 },
        { &m->gu, 2 * F },   { &m->ffn, F },
    };
    for (size_t i = 0; i < sizeof(scr) / sizeof(scr[0]); i++) {
        int64_t sh[2] = { 1, scr[i].w };
        TRY(s2p_tensor_device_alloc(scr[i].t, S2P_DT_BF16, 2, sh));
    }
    fprintf(stderr, "[draft] loaded %s (114M bf16)\n", path);
    return S2P_OK;
}

void s2p_draft_free(s2p_draft_model* m) {
    s2p_tensor* ts[] = { &m->fuse, &m->wqkv, &m->wo, &m->w1, &m->w3, &m->w2,
                         &m->q_norm, &m->k_norm, &m->attn_norm, &m->ffn_norm,
                         &m->final_norm, &m->xcat, &m->x, &m->a, &m->qkv,
                         &m->attn, &m->gu, &m->ffn };
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++)
        s2p_tensor_free(ts[i]);
}

s2p_status s2p_draft_state_init(s2p_draft_state* s) {
    memset(s, 0, sizeof(*s));
    int64_t sh[3] = { S2P_SLOW_KV_HEADS, S2P_DRAFT_CTX, S2P_HEAD_DIM };
    TRY(s2p_tensor_device_alloc(&s->kc, S2P_DT_BF16, 3, sh));
    TRY(s2p_tensor_device_alloc(&s->vc, S2P_DT_BF16, 3, sh));
    return S2P_OK;
}

void s2p_draft_state_free(s2p_draft_state* s) {
    s2p_tensor_free(&s->kc);
    s2p_tensor_free(&s->vc);
}

#define BF(t) ((__nv_bfloat16*)(t).data)

s2p_status s2p_draft_step(s2p_draft_model* m, s2p_draft_state* s,
                          const void* h, const void* e, void* out,
                          cudaStream_t st) {
    const int D = S2P_DRAFT_DIM, F = S2P_DRAFT_FFN;
    if (s->pos >= S2P_DRAFT_CTX) return S2P_ERR_INVALID;
    /* xcat = [h ; e] */
    CU(cudaMemcpyAsync(BF(m->xcat), h, (size_t)D * 2,
                       cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(BF(m->xcat) + D, e, (size_t)D * 2,
                       cudaMemcpyDeviceToDevice, st));
    /* x = fuse(xcat); residual stream */
    TRY(s2p_gemm_bf16(BF(m->xcat), BF(m->fuse), BF(m->x), 1, D, 2 * D, st));
    /* attention */
    CU(s2pk_rms_norm(BF(m->x), BF(m->attn_norm), BF(m->a), 1, D,
                     S2P_NORM_EPS, st));
    TRY(s2p_gemm_bf16(BF(m->a), BF(m->wqkv), BF(m->qkv), 1, 6144, D, st));
    __nv_bfloat16* q = BF(m->qkv);
    __nv_bfloat16* k = BF(m->qkv) + (size_t)S2P_SLOW_Q_HEADS * S2P_HEAD_DIM;
    __nv_bfloat16* v = k + (size_t)S2P_SLOW_KV_HEADS * S2P_HEAD_DIM;
    CU(s2pk_qk_norm(q, BF(m->q_norm), 1, S2P_SLOW_Q_HEADS, S2P_HEAD_DIM,
                    S2P_NORM_EPS, st));
    CU(s2pk_qk_norm(k, BF(m->k_norm), 1, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                    S2P_NORM_EPS, st));
    CU(s2pk_rope(q, k, 1, S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                 s->pos, S2P_ROPE_BASE, st));
    CU(s2pk_kv_append(k, v, BF(s->kc), BF(s->vc), 1, S2P_SLOW_KV_HEADS,
                      S2P_HEAD_DIM, s->pos, S2P_DRAFT_CTX, st));
    CU(s2pk_attention(q, BF(s->kc), BF(s->vc), BF(m->attn), 1,
                      S2P_SLOW_Q_HEADS, S2P_SLOW_KV_HEADS, S2P_HEAD_DIM,
                      s->pos, S2P_DRAFT_CTX, st));
    TRY(s2p_gemm_bf16(BF(m->attn), BF(m->wo), BF(m->a), 1, D, 4096, st));
    CU(s2pk_add(BF(m->x), BF(m->a), D, st));
    /* ffn */
    CU(s2pk_rms_norm(BF(m->x), BF(m->ffn_norm), BF(m->a), 1, D, S2P_NORM_EPS,
                     st));
    TRY(s2p_gemm_bf16(BF(m->a), BF(m->w1), BF(m->gu), 1, F, D, st));
    TRY(s2p_gemm_bf16(BF(m->a), BF(m->w3), BF(m->gu) + F, 1, F, D, st));
    CU(s2pk_silu_mul(BF(m->gu), BF(m->ffn), 1, F, st));
    TRY(s2p_gemm_bf16(BF(m->ffn), BF(m->w2), BF(m->a), 1, D, F, st));
    CU(s2pk_add(BF(m->x), BF(m->a), D, st));
    /* predicted next final-normed hidden */
    CU(s2pk_rms_norm(BF(m->x), BF(m->final_norm), (__nv_bfloat16*)out, 1, D,
                     S2P_NORM_EPS, st));
    s->pos++;
    return S2P_OK;
}
