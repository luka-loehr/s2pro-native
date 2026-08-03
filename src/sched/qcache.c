/* s2pro-native — prequantized-weight sidecar cache (qcache.h impl).
 *
 * File layout (little-endian, payload offsets absolute, 256-aligned):
 *   u8  magic[8] = "S2PQC\1\0\0"
 *   u64 fingerprint          FNV-1a(config string + model file list)
 *   u32 n_entries, u32 reserved
 *   n_entries x entry header:
 *     u32 name_len  u32 kind  i32 N  i32 K  i32 group  u32 pad
 *     u64 q_off u64 q_len  u64 s_off u64 s_len
 *     char name[name_len]   (no NUL)
 *   payload region
 * kind: 1 per-channel INT8 (i8[N,K] + f32[N] scales)
 *       2 group-wise INT4 packed (u8[N,K/2] + f16[N*K/G] scales)
 *       3 group-wise INT4 int8-container (i8[N,K] + f16[N*K/G] scales)
 *
 * Every failure on the read side degrades to a miss (the caller live-
 * quantizes); every failure on the write side degrades to "no cache". */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <cuda_runtime.h>

#include "s2pro/qcache.h"
#include "s2pro/tensor.h"

#define QC_MAGIC "S2PQC\1\0"
#define QC_ALIGN 256
#define QC_MAX_ENTRIES 512

typedef struct {
    char*    name;
    uint32_t kind;
    int32_t  N, K, group;
    uint64_t q_off, q_len, s_off, s_len;
    /* write mode: host copies */
    void*    q_host;
    void*    s_host;
} qc_entry;

struct s2p_qcache {
    char     path[1024];
    int      reading;         /* 1 = serve from buf, 0 = collect for write */
    uint64_t fingerprint;
    char*    buf;             /* read mode: whole file */
    size_t   buf_len;
    qc_entry entries[QC_MAX_ENTRIES];
    int      n;
};

static uint64_t fnv1a(uint64_t h, const void* data, size_t len) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int env_flag_def(const char* name, int def) {
    const char* v = getenv(name);
    if (!v || !v[0]) return def;
    return v[0] == '1' && v[1] == '\0';
}

static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char* const*)a, *(const char* const*)b);
}

/* Fingerprint = quant config + sorted (name, size, mtime) of every
 * *.safetensors in the model dir. */
static s2p_status fingerprint_model(const char* model_dir,
                                    s2p_gemm_mode mode, uint64_t* out) {
    char cfg[256];
    const char* g = getenv("S2P_INT4_GROUP");
    snprintf(cfg, sizeof(cfg), "v1|mode=%d|int4=%d|g=%s|mse=%d|pk=%d|all=%d",
             (int)mode, env_flag_def("S2P_INT4", 0), g ? g : "32",
             env_flag_def("S2P_INT4_MSE", 1),
             env_flag_def("S2P_INT4_PACKED", 1),
             env_flag_def("S2P_INT4_ALL", 0));
    uint64_t h = fnv1a(0xcbf29ce484222325ULL, cfg, strlen(cfg));

    DIR* d = opendir(model_dir);
    if (!d) return S2P_ERR_IO;
    char* names[256];
    int n = 0;
    struct dirent* de;
    while ((de = readdir(d)) != NULL && n < 256) {
        size_t len = strlen(de->d_name);
        if (len > 12 && strcmp(de->d_name + len - 12, ".safetensors") == 0)
            names[n++] = strdup(de->d_name);
    }
    closedir(d);
    if (n == 0) {
        return S2P_ERR_IO;
    }
    qsort(names, n, sizeof(names[0]), cmp_str);
    for (int i = 0; i < n; i++) {
        char full[1200];
        snprintf(full, sizeof(full), "%s/%s", model_dir, names[i]);
        struct stat st;
        if (stat(full, &st) == 0) {
            h = fnv1a(h, names[i], strlen(names[i]));
            h = fnv1a(h, &st.st_size, sizeof(st.st_size));
            h = fnv1a(h, &st.st_mtime, sizeof(st.st_mtime));
        }
        free(names[i]);
    }
    *out = h;
    return S2P_OK;
}

static void qc_read_header(s2p_qcache* qc) {
    /* Any inconsistency: silently stay in write mode. */
    FILE* f = fopen(qc->path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 24) { fclose(f); return; }
    char* buf = (char*)malloc((size_t)sz);
    if (!buf) { fclose(f); return; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return; }
    if (memcmp(buf, QC_MAGIC, 8) != 0) { free(buf); return; }
    uint64_t fp;
    memcpy(&fp, buf + 8, 8);
    if (fp != qc->fingerprint) {
        fprintf(stderr, "[s2pro] qcache: stale fingerprint, rebuilding\n");
        free(buf);
        return;
    }
    uint32_t n;
    memcpy(&n, buf + 16, 4);
    if (n == 0 || n > QC_MAX_ENTRIES) { free(buf); return; }
    size_t off = 24;
    for (uint32_t i = 0; i < n; i++) {
        if (off + 56 > (size_t)sz) { free(buf); return; }
        qc_entry* e = &qc->entries[i];
        uint32_t name_len;
        memcpy(&name_len, buf + off, 4);
        memcpy(&e->kind, buf + off + 4, 4);
        memcpy(&e->N, buf + off + 8, 4);
        memcpy(&e->K, buf + off + 12, 4);
        memcpy(&e->group, buf + off + 16, 4);
        memcpy(&e->q_off, buf + off + 24, 8);
        memcpy(&e->q_len, buf + off + 32, 8);
        memcpy(&e->s_off, buf + off + 40, 8);
        memcpy(&e->s_len, buf + off + 48, 8);
        off += 56;
        if (name_len == 0 || name_len > 512 || off + name_len > (size_t)sz) {
            free(buf);
            return;
        }
        e->name = (char*)malloc(name_len + 1);
        if (!e->name) { free(buf); return; }
        memcpy(e->name, buf + off, name_len);
        e->name[name_len] = '\0';
        off += name_len;
        if (e->q_off + e->q_len > (uint64_t)sz ||
            e->s_off + e->s_len > (uint64_t)sz) {
            free(buf);
            return;
        }
    }
    qc->buf = buf;
    qc->buf_len = (size_t)sz;
    qc->n = (int)n;
    qc->reading = 1;
}

s2p_status s2p_qcache_open(const char* model_dir, s2p_gemm_mode mode,
                           s2p_qcache** out) {
    if (!model_dir || !out) return S2P_ERR_INVALID;
    *out = NULL;
    if (mode != S2P_GEMM_INT8) return S2P_OK; /* only quantized configs */
    if (!env_flag_def("S2P_QCACHE", 1)) return S2P_OK;

    s2p_qcache* qc = (s2p_qcache*)calloc(1, sizeof(*qc));
    if (!qc) return S2P_ERR_OOM;
    if (fingerprint_model(model_dir, mode, &qc->fingerprint) != S2P_OK) {
        free(qc);
        return S2P_OK; /* unusual model dir: run without cache */
    }
    const char* dir = getenv("S2P_QCACHE_DIR");
    if (!dir || !dir[0]) dir = model_dir;
    snprintf(qc->path, sizeof(qc->path), "%s/s2p_qcache.bin", dir);
    qc_read_header(qc);
    if (qc->reading)
        fprintf(stderr, "[s2pro] qcache: serving %d tensors from %s\n",
                qc->n, qc->path);
    else
        fprintf(stderr, "[s2pro] qcache: will write %s after load\n",
                qc->path);
    *out = qc;
    return S2P_OK;
}

int s2p_qcache_reading(const s2p_qcache* qc) { return qc && qc->reading; }

static qc_entry* qc_find(s2p_qcache* qc, const char* name) {
    for (int i = 0; i < qc->n; i++)
        if (strcmp(qc->entries[i].name, name) == 0) return &qc->entries[i];
    return NULL;
}

int s2p_qcache_try_load(s2p_qcache* qc, const char* name, int in_features,
                        int out_features, s2p_linear* lin,
                        cudaStream_t stream) {
    if (!qc || !qc->reading || !name || !lin) return 0;
    qc_entry* e = qc_find(qc, name);
    if (!e) {
        fprintf(stderr, "[s2pro] qcache: miss for %s (live path)\n", name);
        return 0;
    }
    if (e->N != out_features || e->K != in_features) {
        fprintf(stderr, "[s2pro] qcache: shape mismatch for %s\n", name);
        return 0;
    }
    memset(lin, 0, sizeof(*lin));
    lin->in_features = in_features;
    lin->out_features = out_features;

    s2p_status rc = S2P_OK;
    if (e->kind == 1) {
        int64_t qshape[2] = { e->N, e->K };
        int64_t sshape[1] = { e->N };
        rc = s2p_tensor_device_alloc(&lin->w_int8, S2P_DT_I8, 2, qshape);
        if (rc == S2P_OK)
            rc = s2p_tensor_device_alloc(&lin->w_iscale, S2P_DT_F32, 1,
                                         sshape);
    } else if (e->kind == 2) {
        int64_t pshape[2] = { e->N, e->K / 2 };
        int64_t sshape[1] = { (int64_t)e->N * (e->K / e->group) };
        rc = s2p_tensor_device_alloc(&lin->w_pack, S2P_DT_I8, 2, pshape);
        if (rc == S2P_OK)
            rc = s2p_tensor_device_alloc(&lin->w_iscale, S2P_DT_F16, 1,
                                         sshape);
    } else if (e->kind == 3) {
        int64_t qshape[2] = { e->N, e->K };
        int64_t sshape[1] = { (int64_t)e->N * (e->K / e->group) };
        rc = s2p_tensor_device_alloc(&lin->w_int8, S2P_DT_I8, 2, qshape);
        if (rc == S2P_OK)
            rc = s2p_tensor_device_alloc(&lin->w_iscale, S2P_DT_F16, 1,
                                         sshape);
    } else {
        return 0;
    }
    s2p_tensor* qt = (e->kind == 2) ? &lin->w_pack : &lin->w_int8;
    if (rc == S2P_OK && (qt->bytes != e->q_len ||
                         lin->w_iscale.bytes != e->s_len)) {
        fprintf(stderr, "[s2pro] qcache: size mismatch for %s\n", name);
        rc = S2P_ERR_FORMAT;
    }
    if (rc == S2P_OK)
        rc = s2p_tensor_upload(qt, qc->buf + e->q_off, e->q_len, stream);
    if (rc == S2P_OK)
        rc = s2p_tensor_upload(&lin->w_iscale, qc->buf + e->s_off, e->s_len,
                               stream);
    if (rc != S2P_OK) {
        s2p_tensor_free(&lin->w_int8);
        s2p_tensor_free(&lin->w_pack);
        s2p_tensor_free(&lin->w_iscale);
        memset(lin, 0, sizeof(*lin));
        lin->in_features = in_features;
        lin->out_features = out_features;
        return 0;
    }
    lin->int8_ready = 1;
    if (e->kind >= 2) lin->q_group = e->group;
    if (e->kind == 2) lin->q_packed = 1;
    return 1;
}

s2p_status s2p_qcache_put_linear(s2p_qcache* qc, const char* name,
                                 const s2p_linear* lin, cudaStream_t stream) {
    if (!qc || qc->reading || !name || !lin) return S2P_OK;
    if (!lin->int8_ready) return S2P_OK; /* unquantized: nothing to cache */
    if (qc->n >= QC_MAX_ENTRIES) return S2P_OK;

    const s2p_tensor* qt = lin->q_packed ? &lin->w_pack : &lin->w_int8;
    if (!qt->data || !lin->w_iscale.data) return S2P_OK;

    qc_entry* e = &qc->entries[qc->n];
    memset(e, 0, sizeof(*e));
    e->kind = lin->q_packed ? 2 : (lin->q_group > 0 ? 3 : 1);
    e->N = lin->out_features;
    e->K = lin->in_features;
    e->group = lin->q_group;
    e->q_len = qt->bytes;
    e->s_len = lin->w_iscale.bytes;
    e->q_host = malloc(qt->bytes);
    e->s_host = malloc(lin->w_iscale.bytes);
    e->name = strdup(name);
    if (!e->q_host || !e->s_host || !e->name) {
        free(e->q_host);
        free(e->s_host);
        free(e->name);
        memset(e, 0, sizeof(*e));
        return S2P_OK; /* cache degraded, load unaffected */
    }
    S2P_CUDA_TRY(cudaStreamSynchronize(stream));
    S2P_CUDA_TRY(cudaMemcpy(e->q_host, qt->data, qt->bytes,
                            cudaMemcpyDeviceToHost));
    S2P_CUDA_TRY(cudaMemcpy(e->s_host, lin->w_iscale.data,
                            lin->w_iscale.bytes, cudaMemcpyDeviceToHost));
    qc->n++;
    return S2P_OK;
}

static size_t align_up(size_t v) {
    return (v + QC_ALIGN - 1) & ~(size_t)(QC_ALIGN - 1);
}

s2p_status s2p_qcache_finish(s2p_qcache* qc, cudaStream_t stream) {
    if (!qc) return S2P_OK;
    if (qc->reading) {
        /* uploads were issued from qc->buf: they must be complete */
        S2P_CUDA_TRY(cudaStreamSynchronize(stream));
        free(qc->buf);
        qc->buf = NULL;
        return S2P_OK;
    }
    if (qc->n == 0) return S2P_OK;

    size_t off = 24;
    for (int i = 0; i < qc->n; i++)
        off += 56 + strlen(qc->entries[i].name);
    for (int i = 0; i < qc->n; i++) {
        qc_entry* e = &qc->entries[i];
        off = align_up(off);
        e->q_off = off;
        off += e->q_len;
        off = align_up(off);
        e->s_off = off;
        off += e->s_len;
    }

    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", qc->path);
    FILE* f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "[s2pro] qcache: cannot write %s (skipping)\n", tmp);
        return S2P_OK;
    }
    int ok = 1;
    char zeros[QC_ALIGN];
    memset(zeros, 0, sizeof(zeros));
    ok &= fwrite(QC_MAGIC, 1, 8, f) == 8;
    ok &= fwrite(&qc->fingerprint, 1, 8, f) == 8;
    uint32_t n32 = (uint32_t)qc->n, zero32 = 0;
    ok &= fwrite(&n32, 1, 4, f) == 4;
    ok &= fwrite(&zero32, 1, 4, f) == 4;
    for (int i = 0; i < qc->n && ok; i++) {
        qc_entry* e = &qc->entries[i];
        uint32_t name_len = (uint32_t)strlen(e->name), pad32 = 0;
        ok &= fwrite(&name_len, 1, 4, f) == 4;
        ok &= fwrite(&e->kind, 1, 4, f) == 4;
        ok &= fwrite(&e->N, 1, 4, f) == 4;
        ok &= fwrite(&e->K, 1, 4, f) == 4;
        ok &= fwrite(&e->group, 1, 4, f) == 4;
        ok &= fwrite(&pad32, 1, 4, f) == 4;
        ok &= fwrite(&e->q_off, 1, 8, f) == 8;
        ok &= fwrite(&e->q_len, 1, 8, f) == 8;
        ok &= fwrite(&e->s_off, 1, 8, f) == 8;
        ok &= fwrite(&e->s_len, 1, 8, f) == 8;
        ok &= fwrite(e->name, 1, name_len, f) == name_len;
    }
    for (int i = 0; i < qc->n && ok; i++) {
        qc_entry* e = &qc->entries[i];
        long pos = ftell(f);
        while (pos >= 0 && (size_t)pos < e->q_off && ok) {
            size_t pad = e->q_off - (size_t)pos;
            if (pad > sizeof(zeros)) pad = sizeof(zeros);
            ok &= fwrite(zeros, 1, pad, f) == pad;
            pos = ftell(f);
        }
        ok &= fwrite(e->q_host, 1, e->q_len, f) == e->q_len;
        pos = ftell(f);
        while (pos >= 0 && (size_t)pos < e->s_off && ok) {
            size_t pad = e->s_off - (size_t)pos;
            if (pad > sizeof(zeros)) pad = sizeof(zeros);
            ok &= fwrite(zeros, 1, pad, f) == pad;
            pos = ftell(f);
        }
        ok &= fwrite(e->s_host, 1, e->s_len, f) == e->s_len;
    }
    ok &= fclose(f) == 0;
    if (!ok || rename(tmp, qc->path) != 0) {
        fprintf(stderr, "[s2pro] qcache: write failed (%s), removing\n", tmp);
        remove(tmp);
        return S2P_OK;
    }
    fprintf(stderr, "[s2pro] qcache: wrote %d tensors -> %s\n", qc->n,
            qc->path);
    return S2P_OK;
}

void s2p_qcache_free(s2p_qcache* qc) {
    if (!qc) return;
    for (int i = 0; i < qc->n; i++) {
        free(qc->entries[i].name);
        free(qc->entries[i].q_host);
        free(qc->entries[i].s_host);
    }
    free(qc->buf);
    free(qc);
}
