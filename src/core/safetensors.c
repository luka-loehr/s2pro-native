/* s2pro-native — safetensors checkpoint reader (s2pro/safetensors.h).
 *
 * Layout per file: u64-LE header length, JSON header (parsed with s2p_json),
 * then the raw data section. `data_offsets` are relative to the data section.
 * Handles both a single model.safetensors and a sharded checkpoint via
 * model.safetensors.index.json (weight_map values name the shard files; the
 * map itself is only used to enumerate the unique shards — every tensor in
 * every listed shard is indexed).
 *
 * Files are mmap'd read-only for their whole lifetime, so s2p_st_view.data
 * pointers stay valid until s2p_st_close. Lookup is an open-addressing
 * (linear probe) FNV-1a hash table built once after all shards load; it also
 * rejects duplicate tensor names across shards.
 */
#include "s2pro/safetensors.h"
#include "s2pro/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    char*    path;
    int      fd;
    uint8_t* map;
    size_t   size;
} st_file;

typedef struct {
    char*       name; /* owned; s2p_st_view.name aliases this */
    s2p_st_view v;
} st_entry;

struct s2p_st {
    st_file*  files;
    int       nfiles, fcap;
    st_entry* ents;
    int       count, ecap;
    uint32_t* htab;  /* entry index + 1; 0 = empty */
    uint32_t  hmask;
};

/* ----------------------------------------------------------------- helpers */

static uint64_t fnv1a(const char* s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static char* path_join(const char* dir, const char* name)
{
    size_t dl = strlen(dir), nl = strlen(name);
    char* p = (char*)malloc(dl + 1 + nl + 1);
    if (!p) return NULL;
    memcpy(p, dir, dl);
    p[dl] = '/';
    memcpy(p + dl + 1, name, nl + 1);
    return p;
}

static s2p_status read_all(const char* path, char** out, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    long sz;
    char* buf;
    if (!f) return S2P_ERR_IO;
    if (fseek(f, 0, SEEK_END) || (sz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return S2P_ERR_IO;
    }
    buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return S2P_ERR_OOM;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return S2P_ERR_IO;
    }
    fclose(f);
    buf[sz] = '\0';
    *out = buf;
    *out_len = (size_t)sz;
    return S2P_OK;
}

static int dtype_from_str(const char* s, s2p_dtype* out)
{
    if (!strcmp(s, "BF16")) { *out = S2P_DT_BF16; return 1; }
    if (!strcmp(s, "F16"))  { *out = S2P_DT_F16;  return 1; }
    if (!strcmp(s, "F32"))  { *out = S2P_DT_F32;  return 1; }
    if (!strcmp(s, "I64"))  { *out = S2P_DT_I64;  return 1; }
    if (!strcmp(s, "I32"))  { *out = S2P_DT_I32;  return 1; }
    if (!strcmp(s, "U8"))   { *out = S2P_DT_U8;   return 1; }
    if (!strcmp(s, "I8"))   { *out = S2P_DT_I8;   return 1; }
    if (!strcmp(s, "BOOL")) { *out = S2P_DT_U8;   return 1; } /* same width */
    return 0;
}

static s2p_status add_entry(s2p_st* st, const char* name,
                            const s2p_st_view* v)
{
    st_entry* e;
    if (st->count == st->ecap) {
        int nc = st->ecap ? st->ecap * 2 : 1024;
        void* np = realloc(st->ents, (size_t)nc * sizeof(*st->ents));
        if (!np) return S2P_ERR_OOM;
        st->ents = (st_entry*)np;
        st->ecap = nc;
    }
    e = &st->ents[st->count];
    e->name = strdup(name);
    if (!e->name) return S2P_ERR_OOM;
    e->v = *v;
    e->v.name = e->name;
    st->count++;
    return S2P_OK;
}

/* mmap one shard, parse its header, index every tensor. Registers the file
 * in st->files first so s2p_st_close unwinds partial failures. */
static s2p_status st_open_file(s2p_st* st, const char* dir, const char* fname)
{
    char* path = NULL;
    st_file* sf;
    struct stat sb;
    uint64_t hlen;
    s2p_json* j = NULL;
    const s2p_jval* root;
    const uint8_t* data;
    size_t dsize;
    s2p_status rc;
    int i, n;

    if (st->nfiles == st->fcap) {
        int nc = st->fcap ? st->fcap * 2 : 4;
        void* np = realloc(st->files, (size_t)nc * sizeof(*st->files));
        if (!np) return S2P_ERR_OOM;
        st->files = (st_file*)np;
        st->fcap = nc;
    }
    path = path_join(dir, fname);
    if (!path) return S2P_ERR_OOM;

    sf = &st->files[st->nfiles];
    memset(sf, 0, sizeof(*sf));
    sf->path = path;
    sf->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (sf->fd < 0) {
        fprintf(stderr, "[s2pro] cannot open %s\n", path);
        free(path);
        return S2P_ERR_IO;
    }
    st->nfiles++; /* registered: close() unwinds from here on */
    if (fstat(sf->fd, &sb) != 0 || sb.st_size < 8) {
        fprintf(stderr, "[s2pro] bad safetensors file %s\n", path);
        return S2P_ERR_FORMAT;
    }
    sf->size = (size_t)sb.st_size;
    sf->map = (uint8_t*)mmap(NULL, sf->size, PROT_READ, MAP_PRIVATE,
                             sf->fd, 0);
    if (sf->map == MAP_FAILED) {
        sf->map = NULL;
        fprintf(stderr, "[s2pro] mmap failed for %s\n", path);
        return S2P_ERR_IO;
    }
    (void)posix_madvise(sf->map, sf->size, POSIX_MADV_SEQUENTIAL);

    hlen = 0;
    for (i = 7; i >= 0; i--) hlen = (hlen << 8) | sf->map[i]; /* u64 LE */
    if (hlen > sf->size - 8) {
        fprintf(stderr, "[s2pro] safetensors header overruns file: %s\n",
                path);
        return S2P_ERR_FORMAT;
    }
    rc = s2p_json_parse((const char*)sf->map + 8, (size_t)hlen, &j);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] bad safetensors header json: %s\n", path);
        return rc;
    }
    root = s2p_json_root(j);
    data = sf->map + 8 + hlen;
    dsize = sf->size - 8 - (size_t)hlen;

    n = s2p_jobj_len(root);
    if (n == 0) {
        s2p_json_free(j);
        fprintf(stderr, "[s2pro] safetensors header is not an object: %s\n",
                path);
        return S2P_ERR_FORMAT;
    }
    for (i = 0; i < n; i++) {
        const char* key = NULL;
        const s2p_jval* tv = NULL;
        const s2p_jval* jv;
        const char* dts;
        s2p_st_view view;
        int64_t numel = 1, b, e;
        int nd, k;

        if (s2p_jobj_at(root, i, &key, &tv) != S2P_OK) {
            rc = S2P_ERR_INTERNAL;
            goto fail;
        }
        if (!strcmp(key, "__metadata__")) continue;

        memset(&view, 0, sizeof(view));
        dts = s2p_jstr(s2p_jobj_get(tv, "dtype"), NULL);
        if (!dts) {
            fprintf(stderr, "[s2pro] tensor %s: missing dtype\n", key);
            rc = S2P_ERR_FORMAT;
            goto fail;
        }
        if (!dtype_from_str(dts, &view.dtype)) {
            fprintf(stderr, "[s2pro] tensor %s: unsupported dtype %s\n", key,
                    dts);
            rc = S2P_ERR_UNSUPPORTED;
            goto fail;
        }
        jv = s2p_jobj_get(tv, "shape");
        nd = s2p_jarr_len(jv);
        if ((nd == 0 && !jv) || nd > 4) {
            fprintf(stderr, "[s2pro] tensor %s: bad shape (ndim %d)\n", key,
                    nd);
            rc = nd > 4 ? S2P_ERR_UNSUPPORTED : S2P_ERR_FORMAT;
            goto fail;
        }
        view.ndim = nd;
        for (k = 0; k < 4; k++) view.shape[k] = 1;
        for (k = 0; k < nd; k++) {
            int64_t d = s2p_jint(s2p_jarr_at(jv, k));
            if (d < 0) {
                fprintf(stderr, "[s2pro] tensor %s: negative dim\n", key);
                rc = S2P_ERR_FORMAT;
                goto fail;
            }
            view.shape[k] = d;
            numel = d ? numel * d : 0;
        }
        jv = s2p_jobj_get(tv, "data_offsets");
        if (s2p_jarr_len(jv) != 2) {
            fprintf(stderr, "[s2pro] tensor %s: bad data_offsets\n", key);
            rc = S2P_ERR_FORMAT;
            goto fail;
        }
        b = s2p_jint(s2p_jarr_at(jv, 0));
        e = s2p_jint(s2p_jarr_at(jv, 1));
        if (b < 0 || e < b || (uint64_t)e > dsize) {
            fprintf(stderr, "[s2pro] tensor %s: offsets out of range\n", key);
            rc = S2P_ERR_FORMAT;
            goto fail;
        }
        if ((uint64_t)(e - b) !=
            (uint64_t)numel * s2p_dtype_size(view.dtype)) {
            fprintf(stderr,
                    "[s2pro] tensor %s: size mismatch (%lld bytes vs "
                    "numel %lld x %zu)\n",
                    key, (long long)(e - b), (long long)numel,
                    s2p_dtype_size(view.dtype));
            rc = S2P_ERR_FORMAT;
            goto fail;
        }
        view.data = data + b;
        view.nbytes = (size_t)(e - b);
        rc = add_entry(st, key, &view);
        if (rc != S2P_OK) goto fail;
    }
    s2p_json_free(j);
    return S2P_OK;

fail:
    s2p_json_free(j);
    return rc;
}

/* Build the name hash table; detects duplicate tensor names. */
static s2p_status st_build_htab(s2p_st* st)
{
    uint32_t cap = 16;
    int i;
    while (cap < (uint32_t)st->count * 2u) cap <<= 1;
    st->htab = (uint32_t*)calloc(cap, sizeof(uint32_t));
    if (!st->htab) return S2P_ERR_OOM;
    st->hmask = cap - 1;
    for (i = 0; i < st->count; i++) {
        uint32_t h = (uint32_t)fnv1a(st->ents[i].name) & st->hmask;
        while (st->htab[h]) {
            if (!strcmp(st->ents[st->htab[h] - 1].name, st->ents[i].name)) {
                fprintf(stderr, "[s2pro] duplicate tensor across shards: %s\n",
                        st->ents[i].name);
                return S2P_ERR_FORMAT;
            }
            h = (h + 1) & st->hmask;
        }
        st->htab[h] = (uint32_t)i + 1;
    }
    return S2P_OK;
}

/* --------------------------------------------------------------------- API */

s2p_status s2p_st_open_dir(const char* model_dir, s2p_st** out)
{
    s2p_st* st;
    char* ipath = NULL;
    char** shards = NULL;
    int nshards = 0;
    s2p_status rc = S2P_OK;
    struct stat sb;
    int i;

    if (!model_dir || !out) return S2P_ERR_INVALID;
    *out = NULL;
    st = (s2p_st*)calloc(1, sizeof(*st));
    if (!st) return S2P_ERR_OOM;

    ipath = path_join(model_dir, "model.safetensors.index.json");
    if (!ipath) {
        free(st);
        return S2P_ERR_OOM;
    }

    if (stat(ipath, &sb) == 0) {
        /* sharded checkpoint: collect the unique shard files */
        char* buf = NULL;
        size_t len = 0;
        s2p_json* j = NULL;
        const s2p_jval* wm;
        int n, k;

        rc = read_all(ipath, &buf, &len);
        if (rc == S2P_OK) rc = s2p_json_parse(buf, len, &j);
        free(buf);
        if (rc != S2P_OK) {
            fprintf(stderr, "[s2pro] cannot parse %s\n", ipath);
            goto fail;
        }
        wm = s2p_jobj_get(s2p_json_root(j), "weight_map");
        n = s2p_jobj_len(wm);
        if (n == 0) {
            fprintf(stderr, "[s2pro] %s: empty/missing weight_map\n", ipath);
            s2p_json_free(j);
            rc = S2P_ERR_FORMAT;
            goto fail;
        }
        shards = (char**)calloc((size_t)n, sizeof(char*));
        if (!shards) {
            s2p_json_free(j);
            rc = S2P_ERR_OOM;
            goto fail;
        }
        for (k = 0; k < n; k++) {
            const s2p_jval* fv = NULL;
            const char* fname;
            int dup = 0, s;
            if (s2p_jobj_at(wm, k, NULL, &fv) != S2P_OK) continue;
            fname = s2p_jstr(fv, NULL);
            if (!fname) {
                fprintf(stderr, "[s2pro] %s: non-string weight_map entry\n",
                        ipath);
                s2p_json_free(j);
                rc = S2P_ERR_FORMAT;
                goto fail;
            }
            for (s = 0; s < nshards; s++)
                if (!strcmp(shards[s], fname)) { dup = 1; break; }
            if (dup) continue;
            shards[nshards] = strdup(fname);
            if (!shards[nshards]) {
                s2p_json_free(j);
                rc = S2P_ERR_OOM;
                goto fail;
            }
            nshards++;
        }
        s2p_json_free(j);
        for (i = 0; i < nshards; i++) {
            rc = st_open_file(st, model_dir, shards[i]);
            if (rc != S2P_OK) goto fail;
        }
    } else {
        rc = st_open_file(st, model_dir, "model.safetensors");
        if (rc != S2P_OK) goto fail;
    }

    if (st->count == 0) {
        fprintf(stderr, "[s2pro] no tensors found under %s\n", model_dir);
        rc = S2P_ERR_FORMAT;
        goto fail;
    }
    rc = st_build_htab(st);
    if (rc != S2P_OK) goto fail;

    free(ipath);
    if (shards) {
        for (i = 0; i < nshards; i++) free(shards[i]);
        free(shards);
    }
    *out = st;
    return S2P_OK;

fail:
    free(ipath);
    if (shards) {
        for (i = 0; i < nshards; i++) free(shards[i]);
        free(shards);
    }
    s2p_st_close(st);
    return rc;
}

s2p_status s2p_st_find(s2p_st* st, const char* name, s2p_st_view* out)
{
    uint32_t h;
    if (!st || !st->htab || !name || !out) return S2P_ERR_INVALID;
    h = (uint32_t)fnv1a(name) & st->hmask;
    while (st->htab[h]) {
        const st_entry* e = &st->ents[st->htab[h] - 1];
        if (!strcmp(e->name, name)) {
            *out = e->v;
            return S2P_OK;
        }
        h = (h + 1) & st->hmask;
    }
    return S2P_ERR_INVALID; /* not found */
}

int s2p_st_count(const s2p_st* st)
{
    return st ? st->count : 0;
}

s2p_status s2p_st_at(const s2p_st* st, int index, s2p_st_view* out)
{
    if (!st || !out || index < 0 || index >= st->count)
        return S2P_ERR_INVALID;
    *out = st->ents[index].v;
    return S2P_OK;
}

void s2p_st_close(s2p_st* st)
{
    int i;
    if (!st) return;
    for (i = 0; i < st->count; i++) free(st->ents[i].name);
    free(st->ents);
    free(st->htab);
    for (i = 0; i < st->nfiles; i++) {
        if (st->files[i].map) munmap(st->files[i].map, st->files[i].size);
        if (st->files[i].fd >= 0) close(st->files[i].fd);
        free(st->files[i].path);
    }
    free(st->files);
    free(st);
}

s2p_status s2p_st_load_device(s2p_st* st, const char* name, s2p_dtype expect_dt,
                              int ndim, const int64_t* expect_shape,
                              s2p_tensor* out_dev, cudaStream_t stream)
{
    s2p_st_view v;
    s2p_status rc;
    int i;

    if (!out_dev) return S2P_ERR_INVALID;
    rc = s2p_st_find(st, name, &v);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] tensor not found: %s\n", name);
        return rc;
    }
    if (v.dtype != expect_dt) {
        fprintf(stderr, "[s2pro] tensor %s: dtype %d, expected %d\n", name,
                (int)v.dtype, (int)expect_dt);
        return S2P_ERR_FORMAT;
    }
    if (v.ndim != ndim) {
        fprintf(stderr, "[s2pro] tensor %s: ndim %d, expected %d\n", name,
                v.ndim, ndim);
        return S2P_ERR_FORMAT;
    }
    for (i = 0; i < ndim; i++) {
        if (expect_shape && expect_shape[i] >= 0 &&
            expect_shape[i] != v.shape[i]) {
            fprintf(stderr,
                    "[s2pro] tensor %s: shape[%d] = %lld, expected %lld\n",
                    name, i, (long long)v.shape[i],
                    (long long)expect_shape[i]);
            return S2P_ERR_FORMAT;
        }
    }
    S2P_TRY(s2p_tensor_device_alloc(out_dev, expect_dt, v.ndim, v.shape));
    rc = s2p_tensor_upload(out_dev, v.data, v.nbytes, stream);
    if (rc != S2P_OK) {
        s2p_tensor_free(out_dev);
        return rc;
    }
    return S2P_OK;
}
