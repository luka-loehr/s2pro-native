/* s2pro-native — loader for the pre-converted codec weight artifact.
 *
 * On-disk format (produced by fish-s2-native/tools/convert_codec.py):
 *   codec.idx  ASCII, one tensor per line:
 *                <name> <byte_off> <byte_len> <ndim> <d0> [<d1> [<d2> [<d3>]]]
 *              Offsets index into codec.bin. Names are the PyTorch state-dict
 *              keys of the modded-DAC codec with weight-norm ALREADY FOLDED
 *              (g*v/||v|| materialized as plain "<base>.weight"). Sorted by
 *              name; offsets ascend in listing order.
 *   codec.bin  Concatenated raw little-endian float32 tensor data, row-major,
 *              at the offsets given by codec.idx. FP32 end to end.
 *
 * Converter policy (matters for coverage): only decode-path tensors are kept.
 *   dropped: encoder.*  (entire encoder conv stack + its transformer)
 *            quantizer.pre_module.*  (pre-RVQ transformer, encode-only)
 *            *.causal_mask / *.freqs_cis buffers (rematerialized at load)
 *   kept:    decoder.*, quantizer.{downsample,upsample,post_module,
 *            semantic_quantizer,quantizer}.*  (note: downsample and the VQ
 *            in_proj tensors are present even though only encode uses them).
 *
 * The whole .bin is uploaded to ONE device buffer; tensors are views into it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "dac_internal.h"

static int ent_cmp_name(const void* a, const void* b) {
    return strcmp(((const s2p_dacw_ent*)a)->name, ((const s2p_dacw_ent*)b)->name);
}

s2p_status s2p_dacw_load(const char* model_dir, s2p_dacw* w) {
    if (!model_dir || !w) return S2P_ERR_INVALID;
    memset(w, 0, sizeof(*w));

    char path[1024];
    snprintf(path, sizeof(path), "%s/codec.idx", model_dir);
    FILE* fi = fopen(path, "r");
    if (!fi) {
        fprintf(stderr, "[s2pro] dac: cannot open %s\n", path);
        return S2P_ERR_IO;
    }

    int cap = 512, n = 0;
    s2p_dacw_ent* ents = (s2p_dacw_ent*)malloc((size_t)cap * sizeof(*ents));
    if (!ents) { fclose(fi); return S2P_ERR_OOM; }

    char line[512];
    int64_t max_end = 0;
    while (fgets(line, sizeof(line), fi)) {
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') continue;
        if (n == cap) {
            cap *= 2;
            s2p_dacw_ent* ne = (s2p_dacw_ent*)realloc(ents, (size_t)cap * sizeof(*ents));
            if (!ne) { free(ents); fclose(fi); return S2P_ERR_OOM; }
            ents = ne;
        }
        s2p_dacw_ent* e = &ents[n];
        memset(e, 0, sizeof(*e));
        e->d[0] = e->d[1] = e->d[2] = e->d[3] = 1;
        long long off, nbytes, d0 = 1, d1 = 1, d2 = 1, d3 = 1;
        int ndim = 0;
        int got = sscanf(line, "%159s %lld %lld %d %lld %lld %lld %lld",
                         e->name, &off, &nbytes, &ndim, &d0, &d1, &d2, &d3);
        if (got < 5 || ndim < 1 || ndim > 4 || got != 4 + ndim) {
            fprintf(stderr, "[s2pro] dac: bad codec.idx line: %s", line);
            free(ents); fclose(fi);
            return S2P_ERR_FORMAT;
        }
        e->off = off; e->nbytes = nbytes; e->ndim = ndim;
        e->d[0] = d0; e->d[1] = d1; e->d[2] = d2; e->d[3] = d3;
        int64_t prod = e->d[0] * e->d[1] * e->d[2] * e->d[3];
        if (prod * 4 != e->nbytes || (e->off & 3) != 0) {
            fprintf(stderr, "[s2pro] dac: codec.idx shape/offset mismatch for %s\n",
                    e->name);
            free(ents); fclose(fi);
            return S2P_ERR_FORMAT;
        }
        if (e->off + e->nbytes > max_end) max_end = e->off + e->nbytes;
        n++;
    }
    fclose(fi);
    if (n == 0) { free(ents); return S2P_ERR_FORMAT; }
    qsort(ents, (size_t)n, sizeof(*ents), ent_cmp_name);

    snprintf(path, sizeof(path), "%s/codec.bin", model_dir);
    FILE* fb = fopen(path, "rb");
    if (!fb) {
        fprintf(stderr, "[s2pro] dac: cannot open %s\n", path);
        free(ents);
        return S2P_ERR_IO;
    }
    if (fseek(fb, 0, SEEK_END) != 0) { fclose(fb); free(ents); return S2P_ERR_IO; }
    long fsz = ftell(fb);
    if (fsz < max_end) {
        fprintf(stderr, "[s2pro] dac: codec.bin truncated (%ld < %lld)\n",
                fsz, (long long)max_end);
        fclose(fb); free(ents);
        return S2P_ERR_FORMAT;
    }
    rewind(fb);

    float* dev = NULL;
    cudaError_t ce = cudaMalloc((void**)&dev, (size_t)fsz);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[s2pro] dac: cudaMalloc(%ld) failed: %s\n", fsz,
                cudaGetErrorString(ce));
        fclose(fb); free(ents);
        return S2P_ERR_OOM;
    }

    /* staged host read -> device copy (64 MB chunks; the .bin is ~820 MB) */
    const size_t CHUNK = 64u << 20;
    void* stage = malloc(CHUNK);
    if (!stage) { cudaFree(dev); fclose(fb); free(ents); return S2P_ERR_OOM; }
    size_t done = 0, total = (size_t)fsz;
    while (done < total) {
        size_t want = total - done < CHUNK ? total - done : CHUNK;
        if (fread(stage, 1, want, fb) != want) {
            free(stage); cudaFree(dev); fclose(fb); free(ents);
            return S2P_ERR_IO;
        }
        ce = cudaMemcpy((char*)dev + done, stage, want, cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[s2pro] dac: H2D upload failed: %s\n",
                    cudaGetErrorString(ce));
            free(stage); cudaFree(dev); fclose(fb); free(ents);
            return S2P_ERR_CUDA;
        }
        done += want;
    }
    free(stage);
    fclose(fb);

    w->ents = ents;
    w->n_ents = n;
    w->base = dev;
    w->total_bytes = fsz;
    return S2P_OK;
}

void s2p_dacw_free(s2p_dacw* w) {
    if (!w) return;
    free(w->ents);
    if (w->base) cudaFree(w->base);
    memset(w, 0, sizeof(*w));
}

const s2p_dacw_ent* s2p_dacw_ent_find(const s2p_dacw* w, const char* name) {
    if (!w || !w->ents) return NULL;
    int lo = 0, hi = w->n_ents - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int c = strcmp(w->ents[mid].name, name);
        if (c == 0) return &w->ents[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return NULL;
}

const float* s2p_dacw_find(const s2p_dacw* w, const char* name) {
    const s2p_dacw_ent* e = s2p_dacw_ent_find(w, name);
    if (!e) return NULL;
    return (const float*)((const char*)w->base + e->off);
}
