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
        e->h_off = e->f_off = -1;
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

    /* ---- FP16 weight conversion (S2P_DAC_F32=1 keeps everything f32) ----
     * Big conv/matmul weights convert to a half blob; the small keepers
     * (biases, Snake alphas, norms, layer scales, codebooks, the mono
     * output conv) copy into a compact f32 arena; the original blob is
     * freed. Net: roughly half the vocoder weight bytes and traffic. */
    {
        const char* keep = getenv("S2P_DAC_F32");
        if (keep && keep[0] == '1' && keep[1] == '\0') return S2P_OK;
        int64_t hbytes = 0, fbytes = 0;
        for (int i = 0; i < n; i++) {
            s2p_dacw_ent* e = &ents[i];
            size_t len = strlen(e->name);
            int wclass = len > 7 &&
                         strcmp(e->name + len - 7, ".weight") == 0 &&
                         e->ndim >= 2 &&
                         strstr(e->name, "codebook") == NULL &&
                         /* rvq.cu reads out_proj directly as f32 (the
                          * encode-side conv1d use forces f32 explicitly) */
                         strstr(e->name, "out_proj") == NULL;
            if (wclass) {
                e->h_off = hbytes;
                hbytes += (e->nbytes / 2 + 255) & ~(int64_t)255;
            } else {
                e->f_off = fbytes;
                fbytes += (e->nbytes + 255) & ~(int64_t)255;
            }
        }
        void* devh = NULL;
        float* devf = NULL;
        if (cudaMalloc(&devh, (size_t)hbytes) != cudaSuccess ||
            cudaMalloc((void**)&devf, (size_t)fbytes) != cudaSuccess) {
            if (devh) cudaFree(devh);
            fprintf(stderr, "[s2pro] dac: f16 arena alloc failed, staying "
                            "f32\n");
            for (int i = 0; i < n; i++)
                ents[i].h_off = ents[i].f_off = -1;
            return S2P_OK;
        }
        for (int i = 0; i < n; i++) {
            s2p_dacw_ent* e = &ents[i];
            const char* src = (const char*)dev + e->off;
            cudaError_t c2;
            if (e->h_off >= 0)
                c2 = s2pdk_f32_to_f16((const float*)src,
                                      (char*)devh + e->h_off, e->nbytes / 4,
                                      0);
            else
                c2 = cudaMemcpyAsync((char*)devf + e->f_off, src,
                                     (size_t)e->nbytes,
                                     cudaMemcpyDeviceToDevice, 0);
            if (c2 != cudaSuccess) {
                fprintf(stderr, "[s2pro] dac: f16 convert failed (%s), "
                                "staying f32\n", cudaGetErrorString(c2));
                cudaFree(devh);
                cudaFree(devf);
                for (int j = 0; j < n; j++)
                    ents[j].h_off = ents[j].f_off = -1;
                return S2P_OK;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(devh);
            cudaFree(devf);
            for (int j = 0; j < n; j++)
                ents[j].h_off = ents[j].f_off = -1;
            return S2P_OK;
        }
        cudaFree(dev);
        w->base = NULL;
        w->base_h = devh;
        w->base_f = devf;
        w->f16 = 1;
        s2pdk_weights_f16(1);
        fprintf(stderr, "[s2pro] dac: weights f16 (%lld MB, keepers %lld "
                        "KB f32)\n", (long long)(hbytes >> 20),
                (long long)(fbytes >> 10));
    }
    return S2P_OK;
}

void s2p_dacw_free(s2p_dacw* w) {
    if (!w) return;
    free(w->ents);
    if (w->base) cudaFree(w->base);
    if (w->base_h) cudaFree(w->base_h);
    if (w->base_f) cudaFree(w->base_f);
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
    if (w->f16) {
        /* w-class tensors live in the half blob (consumers pass them
         * through the void* launcher params); keepers are real f32 */
        if (e->h_off >= 0)
            return (const float*)((const char*)w->base_h + e->h_off);
        return (const float*)((const char*)w->base_f + e->f_off);
    }
    return (const float*)((const char*)w->base + e->off);
}
