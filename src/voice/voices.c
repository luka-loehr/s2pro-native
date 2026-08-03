/* s2pro-native — voice registry (s2pro/voices.h).
 *
 * Scans <dir>/<name>.wav + <name>.txt pairs, encodes each reference once
 * through the DAC, and serves the cached VQ codes to the HTTP layer. The
 * registry is immutable after load; no locking needed.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>

#include "s2pro/voices.h"
#include "s2pro/wav.h"
#include "s2pro/config.h"

struct s2p_voices {
    s2p_voice* v;
    int        n;
};

static char* read_text_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
        fclose(f);
        return NULL;
    }
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    buf[len] = '\0';
    /* trim trailing whitespace/newlines */
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' ||
                       buf[len - 1] == ' ' || buf[len - 1] == '\t'))
        buf[--len] = '\0';
    return buf;
}


/* ---- encoded-code sidecar cache -----------------------------------------
 * DAC-encoding the registry dominates server start (~6 s of encode per
 * 60 s voice, ~3 min for a 33-voice roster) and its result is a pure
 * function of the wav. Cache it next to the wav as "<name>.codes":
 *   magic "S2PVC1\0\0" | u64 wav size | u64 wav mtime | i32 T | i32 pad
 *   | i32 codes[10*T]  (cb-major, exactly s2p_dac_encode's layout)
 * Any mismatch or read error re-encodes and rewrites. S2P_VOICE_CACHE=0
 * disables. */
#define VC_MAGIC "S2PVC1\0"

static int voice_cache_on(void) {
    const char* e = getenv("S2P_VOICE_CACHE");
    return !(e && e[0] == '0' && e[1] == '\0');
}

static int32_t* voice_cache_load(const char* path, const struct stat* ws,
                                 int* out_T) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[8];
    uint64_t sz = 0, mt = 0;
    int32_t T = 0, pad = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, VC_MAGIC, 8) != 0 ||
        fread(&sz, 8, 1, f) != 1 || fread(&mt, 8, 1, f) != 1 ||
        fread(&T, 4, 1, f) != 1 || fread(&pad, 4, 1, f) != 1 ||
        sz != (uint64_t)ws->st_size || mt != (uint64_t)ws->st_mtime ||
        T <= 0 || T > (1 << 22)) {
        fclose(f);
        return NULL;
    }
    size_t n = (size_t)S2P_NUM_CODEBOOKS * (size_t)T;
    int32_t* codes = (int32_t*)malloc(n * sizeof(int32_t));
    if (!codes) { fclose(f); return NULL; }
    if (fread(codes, sizeof(int32_t), n, f) != n) {
        free(codes);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_T = T;
    return codes;
}

static void voice_cache_store(const char* path, const struct stat* ws,
                              const int32_t* codes, int T) {
    char tmp[1160];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return;
    uint64_t sz = (uint64_t)ws->st_size, mt = (uint64_t)ws->st_mtime;
    int32_t t32 = T, pad = 0;
    size_t n = (size_t)S2P_NUM_CODEBOOKS * (size_t)T;
    int ok = fwrite(VC_MAGIC, 1, 8, f) == 8 && fwrite(&sz, 8, 1, f) == 1 &&
             fwrite(&mt, 8, 1, f) == 1 && fwrite(&t32, 4, 1, f) == 1 &&
             fwrite(&pad, 4, 1, f) == 1 &&
             fwrite(codes, sizeof(int32_t), n, f) == n;
    ok = (fclose(f) == 0) && ok;
    if (!ok || rename(tmp, path) != 0) remove(tmp);
}

s2p_status s2p_voices_load(const char* dir, s2p_dac* dac, s2p_voices** out) {
    if (!dir || !dac || !out) return S2P_ERR_INVALID;
    *out = NULL;
    s2p_voices* reg = (s2p_voices*)calloc(1, sizeof(*reg));
    if (!reg) return S2P_ERR_OOM;

    DIR* d = opendir(dir);
    if (!d) { /* no voices dir: empty registry, zero-shot serving still ok */
        fprintf(stderr, "[s2pro] voices: no directory %s (zero-shot only)\n",
                dir);
        *out = reg;
        return S2P_OK;
    }

    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        size_t nl = strlen(e->d_name);
        if (nl < 5 || strcmp(e->d_name + nl - 4, ".wav") != 0) continue;

        char name[256];
        if (nl - 4 >= sizeof(name)) continue;
        memcpy(name, e->d_name, nl - 4);
        name[nl - 4] = '\0';

        char wav_path[1024], txt_path[1024];
        snprintf(wav_path, sizeof(wav_path), "%s/%s.wav", dir, name);
        snprintf(txt_path, sizeof(txt_path), "%s/%s.txt", dir, name);

        char* transcript = read_text_file(txt_path);
        if (!transcript || transcript[0] == '\0') {
            fprintf(stderr,
                    "[s2pro] voices: skipping %s (missing/empty %s.txt)\n",
                    name, name);
            free(transcript);
            continue;
        }

        char cache_path[1100];
        snprintf(cache_path, sizeof(cache_path), "%s/%s.codes", dir, name);
        struct stat wst;
        int have_stat = (stat(wav_path, &wst) == 0);
        int32_t* codes = NULL;
        int      T = 0;
        double   dur = 0.0;
        s2p_status rc = S2P_OK;
        if (have_stat && voice_cache_on())
            codes = voice_cache_load(cache_path, &wst, &T);
        int cached = 0;
        if (codes) {
            cached = 1;
            dur = (double)T * S2P_FRAME_SAMPLES / S2P_SAMPLE_RATE;
            goto have_codes;
        }

        float*  pcm = NULL;
        int64_t n = 0;
        rc = s2p_wav_read_f32(wav_path, &pcm, &n);
        if (rc != S2P_OK) {
            fprintf(stderr,
                    "[s2pro] voices: skipping %s (wav read failed %d; "
                    "need 44100 Hz mono s16)\n", name, (int)rc);
            free(transcript);
            continue;
        }

        rc = s2p_dac_encode(dac, pcm, n, &codes, &T, 0);
        dur = (double)n / S2P_SAMPLE_RATE;
        free(pcm);
        if (rc != S2P_OK) {
            fprintf(stderr, "[s2pro] voices: skipping %s (encode failed %d)\n",
                    name, (int)rc);
            free(transcript);
            continue;
        }
        if (have_stat && voice_cache_on())
            voice_cache_store(cache_path, &wst, codes, T);

have_codes:
        ;
        s2p_voice* grown =
            (s2p_voice*)realloc(reg->v, (size_t)(reg->n + 1) * sizeof(*grown));
        if (!grown) {
            free(transcript);
            free(codes);
            continue;
        }
        reg->v = grown;
        s2p_voice* vo = &reg->v[reg->n++];
        vo->name = strdup(name);
        vo->transcript = transcript;
        vo->part.codes = codes;
        vo->part.T = T;
        vo->duration_s = dur;
        fprintf(stderr, "[s2pro] voices: %-20s %5.1f s -> %d frames%s\n",
                name, dur, T, cached ? " (cached)" : "");
    }
    closedir(d);
    *out = reg;
    return S2P_OK;
}

int s2p_voices_count(const s2p_voices* v) { return v ? v->n : 0; }

const s2p_voice* s2p_voices_at(const s2p_voices* v, int i) {
    if (!v || i < 0 || i >= v->n) return NULL;
    return &v->v[i];
}

const s2p_voice* s2p_voices_find(const s2p_voices* v, const char* name) {
    if (!v || !name) return NULL;
    for (int i = 0; i < v->n; i++)
        if (strcmp(v->v[i].name, name) == 0) return &v->v[i];
    return NULL;
}

void s2p_voices_free(s2p_voices* v) {
    if (!v) return;
    for (int i = 0; i < v->n; i++) {
        free((void*)v->v[i].name);
        free((void*)v->v[i].transcript);
        free((void*)v->v[i].part.codes);
    }
    free(v->v);
    free(v);
}
