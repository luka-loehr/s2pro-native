/* s2pro-native — voice registry (s2pro/voices.h).
 *
 * Scans <dir>/<name>.wav + <name>.txt pairs, encodes each reference once
 * through the DAC, and serves the cached VQ codes to the HTTP layer. The
 * registry is immutable after load; no locking needed.
 */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
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

        float*  pcm = NULL;
        int64_t n = 0;
        s2p_status rc = s2p_wav_read_f32(wav_path, &pcm, &n);
        if (rc != S2P_OK) {
            fprintf(stderr,
                    "[s2pro] voices: skipping %s (wav read failed %d; "
                    "need 44100 Hz mono s16)\n", name, (int)rc);
            free(transcript);
            continue;
        }

        int32_t* codes = NULL;
        int      T = 0;
        rc = s2p_dac_encode(dac, pcm, n, &codes, &T, 0);
        double dur = (double)n / S2P_SAMPLE_RATE;
        free(pcm);
        if (rc != S2P_OK) {
            fprintf(stderr, "[s2pro] voices: skipping %s (encode failed %d)\n",
                    name, (int)rc);
            free(transcript);
            continue;
        }

        s2p_voice* grown =
            (s2p_voice*)realloc(reg->v, (size_t)(reg->n + 1) * sizeof(*grown));
        if (!grown) {
            free(transcript);
            free(codes);
            continue;
        }
        reg->v = grown;
        /* S2P_REF_MAX_FRAMES caps the reference block that enters the
         * prompt (and therefore every session's KV prefix). Each frame is
         * ~46.4 ms of reference audio and ~78 KB of INT8 KV that every
         * decode step of every stream re-reads, so this is the direct
         * knob on concurrency headroom. Keeping the TAIL preserves the
         * reference's natural ending; the transcript stays whole (it is
         * prompt text, not KV). 0 or unset = no cap. */
        {
            static int cap = -1;
            if (cap < 0) {
                const char* e = getenv("S2P_REF_MAX_FRAMES");
                cap = e ? atoi(e) : 0;
                if (cap < 0) cap = 0;
            }
            if (cap > 0 && T > cap) {
                const int drop = T - cap;
                for (int q = 0; q < S2P_NUM_CODEBOOKS; q++)
                    memmove(codes + (size_t)q * cap,
                            codes + (size_t)q * T + drop,
                            (size_t)cap * sizeof(int32_t));
                T = cap;
            }
        }
        s2p_voice* vo = &reg->v[reg->n++];
        vo->name = strdup(name);
        vo->transcript = transcript;
        vo->part.codes = codes;
        vo->part.T = T;
        vo->duration_s = dur;
        fprintf(stderr, "[s2pro] voices: %-20s %5.1f s -> %d frames\n", name,
                dur, T);
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
