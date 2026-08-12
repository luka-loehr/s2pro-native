#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "s2pro/config.h"
#include "s2pro/sha256.h"
#include "voice_cache.h"

/* V2 binds the payload to WAV contents and the exact producer fingerprint.
 * V1 files intentionally miss because their eight-byte magic differs. */
#define VC_MAGIC "S2PVC2\0"

static int write_u64(FILE* f, uint64_t x) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(x >> (8 * i));
    return fwrite(b, 1, sizeof(b), f) == sizeof(b);
}

static int read_u64(FILE* f, uint64_t* x) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), f) != sizeof(b)) return 0;
    *x = 0;
    for (int i = 0; i < 8; i++) *x |= (uint64_t)b[i] << (8 * i);
    return 1;
}

static int write_i32(FILE* f, int32_t x) {
    uint32_t u = (uint32_t)x;
    uint8_t b[4] = {(uint8_t)u, (uint8_t)(u >> 8),
                    (uint8_t)(u >> 16), (uint8_t)(u >> 24)};
    return fwrite(b, 1, sizeof(b), f) == sizeof(b);
}

static int read_i32(FILE* f, int32_t* x) {
    uint8_t b[4];
    if (fread(b, 1, sizeof(b), f) != sizeof(b)) return 0;
    *x = (int32_t)((uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                   ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24));
    return 1;
}

static int wav_identity(const char* path, uint64_t* size, uint8_t hash[32]) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < 0) return 0;
    *size = (uint64_t)st.st_size;
    return s2p_sha256_file(path, hash);
}

int s2p_voice_cache_load(const char* cache_path, const char* wav_path,
                         const uint8_t producer[32], int32_t** codes_out,
                         int* frames_out) {
    if (!cache_path || !wav_path || !producer || !codes_out || !frames_out)
        return 0;
    *codes_out = NULL;
    *frames_out = 0;
    uint64_t wav_size = 0;
    uint8_t wav_hash[32];
    if (!wav_identity(wav_path, &wav_size, wav_hash)) return 0;

    FILE* f = fopen(cache_path, "rb");
    if (!f) return 0;
    char magic[8];
    uint64_t stored_size = 0;
    uint8_t stored_wav[32], stored_producer[32];
    int32_t frames = 0, codebooks = 0;
    int valid = fread(magic, 1, 8, f) == 8 &&
                memcmp(magic, VC_MAGIC, 8) == 0 &&
                read_u64(f, &stored_size) &&
                fread(stored_wav, 1, 32, f) == 32 &&
                fread(stored_producer, 1, 32, f) == 32 &&
                read_i32(f, &frames) && read_i32(f, &codebooks) &&
                stored_size == wav_size &&
                memcmp(stored_wav, wav_hash, 32) == 0 &&
                memcmp(stored_producer, producer, 32) == 0 &&
                frames > 0 && frames <= (1 << 22) &&
                codebooks == S2P_NUM_CODEBOOKS;
    if (!valid) {
        fclose(f);
        return 0;
    }
    size_t count = (size_t)codebooks * (size_t)frames;
    int32_t* codes = (int32_t*)malloc(count * sizeof(*codes));
    if (!codes) { fclose(f); return 0; }
    for (size_t i = 0; i < count; i++) {
        if (!read_i32(f, &codes[i])) {
            free(codes);
            fclose(f);
            return 0;
        }
    }
    int trailing = fgetc(f);
    if (trailing != EOF || ferror(f) || fclose(f) != 0) {
        free(codes);
        return 0;
    }
    *codes_out = codes;
    *frames_out = frames;
    return 1;
}

int s2p_voice_cache_store(const char* cache_path, const char* wav_path,
                          const uint8_t producer[32], const int32_t* codes,
                          int frames) {
    if (!cache_path || !wav_path || !producer || !codes || frames <= 0)
        return 0;
    uint64_t wav_size = 0;
    uint8_t wav_hash[32];
    if (!wav_identity(wav_path, &wav_size, wav_hash)) return 0;
    char tmp[1160];
    int n = snprintf(tmp, sizeof(tmp), "%s.tmp", cache_path);
    if (n < 0 || (size_t)n >= sizeof(tmp)) return 0;
    FILE* f = fopen(tmp, "wb");
    if (!f) return 0;
    size_t count = (size_t)S2P_NUM_CODEBOOKS * (size_t)frames;
    int ok = fwrite(VC_MAGIC, 1, 8, f) == 8 && write_u64(f, wav_size) &&
             fwrite(wav_hash, 1, 32, f) == 32 &&
             fwrite(producer, 1, 32, f) == 32 && write_i32(f, frames) &&
             write_i32(f, S2P_NUM_CODEBOOKS);
    for (size_t i = 0; ok && i < count; i++) ok = write_i32(f, codes[i]);
    ok = fclose(f) == 0 && ok;
    if (ok) ok = rename(tmp, cache_path) == 0;
    if (!ok) remove(tmp);
    return ok;
}
