#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "s2pro/sha256.h"
#include "../src/voice/voice_cache.h"

static int failures = 0;
#define T(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
} while (0)

static void hex(const uint8_t in[32], char out[65]) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2 * i] = digits[in[i] >> 4];
        out[2 * i + 1] = digits[in[i] & 15];
    }
    out[64] = '\0';
}

static int write_bytes(const char* path, const void* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(data, 1, n, f) == n && fclose(f) == 0;
    return ok;
}

int main(void) {
    uint8_t digest[32];
    char digest_hex[65];
    s2p_sha256 hash;
    s2p_sha256_init(&hash);
    s2p_sha256_update(&hash, "abc", 3);
    s2p_sha256_final(&hash, digest);
    hex(digest, digest_hex);
    T(strcmp(digest_hex,
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0,
      "SHA-256 known vector");

    char dir[] = "/tmp/s2p-voice-cache-test-XXXXXX";
    T(mkdtemp(dir) != NULL, "temporary directory");
    char wav[512], cache[512];
    snprintf(wav, sizeof(wav), "%s/ref.wav", dir);
    snprintf(cache, sizeof(cache), "%s/ref.codes", dir);
    uint8_t wav_a[64], wav_b[64];
    for (int i = 0; i < 64; i++) {
        wav_a[i] = (uint8_t)i;
        wav_b[i] = (uint8_t)(63 - i);
    }
    T(write_bytes(wav, wav_a, sizeof(wav_a)), "write WAV fixture");

    uint8_t producer_a[32] = {0}, producer_b[32] = {0};
    producer_a[0] = 0x11;
    producer_b[0] = 0x22;
    enum { FRAMES = 3, COUNT = 10 * FRAMES };
    int32_t codes[COUNT];
    for (int i = 0; i < COUNT; i++) codes[i] = i * 17 - 9;
    T(s2p_voice_cache_store(cache, wav, producer_a, codes, FRAMES),
      "store V2 cache");

    int32_t* loaded = NULL;
    int frames = 0;
    T(s2p_voice_cache_load(cache, wav, producer_a, &loaded, &frames),
      "matching cache loads");
    T(frames == FRAMES, "frame count round trips");
    T(loaded && memcmp(loaded, codes, sizeof(codes)) == 0,
      "codes round trip");
    free(loaded);

    loaded = NULL;
    T(!s2p_voice_cache_load(cache, wav, producer_b, &loaded, &frames),
      "producer change invalidates cache");
    T(loaded == NULL, "producer mismatch returns no payload");

    struct stat before;
    T(stat(wav, &before) == 0, "stat WAV fixture");
    T(write_bytes(wav, wav_b, sizeof(wav_b)), "replace WAV same size");
#if defined(__APPLE__)
    struct timespec times[2] = {before.st_atimespec, before.st_mtimespec};
#else
    struct timespec times[2] = {before.st_atim, before.st_mtim};
#endif
    T(utimensat(AT_FDCWD, wav, times, 0) == 0,
      "restore WAV mtime after content change");
    loaded = NULL;
    T(!s2p_voice_cache_load(cache, wav, producer_a, &loaded, &frames),
      "same-size same-mtime WAV content change invalidates cache");

    const uint8_t v1_header[8] = {'S','2','P','V','C','1',0,0};
    T(write_bytes(cache, v1_header, sizeof(v1_header)), "write V1 header");
    T(!s2p_voice_cache_load(cache, wav, producer_a, &loaded, &frames),
      "legacy V1 cache misses safely");

    unlink(cache);
    unlink(wav);
    rmdir(dir);
    if (failures) return 1;
    printf("selftest_voice_cache: OK\n");
    return 0;
}
