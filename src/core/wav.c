/* s2pro-native — RIFF/WAVE writer, mono S16LE (s2pro/wav.h). */
#include "s2pro/wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u32(uint8_t* p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static void put_u16(uint8_t* p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}

size_t s2p_wav_header(uint8_t out[44], uint32_t data_bytes_or_max,
                      int sample_rate)
{
    uint32_t data = data_bytes_or_max;
    /* RIFF chunk size = data + 36; saturate for the streaming convention */
    uint32_t riff = data > 0xFFFFFFFFu - 36u ? 0xFFFFFFFFu : data + 36u;
    uint32_t rate = (uint32_t)sample_rate;

    memcpy(out, "RIFF", 4);
    put_u32(out + 4, riff);
    memcpy(out + 8, "WAVE", 4);
    memcpy(out + 12, "fmt ", 4);
    put_u32(out + 16, 16);        /* fmt chunk size */
    put_u16(out + 20, 1);         /* PCM */
    put_u16(out + 22, 1);         /* mono */
    put_u32(out + 24, rate);
    put_u32(out + 28, rate * 2u); /* byte rate = rate * block align */
    put_u16(out + 32, 2);         /* block align: 1 ch x 16 bit */
    put_u16(out + 34, 16);        /* bits per sample */
    memcpy(out + 36, "data", 4);
    put_u32(out + 40, data);
    return 44;
}

s2p_status s2p_wav_write_file(const char* path, const int16_t* pcm, int64_t n,
                              int sample_rate)
{
    uint8_t hdr[44];
    FILE* f;
    s2p_status st = S2P_OK;

    if (!path || n < 0 || (!pcm && n > 0) || sample_rate <= 0)
        return S2P_ERR_INVALID;
    if ((uint64_t)n * 2u > 0xFFFFFFFFull - 36ull)
        return S2P_ERR_INVALID; /* exceeds the RIFF u32 size field */

    s2p_wav_header(hdr, (uint32_t)((uint64_t)n * 2u), sample_rate);
    f = fopen(path, "wb");
    if (!f) return S2P_ERR_IO;
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) st = S2P_ERR_IO;
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    if (st == S2P_OK) {
        int64_t i;
        for (i = 0; i < n; i++) {
            uint8_t b[2];
            b[0] = (uint8_t)((uint16_t)pcm[i] & 0xFF);
            b[1] = (uint8_t)(((uint16_t)pcm[i] >> 8) & 0xFF);
            if (fwrite(b, 1, 2, f) != 2) {
                st = S2P_ERR_IO;
                break;
            }
        }
    }
#else /* little-endian host (aarch64/x86_64): samples are already S16LE */
    if (st == S2P_OK && n > 0 &&
        fwrite(pcm, sizeof(int16_t), (size_t)n, f) != (size_t)n)
        st = S2P_ERR_IO;
#endif
    if (fclose(f) != 0 && st == S2P_OK) st = S2P_ERR_IO;
    return st;
}

void s2p_f32_to_s16(const float* in, int16_t* out, int64_t n)
{
    int64_t i;
    for (i = 0; i < n; i++) {
        float v = in[i];
        float s;
        if (v != v) v = 0.0f; /* NaN */
        if (v > 1.0f) v = 1.0f;
        else if (v < -1.0f) v = -1.0f;
        s = v * 32767.0f;
        /* round half away from zero, no libm */
        out[i] = (int16_t)(int32_t)(s >= 0.0f ? s + 0.5f : s - 0.5f);
    }
}

/* ---- readers (voices/cloning subsystem) --------------------------------- */

static uint32_t get_u32(const uint8_t* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint16_t get_u16(const uint8_t* p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

s2p_status s2p_wav_parse_f32(const void* vbuf, size_t len, float** pcm,
                             int64_t* n_samples)
{
    if (!vbuf || !pcm || !n_samples) return S2P_ERR_INVALID;
    *pcm = NULL;
    *n_samples = 0;
    const uint8_t* buf = (const uint8_t*)vbuf;
    if (len < 44 || memcmp(buf, "RIFF", 4) != 0 ||
        memcmp(buf + 8, "WAVE", 4) != 0)
        return S2P_ERR_FORMAT;

    uint32_t rate = 0;
    uint16_t ch = 0, bits = 0, fmt_tag = 0;
    size_t off = 12;
    while (off + 8 <= len) {
        uint32_t sz = get_u32(buf + off + 4);
        const uint8_t* ck = buf + off;
        if (memcmp(ck, "fmt ", 4) == 0) {
            if (sz < 16 || off + 8 + 16 > len) return S2P_ERR_FORMAT;
            fmt_tag = get_u16(ck + 8);
            ch = get_u16(ck + 10);
            rate = get_u32(ck + 12);
            bits = get_u16(ck + 22);
        } else if (memcmp(ck, "data", 4) == 0) {
            if (fmt_tag != 1 /* PCM */ || ch != 1 || bits != 16 ||
                rate != 44100)
                return S2P_ERR_UNSUPPORTED;
            size_t avail = len - (off + 8);
            size_t db = sz < avail ? sz : avail;
            int64_t n = (int64_t)(db / 2);
            if (n <= 0) return S2P_ERR_FORMAT;
            float* out = (float*)malloc((size_t)n * sizeof(float));
            if (!out) return S2P_ERR_OOM;
            const uint8_t* d = ck + 8;
            for (int64_t i = 0; i < n; i++) {
                int16_t s = (int16_t)((uint16_t)d[2 * i] |
                                      ((uint16_t)d[2 * i + 1] << 8));
                out[i] = (float)s / 32768.0f;
            }
            *pcm = out;
            *n_samples = n;
            return S2P_OK;
        }
        off += 8 + sz + (sz & 1);
    }
    return S2P_ERR_FORMAT;
}

s2p_status s2p_wav_read_f32(const char* path, float** pcm, int64_t* n_samples)
{
    if (!path) return S2P_ERR_INVALID;
    FILE* f = fopen(path, "rb");
    if (!f) return S2P_ERR_IO;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        fclose(f);
        return S2P_ERR_FORMAT;
    }
    void* buf = malloc((size_t)len);
    if (!buf) {
        fclose(f);
        return S2P_ERR_OOM;
    }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (rd != (size_t)len) {
        free(buf);
        return S2P_ERR_IO;
    }
    s2p_status rc = s2p_wav_parse_f32(buf, (size_t)len, pcm, n_samples);
    free(buf);
    return rc;
}
