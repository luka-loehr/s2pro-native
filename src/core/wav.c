/* s2pro-native — RIFF/WAVE writer, mono S16LE (s2pro/wav.h). */
#include "s2pro/wav.h"

#include <stdio.h>
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
