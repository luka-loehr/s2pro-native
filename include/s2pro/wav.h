/* s2pro-native — RIFF/WAVE writer, 16-bit PCM mono. Contract header: frozen.
 *
 * Implemented by core (src/core/wav.c); consumed by serve and tests. The
 * standalone header helper exists for HTTP streaming: emit the 44-byte header
 * with data_bytes_or_max = 0xFFFFFFFF (sizes saturate — the "unknown length"
 * streaming convention), then append raw S16LE frames.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Write a complete mono S16LE WAV file (44-byte header + n samples). */
s2p_status s2p_wav_write_file(const char* path, const int16_t* pcm, int64_t n,
                              int sample_rate);

/* Fill out[44] with a RIFF/fmt/data header for mono S16LE PCM. Pass the
 * final payload size in bytes, or 0xFFFFFFFF for unbounded streaming (RIFF
 * and data sizes saturate). Returns 44. */
size_t s2p_wav_header(uint8_t out[44], uint32_t data_bytes_or_max,
                      int sample_rate);

/* Clamp [-1,1] (NaN -> 0), scale by 32767, round half away from zero. */
void s2p_f32_to_s16(const float* in, int16_t* out, int64_t n);

#ifdef __cplusplus
}
#endif
