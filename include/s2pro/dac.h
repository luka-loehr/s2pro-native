/* s2pro-native — modded-DAC / Firefly-GAN codec (vocoder + encoder).
 * Contract header: frozen.
 *
 * Decode: [10, T] RVQ codes -> 44.1 kHz mono float PCM, T*2048 samples.
 * Streaming: overlapping windows with crossfade per the reference
 * implementation (fishaudio_s2_pro/fish_speech/models/dac/).
 * Encode (voice cloning): 44.1 kHz PCM -> [10, T] codes.
 */
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_dac s2p_dac;

s2p_status s2p_dac_load(const char* model_dir, s2p_dac** out);
void       s2p_dac_free(s2p_dac* d);

/* Whole-buffer decode. codes codebook-major: codes[cb*T + t].
 * *pcm_out malloc'd host float buffer, caller frees. */
s2p_status s2p_dac_decode(s2p_dac* d, const int32_t* codes, int T,
                          float** pcm_out, int64_t* n_samples,
                          cudaStream_t stream);

/* Streaming decode with crossfade. Push one frame (10 codes); chunk may be
 * empty until enough context accumulated. Chunks are malloc'd host buffers. */
typedef struct s2p_dac_stream s2p_dac_stream;
s2p_status s2p_dac_stream_create(s2p_dac* d, s2p_dac_stream** out);
s2p_status s2p_dac_stream_push(s2p_dac_stream* s,
                               const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                               float** pcm_chunk, int64_t* n_out,
                               cudaStream_t stream);
s2p_status s2p_dac_stream_finish(s2p_dac_stream* s, float** pcm_chunk,
                                 int64_t* n_out, cudaStream_t stream);
void       s2p_dac_stream_destroy(s2p_dac_stream* s);

/* Encoder (voice cloning): mono 44.1 kHz PCM in [-1,1] -> codes [10, T],
 * malloc'd, codebook-major. */
s2p_status s2p_dac_encode(s2p_dac* d, const float* pcm, int64_t n_samples,
                          int32_t** codes_out, int* T_out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
