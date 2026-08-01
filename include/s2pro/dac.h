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

/* Streaming decode. Default path is the bit-exact incremental engine (one
 * frame in, its 2048 samples out); S2P_STREAM_REFERENCE=1 selects the ported
 * window/crossfade scheme. Chunks are malloc'd host buffers.
 *
 * Pipelined use (the scheduler's pattern): push_async ENQUEUES the frame's
 * decode on `stream` and returns immediately; collect SYNCS the stream and
 * returns the previously pushed frame's chunk. Exactly one frame may be in
 * flight per s2p_dac_stream. The plain push is push_async + collect. */
typedef struct s2p_dac_stream s2p_dac_stream;
s2p_status s2p_dac_stream_create(s2p_dac* d, s2p_dac_stream** out);
s2p_status s2p_dac_stream_push(s2p_dac_stream* s,
                               const int32_t frame_codes[S2P_NUM_CODEBOOKS],
                               float** pcm_chunk, int64_t* n_out,
                               cudaStream_t stream);
s2p_status s2p_dac_stream_push_async(s2p_dac_stream* s,
                                     const int32_t
                                         frame_codes[S2P_NUM_CODEBOOKS],
                                     cudaStream_t stream);
s2p_status s2p_dac_stream_collect(s2p_dac_stream* s, float** pcm_chunk,
                                  int64_t* n_out, cudaStream_t stream);
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
