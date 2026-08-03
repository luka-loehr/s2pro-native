/* s2pro-native — fast-AR audio decoder (4-layer, per-frame residual codebooks).
 * Private module header, pinned cross-module API (consumed by src/slowar).
 *
 * Implements the FishQwen3AudioDecoder inference path per docs/PORTING.md §6
 * and the reference fish_speech/models/text2semantic/audio_decoder.py:
 *   - 4 transformer layers, dim 2560, 32 q-heads / 8 kv-heads, head_dim 128,
 *     NO qk-norm, interleaved bf16-truncated RoPE (base 1e6), SwiGLU 9728.
 *   - UNTIED output head Linear(2560, 4096); project_in == Identity
 *     (text_dim == dim).
 *   - Two embedding tables: codebook_embeddings [40960, 2560] (slow-AR
 *     reference-VQ injection, offset cb*4096) and embeddings [4096, 2560]
 *     (fast-AR step inputs).
 *   - Tiny KV cache: depth 11 (num_codebooks + 1), zeroed EVERY frame.
 *
 * Pointer conventions (deliberate, see comments per function):
 *   device: rows (vq_embed_add), hidden (decode_frame_batch)
 *   host:   codes, sem_ids, out_codes — staged through internal pinned+device
 *           scratch; decode_frame_batch synchronizes the stream before
 *           returning so out_codes is valid on return.
 * Not thread-safe: one s2pfa per stream, calls serialized by the caller.
 */
#pragma once

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/qcache.h"
#include "s2pro/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2pfa s2pfa;

/* Load all audio_decoder.* tensors from the open checkpoint and allocate
 * decode workspace for up to S2P_MAX_SESSIONS batch rows. mode selects the
 * GEMM path for every linear (FP8 weights are prepared once here).
 * Synchronizes stream before returning (uploads read the st mmap). */
s2p_status s2pfa_load(s2pfa** out, s2p_st* st, s2p_qcache* qc,
                      s2p_gemm_mode mode, cudaStream_t stream);
void s2pfa_free(s2pfa* f);

/* codes [10*T] cb-major (HOST; codes[cb*T + t]); adds the summed 10-codebook
 * embedding of frame t onto rows[t*2560] (device, contiguous [T,2560] bf16).
 * Sum is f32-accumulated over the 10 rows, rounded to bf16, then added to the
 * existing row value in one bf16 add — bit-matching the reference
 * embed_text_dim sum. The base text embedding and the 1/sqrt(11) scale are
 * the CALLER's responsibility (PORTING §4). */
s2p_status s2pfa_vq_embed_add(s2pfa* f, const int32_t* codes, int T,
                              __nv_bfloat16* rows, cudaStream_t stream);
/* Same, but codes already on DEVICE: pure kernel launch, no staging or lazy
 * allocation — safe inside CUDA-graph capture. */
s2p_status s2pfa_vq_embed_add_dev(s2pfa* f, const int32_t* codes_dev, int T,
                                  __nv_bfloat16* rows, cudaStream_t stream);

/* One frame for B sessions: hidden [B,2560] device bf16 = the slow-AR
 * FINAL-NORMED hidden state (post backbone norm — PORTING pitfall 5);
 * sem_ids [B] HOST int32 = the sampled semantic code in 0..4095 (caller has
 * already mapped from vocab id and forced 0 on EOS). Runs the reference loop:
 * zero KV caches, prime pos 0 on hidden (logits discarded), then 9 greedy
 * argmax steps seeded by embeddings(sem_id). Writes out_codes[b*10+1..9]
 * (HOST) with the residual codes; leaves out_codes[b*10+0] untouched.
 * Synchronizes stream before returning. */
s2p_status s2pfa_decode_frame_batch(s2pfa* f, const __nv_bfloat16* hidden,
                                    const int32_t* sem_ids, int B,
                                    int32_t* out_codes, cudaStream_t stream);

/* Device-resident variant: sem_dev [B] DEVICE int32, NO sync, NO D2H. On
 * return *stage_dev points at the internal residual stage [9][B] (device,
 * valid until the next decode call); the caller packs/downloads it. The
 * whole call is a fixed kernel sequence — CUDA-graph capturable. */
s2p_status s2pfa_decode_frame_batch_dev(s2pfa* f, const __nv_bfloat16* hidden,
                                        const int32_t* sem_dev, int B,
                                        const int32_t** stage_dev,
                                        cudaStream_t stream);

#ifdef __cplusplus
}
#endif
