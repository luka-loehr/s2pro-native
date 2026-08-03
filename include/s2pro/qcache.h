/* s2pro-native — prequantized-weight sidecar cache.
 *
 * Load-time quantization (per-channel INT8; group-wise INT4 with the MSE
 * clip search) costs BF16 staging of the full checkpoint plus the search
 * itself on every start. This cache serializes the FINAL quantized device
 * tensors once and lets later starts upload them directly — same bytes in
 * GPU memory, so the audio is bit-identical by construction (gated by the
 * deterministic-smoke MD5).
 *
 * Staleness guard: an FNV-1a fingerprint over the quantization config
 * (mode + every S2P_INT4* knob) and the model directory's *.safetensors
 * (name, size, mtime). Any mismatch ignores the cache and rewrites it.
 * Any read error falls back silently to the live quantization path.
 *
 * Env: S2P_QCACHE=0 disables. S2P_QCACHE_DIR overrides the cache location
 * (default: the model directory; needed when the model mount is
 * read-only). */
#pragma once

#include <cuda_runtime.h>
#include "s2pro/status.h"
#include "s2pro/gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_qcache s2p_qcache;

/* Open the cache for a model directory. *out == NULL (with S2P_OK) when
 * caching is disabled or the mode carries no quantized tensors. A valid
 * matching file puts the cache in read mode; otherwise it collects
 * entries for an atomic write in s2p_qcache_finish. */
s2p_status s2p_qcache_open(const char* model_dir, s2p_gemm_mode mode,
                           s2p_qcache** out);

int s2p_qcache_reading(const s2p_qcache* qc);

/* Read mode: 1 = hit (lin fully populated, quant flags set), 0 = miss or
 * any error (caller runs the live path). Write mode: always 0. */
int s2p_qcache_try_load(s2p_qcache* qc, const char* name, int in_features,
                        int out_features, s2p_linear* lin,
                        cudaStream_t stream);

/* Write mode: snapshot a prepared linear (syncs the stream, copies
 * device->host). Read mode or unquantized linear: no-op. */
s2p_status s2p_qcache_put_linear(s2p_qcache* qc, const char* name,
                                 const s2p_linear* lin, cudaStream_t stream);

/* Read mode: release the file buffer (uploads must be complete: sync the
 * stream first). Write mode: write the collected entries atomically. */
s2p_status s2p_qcache_finish(s2p_qcache* qc, cudaStream_t stream);

void s2p_qcache_free(s2p_qcache* qc);

#ifdef __cplusplus
}
#endif
