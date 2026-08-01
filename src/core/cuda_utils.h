/* s2pro-native — private core header: CUDA device probe + stream/event
 * convenience. Implemented in src/core/cuda_utils.cu. Not part of the frozen
 * contract; other modules may include it via a relative path if useful.
 */
#pragma once

#include <cuda_runtime.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bind the process to `device`, force context creation, optionally print a
 * one-line device summary (name, sm arch, SMs, memory, driver/runtime). */
s2p_status s2p_cuda_init(int device, int verbose);

/* Print the one-line device summary to stderr. */
s2p_status s2p_cuda_device_print(int device);

/* Free/total device memory in bytes; either out pointer may be NULL. */
s2p_status s2p_cuda_mem_info(size_t* free_bytes, size_t* total_bytes);

/* Non-blocking stream / timing-enabled event helpers. Destroy fns are
 * NULL-safe and swallow errors (log-only) so they can run in teardown. */
s2p_status s2p_cuda_stream_create(cudaStream_t* out);
void       s2p_cuda_stream_destroy(cudaStream_t s);
s2p_status s2p_cuda_stream_sync(cudaStream_t s);
s2p_status s2p_cuda_event_create(cudaEvent_t* out);
void       s2p_cuda_event_destroy(cudaEvent_t e);
s2p_status s2p_cuda_event_record(cudaEvent_t e, cudaStream_t s);
/* Synchronizes `end_ev`, then reports end_ev - beg_ev in milliseconds. */
s2p_status s2p_cuda_event_elapsed_ms(cudaEvent_t beg_ev, cudaEvent_t end_ev,
                                     float* out_ms);

#ifdef __cplusplus
}
#endif
