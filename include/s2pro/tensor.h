/* s2pro-native — minimal tensor / device-buffer API. Contract header: frozen. */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "s2pro/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    S2P_DT_BF16 = 0,
    S2P_DT_F32  = 1,
    S2P_DT_F16  = 2,
    S2P_DT_I8   = 3,
    S2P_DT_I32  = 4,
    S2P_DT_I64  = 5,
    S2P_DT_U8   = 6,
} s2p_dtype;

size_t s2p_dtype_size(s2p_dtype dt);

typedef struct {
    void*     data;      /* device or host pointer, see on_device */
    s2p_dtype dtype;
    int       ndim;      /* <= 4 */
    int64_t   shape[4];
    size_t    bytes;
    int       on_device; /* 1 = CUDA device memory */
} s2p_tensor;

int64_t    s2p_tensor_numel(const s2p_tensor* t);
s2p_status s2p_tensor_device_alloc(s2p_tensor* t, s2p_dtype dt, int ndim,
                                   const int64_t* shape);
s2p_status s2p_tensor_host_alloc(s2p_tensor* t, s2p_dtype dt, int ndim,
                                 const int64_t* shape); /* pinned */
s2p_status s2p_tensor_upload(s2p_tensor* dev_dst, const void* host_src,
                             size_t bytes, cudaStream_t stream);
s2p_status s2p_tensor_download(void* host_dst, const s2p_tensor* dev_src,
                               size_t bytes, cudaStream_t stream);
void       s2p_tensor_free(s2p_tensor* t);

#ifdef __cplusplus
}
#endif
