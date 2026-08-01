/* s2pro-native — tensor / device-buffer implementation (s2pro/tensor.h).
 *
 * Host allocations are always pinned (cudaMallocHost) so uploads/downloads
 * on a stream are truly async; device allocations are plain cudaMalloc.
 * s2p_tensor_free dispatches on on_device — do not mix in malloc'd buffers.
 */
#include "s2pro/tensor.h"

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

size_t s2p_dtype_size(s2p_dtype dt)
{
    switch (dt) {
    case S2P_DT_BF16: return 2;
    case S2P_DT_F32:  return 4;
    case S2P_DT_F16:  return 2;
    case S2P_DT_I8:   return 1;
    case S2P_DT_I32:  return 4;
    case S2P_DT_I64:  return 8;
    case S2P_DT_U8:   return 1;
    }
    return 0;
}

int64_t s2p_tensor_numel(const s2p_tensor* t)
{
    int64_t n = 1;
    int i;
    if (!t || t->ndim < 0 || t->ndim > 4) return 0;
    for (i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* Validate dt/ndim/shape, fill everything but data/on_device. */
static s2p_status tensor_init(s2p_tensor* t, s2p_dtype dt, int ndim,
                              const int64_t* shape)
{
    size_t esz;
    int64_t numel = 1;
    int i;
    if (!t || !shape || ndim < 1 || ndim > 4) return S2P_ERR_INVALID;
    esz = s2p_dtype_size(dt);
    if (esz == 0) return S2P_ERR_INVALID;
    for (i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return S2P_ERR_INVALID;
        if (numel > INT64_MAX / shape[i]) return S2P_ERR_INVALID;
        numel *= shape[i];
    }
    if ((uint64_t)numel > SIZE_MAX / esz) return S2P_ERR_INVALID;
    memset(t, 0, sizeof(*t));
    t->dtype = dt;
    t->ndim = ndim;
    for (i = 0; i < 4; i++) t->shape[i] = i < ndim ? shape[i] : 1;
    t->bytes = (size_t)numel * esz;
    return S2P_OK;
}

s2p_status s2p_tensor_device_alloc(s2p_tensor* t, s2p_dtype dt, int ndim,
                                   const int64_t* shape)
{
    cudaError_t e;
    S2P_TRY(tensor_init(t, dt, ndim, shape));
    e = cudaMalloc(&t->data, t->bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[s2pro] cudaMalloc(%zu) failed: %s\n", t->bytes,
                cudaGetErrorString(e));
        t->data = NULL;
        return e == cudaErrorMemoryAllocation ? S2P_ERR_OOM : S2P_ERR_CUDA;
    }
    t->on_device = 1;
    return S2P_OK;
}

s2p_status s2p_tensor_host_alloc(s2p_tensor* t, s2p_dtype dt, int ndim,
                                 const int64_t* shape)
{
    cudaError_t e;
    S2P_TRY(tensor_init(t, dt, ndim, shape));
    e = cudaMallocHost(&t->data, t->bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[s2pro] cudaMallocHost(%zu) failed: %s\n", t->bytes,
                cudaGetErrorString(e));
        t->data = NULL;
        return e == cudaErrorMemoryAllocation ? S2P_ERR_OOM : S2P_ERR_CUDA;
    }
    t->on_device = 0;
    return S2P_OK;
}

s2p_status s2p_tensor_upload(s2p_tensor* dev_dst, const void* host_src,
                             size_t bytes, cudaStream_t stream)
{
    if (!dev_dst || !dev_dst->data || !dev_dst->on_device)
        return S2P_ERR_INVALID;
    if (bytes > dev_dst->bytes || (!host_src && bytes)) return S2P_ERR_INVALID;
    if (bytes == 0) return S2P_OK;
    S2P_CUDA_TRY(cudaMemcpyAsync(dev_dst->data, host_src, bytes,
                                 cudaMemcpyHostToDevice, stream));
    return S2P_OK;
}

s2p_status s2p_tensor_download(void* host_dst, const s2p_tensor* dev_src,
                               size_t bytes, cudaStream_t stream)
{
    if (!dev_src || !dev_src->data || !dev_src->on_device)
        return S2P_ERR_INVALID;
    if (bytes > dev_src->bytes || (!host_dst && bytes)) return S2P_ERR_INVALID;
    if (bytes == 0) return S2P_OK;
    S2P_CUDA_TRY(cudaMemcpyAsync(host_dst, dev_src->data, bytes,
                                 cudaMemcpyDeviceToHost, stream));
    return S2P_OK;
}

void s2p_tensor_free(s2p_tensor* t)
{
    if (!t) return;
    if (t->data) {
        cudaError_t e = t->on_device ? cudaFree(t->data)
                                     : cudaFreeHost(t->data);
        if (e != cudaSuccess)
            fprintf(stderr, "[s2pro] tensor free failed: %s\n",
                    cudaGetErrorString(e));
    }
    memset(t, 0, sizeof(*t));
}
