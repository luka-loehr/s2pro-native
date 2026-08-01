/* s2pro-native — CUDA device probe + stream/event convenience (core). */
#include "cuda_utils.h"

#include <stdio.h>

extern "C" s2p_status s2p_cuda_device_print(int device)
{
    cudaDeviceProp prop;
    int drv = 0, rt = 0;
    size_t free_b = 0, total_b = 0;
    S2P_CUDA_TRY(cudaGetDeviceProperties(&prop, device));
    S2P_CUDA_TRY(cudaDriverGetVersion(&drv));
    S2P_CUDA_TRY(cudaRuntimeGetVersion(&rt));
    /* mem info reflects the current device; best-effort for others */
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        free_b = 0;
        total_b = prop.totalGlobalMem;
    }
    fprintf(stderr,
            "[s2pro] cuda:%d %s sm_%d%d, %d SMs, %.1f/%.1f GiB free, "
            "driver %d.%d, runtime %d.%d\n",
            device, prop.name, prop.major, prop.minor,
            prop.multiProcessorCount, (double)free_b / (1024.0 * 1024 * 1024),
            (double)total_b / (1024.0 * 1024 * 1024), drv / 1000,
            (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    if (prop.major * 10 + prop.minor < 80)
        fprintf(stderr, "[s2pro] warning: sm_%d%d < sm_80 — no native bf16, "
                        "expect breakage\n", prop.major, prop.minor);
    return S2P_OK;
}

extern "C" s2p_status s2p_cuda_init(int device, int verbose)
{
    int n = 0;
    S2P_CUDA_TRY(cudaGetDeviceCount(&n));
    if (device < 0 || device >= n) {
        fprintf(stderr, "[s2pro] cuda device %d out of range (%d present)\n",
                device, n);
        return S2P_ERR_INVALID;
    }
    S2P_CUDA_TRY(cudaSetDevice(device));
    S2P_CUDA_TRY(cudaFree(0)); /* force context creation now, not lazily */
    if (verbose) return s2p_cuda_device_print(device);
    return S2P_OK;
}

extern "C" s2p_status s2p_cuda_mem_info(size_t* free_bytes, size_t* total_bytes)
{
    size_t f = 0, t = 0;
    S2P_CUDA_TRY(cudaMemGetInfo(&f, &t));
    if (free_bytes) *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return S2P_OK;
}

extern "C" s2p_status s2p_cuda_stream_create(cudaStream_t* out)
{
    if (!out) return S2P_ERR_INVALID;
    S2P_CUDA_TRY(cudaStreamCreateWithFlags(out, cudaStreamNonBlocking));
    return S2P_OK;
}

extern "C" void s2p_cuda_stream_destroy(cudaStream_t s)
{
    if (!s) return;
    cudaError_t e = cudaStreamDestroy(s);
    if (e != cudaSuccess)
        fprintf(stderr, "[s2pro] cudaStreamDestroy: %s\n",
                cudaGetErrorString(e));
}

extern "C" s2p_status s2p_cuda_stream_sync(cudaStream_t s)
{
    S2P_CUDA_TRY(cudaStreamSynchronize(s));
    return S2P_OK;
}

extern "C" s2p_status s2p_cuda_event_create(cudaEvent_t* out)
{
    if (!out) return S2P_ERR_INVALID;
    S2P_CUDA_TRY(cudaEventCreate(out));
    return S2P_OK;
}

extern "C" void s2p_cuda_event_destroy(cudaEvent_t e)
{
    if (!e) return;
    cudaError_t err = cudaEventDestroy(e);
    if (err != cudaSuccess)
        fprintf(stderr, "[s2pro] cudaEventDestroy: %s\n",
                cudaGetErrorString(err));
}

extern "C" s2p_status s2p_cuda_event_record(cudaEvent_t e, cudaStream_t s)
{
    S2P_CUDA_TRY(cudaEventRecord(e, s));
    return S2P_OK;
}

extern "C" s2p_status s2p_cuda_event_elapsed_ms(cudaEvent_t beg_ev,
                                                cudaEvent_t end_ev,
                                                float* out_ms)
{
    if (!out_ms) return S2P_ERR_INVALID;
    S2P_CUDA_TRY(cudaEventSynchronize(end_ev));
    S2P_CUDA_TRY(cudaEventElapsedTime(out_ms, beg_ev, end_ev));
    return S2P_OK;
}
