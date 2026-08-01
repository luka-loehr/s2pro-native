/* s2pro-native — status codes and error macros. Contract header: frozen. */
#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    S2P_OK              = 0,
    S2P_ERR_INVALID     = -1,  /* bad argument / precondition */
    S2P_ERR_CUDA        = -2,  /* CUDA runtime/driver error */
    S2P_ERR_IO          = -3,  /* file/socket I/O */
    S2P_ERR_OOM         = -4,  /* host or device allocation failed */
    S2P_ERR_FORMAT      = -5,  /* parse error: safetensors/json/tokenizer */
    S2P_ERR_STATE       = -6,  /* API misuse / wrong lifecycle state */
    S2P_ERR_UNSUPPORTED = -7,  /* dtype/arch/feature not supported */
    S2P_ERR_INTERNAL    = -8,  /* invariant violated */
    S2P_ERR_FULL        = -9,  /* capacity (sessions, queue) exhausted */
} s2p_status;

const char* s2p_status_str(s2p_status s);

/* Propagate non-OK status. */
#define S2P_TRY(expr)                                                          \
    do {                                                                       \
        s2p_status _s = (expr);                                                \
        if (_s != S2P_OK) return _s;                                           \
    } while (0)

/* Wrap a CUDA call; logs file:line on failure. */
#define S2P_CUDA_TRY(expr)                                                     \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[s2pro] CUDA error %s:%d: %s\n", __FILE__,        \
                __LINE__, cudaGetErrorString(_e));                             \
            return S2P_ERR_CUDA;                                               \
        }                                                                      \
    } while (0)

#ifdef __cplusplus
}
#endif
