/* s2pro-native — safetensors checkpoint reader (mmap). Contract header: frozen.
 *
 * Supports a single model.safetensors or a sharded checkpoint with
 * model.safetensors.index.json. Tensor data pointers are valid until close.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "s2pro/status.h"
#include "s2pro/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_st s2p_st;

typedef struct {
    const char* name;      /* owned by the handle */
    s2p_dtype   dtype;
    int         ndim;
    int64_t     shape[4];
    const void* data;      /* mmap'd host pointer */
    size_t      nbytes;
} s2p_st_view;

s2p_status s2p_st_open_dir(const char* model_dir, s2p_st** out);
s2p_status s2p_st_find(s2p_st* st, const char* name, s2p_st_view* out);
int        s2p_st_count(const s2p_st* st);
s2p_status s2p_st_at(const s2p_st* st, int index, s2p_st_view* out);
void       s2p_st_close(s2p_st* st);

/* Convenience: find tensor, validate dtype+shape, upload to fresh device
 * tensor. shape entries of -1 are not checked. */
s2p_status s2p_st_load_device(s2p_st* st, const char* name, s2p_dtype expect_dt,
                              int ndim, const int64_t* expect_shape,
                              s2p_tensor* out_dev, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
