/* s2pro-native — parity dump hooks (slowar_internal.h).
 *
 * When S2P_DUMP_DIR is set, selected device bf16 vectors are downloaded,
 * widened to f32, and written as raw little-endian f32 files for
 * tools/parity_compare.py. Compiled in always; every call is a cheap
 * getenv-cached no-op unless the variable is present, so the hot path is
 * unaffected in normal operation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slowar_internal.h"

const char* s2psl_dump_dir(void) {
    static int init = 0;
    static const char* dir = NULL;
    if (!init) {
        dir = getenv("S2P_DUMP_DIR");
        if (dir != NULL && dir[0] == '\0') dir = NULL;
        init = 1;
    }
    return dir;
}

static inline float dump_b2f(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

void s2psl_dump_vec_bf16(const char* name, const void* dev_bf16, int64_t n,
                         cudaStream_t st) {
    const char* dir = s2psl_dump_dir();
    if (dir == NULL) return;

    uint16_t* h = malloc((size_t)n * sizeof(uint16_t));
    float* f = malloc((size_t)n * sizeof(float));
    if (h == NULL || f == NULL) goto out;
    if (cudaStreamSynchronize(st) != cudaSuccess) goto out;
    if (cudaMemcpy(h, dev_bf16, (size_t)n * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        goto out;
    for (int64_t i = 0; i < n; i++) f[i] = dump_b2f(h[i]);

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", dir, name);
    FILE* fp = fopen(path, "wb");
    if (fp != NULL) {
        fwrite(f, sizeof(float), (size_t)n, fp);
        fclose(fp);
    } else {
        fprintf(stderr, "[s2pro] dump: cannot open %s\n", path);
    }
out:
    free(h);
    free(f);
}
