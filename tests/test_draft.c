/* Draft-forward parity vs the torch draft (docs/SPECULATIVE.md §7 gate 1).
 *
 * Reads the fixture the trainer dumped (fix_h.f32 / fix_e.f32 /
 * fix_pred.f32, each [32, 2560] f32), replays the 32 positions through
 * the native forward with a growing KV state, and reports per-position
 * and overall cosine against the torch predictions. Gate: overall
 * cosine >= 0.999 (bf16 kernel-order differences only).
 *
 * Usage: s2p-draft-test <draft.safetensors> <fixture-dir>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "../src/draft/draft.h"
#include "s2pro/gemm.h"

#define N_POS 32
#define D 2560

static float* read_f32(const char* dir, const char* name, size_t n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FAIL: open %s\n", path); exit(1); }
    float* buf = (float*)malloc(n * sizeof(float));
    if (fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "FAIL: short read %s\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

static unsigned short f32_to_bf16(float x) {
    unsigned int u;
    memcpy(&u, &x, 4);
    unsigned int lsb = (u >> 16) & 1;
    u += 0x7fff + lsb;
    return (unsigned short)(u >> 16);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model fixdir\n", argv[0]); return 2; }
    if (s2p_gemm_init(16) != S2P_OK) { fprintf(stderr, "FAIL gemm init\n"); return 1; }
    s2p_draft_model m;
    if (s2p_draft_load(&m, argv[1]) != S2P_OK) { fprintf(stderr, "FAIL load\n"); return 1; }
    s2p_draft_state st;
    if (s2p_draft_state_init(&st) != S2P_OK) { fprintf(stderr, "FAIL state\n"); return 1; }

    float* h = read_f32(argv[2], "fix_h.f32", (size_t)N_POS * D);
    float* e = read_f32(argv[2], "fix_e.f32", (size_t)N_POS * D);
    float* pr = read_f32(argv[2], "fix_pred.f32", (size_t)N_POS * D);

    unsigned short *dh, *de, *dout;
    cudaMalloc((void**)&dh, (size_t)D * 2);
    cudaMalloc((void**)&de, (size_t)D * 2);
    cudaMalloc((void**)&dout, (size_t)D * 2);
    unsigned short hh[D], he[D], ho[D];

    double cs_min = 2.0, cs_sum = 0.0;
    for (int t = 0; t < N_POS; t++) {
        for (int i = 0; i < D; i++) {
            hh[i] = f32_to_bf16(h[(size_t)t * D + i]);
            he[i] = f32_to_bf16(e[(size_t)t * D + i]);
        }
        cudaMemcpy(dh, hh, sizeof(hh), cudaMemcpyHostToDevice);
        cudaMemcpy(de, he, sizeof(he), cudaMemcpyHostToDevice);
        if (s2p_draft_step(&m, &st, dh, de, dout, 0) != S2P_OK) {
            fprintf(stderr, "FAIL step %d\n", t);
            return 1;
        }
        cudaMemcpy(ho, dout, sizeof(ho), cudaMemcpyDeviceToHost);
        double num = 0, na = 0, nb = 0;
        for (int i = 0; i < D; i++) {
            unsigned int u = (unsigned int)ho[i] << 16;
            float a;
            memcpy(&a, &u, 4);
            float b = pr[(size_t)t * D + i];
            num += (double)a * b;
            na += (double)a * a;
            nb += (double)b * b;
        }
        double cs = num / (sqrt(na) * sqrt(nb) + 1e-30);
        cs_sum += cs;
        if (cs < cs_min) cs_min = cs;
        if (t < 3 || t == N_POS - 1)
            printf("[draft-test] pos %2d cos %.6f\n", t, cs);
    }
    printf("[draft-test] overall: mean cos %.6f, min cos %.6f -> %s\n",
           cs_sum / N_POS, cs_min, cs_min >= 0.999 ? "PASS" : "FAIL");
    return cs_min >= 0.999 ? 0 : 1;
}
