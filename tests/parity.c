/* s2pro-native — parity harness against the PyTorch reference oracle.
 *
 *   s2p-parity MODEL_DIR CODEC_DIR WORK_DIR
 *
 * WORK_DIR is prepared by tools/parity_prep.py from an oracle fixture set:
 *   meta.json          {"n_ids": N, "steps": S}
 *   prompt_ids.bin     int64[N]   — the EXACT oracle prompt (tokenizer
 *                                   bypassed on purpose; prompt-builder
 *                                   parity is checked separately)
 *   fixture_frames.bin int32[S*10] frame-major — oracle greedy frames
 *
 * Outputs into WORK_DIR:
 *   backbone_layerNN.f32 / backbone_final_norm.f32 / prefill_logits.f32 /
 *   step1_logits.f32   — via the S2P_DUMP_DIR hooks (set by this binary)
 *   our_sem.i64          sampled semantic TOKEN ids per frame
 *   our_frames.i32       int32[T*10] frame-major
 *   our_pcm.f32          our frames -> our DAC
 *   fixture_pcm_ours.f32 ORACLE frames -> our DAC (codec isolated from AR)
 *
 * Greedy (temperature 0), repetition penalty 1.1 over a 16-token window —
 * matching tools/oracle.py exactly. tools/parity_compare.py computes the
 * metrics against the .npy fixtures.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/json.h"
#include "s2pro/model.h"
#include "s2pro/dac.h"

#define MAX_STEPS 256

#define CHECK(rc, what)                                                        \
    do {                                                                       \
        s2p_status _r = (rc);                                                  \
        if (_r != S2P_OK) {                                                    \
            fprintf(stderr, "FAIL %s: status %d\n", (what), (int)_r);          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static void* read_file(const char* dir, const char* name, size_t* out_len) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)len);
    if (buf == NULL || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *out_len = (size_t)len;
    return buf;
}

static int write_out(const char* dir, const char* name, const void* data,
                     size_t bytes) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* f = fopen(path, "wb");
    if (f == NULL) return -1;
    fwrite(data, 1, bytes, f);
    fclose(f);
    return 0;
}

/* codes frame-major [T][10] -> malloc'd codebook-major [10][T] */
static int32_t* to_cb_major(const int32_t* fm, int T) {
    int32_t* cm = malloc((size_t)S2P_NUM_CODEBOOKS * T * sizeof(int32_t));
    if (cm == NULL) return NULL;
    for (int t = 0; t < T; t++)
        for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
            cm[(size_t)cb * T + t] = fm[(size_t)t * S2P_NUM_CODEBOOKS + cb];
    return cm;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL_DIR CODEC_DIR WORK_DIR\n", argv[0]);
        return 2;
    }
    const char* model_dir = argv[1];
    const char* codec_dir = argv[2];
    const char* work = argv[3];

    setenv("S2P_DUMP_DIR", work, 1); /* before model touches the hooks */

    /* --- inputs --- */
    size_t len = 0;
    char* meta_buf = read_file(work, "meta.json", &len);
    if (meta_buf == NULL) {
        fprintf(stderr, "FAIL meta.json missing in %s\n", work);
        return 1;
    }
    s2p_json* meta = NULL;
    CHECK(s2p_json_parse(meta_buf, len, &meta), "meta parse");
    const int n_ids =
        (int)s2p_jint(s2p_jobj_get(s2p_json_root(meta), "n_ids"));
    int steps = (int)s2p_jint(s2p_jobj_get(s2p_json_root(meta), "steps"));
    if (steps > MAX_STEPS) steps = MAX_STEPS;
    s2p_json_free(meta);
    free(meta_buf);

    int64_t* ids = read_file(work, "prompt_ids.bin", &len);
    if (ids == NULL || len != (size_t)n_ids * sizeof(int64_t)) {
        fprintf(stderr, "FAIL prompt_ids.bin: want %zu bytes got %zu\n",
                (size_t)n_ids * sizeof(int64_t), len);
        return 1;
    }
    size_t fx_len = 0;
    int32_t* fx_frames = read_file(work, "fixture_frames.bin", &fx_len);
    const int fx_T =
        fx_frames ? (int)(fx_len / (S2P_NUM_CODEBOOKS * sizeof(int32_t))) : 0;

    /* --- engine --- */
    CHECK(s2p_gemm_init(S2P_CTX_LEN_DEFAULT), "gemm init");
    s2p_gemm_mode mode = s2p_gemm_mode_from_env();
    fprintf(stderr, "[parity] gemm mode: %s, prompt %d ids, %d steps\n",
            mode == S2P_GEMM_FP8 ? "FP8" : "BF16", n_ids, steps);

    s2p_model_opts mopts;
    memset(&mopts, 0, sizeof(mopts));
    mopts.gemm_mode = mode;
    mopts.ctx_len = S2P_CTX_LEN_DEFAULT;
    mopts.max_sessions = 1;
    s2p_model* model = NULL;
    CHECK(s2p_model_load(model_dir, &mopts, &model), "model load");
    s2p_dac* dac = NULL;
    CHECK(s2p_dac_load(codec_dir, &dac), "dac load");

    /* greedy + reference repetition penalty (oracle: 1.1 over 16) */
    s2p_sampling_cfg sampling = s2p_sampling_defaults();
    sampling.temperature = 0.0f;
    s2p_session* sess = NULL;
    CHECK(s2p_session_create(model, &sampling, &sess), "session create");
    CHECK(s2p_session_prefill(sess, ids, NULL, n_ids, NULL, 0), "prefill");
    free(ids);

    /* --- frame loop --- */
    static int32_t frames[MAX_STEPS][S2P_NUM_CODEBOOKS];
    static int64_t sems[MAX_STEPS];
    int T = 0, hit_eos = 0;
    for (int t = 0; t < steps; t++) {
        int is_eos = 0;
        CHECK(s2p_session_next_frame(sess, frames[T], &is_eos), "next frame");
        if (is_eos) {
            hit_eos = 1;
            break;
        }
        sems[T] = S2P_TOK_SEMANTIC_START + frames[T][0];
        T++;
    }
    fprintf(stderr, "[parity] generated %d frames%s\n", T,
            hit_eos ? " (EOS)" : "");

    write_out(work, "our_sem.i64", sems, (size_t)T * sizeof(int64_t));
    write_out(work, "our_frames.i32", frames,
              (size_t)T * S2P_NUM_CODEBOOKS * sizeof(int32_t));

    /* --- codec: our frames, and (isolated) the oracle's frames --- */
    if (T > 0) {
        int32_t* cm = to_cb_major(&frames[0][0], T);
        float* pcm = NULL;
        int64_t n = 0;
        CHECK(s2p_dac_decode(dac, cm, T, &pcm, &n, 0), "dac (ours)");
        write_out(work, "our_pcm.f32", pcm, (size_t)n * sizeof(float));
        free(pcm);
        free(cm);
    }
    if (fx_T > 0) {
        int32_t* cm = to_cb_major(fx_frames, fx_T);
        float* pcm = NULL;
        int64_t n = 0;
        CHECK(s2p_dac_decode(dac, cm, fx_T, &pcm, &n, 0), "dac (fixture)");
        write_out(work, "fixture_pcm_ours.f32", pcm,
                  (size_t)n * sizeof(float));
        fprintf(stderr, "[parity] fixture frames %d -> %lld samples\n", fx_T,
                (long long)n);
        free(pcm);
        free(cm);
    }
    free(fx_frames);

    s2p_session_destroy(sess);
    s2p_dac_free(dac);
    s2p_model_free(model);
    s2p_gemm_shutdown();
    fprintf(stderr, "[parity] OK, outputs in %s\n", work);
    return 0;
}
