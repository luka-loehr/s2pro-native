/* s2pro-native — end-to-end smoke test (no HTTP, no scheduler).
 *
 *   s2p-test MODEL_DIR CODEC_DIR [TEXT]
 *
 * Builds a short German+English prompt, generates up to 60 frames with
 * GREEDY sampling (temperature 0 per model.h), decodes with the DAC vocoder
 * and writes /tmp/s2p_smoke.wav. Prints per-stage wall timings (prefill ms,
 * ms/frame, DAC ms, RTF) and the 10 codes of the first 3 frames so a human
 * can eyeball them against the PyTorch reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/model.h"
#include "s2pro/dac.h"
#include "s2pro/tokenizer.h"
#include "s2pro/wav.h"

#define MAX_FRAMES 60
#define OUT_WAV "/tmp/s2p_smoke.wav"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#define CHECK(rc, what)                                                        \
    do {                                                                       \
        s2p_status _r = (rc);                                                  \
        if (_r != S2P_OK) {                                                    \
            fprintf(stderr, "FAIL %s: status %d\n", (what), (int)_r);          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR CODEC_DIR [TEXT]\n", argv[0]);
        return 2;
    }
    const char* model_dir = argv[1];
    const char* codec_dir = argv[2];
    const char* text =
        argc > 3 ? argv[3]
                 : "Hallo und guten Tag! Dies ist ein kurzer Funktionstest. "
                   "And now a short English sentence to check bilingual "
                   "synthesis.";

    CHECK(s2p_gemm_init(S2P_CTX_LEN_DEFAULT), "gemm init");
    s2p_gemm_mode mode = s2p_gemm_mode_from_env();
    fprintf(stderr, "[test] gemm mode: %s\n",
            mode == S2P_GEMM_FP8 ? "FP8" : "BF16");

    s2p_config* cfg = NULL;
    CHECK(s2p_config_load(model_dir, &cfg), "config load");

    double t0 = now_ms();
    s2p_model_opts mopts;
    memset(&mopts, 0, sizeof(mopts));
    mopts.gemm_mode = mode;
    mopts.ctx_len = S2P_CTX_LEN_DEFAULT;
    mopts.max_sessions = 1;
    s2p_model* model = NULL;
    CHECK(s2p_model_load(model_dir, &mopts, &model), "model load");
    fprintf(stderr, "[test] model load: %.0f ms\n", now_ms() - t0);

    t0 = now_ms();
    s2p_dac* dac = NULL;
    CHECK(s2p_dac_load(codec_dir, &dac), "dac load");
    fprintf(stderr, "[test] dac load:   %.0f ms\n", now_ms() - t0);

    s2p_tok* tok = NULL;
    CHECK(s2p_tok_load(model_dir, &tok), "tokenizer load");

    /* Prompt */
    s2p_request_text req;
    memset(&req, 0, sizeof(req));
    req.text = text;
    int64_t*     ids = NULL;
    uint8_t*     vq_mask = NULL;
    s2p_vq_part* parts = NULL;
    int          n_ids = 0, n_parts = 0;
    CHECK(s2p_prompt_build(tok, cfg, &req, &ids, &vq_mask, &n_ids, &parts,
                           &n_parts),
          "prompt build");
    fprintf(stderr, "[test] prompt: %d tokens, %d vq parts\n", n_ids, n_parts);

    /* Greedy session: temperature 0 -> greedy (model.h contract). */
    s2p_sampling_cfg sampling = s2p_sampling_defaults();
    sampling.temperature = 0.0f;
    s2p_session* sess = NULL;
    CHECK(s2p_session_create(model, &sampling, &sess), "session create");

    t0 = now_ms();
    CHECK(s2p_session_prefill(sess, ids, vq_mask, n_ids, parts, n_parts),
          "prefill");
    double prefill_ms = now_ms() - t0;
    free(ids);
    free(vq_mask);
    free(parts);

    /* Frame loop; collect frame-major, transpose to codebook-major after. */
    int32_t frames[MAX_FRAMES][S2P_NUM_CODEBOOKS];
    int T = 0, hit_eos = 0;
    t0 = now_ms();
    for (int f = 0; f < MAX_FRAMES; f++) {
        int is_eos = 0;
        CHECK(s2p_session_next_frame(sess, frames[T], &is_eos), "next frame");
        if (is_eos) { /* the EOS frame carries no codes */
            hit_eos = 1;
            break;
        }
        T++;
    }
    double gen_ms = now_ms() - t0;
    s2p_session_destroy(sess);
    fprintf(stderr, "[test] generated %d frames%s\n", T,
            hit_eos ? " (EOS)" : " (frame cap)");
    if (T == 0) {
        fprintf(stderr, "FAIL: no frames generated\n");
        return 1;
    }

    for (int f = 0; f < (T < 3 ? T : 3); f++) {
        fprintf(stderr, "[test] frame %d codes:", f);
        for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
            fprintf(stderr, " %d", frames[f][cb]);
        fprintf(stderr, "\n");
    }

    /* Transpose to codebook-major [10*T] for the codec. */
    int32_t* codes = (int32_t*)malloc((size_t)S2P_NUM_CODEBOOKS * (size_t)T *
                                      sizeof(int32_t));
    if (!codes) {
        fprintf(stderr, "FAIL: oom\n");
        return 1;
    }
    for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
        for (int t = 0; t < T; t++) codes[cb * T + t] = frames[t][cb];

    t0 = now_ms();
    float*  pcm = NULL;
    int64_t n_samples = 0;
    CHECK(s2p_dac_decode(dac, codes, T, &pcm, &n_samples, 0), "dac decode");
    double dac_ms = now_ms() - t0;
    free(codes);

    int16_t* pcm16 = (int16_t*)malloc((size_t)n_samples * sizeof(int16_t));
    if (!pcm16) {
        fprintf(stderr, "FAIL: oom\n");
        return 1;
    }
    s2p_f32_to_s16(pcm, pcm16, n_samples);
    CHECK(s2p_wav_write_file(OUT_WAV, pcm16, n_samples, S2P_SAMPLE_RATE),
          "wav write");
    free(pcm);
    free(pcm16);

    double audio_s = (double)n_samples / (double)S2P_SAMPLE_RATE;
    double total_s = (prefill_ms + gen_ms + dac_ms) / 1e3;
    fprintf(stderr,
            "[test] prefill  %8.1f ms\n"
            "[test] decode   %8.1f ms  (%.1f ms/frame, %d frames)\n"
            "[test] dac      %8.1f ms\n"
            "[test] audio    %8.2f s   RTF %.2f (compute/audio, <1 = faster than playback)\n"
            "[test] wrote %s (%lld samples)\n",
            prefill_ms, gen_ms, gen_ms / (double)T, T, dac_ms, audio_s,
            total_s / (audio_s > 0 ? audio_s : 1e-9), OUT_WAV,
            (long long)n_samples);

    s2p_tok_free(tok);
    s2p_dac_free(dac);
    s2p_model_free(model);
    s2p_config_free(cfg);
    s2p_gemm_shutdown();
    printf("OK %d frames, %lld samples -> %s\n", T, (long long)n_samples,
           OUT_WAV);
    return 0;
}
