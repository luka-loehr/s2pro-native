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

#define MAX_FRAMES 2048 /* default cap 60; override via S2P_TEST_FRAMES */
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

/* Minimal RIFF reader: 16-bit PCM mono 44.1 kHz -> malloc'd float [-1,1]. */
static float* wav_read_f32(const char* path, int64_t* out_n) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return NULL;
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return NULL;
    }
    uint32_t rate = 0;
    uint16_t ch = 0, bits = 0;
    float* pcm = NULL;
    for (;;) {
        uint8_t ck[8];
        if (fread(ck, 1, 8, f) != 8) break;
        uint32_t sz = (uint32_t)ck[4] | ((uint32_t)ck[5] << 8) |
                      ((uint32_t)ck[6] << 16) | ((uint32_t)ck[7] << 24);
        if (memcmp(ck, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (sz < 16 || fread(fmt, 1, 16, f) != 16) break;
            ch = (uint16_t)(fmt[2] | (fmt[3] << 8));
            rate = (uint32_t)fmt[4] | ((uint32_t)fmt[5] << 8) |
                   ((uint32_t)fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
            bits = (uint16_t)(fmt[14] | (fmt[15] << 8));
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
        } else if (memcmp(ck, "data", 4) == 0) {
            if (ch != 1 || bits != 16 || rate != 44100) {
                fprintf(stderr,
                        "[test] ref wav must be 44100 Hz mono s16 "
                        "(got %u Hz %u ch %u bit)\n", rate, ch, bits);
                break;
            }
            int64_t n = sz / 2;
            int16_t* s = malloc((size_t)sz);
            pcm = malloc((size_t)n * sizeof(float));
            if (s == NULL || pcm == NULL ||
                fread(s, 1, sz, f) != sz) {
                free(s);
                free(pcm);
                pcm = NULL;
                break;
            }
            for (int64_t i = 0; i < n; i++) pcm[i] = (float)s[i] / 32768.0f;
            free(s);
            *out_n = n;
            break;
        } else {
            fseek(f, (long)(sz + (sz & 1)), SEEK_CUR);
        }
    }
    fclose(f);
    return pcm;
}

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
            mode == S2P_GEMM_INT8  ? "INT8"
            : mode == S2P_GEMM_FP8 ? "FP8"
                                   : "BF16");

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

    /* Prompt (optional voice-cloning references: S2P_TEST_REF is one wav
     * path or a comma-separated list; S2P_TEST_REF_TEXT is the ONE combined
     * transcript covering all clips in order — the prompt builder emits a
     * single system block for all refs. Needs the full codec artifact.) */
    s2p_request_text req;
    memset(&req, 0, sizeof(req));
    req.text = text;
#define MAX_REFS 8
    s2p_vq_part refparts[MAX_REFS];
    int n_refwavs = 0;
    const char* refenv = getenv("S2P_TEST_REF");
    if (refenv != NULL) {
        char reflist[2048];
        snprintf(reflist, sizeof(reflist), "%s", refenv);
        for (char* tokp = strtok(reflist, ","); tokp != NULL;
             tokp = strtok(NULL, ",")) {
            if (n_refwavs >= MAX_REFS) break;
            int64_t rn = 0;
            float* rpcm = wav_read_f32(tokp, &rn);
            if (rpcm == NULL) {
                fprintf(stderr, "FAIL: cannot read ref wav %s\n", tokp);
                return 1;
            }
            int refT = 0;
            int32_t* rc32 = NULL;
            double tr = now_ms();
            CHECK(s2p_dac_encode(dac, rpcm, rn, &rc32, &refT, 0),
                  "dac encode (reference)");
            fprintf(stderr,
                    "[test] reference[%d]: %s  %.2f s -> %d frames (%.0f ms)\n",
                    n_refwavs, tokp, (double)rn / S2P_SAMPLE_RATE, refT,
                    now_ms() - tr);
            free(rpcm);
            refparts[n_refwavs].codes = rc32;
            refparts[n_refwavs].T = refT;
            n_refwavs++;
        }
        req.refs = refparts;
        req.n_refs = n_refwavs;
        req.ref_text = getenv("S2P_TEST_REF_TEXT");
        if (req.ref_text == NULL) req.ref_text = "";
    }
    int64_t*     ids = NULL;
    uint8_t*     vq_mask = NULL;
    s2p_vq_part* parts = NULL;
    int          n_ids = 0, n_parts = 0;
    CHECK(s2p_prompt_build(tok, cfg, &req, &ids, &vq_mask, &n_ids, &parts,
                           &n_parts, NULL),
          "prompt build");
    fprintf(stderr, "[test] prompt: %d tokens, %d vq parts\n", n_ids, n_parts);

    /* Deterministic greedy by default; the reference sampler is opt-in for
     * listening runs: S2P_TEST_TEMP (e.g. 0.8), S2P_TEST_SEED,
     * S2P_TEST_FRAMES (cap, <= 512). */
    s2p_sampling_cfg sampling = s2p_sampling_defaults();
    sampling.temperature = 0.0f;
    const char* env;
    if ((env = getenv("S2P_TEST_TEMP")) != NULL)
        sampling.temperature = (float)atof(env);
    if ((env = getenv("S2P_TEST_SEED")) != NULL)
        sampling.seed = (uint64_t)strtoull(env, NULL, 10);
    int frame_cap = 60;
    if ((env = getenv("S2P_TEST_FRAMES")) != NULL) {
        frame_cap = atoi(env);
        if (frame_cap < 1) frame_cap = 1;
        if (frame_cap > MAX_FRAMES) frame_cap = MAX_FRAMES;
    }
    fprintf(stderr, "[test] sampling: temp %.2f top_p %.2f seed %llu cap %d\n",
            sampling.temperature, sampling.top_p,
            (unsigned long long)sampling.seed, frame_cap);
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
    for (int f = 0; f < frame_cap; f++) {
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

    /* Diagnostic: S2P_TEST_STREAM_WAV=path additionally decodes the SAME
     * frames through the streaming vocoder (window + crossfade) so the two
     * DAC paths can be diffed on identical codes. */
    const char* stream_out = getenv("S2P_TEST_STREAM_WAV");
    if (stream_out != NULL) {
        s2p_dac_stream* ds = NULL;
        CHECK(s2p_dac_stream_create(dac, &ds), "dac stream create");
        float*  sp = NULL;
        int64_t sn = 0, cap = 0;
        for (int t = 0; t < T; t++) {
            int32_t frame[S2P_NUM_CODEBOOKS];
            for (int cb = 0; cb < S2P_NUM_CODEBOOKS; cb++)
                frame[cb] = codes[cb * T + t];
            float*  ch = NULL;
            int64_t cn = 0;
            CHECK(s2p_dac_stream_push(ds, frame, &ch, &cn, 0), "stream push");
            if (cn > 0) {
                if (sn + cn > cap) {
                    cap = (sn + cn) * 2;
                    sp = (float*)realloc(sp, (size_t)cap * sizeof(float));
                }
                memcpy(sp + sn, ch, (size_t)cn * sizeof(float));
                sn += cn;
            }
            free(ch);
        }
        float*  ch = NULL;
        int64_t cn = 0;
        CHECK(s2p_dac_stream_finish(ds, &ch, &cn, 0), "stream finish");
        if (cn > 0) {
            sp = (float*)realloc(sp, (size_t)(sn + cn) * sizeof(float));
            memcpy(sp + sn, ch, (size_t)cn * sizeof(float));
            sn += cn;
        }
        free(ch);
        s2p_dac_stream_destroy(ds);
        /* bit-exactness gate vs the whole-buffer decode above */
        {
            int64_t ncmp = sn < n_samples ? sn : n_samples;
            double maxd = 0.0;
            int64_t ndiff = 0;
            for (int64_t i = 0; i < ncmp; i++) {
                double dd = sp[i] - pcm[i];
                if (dd < 0) dd = -dd;
                if (dd > 0) ndiff++;
                if (dd > maxd) maxd = dd;
            }
            fprintf(stderr,
                    "[test] stream vs whole-buffer: len %lld/%lld, "
                    "differing samples %lld, max|diff| %.3g%s\n",
                    (long long)sn, (long long)n_samples, (long long)ndiff,
                    maxd,
                    (sn == n_samples && ndiff == 0) ? "  BIT-EXACT" : "");
        }
        int16_t* s16 = (int16_t*)malloc((size_t)sn * sizeof(int16_t));
        if (s16 != NULL) {
            s2p_f32_to_s16(sp, s16, sn);
            CHECK(s2p_wav_write_file(stream_out, s16, sn, S2P_SAMPLE_RATE),
                  "stream wav write");
            fprintf(stderr, "[test] stream decode: %lld samples -> %s\n",
                    (long long)sn, stream_out);
        }
        free(s16);
        free(sp);
    }
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
