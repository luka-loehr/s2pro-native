/* s2pro-native — server entry point.
 *
 *   s2pro-server --model-dir DIR [--codec-dir DIR] [--port N] [--bind ADDR]
 *                [--token TOK] [--fp8] [--ctx N]
 *
 * Init order: gemm -> config -> model -> dac -> tokenizer -> scheduler ->
 * server. s2p_server_run blocks until SIGINT/SIGTERM, then everything is
 * torn down in reverse. FP8 GEMM is opt-in via --fp8 or S2P_FP8=1
 * (docs/SPARK.md: BF16 is the validated default).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "s2pro/status.h"
#include "s2pro/config.h"
#include "s2pro/gemm.h"
#include "s2pro/model.h"
#include "s2pro/dac.h"
#include "s2pro/tokenizer.h"
#include "s2pro/scheduler.h"
#include "s2pro/server.h"

static void usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s --model-dir DIR [options]\n"
            "  --model-dir DIR   checkpoint + tokenizer directory (required)\n"
            "  --codec-dir DIR   DAC codec directory (default: model dir)\n"
            "  --port N          listen port (default 8010)\n"
            "  --bind ADDR       bind address (default 127.0.0.1)\n"
            "  --token TOK       require 'Authorization: Bearer TOK'\n"
            "  --fp8             use the fish-scales-ops FP8 GEMM path\n"
            "  --ctx N           context length (default %d)\n",
            argv0, S2P_CTX_LEN_DEFAULT);
}

int main(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* codec_dir = NULL;
    const char* bind_addr = NULL;
    const char* token = NULL;
    int port = 0;
    int fp8 = 0;
    int ctx = 0;

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        int has_next = i + 1 < argc;
        if (strcmp(a, "--model-dir") == 0 && has_next) {
            model_dir = argv[++i];
        } else if (strcmp(a, "--codec-dir") == 0 && has_next) {
            codec_dir = argv[++i];
        } else if (strcmp(a, "--port") == 0 && has_next) {
            port = atoi(argv[++i]);
        } else if (strcmp(a, "--bind") == 0 && has_next) {
            bind_addr = argv[++i];
        } else if (strcmp(a, "--token") == 0 && has_next) {
            token = argv[++i];
        } else if (strcmp(a, "--fp8") == 0) {
            fp8 = 1;
        } else if (strcmp(a, "--ctx") == 0 && has_next) {
            ctx = atoi(argv[++i]);
        } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown/incomplete argument: %s\n", a);
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_dir) {
        usage(argv[0]);
        return 2;
    }
    if (!codec_dir) codec_dir = model_dir;
    if (ctx <= 0) ctx = S2P_CTX_LEN_DEFAULT;

    s2p_status rc;
    s2p_config* cfg = NULL;
    s2p_model* model = NULL;
    s2p_dac* dac = NULL;
    s2p_tok* tok = NULL;
    s2p_sched* sched = NULL;
    int exit_code = 1;

    /* 1. GEMM context (cuBLAS handle + FP8 scratch sized for prefill). */
    rc = s2p_gemm_init(ctx);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] gemm init failed: %d\n", (int)rc);
        return 1;
    }
    s2p_gemm_mode mode = fp8 ? S2P_GEMM_FP8 : s2p_gemm_mode_from_env();
    if (mode == S2P_GEMM_FP8 && !s2p_fso_available()) {
        fprintf(stderr, "[s2pro] FP8 requested but unavailable; using BF16\n");
        mode = S2P_GEMM_BF16;
    }
    fprintf(stderr, "[s2pro] gemm mode: %s\n",
            mode == S2P_GEMM_FP8 ? "FP8 (fish-scales-ops)" : "BF16 (cuBLAS)");

    /* 2. config */
    rc = s2p_config_load(model_dir, &cfg);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] config load failed: %d (%s)\n", (int)rc,
                model_dir);
        goto out;
    }

    /* 3. model (slow-AR + fast-AR weights) */
    s2p_model_opts mopts;
    memset(&mopts, 0, sizeof(mopts));
    mopts.gemm_mode = mode;
    mopts.ctx_len = ctx;
    mopts.max_sessions = S2P_MAX_SESSIONS;
    fprintf(stderr, "[s2pro] loading model from %s ...\n", model_dir);
    rc = s2p_model_load(model_dir, &mopts, &model);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] model load failed: %d\n", (int)rc);
        goto out;
    }

    /* 4. DAC codec */
    fprintf(stderr, "[s2pro] loading codec from %s ...\n", codec_dir);
    rc = s2p_dac_load(codec_dir, &dac);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] dac load failed: %d\n", (int)rc);
        goto out;
    }

    /* 5. tokenizer */
    rc = s2p_tok_load(model_dir, &tok);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] tokenizer load failed: %d\n", (int)rc);
        goto out;
    }

    /* 6. scheduler */
    s2p_sched_opts sopts;
    memset(&sopts, 0, sizeof(sopts));
    rc = s2p_sched_create(model, dac, tok, cfg, &sopts, &sched);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] scheduler create failed: %d\n", (int)rc);
        goto out;
    }
    rc = s2p_sched_start(sched);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] scheduler start failed: %d\n", (int)rc);
        goto out;
    }

    /* 7. HTTP server (blocks until SIGINT/SIGTERM). */
    s2p_server_opts srvopts;
    memset(&srvopts, 0, sizeof(srvopts));
    srvopts.port = port;
    srvopts.bind_addr = bind_addr;
    srvopts.auth_token = token;
    rc = s2p_server_run(sched, &srvopts);
    exit_code = (rc == S2P_OK) ? 0 : 1;

out:
    if (sched) s2p_sched_destroy(sched); /* stops + joins the worker */
    if (tok) s2p_tok_free(tok);
    if (dac) s2p_dac_free(dac);
    if (model) s2p_model_free(model);
    if (cfg) s2p_config_free(cfg);
    s2p_gemm_shutdown();
    fprintf(stderr, "[s2pro] bye\n");
    return exit_code;
}
