/* s2pro-native — runtime config loader (s2pro/config.h).
 *
 * Parses <model_dir>/config.json once and keeps the DOM alive; the compile-
 * time dims in config.h stay authoritative — this accessor exists for load-
 * time sanity checks and the handful of values only known at runtime.
 *
 * s2p_config_i64 takes dotted paths into nested objects
 * ("text_config.n_layer", "audio_decoder_config.num_codebooks") as well as
 * plain top-level keys ("semantic_start_token_id", "eos_token_id", ...).
 * Booleans read as 0/1; a one-element array reads as its first element
 * (HF-style eos lists); anything missing or non-numeric reads as 0.
 */
#include "s2pro/config.h"
#include "s2pro/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct s2p_config {
    char*     model_dir;
    s2p_json* j;
};

static s2p_status read_all(const char* path, char** out, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    long sz;
    char* buf;
    if (!f) return S2P_ERR_IO;
    if (fseek(f, 0, SEEK_END) || (sz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return S2P_ERR_IO;
    }
    buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return S2P_ERR_OOM;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return S2P_ERR_IO;
    }
    fclose(f);
    buf[sz] = '\0';
    *out = buf;
    *out_len = (size_t)sz;
    return S2P_OK;
}

s2p_status s2p_config_load(const char* model_dir, s2p_config** out)
{
    s2p_config* c;
    char* path;
    char* buf = NULL;
    size_t len = 0;
    s2p_status rc;

    if (!model_dir || !out) return S2P_ERR_INVALID;
    *out = NULL;

    c = (s2p_config*)calloc(1, sizeof(*c));
    if (!c) return S2P_ERR_OOM;
    c->model_dir = strdup(model_dir);
    if (!c->model_dir) {
        free(c);
        return S2P_ERR_OOM;
    }

    {
        size_t dl = strlen(model_dir);
        path = (char*)malloc(dl + sizeof("/config.json"));
        if (!path) {
            s2p_config_free(c);
            return S2P_ERR_OOM;
        }
        memcpy(path, model_dir, dl);
        memcpy(path + dl, "/config.json", sizeof("/config.json"));
    }

    rc = read_all(path, &buf, &len);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] cannot read %s\n", path);
        free(path);
        s2p_config_free(c);
        return rc;
    }
    rc = s2p_json_parse(buf, len, &c->j);
    free(buf);
    if (rc != S2P_OK) {
        fprintf(stderr, "[s2pro] cannot parse %s\n", path);
        free(path);
        s2p_config_free(c);
        return rc;
    }
    free(path);
    if (s2p_jobj_len(s2p_json_root(c->j)) == 0) {
        fprintf(stderr, "[s2pro] config.json root is not an object\n");
        s2p_config_free(c);
        return S2P_ERR_FORMAT;
    }
    *out = c;
    return S2P_OK;
}

void s2p_config_free(s2p_config* c)
{
    if (!c) return;
    s2p_json_free(c->j);
    free(c->model_dir);
    free(c);
}

int64_t s2p_config_i64(const s2p_config* c, const char* key)
{
    const s2p_jval* v;
    const char* seg;

    if (!c || !c->j || !key) return 0;
    v = s2p_json_root(c->j);
    seg = key;
    while (v) {
        const char* dot = strchr(seg, '.');
        size_t sl = dot ? (size_t)(dot - seg) : strlen(seg);
        char kb[128];
        if (sl == 0 || sl >= sizeof(kb)) return 0;
        memcpy(kb, seg, sl);
        kb[sl] = '\0';
        v = s2p_jobj_get(v, kb);
        if (!dot) break;
        seg = dot + 1;
    }
    if (!v) return 0;
    if (s2p_jarr_len(v) > 0) v = s2p_jarr_at(v, 0); /* eos-style id lists */
    {
        int64_t i = s2p_jint(v);
        if (i) return i;
    }
    return s2p_jbool(v);
}

const char* s2p_config_model_dir(const s2p_config* c)
{
    return c ? c->model_dir : NULL;
}
