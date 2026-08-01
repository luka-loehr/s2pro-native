/* s2pro-native — voice registry: named, pre-encoded reference voices.
 * Additive contract extension (2026-08-01).
 *
 * A voice is a pair of files in the voices directory:
 *   <name>.wav   RIFF 16-bit PCM mono 44100 Hz reference audio
 *   <name>.txt   the EXACT transcript of that audio (UTF-8)
 *
 * Every voice is encoded through the DAC ONCE at load time; requests then
 * reuse the cached VQ codes, so voice selection adds no per-request encode
 * cost. Reference material must come from a source with native, accent-free
 * multilinguality — see docs/VOICES.md for why and how.
 */
#pragma once

#include "s2pro/status.h"
#include "s2pro/dac.h"
#include "s2pro/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2p_voices s2p_voices;

typedef struct {
    const char* name;       /* file basename, e.g. "neutral-female" */
    const char* transcript; /* exact reference transcript */
    s2p_vq_part part;       /* pre-encoded [10*T] codes, registry-owned */
    double      duration_s; /* reference audio length */
} s2p_voice;

/* Scan dir for <name>.wav + <name>.txt pairs and encode each through the
 * DAC (requires the full codec artifact / active encoder). A missing or
 * empty directory yields an empty registry (S2P_OK) — zero-shot serving
 * still works. Individual bad pairs are skipped with a logged warning. */
s2p_status s2p_voices_load(const char* dir, s2p_dac* dac, s2p_voices** out);

int              s2p_voices_count(const s2p_voices* v);
const s2p_voice* s2p_voices_at(const s2p_voices* v, int i);
const s2p_voice* s2p_voices_find(const s2p_voices* v, const char* name);
void             s2p_voices_free(s2p_voices* v);

#ifdef __cplusplus
}
#endif
