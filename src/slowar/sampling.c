/* s2pro-native — slow-AR semantic sampler (exact reference algorithm).
 *
 * Replicates `S2ProSGLangTextModel._decode_codebooks` (sglang_model.py) plus
 * the history bookkeeping in model_runner.py, in host fp32:
 *
 *   bias -> bf16 round -> RAS select -> repetition penalty (gather/scatter on
 *   the RAW biased logits) -> top-k(30) -> top-p over an UN-temperatured
 *   softmax (rank 0 always kept) -> SECOND softmax with temperature
 *   (clamp min 1e-5) -> multinomial (seeded hash-Gumbel or per-session RNG).
 *
 * The semantic bias leaves exactly 4097 candidates finite (4096 semantic ids
 * + im_end); every other logit is -inf and can never enter the top-30, so the
 * sampler operates on just that slice (bit-identical to running on the full
 * 155776 row). The bf16 round of (logits + bias) is a no-op on the candidate
 * slice (bias 0.0 added to an already-bf16 logit), so widening bf16 -> f32 is
 * exact.
 *
 * Seeded draws replicate sglang's `multinomial_with_seed` hash + Gumbel-max
 * (constants 19349663 / 73856093 / 8589934591 / 479001599, 24-bit uniforms,
 * clamp, -log(-log(u)), argmax over log(probs + 1e-10) + gumbel) with the
 * seed masked to a positive int32 (resolve_row_seed) and the UNCAPPED step
 * counter. NOTE: the reference call site passes log(probs) into a function
 * that logs again — log of a negative number, NaN everywhere, which torch
 * argmax resolves to column 0 (rank 0). That is a reference-side bug; we
 * apply the log ONCE (the function's documented semantics) so seeded rows
 * genuinely sample. Flagged in the module report.
 *
 * History semantics (collect_s2pro_step_outputs / _sync_decode_row_state):
 * only non-EOS semantic tokens are appended; the RAS/rep-penalty window is
 * capped at `window`, the seeded step counter is uncapped; both are read
 * BEFORE this frame's token is appended.
 */
#include <math.h>
#include <string.h>
#include <time.h>

#include "slowar_internal.h"

/* ----- bf16 -> f32 (exact widening) -------------------------------------- */

static inline float b2f(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* ----- xoshiro256** (unseeded rows; reference uses torch.multinomial whose
 * stream is not reproducible outside torch, so any sound uniform source is
 * equally faithful) ------------------------------------------------------- */

static uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro_next(uint64_t s[4]) {
    const uint64_t result = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static double xoshiro_double(uint64_t s[4]) {
    return (double)(xoshiro_next(s) >> 11) * 0x1.0p-53;
}

/* ----- public defaults (model.h) ----------------------------------------- */

s2p_sampling_cfg s2p_sampling_defaults(void) {
    s2p_sampling_cfg c;
    c.temperature = 0.8f;       /* _DEFAULT_TEMPERATURE */
    c.top_p = 0.8f;             /* _DEFAULT_TOP_P */
    c.repetition_penalty = 1.1f; /* _DEFAULT_REP_PENALTY */
    c.repetition_window = S2PS_DEF_WINDOW;
    c.seed = 0; /* 0 -> unseeded (fresh RNG), else reproducible */
    return c;
}

/* ----- sampler ----------------------------------------------------------- */

void s2ps_sampler_init(s2ps_sampler* sp, const s2p_sampling_cfg* cfg,
                       uint64_t entropy) {
    s2p_sampling_cfg def = s2p_sampling_defaults();
    if (cfg == NULL) cfg = &def;
    memset(sp, 0, sizeof(*sp));
    sp->temperature = cfg->temperature < 0.f ? 0.f : cfg->temperature;
    sp->top_p = cfg->top_p;
    if (sp->top_p <= 0.f || sp->top_p > 1.f) sp->top_p = 1.f;
    sp->rep_penalty =
        cfg->repetition_penalty > 0.f ? cfg->repetition_penalty : 1.f;
    sp->window = cfg->repetition_window > 0 ? cfg->repetition_window
                                            : S2PS_DEF_WINDOW;
    if (sp->window > S2PS_MAX_WINDOW) sp->window = S2PS_MAX_WINDOW;
    sp->seeded = cfg->seed != 0;
    sp->seed31 = (uint32_t)(cfg->seed & 0x7FFFFFFFULL); /* resolve_row_seed */
    uint64_t sm = entropy ^ 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < 4; i++) sp->rng[i] = splitmix64(&sm);
}

static int64_t cand_to_vocab(int ci) {
    return ci == S2PS_CAND_EOS ? (int64_t)S2P_TOK_EOS
                               : (int64_t)(S2P_TOK_SEMANTIC_START + ci);
}

int64_t s2ps_sample(s2ps_sampler* sp, const uint16_t* sem_bf16,
                    uint16_t eos_bf16, int32_t* out_sem_id, int* out_eos) {
    float cand[S2PS_N_CAND];
    for (int i = 0; i < S2PS_N_SEM; i++) cand[i] = b2f(sem_bf16[i]);
    cand[S2PS_CAND_EOS] = b2f(eos_bf16);

    const int greedy = sp->temperature == 0.0f;
    const int capped =
        sp->count < (uint64_t)sp->window ? (int)sp->count : sp->window;

    /* RAS detect: any duplicate among the last 4 (capped-window gather with
     * clamp-to-0, exactly the reference formula), gated on capped count>=4. */
    int use_ras = 0;
    if (!greedy && capped >= 4) {
        int64_t last4[4];
        for (int r = 0; r < 4; r++) {
            int idx = capped - (4 - r);
            if (idx < 0) idx = 0;
            last4[r] = sp->prev[idx];
        }
        for (int a = 0; a < 3; a++) /* tiny insertion sort */
            for (int b = a + 1; b < 4; b++)
                if (last4[b] < last4[a]) {
                    int64_t t = last4[a];
                    last4[a] = last4[b];
                    last4[b] = t;
                }
        for (int a = 0; a < 3; a++)
            if (last4[a] == last4[a + 1]) use_ras = 1;
    }
    const float temp = use_ras ? S2PS_RAS_TEMPERATURE : sp->temperature;
    const float topp = use_ras ? S2PS_RAS_TOP_P : sp->top_p;

    /* Repetition penalty on the raw biased logits, BEFORE top-k/top-p.
     * Gather all originals first, then scatter — duplicated history tokens
     * receive the identical singly-penalized value (torch scatter_). */
    if (capped > 0) {
        int ridx[S2PS_MAX_WINDOW];
        float rval[S2PS_MAX_WINDOW];
        int rn = 0;
        for (int j = 0; j < capped; j++) {
            const int64_t tok = sp->prev[j];
            int ci;
            if (tok >= S2P_TOK_SEMANTIC_START && tok <= S2P_TOK_SEMANTIC_END)
                ci = (int)(tok - S2P_TOK_SEMANTIC_START);
            else if (tok == S2P_TOK_EOS)
                ci = S2PS_CAND_EOS; /* unreachable: EOS never enters history */
            else
                continue;
            const float v = cand[ci];
            ridx[rn] = ci;
            rval[rn] = v < 0.f ? v * sp->rep_penalty : v / sp->rep_penalty;
            rn++;
        }
        for (int j = 0; j < rn; j++) cand[ridx[j]] = rval[j];
    }

    /* top-k 30: descending value, ascending index on ties (torch.topk). */
    float tv[S2PS_TOP_K];
    int ti[S2PS_TOP_K];
    int tn = 0;
    for (int i = 0; i < S2PS_N_CAND; i++) {
        const float v = cand[i];
        if (tn == S2PS_TOP_K && v <= tv[S2PS_TOP_K - 1]) continue;
        int p = tn < S2PS_TOP_K ? tn : S2PS_TOP_K - 1;
        while (p > 0 && v > tv[p - 1]) {
            tv[p] = tv[p - 1];
            ti[p] = ti[p - 1];
            p--;
        }
        tv[p] = v;
        ti[p] = i;
        if (tn < S2PS_TOP_K) tn++;
    }

    int choice = 0;
    if (!greedy) {
        /* top-p over the UN-temperatured softmax; rank 0 always survives.
         * cumsum includes the current element (torch.cumsum). */
        float e[S2PS_TOP_K], sum = 0.f;
        const float m = tv[0];
        for (int j = 0; j < tn; j++) {
            e[j] = expf(tv[j] - m);
            sum += e[j];
        }
        int masked[S2PS_TOP_K] = {0};
        float cum = 0.f;
        for (int j = 0; j < tn; j++) {
            cum += e[j] / sum;
            if (j > 0 && cum > topp) masked[j] = 1;
        }

        /* second softmax WITH temperature (clamp min 1e-5) */
        const float t = temp < 1e-5f ? 1e-5f : temp;
        float p2[S2PS_TOP_K], s2 = 0.f;
        const float m2 = tv[0] / t; /* rank 0 unmasked and maximal (t > 0) */
        for (int j = 0; j < tn; j++) {
            p2[j] = masked[j] ? 0.f : expf(tv[j] / t - m2);
            s2 += p2[j];
        }
        for (int j = 0; j < tn; j++) p2[j] /= s2;

        if (sp->seeded) {
            /* multinomial_with_seed: hash -> 24-bit uniform -> Gumbel-max.
             * int64 wrap == uint64 wrap; % 2^24 == low 24 bits (two's
             * complement, power-of-two divisor); upper clamp 1-1e-10 == 1.0f
             * in fp32 and u < 1 by construction, so only the lower clamp can
             * fire. positions = uncapped pre-draw step count. */
            const uint64_t step_seed =
                ((uint64_t)sp->seed31 * 19349663ULL) ^
                (sp->count * 73856093ULL);
            float best = -INFINITY;
            for (int j = 0; j < tn; j++) {
                const uint64_t hashed = (step_seed * 8589934591ULL) ^
                                        ((uint64_t)j * 479001599ULL);
                float u = (float)(uint32_t)(hashed & 0xFFFFFFULL) /
                          16777216.0f;
                if (u < 1e-10f) u = 1e-10f;
                const float g = -logf(-logf(u));
                const float pert = logf(p2[j] + 1e-10f) + g;
                if (pert > best) { /* first max wins (torch.argmax) */
                    best = pert;
                    choice = j;
                }
            }
        } else {
            /* torch.multinomial equivalent: inverse-CDF on a fresh uniform */
            const double r = xoshiro_double(sp->rng);
            double cum2 = 0.0;
            int lastnz = 0;
            choice = -1;
            for (int j = 0; j < tn; j++) {
                if (p2[j] <= 0.f) continue;
                lastnz = j;
                cum2 += (double)p2[j];
                if (r < cum2) {
                    choice = j;
                    break;
                }
            }
            if (choice < 0) choice = lastnz;
        }
    }

    const int ci = ti[choice];
    const int64_t tok = cand_to_vocab(ci);
    if (tok == S2P_TOK_EOS) {
        /* EOS frame: forced sem_id 0, no history/step update. */
        *out_sem_id = 0;
        *out_eos = 1;
        return tok;
    }
    *out_sem_id = ci;
    *out_eos = 0;
    if (sp->count < (uint64_t)sp->window) {
        sp->prev[sp->count] = tok;
    } else {
        memmove(sp->prev, sp->prev + 1,
                (size_t)(sp->window - 1) * sizeof(int64_t));
        sp->prev[sp->window - 1] = tok;
    }
    sp->count++;
    return tok;
}
