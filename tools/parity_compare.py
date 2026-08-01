#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare s2p-parity outputs against the oracle fixture set.

Pure Python (no numpy).

  parity_compare.py FIXTURE_DIR WORK_DIR [--json OUT.json]

Metrics:
  - per-layer residual cosine + max abs diff (36 layers + final norm)
  - prefill/step1 logits: cosine over the full 155776 row, argmax agreement
  - semantic tokens: exact-match prefix length and total matches
  - frames: exact row matches, per-codebook agreement
  - codec (isolated): our DAC on the ORACLE frames vs fixture pcm
    (cos, SNR dB) — separates vocoder fidelity from AR divergence
  - end to end: our pcm vs fixture pcm (informative only once AR diverges)
"""
import json
import math
import os
import struct
import sys

from parity_prep import npy_values  # same directory


def read_raw(path, fmt):
    raw = open(path, "rb").read()
    n = len(raw) // struct.calcsize(fmt)
    return list(struct.unpack(f"<{n}{fmt}", raw))


def cos(a, b):
    n = min(len(a), len(b))
    dot = sa = sb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        sa += a[i] * a[i]
        sb += b[i] * b[i]
    return dot / math.sqrt(sa * sb) if sa > 0 and sb > 0 else 0.0


def max_abs_diff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def snr_db(ref, test):
    n = min(len(ref), len(test))
    sig = err = 0.0
    for i in range(n):
        sig += ref[i] * ref[i]
        d = ref[i] - test[i]
        err += d * d
    if err == 0.0:
        return float("inf")
    return 10.0 * math.log10(sig / err) if sig > 0 else float("-inf")


def argmax(v):
    bi, bv = 0, v[0]
    for i, x in enumerate(v):
        if x > bv:
            bi, bv = i, x
    return bi


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    fixdir, work = sys.argv[1], sys.argv[2]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]

    r = {"fixture_dir": fixdir, "work_dir": work, "layers": {}, "summary": {}}

    # --- backbone layers + final norm ---
    worst = (1.0, None)
    for name in [f"backbone_layer{i:02d}" for i in range(36)] + \
                ["backbone_final_norm"]:
        fx_path = os.path.join(fixdir, name + ".npy")
        our_path = os.path.join(work, name + ".f32")
        if not (os.path.exists(fx_path) and os.path.exists(our_path)):
            continue
        _, fx = npy_values(fx_path)
        ours = read_raw(our_path, "f")
        c = cos(fx, ours)
        r["layers"][name] = {"cos": round(c, 6),
                             "max_abs": round(max_abs_diff(fx, ours), 4)}
        if c < worst[0]:
            worst = (c, name)
    r["summary"]["worst_layer"] = {"name": worst[1], "cos": round(worst[0], 6)}

    # --- logits ---
    for fx_name, our_name in [("prefill_logits", "prefill_logits"),
                              ("step1_logits", "step1_logits")]:
        fx_path = os.path.join(fixdir, fx_name + ".npy")
        our_path = os.path.join(work, our_name + ".f32")
        if not (os.path.exists(fx_path) and os.path.exists(our_path)):
            continue
        _, fx = npy_values(fx_path)
        ours = read_raw(our_path, "f")
        r["summary"][fx_name] = {
            "cos": round(cos(fx, ours), 6),
            "argmax_fixture": argmax(fx),
            "argmax_ours": argmax(ours),
            "argmax_match": argmax(fx) == argmax(ours),
        }

    # --- fast-AR residual-step logits of the first frame ---
    fx_fa_path = os.path.join(fixdir, "fast_ar_step0_logits.npy")
    if os.path.exists(fx_fa_path):
        shape, fx_fa = npy_values(fx_fa_path)  # [9, 4096]
        steps = []
        for k in range(shape[0]):
            our_path = os.path.join(work, f"fast_ar_step0_cb{k + 1}.f32")
            if not os.path.exists(our_path):
                continue
            ours = read_raw(our_path, "f")
            fx = fx_fa[k * shape[1]:(k + 1) * shape[1]]
            steps.append({"cb": k + 1, "cos": round(cos(fx, ours), 6),
                          "argmax_fixture": argmax(fx),
                          "argmax_ours": argmax(ours),
                          "argmax_match": argmax(fx) == argmax(ours)})
        if steps:
            r["summary"]["fast_ar_step0"] = steps

    # --- semantic tokens ---
    fx_sem_path = os.path.join(fixdir, "semantic_tokens.npy")
    our_sem_path = os.path.join(work, "our_sem.i64")
    if os.path.exists(fx_sem_path) and os.path.exists(our_sem_path):
        _, fx_sem = npy_values(fx_sem_path)
        our_sem = read_raw(our_sem_path, "q")
        n = min(len(fx_sem), len(our_sem))
        prefix = 0
        while prefix < n and int(fx_sem[prefix]) == int(our_sem[prefix]):
            prefix += 1
        total = sum(1 for i in range(n)
                    if int(fx_sem[i]) == int(our_sem[i]))
        r["summary"]["semantic_tokens"] = {
            "fixture": len(fx_sem), "ours": len(our_sem),
            "match_prefix": prefix, "match_total": total,
            "first_divergence": None if prefix == n else
            {"step": prefix, "fixture": int(fx_sem[prefix]),
             "ours": int(our_sem[prefix])},
        }

    # --- frames ---
    fx_fr_path = os.path.join(fixdir, "frames.npy")
    our_fr_path = os.path.join(work, "our_frames.i32")
    if os.path.exists(fx_fr_path) and os.path.exists(our_fr_path):
        shape, fx_fr = npy_values(fx_fr_path)
        our_fr = read_raw(our_fr_path, "i")
        T = min(shape[0], len(our_fr) // 10)
        rows = sum(1 for t in range(T)
                   if all(int(fx_fr[t * 10 + c]) == our_fr[t * 10 + c]
                          for c in range(10)))
        per_cb = [sum(1 for t in range(T)
                      if int(fx_fr[t * 10 + c]) == our_fr[t * 10 + c])
                  for c in range(10)]
        r["summary"]["frames"] = {"rows_compared": T, "rows_exact": rows,
                                  "per_codebook_matches": per_cb}

    # --- codec isolated: our DAC on oracle frames vs fixture pcm ---
    fx_pcm_path = os.path.join(fixdir, "pcm.npy")
    iso_path = os.path.join(work, "fixture_pcm_ours.f32")
    if os.path.exists(fx_pcm_path) and os.path.exists(iso_path):
        _, fx_pcm = npy_values(fx_pcm_path)
        iso = read_raw(iso_path, "f")
        r["summary"]["codec_isolated"] = {
            "cos": round(cos(fx_pcm, iso), 6),
            "snr_db": round(snr_db(fx_pcm, iso), 2),
            "n_fixture": len(fx_pcm), "n_ours": len(iso),
        }

    # --- end to end pcm (informative) ---
    our_pcm_path = os.path.join(work, "our_pcm.f32")
    if os.path.exists(fx_pcm_path) and os.path.exists(our_pcm_path):
        _, fx_pcm = npy_values(fx_pcm_path)
        ours = read_raw(our_pcm_path, "f")
        r["summary"]["pcm_end_to_end"] = {
            "cos": round(cos(fx_pcm, ours), 6),
            "n_fixture": len(fx_pcm), "n_ours": len(ours),
        }

    text = json.dumps(r, indent=1)
    print(text)
    if out_json:
        open(out_json, "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
