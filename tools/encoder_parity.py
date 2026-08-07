#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Encoder parity: native DAC encode vs the PyTorch codec on one wav.

Closes the last unvalidated numeric stage. The decode path has been
oracle-gated since the beginning (benchmarks/parity), but the encoder —
which turns reference audio into the VQ codes that pin every cloned
voice — was only ever judged by how the clone sounded.

Metric: per-codebook code agreement between the native encoder's output
(dumped by `s2p-test` with S2P_TEST_ENC_DUMP) and the reference codec's
`encode()` on the same samples. Codebook 0 (semantic, 4096 entries)
carries the identity and is the acceptance metric; residual codebooks
are reported for completeness — like the fast-AR's greedy cascade they
contain near-ties that flip harmlessly.

Usage (inside a torch container with the codec checkpoint mounted):
  python3 encoder_parity.py --codec /model/codec.pth --wav ref.wav \
      --native codes.bin [--json out.json]

`codes.bin` is the native dump: int32, cb-major [10][T].
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True, help="codec.pth (torch)")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--native", required=True, help="native codes.bin")
    ap.add_argument("--json")
    args = ap.parse_args()

    import numpy as np
    import torch
    import soundfile as sf

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if sr != 44100:
        print(f"[enc-parity] wav is {sr} Hz, need 44100", file=sys.stderr)
        return 2
    if audio.ndim > 1:
        audio = audio[:, 0]

    sd = torch.load(args.codec, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        from fish_speech.models.dac.modded_dac import DAC
    except ImportError:
        print("[enc-parity] reference package not importable; run inside "
              "the oracle container", file=sys.stderr)
        return 3
    model = DAC()
    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(audio)[None, :]
        ref = model.encode(x)
        ref_codes = (ref[0] if isinstance(ref, (tuple, list)) else ref)
        ref_codes = ref_codes[0].to(torch.int32).cpu().numpy()  # [10, T]

    raw = open(args.native, "rb").read()
    n = len(raw) // 4
    native = np.frombuffer(raw, dtype=np.int32)[: n]
    T = n // 10
    native = native[: 10 * T].reshape(10, T)

    Tc = min(T, ref_codes.shape[1])
    out = {"wav": args.wav, "frames_native": int(T),
           "frames_reference": int(ref_codes.shape[1]),
           "frames_compared": int(Tc), "per_codebook": []}
    for q in range(10):
        agree = float((native[q, :Tc] == ref_codes[q, :Tc]).mean())
        out["per_codebook"].append({"cb": q, "agreement": round(agree, 6)})
    out["semantic_agreement"] = out["per_codebook"][0]["agreement"]
    out["residual_mean_agreement"] = round(
        sum(c["agreement"] for c in out["per_codebook"][1:]) / 9, 6)

    print(json.dumps(out, indent=1))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
