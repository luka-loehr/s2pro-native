#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert codec.pth into the flat FP32 native artifact — FULL variant.

Same bin/idx format as the historic decode-only converter, but KEEPS the
encoder conv stack (`encoder.*`) and the quantizer pre-module
(`quantizer.pre_module.*`) so the native voice-cloning encode path
(`s2p_dac_encode`) can activate. Weight-norm (legacy `weight_g`/`weight_v`
and parametrizations layouts) is folded into plain `weight` tensors; mask /
rope buffers are dropped. FP32 end-to-end (continuous audio has no INT8
cushion).

Run inside any torch-capable container:
  python3 convert_codec_full.py --model /model --out /out
"""
import argparse
import os

import torch


def fold_wn(g, v):
    dims = tuple(range(1, v.ndim))
    norm = v.float().pow(2).sum(dim=dims, keepdim=True).sqrt()
    return g.float() * v.float() / norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir containing codec.pth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(os.path.join(args.model, "codec.pth"),
                    map_location="cpu", weights_only=True)

    folded = {}
    keys = set(sd.keys())
    consumed = set()
    for k in list(keys):
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            v = sd[base + ".parametrizations.weight.original1"]
            folded[base + ".weight"] = fold_wn(sd[k], v)
            consumed.add(k)
            consumed.add(base + ".parametrizations.weight.original1")
        elif k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            folded[base + ".weight"] = fold_wn(sd[k], sd[base + ".weight_v"])
            consumed.add(k)
            consumed.add(base + ".weight_v")
    for k in keys:
        if k not in consumed:
            folded[k] = sd[k].float()

    def keep(n):
        if n.endswith("causal_mask") or n.endswith("freqs_cis"):
            return False
        return (n.startswith("decoder") or n.startswith("quantizer")
                or n.startswith("encoder"))

    binf = open(os.path.join(args.out, "codec.bin"), "wb")
    idx = open(os.path.join(args.out, "codec.idx"), "w")
    off = 0
    n_kept = n_enc = 0
    for name in sorted(folded):
        if not keep(name):
            continue
        t = folded[name].contiguous().float().cpu().numpy().astype("<f4")
        b = t.tobytes()
        binf.write(b)
        idx.write("%s %d %d %d %s\n"
                  % (name, off, len(b), t.ndim,
                     " ".join(str(x) for x in t.shape)))
        off += len(b)
        n_kept += 1
        if name.startswith("encoder") or "pre_module" in name:
            n_enc += 1
    binf.close()
    idx.close()
    print(f"wrote {off / 1e6:.1f} MB, {n_kept} tensors "
          f"({n_enc} encode-path) -> {args.out}")


if __name__ == "__main__":
    main()
