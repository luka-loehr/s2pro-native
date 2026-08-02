#!/usr/bin/env python3
"""Apply a QAT patch (tools/qat_fastar.py output) to a checkpoint copy.

Copies non-safetensors files, rewrites every shard with patched
audio_decoder.* tensors substituted, keeps all other tensors and the shard
layout (index json stays valid: same tensor->shard mapping).

  python3 apply_qat_patch.py --model /model --patch qat_patch.safetensors \
      --out /out/model-qat
"""
import argparse
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    patch = {}
    with safe_open(args.patch, framework="pt") as sf:
        for k in sf.keys():
            patch[k] = sf.get_tensor(k)
    print(f"[patch] {len(patch)} tensors to substitute", flush=True)

    os.makedirs(args.out, exist_ok=True)
    used = set()
    for name in sorted(os.listdir(args.model)):
        src = os.path.join(args.model, name)
        dst = os.path.join(args.out, name)
        if not os.path.isfile(src):
            continue
        if not name.endswith(".safetensors"):
            shutil.copy2(src, dst)
            continue
        tensors = {}
        n_sub = 0
        with safe_open(src, framework="pt") as sf:
            meta = sf.metadata()
            for k in sf.keys():
                if k in patch:
                    t = sf.get_tensor(k)
                    p = patch[k]
                    assert t.shape == p.shape and t.dtype == p.dtype, \
                        f"{k}: {t.shape}/{t.dtype} vs {p.shape}/{p.dtype}"
                    tensors[k] = p
                    used.add(k)
                    n_sub += 1
                else:
                    tensors[k] = sf.get_tensor(k)
        save_file(tensors, dst, metadata=meta)
        print(f"[patch] {name}: {n_sub} substituted, "
              f"{len(tensors) - n_sub} kept", flush=True)
    missing = set(patch) - used
    if missing:
        raise SystemExit(f"[patch] ERROR: {len(missing)} patch tensors not "
                         f"found in checkpoint: {sorted(missing)[:4]}...")
    print("[patch] done", flush=True)


if __name__ == "__main__":
    main()
