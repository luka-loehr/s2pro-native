#!/usr/bin/env python3
"""Draft-model corpus generator over the workflow-generated text set.

Sequential requests against the local server (batch 1, so the EOS
sentinel in S2P_DUMP_FRAMES cuts clean per-take sequences). The server
decides the dump file via its environment; run once against the
training server (corpus.bin) with no argument, and once against a
holdout server (holdout.bin) with --holdout — the holdout texts never
appear in training generation, so eval measures unseen-text behavior.

Grid: every training text x N seeds, voice deterministic per (text,
seed) index over the full registry — no RNG state, reproducible.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from draft_texts import TRAIN_SHORT, TRAIN_LONG, HOLDOUT  # noqa: E402

PORT = 8011
SEEDS_SHORT = (7, 23)
SEEDS_LONG = (7, 23, 51)
SEEDS_HOLDOUT = (13, 37)


def voices():
    d = json.load(urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/v1/voices", timeout=60))
    lst = d["voices"] if isinstance(d, dict) and "voices" in d else d
    names = [v["name"] if isinstance(v, dict) else v for v in lst]
    if not names:
        raise SystemExit("no voices registered")
    return sorted(names)


def gen(text, voice, seed):
    body = json.dumps({"text": text, "voice": voice, "format": "wav",
                       "seed": seed, "stream": False},
                      ensure_ascii=False).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/tts", data=body,
        headers={"Content-Type": "application/json"})
    data = urllib.request.urlopen(req, timeout=1800).read()
    return (len(data) - 44) / 2 / 44100


def main():
    holdout = "--holdout" in sys.argv
    vs = voices()
    jobs = []
    if holdout:
        for ti, t in enumerate(HOLDOUT):
            for si, s in enumerate(SEEDS_HOLDOUT):
                jobs.append((t, vs[(ti * 7 + si * 13) % len(vs)],
                             s + 1000 * ti))
    else:
        for ti, t in enumerate(TRAIN_SHORT):
            for si, s in enumerate(SEEDS_SHORT):
                jobs.append((t, vs[(ti * 7 + si * 13) % len(vs)],
                             s + 1000 * ti))
        for ti, t in enumerate(TRAIN_LONG):
            for si, s in enumerate(SEEDS_LONG):
                jobs.append((t, vs[(ti * 11 + si * 5) % len(vs)],
                             s + 1000 * (ti + 5000)))
    tag = "holdout" if holdout else "train"
    print(f"[corpus] {tag}: {len(jobs)} takes over {len(vs)} voices",
          flush=True)
    total, n, fails, t0 = 0.0, 0, 0, time.time()
    for text, v, s in jobs:
        try:
            a = gen(text, v, s)
            total += a
            n += 1
            fails = 0
        except Exception as e:
            fails += 1
            print(f"[corpus] FAIL {v} s{s}: {e}", flush=True)
            if fails >= 5:
                print("[corpus] ABORT: 5 consecutive failures", flush=True)
                break
            time.sleep(5)
            continue
        if n % 50 == 0:
            print(f"[corpus-mile] {tag} {n}/{len(jobs)} takes, "
                  f"{total / 60:.1f} min audio, "
                  f"wall {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[corpus] DONE {tag} {n} takes, {total / 60:.1f} min audio, "
          f"wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
