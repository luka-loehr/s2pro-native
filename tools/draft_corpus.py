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
SEEDS_PHASE2 = (51, 77, 101, 123)
# whole voices excluded from EVERY training take (deterministic picks
# from the sorted registry) — the eval's unseen-voice cell. Review
# finding: without held-out voices the reported acceptance conflates
# voice memorization with generalization.
HOLDOUT_VOICE_IX = (5, 15, 25)


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
    phase2 = "--phase2" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--port="):
            global PORT
            PORT = int(a.split("=")[1])
    vs_all = voices()
    held = [vs_all[i % len(vs_all)] for i in HOLDOUT_VOICE_IX]
    vs = [v for v in vs_all if v not in held]
    jobs = []
    if holdout:
        # three cells: unseen text x seen voice, unseen text x unseen
        # voice, seen text x unseen voice — the manifest records which.
        for ti, t in enumerate(HOLDOUT):
            for si, sd in enumerate(SEEDS_HOLDOUT):
                jobs.append((t, vs[(ti * 7 + si * 13) % len(vs)],
                             sd + 1000 * ti, "utext-svoice"))
        for ti, t in enumerate(HOLDOUT):
            jobs.append((t, held[ti % len(held)], 13 + 1000 * ti,
                         "utext-uvoice"))
        for ti in range(0, len(TRAIN_SHORT), 20):
            jobs.append((TRAIN_SHORT[ti], held[(ti // 20) % len(held)],
                         13 + 1000 * ti, "stext-uvoice"))
    elif phase2:
        for ti, t in enumerate(TRAIN_SHORT):
            for si, sd in enumerate(SEEDS_PHASE2):
                jobs.append((t, vs[(ti * 3 + si * 17) % len(vs)],
                             sd + 1000 * ti, "train2"))
        for ti, t in enumerate(TRAIN_LONG):
            for si, sd in enumerate(SEEDS_PHASE2):
                jobs.append((t, vs[(ti * 13 + si * 7) % len(vs)],
                             sd + 1000 * (ti + 5000), "train2"))
    else:
        for ti, t in enumerate(TRAIN_SHORT):
            for si, sd in enumerate(SEEDS_SHORT):
                jobs.append((t, vs[(ti * 7 + si * 13) % len(vs)],
                             sd + 1000 * ti, "train"))
        for ti, t in enumerate(TRAIN_LONG):
            for si, sd in enumerate(SEEDS_LONG):
                jobs.append((t, vs[(ti * 11 + si * 5) % len(vs)],
                             sd + 1000 * (ti + 5000), "train"))
    tag = "holdout" if holdout else ("train2" if phase2 else "train")
    mpath = None
    for a in sys.argv[1:]:
        if a.startswith("--manifest="):
            mpath = a.split("=", 1)[1]
    if mpath:
        json.dump([{"voice": v, "seed": sd, "cell": c,
                    "text": t[:60]} for t, v, sd, c in jobs],
                  open(mpath, "w"), ensure_ascii=False)
    print(f"[corpus] {tag}: {len(jobs)} takes, {len(vs)} train voices, "
          f"held-out voices: {', '.join(held)}", flush=True)
    total, n, fails, t0 = 0.0, 0, 0, time.time()
    for text, v, sd, _cell in jobs:
        try:
            a = gen(text, v, sd)
            total += a
            n += 1
            fails = 0
        except Exception as e:
            fails += 1
            print(f"[corpus] FAIL {v} s{sd}: {e}", flush=True)
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
