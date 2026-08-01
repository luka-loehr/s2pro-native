# Voice registry

One voice = `<name>.wav` (RIFF, 16-bit PCM, mono, 44100 Hz) + `<name>.txt`
(the exact transcript). The server loads and DAC-encodes every pair at
startup; select with `"voice": "<name>"` on `POST /v1/tts`.

References must come from an accent-free multilingual source, cycling all
target languages in one take — see [docs/VOICES.md](../docs/VOICES.md) for
the why and how.

## This directory ships empty — generate your voices

Reference audio is **not committed**: it is ~5.4 MB per voice (~160 MB for a
full set), and it is reproducible from one command, so the repository carries
the generator instead of the output. `voices/*.wav` and `voices/*.txt` are
gitignored.

Populate it with [`tools/voicegen`](../tools/voicegen/README.md), which
synthesizes any or all 30 Google Gemini TTS prebuilt voices — each one take
of the same multilingual passage, named after the voice that produced it
(`sulafat`, `charon`, `kore`, ...), so a set differs only in speaker identity:

```sh
cd tools/voicegen
cargo run -- --languages de,en,fr,es,ru,uk,tr --seconds 65 \
    --voice sulafat,charon,gacrux --out-dir ../../voices --verify
```

An empty or missing directory is not an error — the server logs it and serves
zero-shot only (a random, unpinned speaker per request).

**Generate only what you serve.** Every pair is DAC-encoded once before the
server accepts traffic, at roughly 1 s of encode per 10 s of reference audio,
so a full 30-voice set of ~60 s takes delays first traffic by a few minutes.
Three voices cost seconds. (The cached codes are tiny — ~1.5 MB for all 30 —
and voice selection stays free per request.)
