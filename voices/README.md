# Voice registry

One voice = `<name>.wav` (RIFF, 16-bit PCM, mono, 44100 Hz) + `<name>.txt`
(the exact transcript). The server loads and DAC-encodes every pair at
startup; select with `"voice": "<name>"` on `POST /v1/tts`.

References must come from an accent-free multilingual source, cycling all
target languages in one take — see [docs/VOICES.md](../docs/VOICES.md) for
the why and how. The shipped samples (`neutral-female`, `young-male`,
`deeper-male`) each speak the same seven-language passage (DE, EN, FR, ES,
RU, UK, TR); their shared transcript is in the `.txt` files.
