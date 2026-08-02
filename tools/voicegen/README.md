# voicegen

Offline asset generator for the voice registry: synthesizes reference
voices with Google Gemini TTS on Vertex AI and writes exactly what the
server consumes — `voices/<name>.wav` (RIFF, mono, 16-bit, 44.1 kHz) plus
`voices/<name>.txt` (the verbatim transcript). It is never linked into
the C/CUDA runtime and never invoked by the Makefile; run it by hand when
the roster changes. Why references must be one continuous multilingual
take is specified in [docs/VOICES.md](../../docs/VOICES.md).

## 1. Prerequisites

A gcloud login with Vertex AI access — the tool itself stores nothing:

```sh
gcloud auth login
gcloud config set project <your-project>
```

(Or set `GOOGLE_CLOUD_PROJECT` and `S2P_GEMINI_ACCESS_TOKEN` for CI /
workload-identity setups.)

## 2. Usage

```sh
cd tools/voicegen

# the usual workflow: author a ~50 s passage cycling seven languages,
# synthesize three specific voices, verify each take
# (voices/ ships empty — this is how it is populated)
cargo run -- --languages de,en,fr,es,ru,uk,tr \
    --voice sulafat,charon,gacrux --verify --out-dir ../../voices

# exact passage, all 30 prebuilt voices, 4 at a time
cargo run -- --text-file passage.txt

# inspect the roster / plan without spending tokens
cargo run -- --list-voices
cargo run -- --languages de,en --voice puck --dry-run
```

For a whole set, author the passage once and reuse it verbatim, so all 30
voices share a byte-identical transcript and the passage can be reviewed
before any TTS call is spent:

```sh
cargo run -- --languages de,en,fr,es,ru,uk,tr --seconds 65 \
    --passage-out passage.txt --dry-run     # author + review
cargo run -- --text-file passage.txt --out-dir ../../voices \
    --concurrency 5 --verify                # then synthesize
```

## 3. Verification semantics

`--verify` transcribes every take back through the text model and scores
it against the passage (word-level LCS). Takes under 95 % match are
flagged — the `.txt` is injected into the generation prompt, so a
transcript that no longer describes its audio quietly degrades every
request that uses the voice.

Read a flag before acting on it: the score bounds the *transcriber's*
agreement, not the take's correctness. On a full 30-voice run over a
seven-language passage, 8 takes were flagged and all 8 were transcription
artifacts — the transcriber romanized the Russian block
(`мы искренне рады` → `mi iskrenie rade`) or blended the adjacent Russian
and Ukrainian blocks (`мы искренне рады` → `ми щиро раді`). The tell is
word count: those takes came back 124 words against a 124-word reference,
with every difference inside the Cyrillic region. A genuinely broken take
is materially shorter.

Truncated takes are caught separately and reliably by `finishReason`
rather than by scoring: in that same run one voice returned `OTHER`
mid-generation and was rejected outright with no files written. Rerun to
fill the gap.

## 4. Notes

- Gemini TTS returns 24 kHz PCM; the tool resamples to the codec's native
  44.1 kHz with an exact polyphase 147/80 windowed-sinc bank
  (`resample.rs`), with a headroom guard — band-limited interpolation
  overshoots a 0 dBFS source, so takes are scaled rather than clipped.
  The band above 12 kHz stays empty — a property of the source,
  faithfully imitated by the model (see docs/VOICES.md).
- The prebuilt-voice roster has no list endpoint; `roster.rs` mirrors
  Google's documented set (30 voices) and is the only place to touch when
  Google adds one.
- Reruns skip existing pairs, so a partial failure is fixed by running
  the same command again.
- Model/location overrides: `S2P_TTS_MODEL`, `S2P_TTS_LOCATION`,
  `S2P_TEXT_MODEL`, `S2P_TEXT_LOCATION`.
