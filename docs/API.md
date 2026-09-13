# HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /healthz` | Engine and scheduler statistics. |
| `GET /v1/voices` | Named-voice registry listing. |
| `POST /v1/tts` | Chunked streaming synthesis (WAV or raw PCM). |

`POST /v1/tts` accepts `{"text", "format": "wav"|"pcm", "temperature",
"top_p", "seed", "stream", "chunk_length", "chunk_sentences",
"chunk_gap_ms", "chunk_parallel", "voice", "reference_audio_b64",
"reference_text"}`; with `--token` set, requests require
`Authorization: Bearer <token>`.

```bash
curl -X POST localhost:8010/v1/tts -d '{"text":"Hello.","format":"wav"}' -o hello.wav
```

## Voice selection

`voice` selects a pre-encoded registry voice; `reference_audio_b64` +
`reference_text` clone a per-request wav (max 15 s) on the fly; neither
selects zero-shot (a random, unpinned speaker). Mixed-language text is
one generation; voices are multilingual by construction
([VOICES.md](VOICES.md)).

## Long-form chunking

Voiced requests are served as a chunk chain: the text splits at sentence
boundaries into chunks that close after `chunk_sentences` sentences
(default 2, env `S2P_CHUNK_SENTENCES`) or `chunk_length` bytes (default
300, env `S2P_CHUNK_BYTES`), whichever comes first, and each chunk
generates against the fresh voice reference, all audio in one response.
The measured reason: single-shot prosody flattens (punctuation pauses per
10 s bucket decay from ~0.5–1.0 s early to ~0–0.2 s after ~40 s at every
weight precision), while chunked generation holds the opening-quality
prosody across the whole take (2–5 pauses per bucket through 130+ s; the
same text runs ~26 % longer because the rushing is gone). Zero-shot
requests never chunk (each chunk would draw a new voice).

## Parallel chunks

The chunks of one request are generated concurrently rather than one
after another: up to `chunk_parallel` of them (default 4, max 8, env
`S2P_CHUNK_PARALLEL`; `1` restores the sequential chain) sit in the
lockstep batch at once, chunks after the first joining only once the
first has produced audio so time-to-first-audio is unchanged. The
lockstep scheduler reads the weight stream once per tick for every
session, so the extra chunks cost little memory-bus traffic while the
request's audio arrives much faster: on the DGX Spark a 58 s German take
drops from wall RTF 0.56 to 0.23 at the same 0.27 s first-audio latency.
Wire order and the join filter are unchanged: the chunk whose turn it is
streams live, later chunks buffer until their turn. Per-request cloning
stays sequential (chunk 1 memoizes the codes).

## Take-length guard

The model occasionally fails to emit its end token for a short chunk and
continues to its context bound, or ends a chunk before producing speech.
Every generation therefore carries a duration cap from its text (bytes /
15 per second, ×2, +3 s, at least 8 s). A chunk still buffered when it
overruns, or that ends nearly empty, is regenerated with a fresh seed (up
to 2×) before the listener reaches it; the chunk on the wire is cut at
the cap. Measured under 48 heavy concurrent requests: one cut, no crash.
`S2P_CHUNK_GUARD=0` disables it. The root cause is not yet known; the
guard bounds it.

## Join normalization

Boundary silence at chunk joins is trimmed on both sides and replaced by
exactly `chunk_gap_ms` of silence (default 1000, env `S2P_CHUNK_GAP_MS`;
`0` = raw concatenation). The model's own sentence pauses inside a chunk
are untouched; only the stitched joins get a deterministic,
runtime-tunable pause.

## Streaming semantics

`wav` streams a header followed by S16LE frames as they are generated;
playback can start while generation runs. A streamed WAV necessarily
advertises saturated RIFF sizes: the header is on the wire before the
first frame is sampled, and chunked transfer cannot rewrite sent bytes.
Tolerant players handle that; strict ones (Apple's, notably) refuse it.
`"stream": false` buffers server-side and responds with exact
`Content-Length` and RIFF sizes, a well-formed file at the cost of
time-to-first-audio.

## Deployment

The server binds loopback by default and does not terminate TLS. Public
deployments must place it behind an authenticated, rate-limited proxy.
Review [SECURITY.md](../SECURITY.md) before exposing the service.
