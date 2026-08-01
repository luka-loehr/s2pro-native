# Voices: the reference system

s2pro-native pins the speaker identity of every generation through
**reference audio** (voice cloning). This document explains how references
work, why they must come from a specific kind of source, and how to add
your own.

## How a voice works

S2-Pro conditions generation on a reference clip plus its exact transcript:
the clip is encoded to 21.5 Hz RVQ codes and injected into the prompt, and
the model then speaks the request text "as this speaker". Without a
reference the model casts a random plausible speaker per request — natural,
but unpinned.

## The accent rule (read this before recording anything)

Cloning copies **everything** about how the reference speaker talks —
timbre, prosody, and *phonetic habits*. A monolingual reference therefore
transfers its accent onto every other language: an English-only reference
speaking German produces heavily English-accented German. This is inherent
to the S2 model family ([known upstream issue](https://github.com/fishaudio/fish-speech/issues/1263))
and is NOT steerable with inline `[tags]` (verified experimentally: prosody
tags like `[whispering]` work, accent instructions do nothing).

**Therefore: reference audio MUST be produced by a voice that speaks every
target language with native pronunciation — in ONE continuous recording
that actually cycles through all of them.**

In practice that means generating the reference with an AI voice built for
accent-free multilinguality (e.g. ElevenLabs Multilingual, or comparable
systems). One take, one identity, every language spoken natively — the
model then hears "this speaker is native in all of these" and switches
pronunciation per language instead of dragging an accent along. Note that
S2-family TTS itself does NOT qualify as a source: its own voices carry
their reference accent across languages, which is the problem being solved.

A single multilingual file beats per-language files: the identity stays
perfectly continuous across language switches, and the registry treats it
as one reference block anyway. Mixed-language request text then works in
ONE generation — no language detection, no per-language calls, no audio
stitching.

## Adding a voice (two files, done)

Drop a pair into the voices directory (`--voices-dir`, default `./voices`):

```
voices/
  my-voice.wav   RIFF wav, 16-bit PCM, mono, 44100 Hz, ~30-90 s
  my-voice.txt   the EXACT transcript of the wav (UTF-8, one line)
```

The name is the file basename. Convert anything to the required format
with, e.g.:

```sh
afconvert -f WAVE -d LEI16@44100 -c 1 input.mp3 voices/my-voice.wav   # macOS
ffmpeg -i input.mp3 -ar 44100 -ac 1 -sample_fmt s16 voices/my-voice.wav
```

Rules of thumb:

- 30–90 s total; cycle through every language your deployment serves.
- The transcript must match the audio word for word — it is part of the
  prompt, not metadata.
- Clean, dry audio; no music, no second speaker.
- Every voice is DAC-encoded once at server start and cached; voice
  selection costs nothing per request.

The three shipped sample voices (`neutral-female`, `young-male`,
`deeper-male`) were generated exactly this way: one ElevenLabs-Multilingual
take each, cycling German, English, French, Spanish, Russian, Ukrainian,
and Turkish, with the shared transcript in the matching `.txt`.

## Using voices over the API

```sh
# list
curl -s localhost:8010/v1/voices

# speak with a named voice (mixed-language text in ONE call)
curl -s -X POST localhost:8010/v1/tts \
  -d '{"text":"Das französische Wort bibliothèque bedeutet Bücherei.",
       "voice":"neutral-female","format":"wav"}' -o out.wav

# on-the-fly clone: attach a wav (<= 15 s, 44.1k mono s16) per request
curl -s -X POST localhost:8010/v1/tts \
  -d "{\"text\":\"Hallo!\",\"format\":\"wav\",
       \"reference_text\":\"<exact transcript>\",
       \"reference_audio_b64\":\"$(base64 -i ref.wav)\"}" -o out.wav
```

`reference_audio_b64` wins over `voice`; neither field selects zero-shot
(a random, unpinned speaker). The on-the-fly clone pays one DAC encode
(~1 s for 10 s of audio) on the scheduler thread per request; named voices
do not.
