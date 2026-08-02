# The Voice Reference System

Guide. s2pro-native pins the speaker identity of every generation through
reference audio (voice cloning). This document specifies how references
work, the constraint that governs where they may come from, and how to add
one.

## 1. Mechanism

S2-Pro conditions generation on a reference clip plus its exact
transcript: the clip is encoded to 21.5 Hz RVQ codes and injected into the
prompt, and the model then speaks the request text as that speaker.
Without a reference, the model casts a random plausible speaker per
request — natural-sounding, but unpinned.

## 2. The accent constraint

Cloning copies everything about how the reference speaker talks — timbre,
prosody, and phonetic habits. A monolingual reference therefore transfers
its accent onto every other language: an English-only reference speaking
German produces heavily English-accented German. This is inherent to the
S2 model family
([known upstream issue](https://github.com/fishaudio/fish-speech/issues/1263))
and is not steerable with inline `[tags]` — verified experimentally:
prosody tags such as `[whispering]` work, accent instructions do nothing.

Reference audio must therefore be produced by a voice that speaks every
target language with native pronunciation, in one continuous recording
that cycles through all of them. In practice that means an AI voice built
for accent-free multilinguality (e.g. Google Gemini TTS, ElevenLabs
Multilingual, or comparable systems). S2-family TTS itself does not
qualify as a source: its own voices carry their reference accent across
languages, which is precisely the problem being solved.

A single multilingual file beats per-language files: the identity stays
continuous across language switches, and the registry treats it as one
reference block. Mixed-language request text is then a single generation —
no language detection, no per-language calls, no audio stitching.

## 3. Adding a voice

Drop a pair into the voices directory (`--voices-dir`, default
`./voices`):

```
voices/
  my-voice.wav   RIFF wav, 16-bit PCM, mono, 44100 Hz, ~30–90 s
  my-voice.txt   the EXACT transcript of the wav (UTF-8, one line)
```

The name is the file basename. The supported way to produce a pair is
[`tools/voicegen`](../tools/voicegen/README.md), which authors a
multilingual passage, synthesizes any of the 30 Gemini prebuilt voices,
resamples to 44.1 kHz, and verifies each take against its transcript. For
material from other sources, convert to the required format:

```sh
afconvert -f WAVE -d LEI16@44100 -c 1 input.mp3 voices/my-voice.wav   # macOS
ffmpeg -i input.mp3 -ar 44100 -ac 1 -sample_fmt s16 voices/my-voice.wav
```

Requirements:

- 30–90 s total; cycle through every language the deployment serves.
- The transcript must match the audio word for word — it is part of the
  prompt, not metadata.
- Clean, dry audio; no music, no second speaker.
- Every voice is DAC-encoded once at server start and cached; voice
  selection costs nothing per request.

`voices/` ships empty; reference audio is generated per deployment, not
committed (~5.4 MB per voice, reproducible from one command —
`voices/*.wav` and `voices/*.txt` are gitignored). An empty directory is
not an error: the server logs it and serves zero-shot only.

Note on bandwidth: Gemini TTS returns 24 kHz PCM, so those references
carry no energy above 12 kHz, and the model faithfully imitates the
band-limit. `voicegen` resamples to the codec's native 44.1 kHz (required
— the loader rejects other rates), but resampling cannot invent the
missing top octave; export at 44.1 kHz natively where a provider offers
it.

## 4. API usage

```sh
# list
curl -s localhost:8010/v1/voices

# speak with a named voice (mixed-language text in ONE call)
curl -s -X POST localhost:8010/v1/tts \
  -d '{"text":"Das französische Wort bibliothèque bedeutet Bücherei.",
       "voice":"sulafat","format":"wav"}' -o out.wav

# on-the-fly clone: attach a wav (<= 15 s, 44.1 kHz mono s16) per request
curl -s -X POST localhost:8010/v1/tts \
  -d "{\"text\":\"Hallo!\",\"format\":\"wav\",
       \"reference_text\":\"<exact transcript>\",
       \"reference_audio_b64\":\"$(base64 -i ref.wav)\"}" -o out.wav
```

`reference_audio_b64` wins over `voice`; neither field selects zero-shot.
The on-the-fly clone pays one DAC encode (~1 s per 10 s of audio) on the
scheduler thread per request; named voices do not.
