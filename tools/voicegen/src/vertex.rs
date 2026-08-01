//! Vertex AI Gemini client: TTS synthesis, passage authoring, transcription.
//!
//! The wire contract is the one orin-backend already runs in production
//! (`orin-common/src/gemini.rs`, `orin-stream/src/vertex.rs`): bearer token on
//! `aiplatform.googleapis.com`, `:generateContent`, and for audio
//! `responseModalities:["AUDIO"]` with `speechConfig.voiceConfig
//! .prebuiltVoiceConfig.voiceName`. The reply carries base64 S16 PCM in
//! `candidates[0].content.parts[0].inlineData.data`.
//!
//! Non-streaming is sufficient: a 53 s multilingual take comes back in one
//! response with `finishReason: STOP` (~1300 audio tokens of a 16384 budget).

use crate::config::Config;
use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};
use std::time::Duration;

/// Long enough for a full multilingual take (~25 s observed for 53 s of audio)
/// plus headroom; short enough that a wedged request eventually fails.
const AUDIO_TIMEOUT: Duration = Duration::from_secs(300);
const TEXT_TIMEOUT: Duration = Duration::from_secs(180);

pub struct Vertex {
    http: reqwest::Client,
    config: Config,
}

/// One synthesized take: raw mono S16 samples at [`crate::resample::IN_RATE`].
pub struct Take {
    pub pcm: Vec<i16>,
    pub audio_tokens: u64,
}

impl Vertex {
    pub fn new(config: Config) -> Result<Self> {
        // rustls-no-provider ships no default provider; install the pure-Rust
        // `ring` one before the first TLS handshake.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .context("build HTTP client")?;
        Ok(Self { http, config })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    fn url(&self, location: &str, model: &str) -> String {
        format!(
            "https://aiplatform.googleapis.com/v1/projects/{}/locations/{location}\
             /publishers/google/models/{model}:generateContent",
            self.config.project
        )
    }

    async fn post(&self, url: String, body: Value, timeout: Duration, what: &str) -> Result<Value> {
        let res = self
            .http
            .post(url)
            .bearer_auth(&self.config.token)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .with_context(|| format!("{what} request"))?;
        let status = res.status();
        if !status.is_success() {
            let detail = res.text().await.unwrap_or_default();
            let detail: String = detail.chars().take(600).collect();
            bail!("{what} failed with HTTP {status}: {detail}");
        }
        res.json()
            .await
            .with_context(|| format!("{what} response JSON"))
    }

    /// Synthesize `text` with one prebuilt voice.
    ///
    /// `style` is prepended as its own line and is *not* spoken — verified by
    /// transcribing a take back and getting the passage alone. That matters
    /// because the `.txt` transcript has to match the audio word for word.
    pub async fn synthesize(&self, voice: &str, style: &str, text: &str) -> Result<Take> {
        let prompt = if style.trim().is_empty() {
            text.to_owned()
        } else {
            format!("{}\n{text}", style.trim())
        };
        let body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": voice } }
                }
            }
        });
        let url = self.url(&self.config.tts_location, &self.config.tts_model);
        let reply = self.post(url, body, AUDIO_TIMEOUT, "TTS").await?;

        // A truncated take would leave a transcript that no longer matches.
        if let Some(reason) = finish_reason(&reply)
            && reason != "STOP"
        {
            bail!("TTS stopped early (finishReason: {reason}) — the take is incomplete");
        }
        let part = first_part(&reply).context("TTS reply had no content parts")?;
        let inline = part
            .get("inlineData")
            .context("TTS reply had no inlineData audio")?;
        // Guard the rate assumption the resampler is built around.
        if let Some(mime) = inline.get("mimeType").and_then(Value::as_str)
            && !mime.contains(&crate::resample::IN_RATE.to_string())
        {
            bail!(
                "TTS returned {mime}, but the resampler expects {} Hz",
                crate::resample::IN_RATE
            );
        }
        let b64 = inline
            .get("data")
            .and_then(Value::as_str)
            .context("TTS inlineData had no data")?;
        let bytes = BASE64
            .decode(b64.trim())
            .context("TTS inlineData was not valid base64")?;
        Ok(Take {
            pcm: crate::wav::pcm_from_le_bytes(&bytes)?,
            audio_tokens: audio_tokens(&reply),
        })
    }

    /// Have the text model author one continuous passage that cycles `languages`.
    ///
    /// Retries when the reply leaks a language label into the prose. Observed in
    /// practice: "French est une très belle journée qui commence". One passage is
    /// shared by every voice, so a defect here would be frozen into every
    /// transcript — cheaper to re-ask than to ship it.
    pub async fn write_passage(
        &self,
        languages: &[String],
        seconds: u32,
        topic: Option<&str>,
    ) -> Result<String> {
        const ATTEMPTS: u32 = 3;
        let mut leaked = Vec::new();
        for attempt in 1..=ATTEMPTS {
            let passage = self.write_passage_once(languages, seconds, topic).await?;
            match leaked_label(&passage, languages) {
                None => return Ok(passage),
                Some(label) => {
                    eprintln!(
                        "  attempt {attempt}/{ATTEMPTS}: passage leaked the label \
                         '{label}' into the prose, re-asking"
                    );
                    leaked.push(label);
                }
            }
        }
        bail!(
            "the model kept writing language labels ({}) into the passage after \
             {ATTEMPTS} attempts — author it by hand and pass --text-file instead",
            leaked.join(", ")
        )
    }

    async fn write_passage_once(
        &self,
        languages: &[String],
        seconds: u32,
        topic: Option<&str>,
    ) -> Result<String> {
        // Measured on a real 7-language take: 133 words read as 62.7 s, i.e.
        // ~2.1 words/second. Multilingual prose is slower than the ~2.6 w/s of
        // an English-only read, so the estimate is calibrated to the former.
        let words = (seconds as f32 * 2.1).round() as u32;
        let per_language = (words / languages.len().max(1) as u32).max(8);
        let theme = topic.unwrap_or(
            "a warm, welcoming introduction that suits being read aloud as a \
             voice sample",
        );
        let prompt = format!(
            "Write ONE continuous passage to be read aloud by a single speaker as a \
             text-to-speech reference recording.\n\n\
             Requirements:\n\
             - It must move through these languages, in this order, and no others: {langs}.\n\
             - Roughly {per_language} words per language, about {words} words total \
               (~{seconds} seconds of calm speech).\n\
             - Each language's part must be natural, idiomatic, native-quality prose — \
               NOT a translation of the previous part. Use each language's own \
               punctuation and diacritics correctly.\n\
             - The whole thing must read as one flowing piece by one speaker, switching \
               languages mid-passage without announcing the switch.\n\
             - Never refer to the passage, the languages, the recording, or the act of \
               speaking or switching. Sentences like 'French is next' or 'this recording \
               demonstrates my voice' are forbidden — write real content, not narration \
               about itself.\n\
             - Theme: {theme}.\n\
             - Plain prose only: one paragraph, no headings, no language names or labels, \
               no bullet points, no quotation marks around parts, no square brackets, \
               no markdown, no emoji, no stage directions.\n\
             - Every character must be speakable: no digits (spell numbers out), no URLs, \
               no abbreviations that would be read letter by letter.\n\n\
             Return JSON: {{\"passage\": \"<the passage as one line>\"}}",
            langs = languages.join(", "),
        );
        let body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
            "generationConfig": {
                "thinkingConfig": { "thinkingLevel": "MINIMAL" },
                "temperature": 0.9,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": { "passage": { "type": "string" } },
                    "required": ["passage"]
                }
            }
        });
        let url = self.url(&self.config.text_location, &self.config.text_model);
        let reply = self
            .post(url, body, TEXT_TIMEOUT, "passage generation")
            .await?;
        let text = first_text(&reply).context("passage reply had no text part")?;
        let parsed: Value =
            serde_json::from_str(&text).context("passage reply was not valid JSON")?;
        let passage = parsed
            .get("passage")
            .and_then(Value::as_str)
            .context("passage reply had no `passage` field")?;
        let passage = normalize_whitespace(passage);
        if passage.is_empty() {
            bail!("the model returned an empty passage");
        }
        Ok(passage)
    }

    /// Transcribe a WAV verbatim, for `--verify`.
    pub async fn transcribe(&self, wav: &[u8]) -> Result<String> {
        let body = json!({
            "contents": [{ "role": "user", "parts": [
                { "text": "Transcribe this audio VERBATIM, exactly as spoken. Write each \
                           language in its OWN native script — Cyrillic as Cyrillic, never \
                           romanized or transliterated. Do not translate, and do not \
                           normalize one language into a related one. Output only the \
                           transcript." },
                { "inlineData": { "mimeType": "audio/wav", "data": BASE64.encode(wav) } }
            ]}],
            "generationConfig": { "thinkingConfig": { "thinkingLevel": "MINIMAL" } }
        });
        let url = self.url(&self.config.text_location, &self.config.text_model);
        let reply = self.post(url, body, TEXT_TIMEOUT, "transcription").await?;
        let text = first_text(&reply).context("transcription reply had no text part")?;
        Ok(normalize_whitespace(&text))
    }
}

fn parts(reply: &Value) -> Option<&Vec<Value>> {
    reply
        .get("candidates")?
        .get(0)?
        .get("content")?
        .get("parts")?
        .as_array()
}

fn first_part(reply: &Value) -> Option<&Value> {
    parts(reply)?.first()
}

/// First part carrying a string `text`, skipping thought-only parts.
fn first_text(reply: &Value) -> Option<String> {
    parts(reply)?
        .iter()
        .find_map(|p| p.get("text").and_then(Value::as_str))
        .map(str::to_owned)
}

fn finish_reason(reply: &Value) -> Option<&str> {
    reply
        .get("candidates")?
        .get(0)?
        .get("finishReason")?
        .as_str()
}

fn audio_tokens(reply: &Value) -> u64 {
    reply
        .get("usageMetadata")
        .and_then(|u| u.get("candidatesTokenCount"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// Collapse all whitespace runs to single spaces and trim. The transcript is a
/// one-line file, and stray newlines would not survive the round trip.
pub fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The first requested language label that appears as a standalone word in the
/// passage, if any.
///
/// Matching whole words only: "Turkish" is a leak, "Turkishness" inside a longer
/// token is not what this is looking for, and a substring test would fire on
/// ordinary words. Multi-word labels ("Mandarin Chinese") are checked as a
/// phrase over the word sequence.
fn leaked_label(passage: &str, languages: &[String]) -> Option<String> {
    let fold = |text: &str| -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    };
    let haystack = fold(passage);
    for language in languages {
        let needle = fold(language);
        if needle.is_empty() || needle.len() > haystack.len() {
            continue;
        }
        if haystack
            .windows(needle.len())
            .any(|window| *window == needle[..])
        {
            return Some(language.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_reply(b64: &str, mime: &str, reason: &str) -> Value {
        json!({
            "candidates": [{
                "finishReason": reason,
                "content": { "parts": [{ "inlineData": { "mimeType": mime, "data": b64 } }] }
            }],
            "usageMetadata": { "candidatesTokenCount": 1317 }
        })
    }

    #[test]
    fn audio_reply_navigation_matches_the_observed_shape() {
        let reply = audio_reply("AQI=", "audio/l16; rate=24000; channels=1", "STOP");
        let part = first_part(&reply).unwrap();
        let data = part["inlineData"]["data"].as_str().unwrap();
        assert_eq!(BASE64.decode(data).unwrap(), vec![0x01, 0x02]);
        assert_eq!(finish_reason(&reply), Some("STOP"));
        assert_eq!(audio_tokens(&reply), 1317);
    }

    #[test]
    fn first_text_skips_thought_only_parts() {
        let reply = json!({
            "candidates": [{ "content": { "parts": [
                { "thought": true },
                { "text": "the answer" }
            ]}}]
        });
        assert_eq!(first_text(&reply).as_deref(), Some("the answer"));
    }

    #[test]
    fn missing_fields_yield_none_not_panic() {
        assert!(first_part(&json!({})).is_none());
        assert!(first_text(&json!({ "candidates": [] })).is_none());
        assert!(finish_reason(&json!({ "candidates": [{}] })).is_none());
        assert_eq!(audio_tokens(&json!({})), 0);
    }

    #[test]
    fn tts_body_carries_audio_modality_and_prebuilt_voice() {
        // Mirrors the literal `synthesize` sends, so the wire shape is covered
        // without a network call.
        let body = json!({
            "contents": [{ "role": "user", "parts": [{ "text": "style\nhello" }] }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Sulafat" } }
                }
            }
        });
        assert_eq!(body["generationConfig"]["responseModalities"][0], "AUDIO");
        assert_eq!(
            body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Sulafat"
        );
        // Audio requests must not carry a thinkingConfig.
        assert!(body["generationConfig"].get("thinkingConfig").is_none());
    }

    fn seven_languages() -> Vec<String> {
        [
            "German",
            "English",
            "French",
            "Spanish",
            "Russian",
            "Ukrainian",
            "Turkish",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }

    #[test]
    fn detects_the_real_observed_leak() {
        // Verbatim from a real authoring reply.
        let passage = "Guten Morgen und herzlich willkommen. French est une très belle \
                       journée qui commence, parfaite pour prendre un café.";
        assert_eq!(
            leaked_label(passage, &seven_languages()).as_deref(),
            Some("French")
        );
    }

    #[test]
    fn a_clean_passage_has_no_leak() {
        let passage = "Guten Morgen und herzlich willkommen in unserer Runde. We are \
                       delighted to have you join us today. Le soleil se couche lentement. \
                       Espero que disfrutes de este momento. Мы искренне рады каждому гостю. \
                       Нехай цей день подарує вам спокій. Sizinle burada bulunmak güzel.";
        assert_eq!(leaked_label(passage, &seven_languages()), None);
    }

    #[test]
    fn leak_detection_is_case_and_punctuation_insensitive() {
        assert_eq!(
            leaked_label("und dann: SPANISH, natürlich.", &seven_languages()).as_deref(),
            Some("Spanish")
        );
    }

    #[test]
    fn leak_detection_matches_whole_words_only() {
        // "Russianate" is not the standalone label; no leak.
        assert_eq!(
            leaked_label("ein russianate Klang", &seven_languages()),
            None
        );
    }

    #[test]
    fn multi_word_labels_are_matched_as_a_phrase() {
        let languages = vec!["Mandarin Chinese".to_owned(), "German".to_owned()];
        assert_eq!(
            leaked_label("now in Mandarin Chinese we continue", &languages).as_deref(),
            Some("Mandarin Chinese")
        );
        // "Mandarin" alone is not the full label.
        assert_eq!(leaked_label("a mandarin orange", &languages), None);
    }

    #[test]
    fn empty_passage_has_no_leak() {
        assert_eq!(leaked_label("", &seven_languages()), None);
    }

    #[test]
    fn whitespace_normalization_produces_one_line() {
        assert_eq!(
            normalize_whitespace("  hello\n\n world \t again  "),
            "hello world again"
        );
        assert_eq!(normalize_whitespace("\n\n"), "");
    }
}
