//! voicegen — generate s2pro-native reference voices with Gemini TTS on Vertex AI.
//!
//! Produces exactly what the voice registry consumes: `<voice>.wav` (RIFF, mono,
//! 16-bit, 44100 Hz) plus `<voice>.txt` (the verbatim transcript). See
//! tools/voicegen/README.md for usage and docs/VOICES.md for why references have
//! to be one continuous multilingual take.

mod config;
mod resample;
mod roster;
mod verify;
mod vertex;
mod wav;

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const USAGE: &str = "\
voicegen — generate s2pro-native reference voices with Gemini TTS (Vertex AI)

USAGE
  cargo run --manifest-path tools/voicegen/Cargo.toml -- [OPTIONS]

PASSAGE (pick exactly one)
  --text <STRING>        Use this exact passage verbatim.
  --text-file <PATH>     Read the passage from a file ('-' for stdin).
  --languages <LIST>     Have the text model author the passage; comma-separated,
                         e.g. 'de,en,fr,es,ru,uk,tr' or 'German,Swiss German'.

PASSAGE AUTHORING (with --languages)
  --seconds <N>          Target spoken length, default 50.
  --topic <STRING>       Theme for the passage. Default: a welcoming intro.

VOICES
  --voice <LIST>         Comma-separated voice ids. Default: all 30.
  --list-voices          Print the roster and exit.

OUTPUT
  --out-dir <PATH>       Default './voices'.
  --overwrite            Replace existing <voice>.wav / <voice>.txt pairs.
  --style <STRING>       Delivery instruction sent with each take (not spoken).
  --concurrency <N>      Parallel TTS requests, default 4.
  --verify               Transcribe each take and report how well it matches.
  --dry-run              Resolve the passage and print the plan; no TTS calls.

ENVIRONMENT (nothing is hardcoded; this repo is public)
  GOOGLE_CLOUD_PROJECT      else `gcloud config get-value project`
  S2P_GEMINI_ACCESS_TOKEN   else `gcloud auth print-access-token`
  S2P_TTS_MODEL             default gemini-3.1-flash-tts-preview
  S2P_TTS_LOCATION          default global
  S2P_TEXT_MODEL            default gemini-3.6-flash
  S2P_TEXT_LOCATION         default europe-west4
";

const DEFAULT_STYLE: &str = "Read the following passage aloud naturally, at a calm and steady \
     pace, with fully native pronunciation for every language it contains.";

/// Below this word-level match, a transcript probably no longer describes its
/// audio — which silently degrades every generation that uses the voice.
const VERIFY_WARN_BELOW: f32 = 0.95;

struct Args {
    text: Option<String>,
    text_file: Option<PathBuf>,
    languages: Option<Vec<String>>,
    seconds: u32,
    topic: Option<String>,
    voices: Option<Vec<String>>,
    out_dir: PathBuf,
    style: String,
    concurrency: usize,
    overwrite: bool,
    verify: bool,
    dry_run: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            text: None,
            text_file: None,
            languages: None,
            seconds: 50,
            topic: None,
            voices: None,
            out_dir: PathBuf::from("voices"),
            style: DEFAULT_STYLE.to_owned(),
            concurrency: 4,
            overwrite: false,
            verify: false,
            dry_run: false,
        }
    }
}

/// Hand-rolled parsing keeps the dependency list at five crates. Accepts both
/// `--flag value` and `--flag=value`.
fn parse_args() -> Result<Option<Args>> {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(1).peekable();

    while let Some(arg) = argv.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f.to_owned(), Some(v.to_owned())),
            None => (arg.clone(), None),
        };
        let mut value = |flag: &str| -> Result<String> {
            match inline.clone().or_else(|| argv.next()) {
                Some(v) => Ok(v),
                None => bail!("{flag} needs a value"),
            }
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--list-voices" => {
                println!("{} Gemini TTS prebuilt voices:\n", roster::VOICES.len());
                for (id, characteristic) in roster::VOICES {
                    println!("  {:<16} {:<15} -> {}.wav", id, characteristic, roster::basename(id));
                }
                return Ok(None);
            }
            "--text" => args.text = Some(value(&flag)?),
            "--text-file" => args.text_file = Some(PathBuf::from(value(&flag)?)),
            "--languages" => args.languages = Some(split_list(&value(&flag)?)),
            "--topic" => args.topic = Some(value(&flag)?),
            "--voice" | "--voices" => args.voices = Some(split_list(&value(&flag)?)),
            "--out-dir" => args.out_dir = PathBuf::from(value(&flag)?),
            "--style" => args.style = value(&flag)?,
            "--seconds" => {
                let raw = value(&flag)?;
                args.seconds = raw
                    .parse()
                    .with_context(|| format!("--seconds expects a number, got '{raw}'"))?;
                if args.seconds < 10 {
                    bail!("--seconds must be at least 10; docs/VOICES.md wants 30-90 s");
                }
            }
            "--concurrency" => {
                let raw = value(&flag)?;
                args.concurrency = raw
                    .parse()
                    .with_context(|| format!("--concurrency expects a number, got '{raw}'"))?;
                if args.concurrency == 0 {
                    bail!("--concurrency must be at least 1");
                }
            }
            "--overwrite" => args.overwrite = true,
            "--verify" => args.verify = true,
            "--dry-run" => args.dry_run = true,
            other => bail!("unknown option '{other}' (try --help)"),
        }
    }

    let sources = [
        args.text.is_some(),
        args.text_file.is_some(),
        args.languages.is_some(),
    ]
    .iter()
    .filter(|set| **set)
    .count();
    match sources {
        1 => Ok(Some(args)),
        0 => bail!("pick a passage source: --text, --text-file, or --languages (try --help)"),
        _ => bail!("--text, --text-file, and --languages are mutually exclusive"),
    }
}

fn split_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|part| part.trim().to_owned())
        .filter(|part| !part.is_empty())
        .collect()
}

/// Expand ISO 639-1 codes to English names so the authoring prompt is
/// unambiguous. Anything unrecognized passes through verbatim, which is what
/// lets `--languages 'Swiss German,Bavarian'` work.
fn language_name(token: &str) -> String {
    const CODES: [(&str, &str); 26] = [
        ("ar", "Arabic"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("de", "German"),
        ("el", "Greek"),
        ("en", "English"),
        ("es", "Spanish"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("he", "Hebrew"),
        ("hi", "Hindi"),
        ("hu", "Hungarian"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("nl", "Dutch"),
        ("no", "Norwegian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("sv", "Swedish"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian"),
        ("zh", "Mandarin Chinese"),
    ];
    CODES
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(token))
        .map(|(_, name)| (*name).to_owned())
        .unwrap_or_else(|| token.to_owned())
}

fn resolve_voices(requested: &Option<Vec<String>>) -> Result<Vec<&'static str>> {
    match requested {
        None => Ok(roster::VOICES.iter().map(|(id, _)| *id).collect()),
        Some(names) => {
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                let id = roster::canonical(name).with_context(|| {
                    format!("unknown voice '{name}' — run --list-voices for the roster")
                })?;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            if out.is_empty() {
                bail!("--voice matched no voices");
            }
            Ok(out)
        }
    }
}

fn read_passage_from_file(path: &Path) -> Result<String> {
    if path == Path::new("-") {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading the passage from stdin")?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("reading the passage from {}", path.display()))
    }
}

struct Outcome {
    seconds: f32,
    bytes: usize,
    audio_tokens: u64,
    match_ratio: Option<f32>,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("voicegen: {error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let Some(args) = parse_args()? else { return Ok(()) };
    let voices = resolve_voices(&args.voices)?;

    // The passage is authored/read once and shared by every voice: one identity
    // per take, one transcript, exactly as the three shipped samples were made.
    let needs_api = args.languages.is_some() || !args.dry_run;
    let vertex = if needs_api {
        Some(Arc::new(vertex::Vertex::new(config::Config::from_env()?)?))
    } else {
        None
    };

    let passage = if let Some(languages) = &args.languages {
        let names: Vec<String> = languages.iter().map(|l| language_name(l)).collect();
        eprintln!("Authoring a ~{} s passage in: {}", args.seconds, names.join(", "));
        let vertex = vertex.as_ref().expect("--languages implies an API client");
        vertex
            .write_passage(&names, args.seconds, args.topic.as_deref())
            .await?
    } else {
        let raw = match (&args.text, &args.text_file) {
            (Some(text), _) => text.clone(),
            (_, Some(path)) => read_passage_from_file(path)?,
            _ => unreachable!("parse_args guarantees one passage source"),
        };
        vertex::normalize_whitespace(&raw)
    };
    if passage.is_empty() {
        bail!("the passage is empty");
    }

    if let Some(vertex) = &vertex {
        let config = vertex.config();
        eprintln!(
            "Project {}  TTS {} @ {}  text {} @ {}",
            config.project,
            config.tts_model,
            config.tts_location,
            config.text_model,
            config.text_location
        );
    }
    eprintln!(
        "\nPassage ({} chars, {} words):\n{passage}\n",
        passage.chars().count(),
        passage.split_whitespace().count()
    );

    std::fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("creating {}", args.out_dir.display()))?;

    // Skip pairs that already exist unless asked to replace them, so a rerun
    // after a partial failure only fills the gaps.
    let mut pending = Vec::new();
    for voice in voices {
        let wav_path = args.out_dir.join(format!("{}.wav", roster::basename(voice)));
        if wav_path.exists() && !args.overwrite {
            eprintln!("skip  {voice}: {} exists (--overwrite to replace)", wav_path.display());
            continue;
        }
        pending.push(voice);
    }
    if pending.is_empty() {
        eprintln!("Nothing to generate.");
        return Ok(());
    }

    if args.dry_run {
        eprintln!("Would write {} pair(s) to {}:", pending.len(), args.out_dir.display());
        for voice in &pending {
            let base = roster::basename(voice);
            eprintln!("  {base}.wav + {base}.txt  ({voice}, {})", roster::characteristic(voice));
        }
        return Ok(());
    }

    let vertex = vertex.expect("a non-dry run implies an API client");
    eprintln!(
        "Generating {} voice(s), {} at a time...\n",
        pending.len(),
        args.concurrency
    );

    let mut outcomes: Vec<Outcome> = Vec::new();
    let mut failures: Vec<(String, String)> = Vec::new();

    // Bounded fan-out: chunk the roster rather than firing 30 requests at once.
    for chunk in pending.chunks(args.concurrency) {
        let mut set = tokio::task::JoinSet::new();
        for voice in chunk {
            let voice = *voice;
            let vertex = Arc::clone(&vertex);
            let passage = passage.clone();
            let style = args.style.clone();
            let out_dir = args.out_dir.clone();
            let verify = args.verify;
            set.spawn(async move {
                let result =
                    generate_one(&vertex, voice, &style, &passage, &out_dir, verify).await;
                (voice, result)
            });
        }
        while let Some(joined) = set.join_next().await {
            let (voice, result) = joined.context("a generation task panicked")?;
            match result {
                Ok(outcome) => {
                    let verified = match outcome.match_ratio {
                        Some(ratio) if ratio < VERIFY_WARN_BELOW => {
                            format!("  transcript match {:.1}% <-- CHECK THIS", ratio * 100.0)
                        }
                        Some(ratio) => format!("  transcript match {:.1}%", ratio * 100.0),
                        None => String::new(),
                    };
                    eprintln!(
                        "ok    {:<16} {:>6.1} s  {:>7.1} MiB  {} audio tokens{}",
                        voice,
                        outcome.seconds,
                        outcome.bytes as f64 / (1024.0 * 1024.0),
                        outcome.audio_tokens,
                        verified
                    );
                    outcomes.push(outcome);
                }
                Err(error) => {
                    eprintln!("FAIL  {voice}: {error:#}");
                    failures.push((voice.to_owned(), format!("{error:#}")));
                }
            }
        }
    }

    eprintln!("\n{} generated, {} failed.", outcomes.len(), failures.len());
    if let Some(shortest) = outcomes.iter().map(|o| o.seconds).fold(None, |acc: Option<f32>, s| {
        Some(acc.map_or(s, |a| a.min(s)))
    }) {
        let longest = outcomes.iter().map(|o| o.seconds).fold(0.0f32, f32::max);
        eprintln!("Duration {shortest:.1}-{longest:.1} s (docs/VOICES.md wants 30-90 s).");
        if shortest < 30.0 {
            eprintln!("Note: some takes are under 30 s — consider a longer passage.");
        }
    }
    let flagged = outcomes
        .iter()
        .filter(|o| o.match_ratio.is_some_and(|r| r < VERIFY_WARN_BELOW))
        .count();
    if flagged > 0 {
        eprintln!(
            "Warning: {flagged} take(s) diverge from the transcript. The .txt is part of \
             the prompt, not metadata — regenerate those voices before shipping them."
        );
    }
    if !failures.is_empty() {
        bail!("{} voice(s) failed; rerun to fill the gaps", failures.len());
    }
    Ok(())
}

/// Synthesize one voice and write its `.wav` + `.txt` pair.
async fn generate_one(
    vertex: &vertex::Vertex,
    voice: &'static str,
    style: &str,
    passage: &str,
    out_dir: &Path,
    verify: bool,
) -> Result<Outcome> {
    let take = vertex.synthesize(voice, style, passage).await?;
    if take.pcm.is_empty() {
        bail!("TTS returned no audio");
    }
    let pcm = resample::resample(&take.pcm);
    let bytes = wav::encode(&pcm, resample::OUT_RATE)?;
    let seconds = pcm.len() as f32 / resample::OUT_RATE as f32;

    let base = roster::basename(voice);
    // Transcript first: a .wav without its .txt is skipped by the registry with
    // a warning, which is a clearer failure than the reverse.
    let txt_path = out_dir.join(format!("{base}.txt"));
    std::fs::write(&txt_path, format!("{passage}\n"))
        .with_context(|| format!("writing {}", txt_path.display()))?;
    let wav_path = out_dir.join(format!("{base}.wav"));
    std::fs::write(&wav_path, &bytes)
        .with_context(|| format!("writing {}", wav_path.display()))?;

    let match_ratio = if verify {
        let transcript = vertex.transcribe(&bytes).await?;
        Some(verify::similarity(passage, &transcript))
    } else {
        None
    };

    Ok(Outcome {
        seconds,
        bytes: bytes.len(),
        audio_tokens: take.audio_tokens,
        match_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_codes_expand_and_unknowns_pass_through() {
        assert_eq!(language_name("de"), "German");
        assert_eq!(language_name("UK"), "Ukrainian");
        assert_eq!(language_name("Swiss German"), "Swiss German");
        assert_eq!(language_name("zh"), "Mandarin Chinese");
    }

    #[test]
    fn list_splitting_trims_and_drops_blanks() {
        assert_eq!(
            split_list(" de , en ,, fr "),
            vec!["de".to_owned(), "en".to_owned(), "fr".to_owned()]
        );
        assert!(split_list(" , ").is_empty());
    }

    #[test]
    fn voice_resolution_defaults_to_the_whole_roster() {
        assert_eq!(resolve_voices(&None).unwrap().len(), roster::VOICES.len());
    }

    #[test]
    fn voice_resolution_canonicalizes_and_dedupes() {
        let voices = resolve_voices(&Some(vec![
            "sulafat".to_owned(),
            "SULAFAT".to_owned(),
            "charon".to_owned(),
        ]))
        .unwrap();
        assert_eq!(voices, vec!["Sulafat", "Charon"]);
    }

    #[test]
    fn unknown_voice_is_rejected() {
        let error = resolve_voices(&Some(vec!["Nonexistent".to_owned()])).unwrap_err();
        assert!(error.to_string().contains("unknown voice"));
    }
}
