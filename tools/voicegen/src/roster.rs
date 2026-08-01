//! The Gemini TTS prebuilt-voice roster.
//!
//! There is no voices.list endpoint on the `aiplatform.googleapis.com` path —
//! the prebuilt set is fixed per model family and published only as
//! documentation, so it lives here as data. (Google Cloud's separate
//! `texttospeech.googleapis.com` product does expose voices.list, but it does
//! not enumerate Gemini prebuilt voices either.)
//!
//! Names and characteristic labels are Google's own, from
//! <https://ai.google.dev/gemini-api/docs/speech-generation>. If Google adds a
//! voice, add the row here — that is the only place the tool needs to change.

/// `(voice id, official characteristic label)`, in the order Google documents.
pub const VOICES: [(&str, &str); 30] = [
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
];

/// Resolve a user-supplied voice name to its canonical spelling, accepting any
/// case (so `--voice sulafat` works and still writes `sulafat.wav`).
pub fn canonical(name: &str) -> Option<&'static str> {
    VOICES
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(name.trim()))
        .map(|(id, _)| *id)
}

/// The characteristic label for a canonical voice id.
pub fn characteristic(id: &str) -> &'static str {
    VOICES
        .iter()
        .find(|(v, _)| *v == id)
        .map(|(_, c)| *c)
        .unwrap_or("")
}

/// The registry basename for a voice: its id, lowercased.
pub fn basename(id: &str) -> String {
    id.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roster_has_thirty_unique_voices() {
        let mut ids: Vec<&str> = VOICES.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate voice id in the roster");
        assert_eq!(count, 30);
    }

    #[test]
    fn basenames_are_unique_and_filesystem_safe() {
        let mut names: Vec<String> = VOICES.iter().map(|(id, _)| basename(id)).collect();
        for name in &names {
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase()),
                "{name} is not a plain lowercase basename"
            );
        }
        names.sort();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two voices collide on one basename");
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(canonical("sulafat"), Some("Sulafat"));
        assert_eq!(canonical("  SULAFAT "), Some("Sulafat"));
        assert_eq!(canonical("Sulafat"), Some("Sulafat"));
        assert_eq!(canonical("nope"), None);
    }

    #[test]
    fn characteristics_are_populated() {
        assert_eq!(characteristic("Sulafat"), "Warm");
        assert!(VOICES.iter().all(|(_, c)| !c.is_empty()));
    }
}
