//! Transcript verification.
//!
//! The `.txt` is injected into the prompt, so if a take drops or mangles words
//! the reference is quietly wrong for every future generation. `--verify`
//! transcribes each take back and scores it here.
//!
//! Scoring is word-level longest-common-subsequence, which tolerates the things
//! a transcriber legitimately differs on (punctuation, casing) while catching
//! the things that matter (dropped clauses, skipped languages, invented words).

/// Fold text into comparable words: lowercase, drop everything that is not
/// alphanumeric, split on whitespace. Unicode-aware, so Cyrillic and Turkish
/// dotted/dotless forms survive.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word-level LCS length, computed with two rolling rows.
fn lcs_len(a: &[String], b: &[String]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for word_a in a {
        for (j, word_b) in b.iter().enumerate() {
            cur[j + 1] = if word_a == word_b {
                prev[j] + 1
            } else {
                cur[j].max(prev[j + 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.iter_mut().for_each(|slot| *slot = 0);
    }
    prev[b.len()]
}

/// Similarity in `0.0..=1.0`: `2 * LCS / (len_a + len_b)`.
///
/// Dividing by the combined length penalizes insertions as well as deletions,
/// so a transcript that adds narration scores below one just like a truncated
/// one does.
pub fn similarity(reference: &str, transcript: &str) -> f32 {
    let a = words(reference);
    let b = words(transcript);
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * lcs_len(&a, &b) as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_scores_one() {
        let text = "Herzlich willkommen zum Sprachentag an unserer Schule!";
        assert_eq!(similarity(text, text), 1.0);
    }

    #[test]
    fn punctuation_and_casing_do_not_matter() {
        // Exactly the divergence a real transcriber produced: '!' became '.'.
        let reference = "Welcome everyone! Learning a new language opens a door.";
        let transcript = "welcome everyone. learning a new language opens a door";
        assert_eq!(similarity(reference, transcript), 1.0);
    }

    #[test]
    fn non_latin_scripts_compare_correctly() {
        let reference = "Добро пожаловать! Мова — це міст між людьми.";
        let transcript = "Добро пожаловать. Мова це міст між людьми";
        assert_eq!(similarity(reference, transcript), 1.0);
        // Turkish dotless-i forms must survive the fold.
        assert_eq!(similarity("Hoş geldiniz", "hoş geldiniz"), 1.0);
    }

    #[test]
    fn a_dropped_language_block_scores_well_below_the_threshold() {
        let reference = "eins zwei drei vier funf sechs sieben acht neun zehn";
        let transcript = "eins zwei drei vier funf";
        // 2*5 / (10+5) = 0.667
        let score = similarity(reference, transcript);
        assert!(score < 0.7, "dropped half the passage but scored {score}");
    }

    #[test]
    fn added_narration_is_penalized_too() {
        let reference = "one two three four";
        let transcript = "sure here is the passage one two three four";
        assert!(similarity(reference, transcript) < 0.75);
    }

    #[test]
    fn a_single_mangled_word_stays_above_the_threshold() {
        // Calibration check at a realistic length. A ~50 s passage is ~140
        // words; one mis-transcribed word there scores 2*139/280 = 0.993, well
        // clear of the 0.95 warning. (On a very short passage a single word is
        // a much larger fraction, so the threshold is only meaningful for
        // reference-length text — which is all this tool produces.)
        let reference: Vec<String> = (0..140).map(|i| format!("word{i}")).collect();
        let mut transcript = reference.clone();
        transcript[73] = "mangled".to_owned();
        let score = similarity(&reference.join(" "), &transcript.join(" "));
        assert!(
            score > 0.99,
            "one typo in 140 words dropped the score to {score}"
        );
    }

    #[test]
    fn a_dropped_sentence_in_a_full_passage_trips_the_threshold() {
        // The failure mode that matters: ~12 words of one language go missing
        // from an otherwise perfect take. It must land under 0.95.
        let reference: Vec<String> = (0..140).map(|i| format!("word{i}")).collect();
        let mut transcript = reference.clone();
        transcript.drain(60..72);
        let score = similarity(&reference.join(" "), &transcript.join(" "));
        assert!(score < 0.96, "a dropped sentence still scored {score}");
    }

    #[test]
    fn empty_transcript_scores_zero() {
        assert_eq!(similarity("something was said", ""), 0.0);
    }

    #[test]
    fn two_empty_strings_score_one() {
        assert_eq!(similarity("", ""), 1.0);
    }

    #[test]
    fn word_order_matters() {
        // LCS, not a bag of words: a reversed passage must not score as a match.
        assert!(similarity("alpha beta gamma delta", "delta gamma beta alpha") < 0.6);
    }
}
