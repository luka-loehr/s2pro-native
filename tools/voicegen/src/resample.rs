//! 24 kHz -> 44.1 kHz polyphase resampler.
//!
//! Gemini TTS returns `audio/l16; rate=24000; channels=1`; the codec's native
//! rate is 44.1 kHz and `s2p_wav_parse_f32` rejects anything else outright, so
//! every take is resampled here before it is written.
//!
//! gcd(44100, 24000) = 300, so the exact conversion is interpolate-by-147 /
//! decimate-by-80. That is not a power of two, which is why orin-stream's 2x
//! half-band `TtsUpsampler` does not apply. Implemented as a polyphase
//! windowed-sinc bank: 147 phases, each a 64-tap Blackman-windowed sinc, built
//! once and normalized to unity DC gain.

/// Interpolation factor: 44100 / gcd(44100, 24000).
const L: usize = 147;
/// Decimation factor: 24000 / gcd(44100, 24000).
const M: usize = 80;
/// Kernel half-width in input samples; 64 taps total.
const HALF: usize = 32;
const TAPS: usize = HALF * 2;
/// Cutoff as a fraction of the *input* sample rate (10.8 kHz of 12 kHz Nyquist).
/// With 64 Blackman taps the transition half-width is ~0.043, so the stopband
/// starts at ~0.493 — just inside Nyquist, i.e. no fold-back aliasing.
const FC: f64 = 0.45;

pub const IN_RATE: u32 = 24_000;
pub const OUT_RATE: u32 = 44_100;

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

/// Blackman window over `u` in [-HALF, HALF], zero outside.
fn blackman(u: f64) -> f64 {
    let n = (u + HALF as f64) / (2.0 * HALF as f64); // 0..1
    if !(0.0..=1.0).contains(&n) {
        return 0.0;
    }
    let two_pi_n = 2.0 * std::f64::consts::PI * n;
    0.42 - 0.5 * two_pi_n.cos() + 0.08 * (2.0 * two_pi_n).cos()
}

/// The polyphase bank, flattened to `L * TAPS`. Row `p` covers fractional
/// position `p / L` between two input samples.
fn build_bank() -> Vec<f32> {
    let mut bank = vec![0.0f32; L * TAPS];
    for p in 0..L {
        let frac = p as f64 / L as f64;
        let row = &mut bank[p * TAPS..(p + 1) * TAPS];
        let mut sum = 0.0f64;
        for (k, tap) in row.iter_mut().enumerate() {
            // Tap k sits at input offset k - (HALF - 1), i.e. -(HALF-1)..=HALF.
            let offset = k as f64 - (HALF as f64 - 1.0);
            let u = offset - frac;
            let h = 2.0 * FC * sinc(2.0 * FC * u) * blackman(u);
            *tap = h as f32;
            sum += h;
        }
        // Unity DC gain per phase, so the output level never drifts with phase.
        if sum.abs() > 1e-12 {
            for tap in row.iter_mut() {
                *tap = (*tap as f64 / sum) as f32;
            }
        }
    }
    bank
}

/// Resample mono 24 kHz S16 to mono 44.1 kHz S16.
pub fn resample(input: &[i16]) -> Vec<i16> {
    if input.is_empty() {
        return Vec::new();
    }
    let bank = build_bank();
    let out_len = input.len() * L / M;
    let mut out = Vec::with_capacity(out_len);

    for n in 0..out_len {
        // Output n samples the input at t = n * M / L, so the integer part is
        // the base index and the remainder selects the polyphase row.
        let t = n * M;
        let base = (t / L) as isize;
        let row = &bank[(t % L) * TAPS..(t % L + 1) * TAPS];

        let mut acc = 0.0f32;
        for (k, tap) in row.iter().enumerate() {
            let idx = base + k as isize - (HALF as isize - 1);
            // Zero outside the signal: the takes start and end in silence.
            if idx >= 0 && (idx as usize) < input.len() {
                acc += input[idx as usize] as f32 * *tap;
            }
        }
        out.push(acc.round().clamp(-32768.0, 32767.0) as i16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_is_exact_147_over_80() {
        // 300 is gcd(44100, 24000); the factors must reproduce both rates.
        assert_eq!(L * 300, OUT_RATE as usize);
        assert_eq!(M * 300, IN_RATE as usize);
    }

    #[test]
    fn output_length_follows_the_rate_ratio() {
        // One second in, one second out (within the floor of the last sample).
        let input = vec![0i16; IN_RATE as usize];
        let out = resample(&input);
        assert_eq!(out.len(), OUT_RATE as usize);
    }

    #[test]
    fn every_phase_has_unity_dc_gain() {
        let bank = build_bank();
        for p in 0..L {
            let sum: f32 = bank[p * TAPS..(p + 1) * TAPS].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "phase {p} DC gain {sum} is not unity"
            );
        }
    }

    #[test]
    fn dc_signal_passes_through_at_the_same_level() {
        let input = vec![8000i16; IN_RATE as usize];
        let out = resample(&input);
        // Skip the kernel's edge ramp at both ends.
        let interior = &out[HALF * 4..out.len() - HALF * 4];
        for &s in interior {
            assert!((s as i32 - 8000).abs() <= 2, "DC drifted to {s}");
        }
    }

    #[test]
    fn a_1_khz_tone_keeps_its_amplitude_and_frequency() {
        // 1 kHz at 24 kHz in; measure the same tone at 44.1 kHz out.
        let n_in = IN_RATE as usize;
        let input: Vec<i16> = (0..n_in)
            .map(|i| {
                let t = i as f64 / IN_RATE as f64;
                (10_000.0 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as i16
            })
            .collect();
        let out = resample(&input);

        // Peak amplitude survives (well inside the passband).
        let interior = &out[HALF * 4..out.len() - HALF * 4];
        let peak = interior.iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!(
            (9_500..=10_500).contains(&(peak as i32)),
            "1 kHz peak drifted to {peak}"
        );

        // Zero crossings pin the frequency: 1 kHz over ~1 s is ~2000 crossings.
        let crossings = interior
            .windows(2)
            .filter(|w| (w[0] >= 0) != (w[1] >= 0))
            .count();
        assert!(
            (1_960..=2_040).contains(&crossings),
            "1 kHz produced {crossings} zero crossings"
        );
    }

    #[test]
    fn above_nyquist_content_is_attenuated() {
        // 11.9 kHz sits in the stopband; it must not survive into the output.
        let n_in = IN_RATE as usize;
        let input: Vec<i16> = (0..n_in)
            .map(|i| {
                let t = i as f64 / IN_RATE as f64;
                (10_000.0 * (2.0 * std::f64::consts::PI * 11_900.0 * t).sin()) as i16
            })
            .collect();
        let out = resample(&input);
        let interior = &out[HALF * 4..out.len() - HALF * 4];
        let peak = interior.iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!(peak < 1_000, "stopband tone leaked through at {peak}");
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(resample(&[]).is_empty());
    }
}
