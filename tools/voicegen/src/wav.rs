//! Minimal RIFF/WAVE writer, mono S16LE.
//!
//! Byte-for-byte the layout `s2p_wav_header` emits (src/core/wav.c) and the one
//! `s2p_wav_parse_f32` accepts: canonical 44-byte header, fmt tag 1, 1 channel,
//! 16 bits, and — non-negotiable for the registry — 44100 Hz.

use anyhow::{Result, bail};

pub const HEADER_LEN: usize = 44;

/// Serialize `pcm` as a complete mono S16LE WAV file.
pub fn encode(pcm: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let data_bytes = pcm.len() * 2;
    if data_bytes as u64 > u32::MAX as u64 - 36 {
        bail!("audio exceeds the RIFF 32-bit size field");
    }
    let mut out = Vec::with_capacity(HEADER_LEN + data_bytes);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(data_bytes as u32 + 36).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for sample in pcm {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(out)
}

/// Reinterpret raw little-endian S16 bytes (Gemini's `audio/l16` payload) as
/// samples. A trailing odd byte would mean a truncated stream.
pub fn pcm_from_le_bytes(bytes: &[u8]) -> Result<Vec<i16>> {
    if bytes.len() % 2 != 0 {
        bail!("l16 payload has an odd byte count ({})", bytes.len());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_the_c_writer_layout() {
        let wav = encode(&[1i16, -1], 44_100).unwrap();
        assert_eq!(wav.len(), HEADER_LEN + 4);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1); // PCM
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1); // mono
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 44_100);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 88_200);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4);
        // RIFF size = data + 36.
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 40);
    }

    #[test]
    fn samples_are_little_endian() {
        let wav = encode(&[0x0102], 44_100).unwrap();
        assert_eq!(&wav[44..46], &[0x02, 0x01]);
    }

    #[test]
    fn l16_round_trips() {
        let pcm = pcm_from_le_bytes(&[0x02, 0x01, 0xFF, 0xFF]).unwrap();
        assert_eq!(pcm, vec![0x0102, -1]);
    }

    #[test]
    fn odd_l16_length_is_rejected() {
        assert!(pcm_from_le_bytes(&[0x00]).is_err());
    }
}
