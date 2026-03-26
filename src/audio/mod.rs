//! Audio processing utilities

pub mod istft;
pub mod mel;

pub use istft::ISTFT;
pub use mel::compute_mel_spectrogram;

use hound::{WavReader, WavSpec};
use std::io::Cursor;
use std::path::Path;

use crate::error::Result;

/// Load audio file (WAV) and convert to f32 samples
/// Supports both float and int formats
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                crate::error::LFM2Error::Audio(format!("Failed to read float samples: {}", e))
            })?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                crate::error::LFM2Error::Audio(format!("Failed to read int samples: {}", e))
            })?,
    };

    Ok((samples, spec))
}

/// Save audio to WAV file
pub fn save_audio<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Decode WAV bytes into mono f32 samples.
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<(Vec<f32>, WavSpec)> {
    let cursor = Cursor::new(bytes);
    let mut reader = WavReader::new(cursor)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                crate::error::LFM2Error::Audio(format!("Failed to read float samples: {}", e))
            })?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
                crate::error::LFM2Error::Audio(format!("Failed to read int samples: {}", e))
            })?,
    };

    Ok((samples, spec))
}

/// Encode mono f32 samples as WAV bytes.
pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let channels = 1u16;
    let bits_per_sample = 16u16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_len = samples.len() as u32 * block_align as u32;
    let riff_len = 36 + data_len;

    let mut bytes = Vec::with_capacity(44 + data_len as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_len.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());

    for &sample in samples {
        let quantized = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }

    Ok(bytes)
}

/// Resample audio using simple linear interpolation
/// Note: This is a basic implementation. For production, consider using a proper resampling library.
pub fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let new_len = (audio.len() as f64 * ratio) as usize;
    let mut result = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - idx_floor as f64;

        let val = audio[idx_floor] as f64 * (1.0 - frac) + audio[idx_ceil] as f64 * frac;
        result.push(val as f32);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::{decode_wav_bytes, encode_wav_bytes};

    #[test]
    fn wav_bytes_roundtrip_preserves_shape_and_rate() {
        let sample_rate = 16_000;
        let samples: Vec<f32> = (0..512)
            .map(|idx| ((idx as f32 / 32.0).sin() * 0.5).clamp(-1.0, 1.0))
            .collect();

        let wav = encode_wav_bytes(&samples, sample_rate).expect("should encode wav bytes");
        let (decoded, spec) = decode_wav_bytes(&wav).expect("should decode wav bytes");

        assert_eq!(spec.sample_rate, sample_rate);
        assert_eq!(decoded.len(), samples.len());
        assert!(decoded.iter().all(|sample| sample.is_finite()));
    }
}
