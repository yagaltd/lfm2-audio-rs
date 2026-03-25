//! Mel spectrogram computation
//! Extracted from parakeet-rs/src/audio.rs with modifications for LFM2.5

use ndarray::Array2;
use realfft::RealFftPlanner;
use std::f32::consts::PI;

use crate::config::PreprocessorConfig;
use crate::error::{LFM2Error, Result};

/// Compute mel spectrogram from audio samples
pub fn compute_mel_spectrogram(
    audio: &[f32],
    config: &PreprocessorConfig,
) -> Result<Array2<f32>> {
    // 1. Apply preemphasis
    let audio = apply_preemphasis(audio, config.dither as f32);
    
    // 2. Compute STFT
    let hop_length = (config.window_stride * config.sample_rate as f64) as usize;
    let win_length = (config.window_size * config.sample_rate as f64) as usize;
    
    let spectrogram = stft(&audio, config.n_fft, hop_length, win_length)?;
    
    // 3. Create mel filterbank
    let mel_filterbank = create_mel_filterbank(
        config.n_fft,
        config.features,
        config.sample_rate as usize,
    );
    
    // 4. Apply mel filterbank
    let mel_spectrogram = mel_filterbank.dot(&spectrogram);
    
    // 5. Log compression
    let log_zero_guard: f32 = 5.960464477539063e-08; // From mel_config.json
    let mel_spectrogram = mel_spectrogram.mapv(|x| (x + log_zero_guard).ln());
    
    // 6. Transpose to [num_frames, n_mels]
    let mel_spectrogram = mel_spectrogram.t().to_owned();
    
    // 7. Per-feature normalization if configured
    if config.normalize == "per_feature" {
        Ok(normalize_per_feature(mel_spectrogram))
    } else {
        Ok(mel_spectrogram)
    }
}

/// Apply preemphasis filter: y[t] = x[t] - coef * x[t-1]
pub fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(audio.len());
    
    if audio.is_empty() {
        return result;
    }
    
    // First sample passes through unchanged (or with minimal adjustment)
    result.push(audio[0] * (1.0 - coef * 0.5));
    
    // Apply preemphasis to rest
    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }
    
    result
}

/// Hann window function
fn hann_window(window_length: usize) -> Vec<f32> {
    if window_length <= 1 {
        return vec![1.0; window_length];
    }
    
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

/// Short-Time Fourier Transform
/// Returns [n_freqs, n_frames] where n_freqs = n_fft / 2 + 1
fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<f32>> {
    let pad_amount = n_fft / 2;
    let mut padded = vec![0.0f32; pad_amount];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad_amount, 0.0);
    
    let window = hann_window(win_length);
    let num_frames = (padded.len().saturating_sub(n_fft)) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
    
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);
    let mut input = vec![0.0f32; n_fft];
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    
    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        
        input.fill(0.0);
        for i in 0..win_length.min(padded.len().saturating_sub(start)) {
            input[i] = padded[start + i] * window[i];
        }
        
        r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| LFM2Error::Audio(format!("FFT failed: {}", e)))?;
        
        for k in 0..freq_bins {
            spectrogram[[k, frame_idx]] = output[k].norm_sqr();
        }
    }
    
    Ok(spectrogram)
}

// Slaney mel scale constants
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.06875177742094912;

/// Convert Hz to mel (Slaney scale)
fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

/// Convert mel to Hz (Slaney scale)
fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

/// Create mel filterbank
/// Returns [n_mels, n_freqs]
fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));
    
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);
    
    // Mel points evenly spaced
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
        .collect();
    
    // Convert to FFT bin indices
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&mel| {
            let hz = mel_to_hz_slaney(mel);
            fft_freqs
                .iter()
                .position(|&f| f >= hz)
                .unwrap_or(freq_bins - 1)
        })
        .collect();
    
    // Build triangular filters
    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];
        
        // Rising slope
        if center > left {
            for j in left..center {
                let weight = (j - left) as f32 / (center - left) as f32;
                filterbank[[i, j]] = weight;
            }
        }
        
        // Falling slope
        if right > center {
            for j in center..right {
                let weight = (right - j) as f32 / (right - center) as f32;
                filterbank[[i, j]] = weight;
            }
        }
        
        if center < freq_bins {
            filterbank[[i, center]] = 1.0;
        }
    }
    
    // Slaney normalization
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]) as f32;
        for j in 0..freq_bins {
            filterbank[[i, j]] *= enorm;
        }
    }
    
    filterbank
}

/// Per-feature normalization (zero mean, unit variance)
pub fn normalize_per_feature(mel: Array2<f32>) -> Array2<f32> {
    let num_frames = mel.shape()[0];
    let num_features = mel.shape()[1];
    
    let mut normalized = mel;
    
    for feat_idx in 0..num_features {
        let column = normalized.column(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 = column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_frames as f32;
        let std = variance.sqrt().max(1e-5);
        
        let mut column = normalized.column_mut(feat_idx);
        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }
    
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preemphasis() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let coef = 0.97;
        let result = apply_preemphasis(&audio, coef);
        
        assert_eq!(result.len(), audio.len());
        // First sample passes through with adjustment
        assert!((result[0] - audio[0] * (1.0 - coef * 0.5)).abs() < 1e-6);
        // Rest have preemphasis applied
        assert!((result[1] - (audio[1] - coef * audio[0])).abs() < 1e-6);
    }
    
    #[test]
    fn test_hann_window() {
        let window = hann_window(400);
        assert_eq!(window.len(), 400);
        // Window should be symmetric
        assert!((window[0] - window[399]).abs() < 1e-6);
        // Center should be 1.0
        // For even-length window, peak is between indices 199 and 200
        // Both should be very close to 1.0 (~0.99997)
        assert!(window[199] > 0.999);
        assert!(window[200] > 0.999);
        assert!(window[199] > window[0]);
    }
    
    #[test]
    fn test_mel_filterbank_shape() {
        let filterbank = create_mel_filterbank(512, 128, 16000);
        assert_eq!(filterbank.shape(), &[128, 257]); // [n_mels, n_freqs]
    }
    
    #[test]
    fn test_compute_mel_spectrogram_shape() {
        // 1 second of audio at 16kHz
        let audio = vec![0.0f32; 16000];
        let config = PreprocessorConfig::default();
        
        let mel = compute_mel_spectrogram(&audio, &config).unwrap();
        
        // Should have approximately 100 frames (1s / 0.01s hop)
        assert_eq!(mel.shape()[1], 128); // n_mels
        assert!(mel.shape()[0] >= 99 && mel.shape()[0] <= 101);
    }
}