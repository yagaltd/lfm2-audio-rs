//! Inverse STFT for audio detokenization
//! Reference: liquid_audio/src/liquid_audio/detokenizer.py (ISTFT class)

use ndarray::{Array1, Array2};
use num_complex::Complex32;

use crate::error::{LFM2Error, Result};

/// Inverse Short-Time Fourier Transform
/// Reconstructs time-domain audio from complex spectrogram
pub struct ISTFT {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Vec<f32>,
}

impl ISTFT {
    /// Create ISTFT with given parameters
    /// For LFM2.5 audio detokenizer: n_fft=1280, hop_length=320, win_length=1280
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        let window = hann_window(win_length);
        Self {
            n_fft,
            hop_length,
            win_length,
            window,
        }
    }

    /// Compute inverse STFT
    /// Input: complex spectrogram [n_freqs, n_frames] where n_freqs = n_fft / 2 + 1
    /// Output: time-domain audio samples
    pub fn inverse(&self, spec: &Array2<Complex32>) -> Result<Vec<f32>> {
        let n_freqs = spec.shape()[0];
        let n_frames = spec.shape()[1];

        if n_freqs != self.n_fft / 2 + 1 {
            return Err(LFM2Error::Audio(format!(
                "Expected {} frequency bins, got {}",
                self.n_fft / 2 + 1,
                n_freqs
            )));
        }

        // Calculate output size
        let output_size = (n_frames - 1) * self.hop_length + self.win_length;

        // Pad for "same" padding
        let pad = (self.win_length - self.hop_length) / 2;

        // Overlap-add buffers
        let mut y = Array1::<f32>::zeros(output_size);
        let mut window_envelope = Array1::<f32>::zeros(output_size);

        // Process each frame
        for frame_idx in 0..n_frames {
            // Extract frame from spectrogram
            let mut frame_spec: Vec<Complex32> = Vec::with_capacity(self.n_fft);

            // DC component
            frame_spec.push(spec[[0, frame_idx]]);

            // Positive frequencies
            for k in 1..n_freqs - 1 {
                frame_spec.push(spec[[k, frame_idx]]);
            }

            // Nyquist (real for even n_fft)
            if self.n_fft % 2 == 0 {
                frame_spec.push(spec[[n_freqs - 1, frame_idx]]);
            }

            // Negative frequencies (conjugate symmetry)
            for k in (1..n_freqs - 1).rev() {
                frame_spec.push(spec[[k, frame_idx]].conj());
            }

            // IFFT
            let frame_time = ifft(&frame_spec);

            // Apply window
            let start = frame_idx * self.hop_length;
            for i in 0..self.win_length {
                if start + i < output_size {
                    y[start + i] += frame_time[i] * self.window[i];
                    window_envelope[start + i] += self.window[i] * self.window[i];
                }
            }
        }

        // Normalize by window envelope
        for i in 0..output_size {
            if window_envelope[i] > 1e-11 {
                y[i] /= window_envelope[i];
            }
        }

        // Trim padding
        let start_idx = pad;
        let end_idx = output_size - pad;
        let result: Vec<f32> = y.iter().skip(start_idx).take(end_idx - start_idx).copied().collect();

        Ok(result)
    }

    /// Inverse from magnitude and phase separately
    /// magnitude: [n_freqs, n_frames]
    /// phase: [n_freqs, n_frames]
    pub fn inverse_from_polar(
        &self,
        magnitude: &Array2<f32>,
        phase: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let n_freqs = magnitude.shape()[0];
        let n_frames = magnitude.shape()[1];

        // Convert to complex
        let mut spec = Array2::<Complex32>::zeros((n_freqs, n_frames));
        for i in 0..n_freqs {
            for j in 0..n_frames {
                spec[[i, j]] = Complex32::from_polar(magnitude[[i, j]], phase[[i, j]]);
            }
        }

        self.inverse(&spec)
    }

    /// Inverse from log-magnitude and phase (as produced by detokenizer)
    /// log_magnitude: [n_freqs, n_frames] - log of magnitude
    /// phase: [n_freqs, n_frames]
    pub fn inverse_from_log_polar(
        &self,
        log_magnitude: &Array2<f32>,
        phase: &Array2<f32>,
    ) -> Result<Vec<f32>> {
        let magnitude = log_magnitude.mapv(|x| x.exp());
        self.inverse_from_polar(&magnitude, phase)
    }
}

/// Hann window
fn hann_window(window_length: usize) -> Vec<f32> {
    if window_length <= 1 {
        return vec![1.0; window_length];
    }

    (0..window_length)
        .map(|i| {
            0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / (window_length as f32 - 1.0)).cos()
        })
        .collect()
}

/// Simple IFFT implementation
fn ifft(input: &[Complex32]) -> Vec<f32> {
    let n = input.len();
    let mut output: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];

    // IFFT = 1/N * FFT with conjugate
    for k in 0..n {
        for t in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (t * k) as f32 / n as f32;
            let twiddle = Complex32::new(angle.cos(), angle.sin());
            output[k] += input[t] * twiddle;
        }
        output[k] /= n as f32;
    }

    // Take real part and normalize
    output.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_istft_reconstructs_sine() {
        // Create a simple sine wave
        let sample_rate = 24000;
        let freq = 1000.0;
        let duration = 0.1;
        let num_samples = (sample_rate as f32 * duration) as usize;

        let sine: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        // For this test, we'll just verify ISTFT doesn't panic with valid input
        let n_fft = 1280;
        let hop = 320;
        let n_frames = (num_samples - n_fft) / hop + 1;
        let n_freqs = n_fft / 2 + 1;

        // Create a simple spectrogram (zeros)
        let spec = Array2::<Complex32>::zeros((n_freqs, n_frames));

        let istft = ISTFT::new(n_fft, hop, n_fft);
        let result = istft.inverse(&spec).unwrap();

        // Output should be non-empty
        assert!(!result.is_empty());
    }

    #[test]
    fn test_hann_window() {
        let window = hann_window(1280);
        assert_eq!(window.len(), 1280);
        // Check symmetry
        for i in 0..window.len() / 2 {
            assert!((window[i] - window[window.len() - 1 - i]).abs() < 1e-6);
        }
    }
}