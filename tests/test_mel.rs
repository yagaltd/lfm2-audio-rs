//! Mel spectrogram computation tests

use lfm2_audio::audio::mel::compute_mel_spectrogram;
use lfm2_audio::config::PreprocessorConfig;

#[test]
fn test_mel_shape() {
    // 1 second of audio at 16kHz
    let audio = vec![0.0f32; 16000];
    let config = PreprocessorConfig::default();
    
    let mel = compute_mel_spectrogram(&audio, &config).unwrap();
    
    // Should have approximately 100 frames (1s / 0.01s hop)
    assert_eq!(mel.shape()[1], 128, "Should have 128 mel bins");
    assert!(
        mel.shape()[0] >= 99 && mel.shape()[0] <= 101,
        "Should have ~100 frames, got {}",
        mel.shape()[0]
    );
    
    println!("✓ Mel spectrogram shape: {:?}", mel.shape());
}

#[test]
fn test_mel_different_durations() {
    let config = PreprocessorConfig::default();
    
    let test_cases = vec![
        (8000, 50),    // 0.5s -> ~50 frames
        (16000, 100),  // 1s -> ~100 frames
        (32000, 200),  // 2s -> ~200 frames
    ];
    
    for (num_samples, expected_frames) in test_cases {
        let audio = vec![0.0f32; num_samples];
        let mel = compute_mel_spectrogram(&audio, &config).unwrap();
        
        let tolerance = 5;
        assert!(
            (mel.shape()[0] as i32 - expected_frames as i32).abs() <= tolerance,
            "For {} samples: expected ~{} frames, got {}",
            num_samples,
            expected_frames,
            mel.shape()[0]
        );
    }
    
    println!("✓ Mel spectrogram scales correctly with duration");
}

#[test]
fn test_mel_normalization() {
    use lfm2_audio::audio::mel::normalize_per_feature;
    use ndarray::Array2;
    
    // Create test array
    let mut data = Array2::<f32>::zeros((100, 128));
    for i in 0..100 {
        for j in 0..128 {
            data[[i, j]] = (i * 128 + j) as f32;
        }
    }
    
    let normalized = normalize_per_feature(data);
    
    // Check that mean is approximately 0 and std is approximately 1
    for j in 0..128 {
        let column = normalized.column(j);
        let mean: f32 = column.iter().sum::<f32>() / column.len() as f32;
        let variance: f32 = column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / column.len() as f32;
        let std = variance.sqrt();
        
        assert!(mean.abs() < 0.01, "Mean should be ~0, got {}", mean);
        assert!((std - 1.0).abs() < 0.01, "Std should be ~1, got {}", std);
    }
    
    println!("✓ Per-feature normalization verified");
}