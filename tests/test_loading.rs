//! Model loading tests
//! 
//! These tests require the LFM2.5-Audio-1.5B-ONNX model to be present at:
//! tests/models/LFM2.5-Audio-1.5B-ONNX/

use lfm2_audio::{Device, LFM2Audio, Precision};
use std::path::PathBuf;

fn get_model_path() -> Option<PathBuf> {
    let path = PathBuf::from("tests/models/LFM2.5-Audio-1.5B-ONNX");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

#[test]
#[ignore = "requires model files"]
fn test_load_model_q4() {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    
    let model = LFM2Audio::from_pretrained(&model_path, Precision::Q4, Device::CPU)
        .expect("Failed to load model");
    
    let info = model.info();
    assert_eq!(info.hidden_size, 2048);
    assert_eq!(info.num_codebooks, 8);
    
    println!("✓ Model loaded successfully (Q4)");
    println!("  Hidden size: {}", info.hidden_size);
    println!("  Vocab size: {}", info.vocab_size);
    println!("  Num layers: {}", info.num_layers);
}

#[test]
#[ignore = "requires model files"]
fn test_load_model_fp16() {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    
    let _model = LFM2Audio::from_pretrained(&model_path, Precision::FP16, Device::CPU)
        .expect("Failed to load model");
    
    println!("✓ Model loaded successfully (FP16)");
}

#[test]
#[ignore = "requires model files"]
fn test_model_info() {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    
    let model = LFM2Audio::from_pretrained(&model_path, Precision::Q4, Device::CPU)
        .unwrap();
    
    let info = model.info();
    
    assert_eq!(info.hidden_size, 2048, "Hidden size should be 2048");
    assert_eq!(info.num_codebooks, 8, "Should have 8 codebooks");
    assert!(info.vocab_size > 0, "Vocab size should be > 0");
    assert_eq!(info.num_layers, 16, "Should have 16 layers");
    
    println!("✓ Model info verified");
}

#[test]
#[ignore = "requires model files"]
fn test_embeddings_loaded() {
    let model_path = get_model_path().expect("Model not found. Skipping test.");
    
    let model = LFM2Audio::from_pretrained(&model_path, Precision::Q4, Device::CPU)
        .unwrap();
    
    // Test text embedding lookup
    let ids = vec![1, 2, 3, 4, 5];
    let embeds = model.get_text_embeddings(&ids);
    
    assert_eq!(embeds.shape(), &[1, 5, 2048]);
    println!("✓ Text embeddings shape verified: {:?}", embeds.shape());
}
