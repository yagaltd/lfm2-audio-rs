//! LFM2-Audio-RS
//! 
//! Rust implementation of LFM2.5-Audio multimodal model supporting:
//! - ASR (Automatic Speech Recognition): Audio → Text
//! - TTS (Text-to-Speech): Text → Audio
//! - Interleaved: Audio ↔ Text + Audio
//! 
//! # Quick Start
//! 
//! ```rust,no_run
//! use lfm2_audio::{LFM2Audio, Precision, Device, ASROptions};
//! 
//! fn main() -> anyhow::Result<()> {
//!     // Load model
//!     let model = LFM2Audio::from_pretrained(
//!         "./LFM2.5-Audio-1.5B-ONNX",
//!         Precision::Q4,
//!         Device::CPU,
//!     )?;
//! 
//!     // ASR
//!     let audio = load_audio("input.wav")?;
//!     let text = model.asr().transcribe(&audio, 16000, &ASROptions::default())?;
//!     println!("Transcription: {}", text);
//! 
//!     // TTS
//!     let speech = model.tts().synthesize("Hello, world!", &Default::default())?;
//!     save_audio("output.wav", &speech, 24000)?;
//! 
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod config;
pub mod embeddings;
pub mod tokenizer;
pub mod cache;
pub mod sessions;
pub mod model;
pub mod asr;
pub mod tts;
pub mod interleaved;
pub mod chat;
pub mod audio;

// Re-exports
pub use error::{LFM2Error, Result};
pub use config::{ModelConfig, Precision, Device, PreprocessorConfig};
pub use embeddings::{EmbedTokens, AudioEmbedding};
pub use tokenizer::LFM2Tokenizer;
pub use model::{LFM2Audio, ModelInfo};
pub use asr::{ASRPipeline, ASROptions};
pub use tts::{TTSPipeline, TTSOptions};
pub use interleaved::{InterleavedPipeline, InterleavedResponse};
pub use chat::{ChatSession, AssistantResponse, Turn};
pub use audio::{load_audio, save_audio, compute_mel_spectrogram};

// Audio utilities
pub use audio::mel;
pub use audio::istft;

/// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Model constants
pub mod constants {
    /// Number of audio codebooks
    pub const NUM_CODEBOOKS: usize = 8;
    
    /// Vocabulary size per codebook (2048 audio + 1 EOS)
    pub const CODEBOOK_VOCAB: usize = 2049;
    
    /// End-of-audio token ID
    pub const END_OF_AUDIO_TOKEN: u16 = 2048;
    
    /// Hidden size of LFM2 model
    pub const HIDDEN_SIZE: usize = 2048;
    
    /// Vocabulary size for text
    pub const VOCAB_SIZE: usize = 65536;
    
    /// Number of transformer layers
    pub const NUM_LAYERS: usize = 16;
    
    /// Number of attention heads
    pub const NUM_ATTENTION_HEADS: usize = 32;
    
    /// Number of key-value heads (GQA)
    pub const NUM_KV_HEADS: usize = 8;
    
    /// Head dimension
    pub const HEAD_DIM: usize = 64; // HIDDEN_SIZE / NUM_ATTENTION_HEADS
    
    /// Input sample rate for ASR
    pub const INPUT_SAMPLE_RATE: u32 = 16000;
    
    /// Output sample rate for TTS
    pub const OUTPUT_SAMPLE_RATE: u32 = 24000;
}