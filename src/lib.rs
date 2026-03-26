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
//! use lfm2_audio::{LFM2Audio, Precision, Device, ASROptions, load_audio, save_audio};
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
//!     let (audio, _spec) = load_audio("input.wav")?;
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

pub mod asr;
pub mod audio;
pub mod cache;
pub mod chat;
pub mod config;
pub mod embeddings;
pub mod error;
pub mod interleaved;
pub mod model;
pub mod sessions;
pub mod tokenizer;
pub mod tts;

// Re-exports
pub use asr::{ASROptions, ASRPipeline};
pub use audio::{
    compute_mel_spectrogram, decode_wav_bytes, encode_wav_bytes, load_audio, save_audio,
};
pub use chat::{AssistantResponse, ChatSession, Turn};
pub use config::{Device, ModelConfig, Precision, PreprocessorConfig};
pub use embeddings::{AudioEmbedding, EmbedTokens};
pub use error::{LFM2Error, Result};
pub use interleaved::{
    InterleavedEvent, InterleavedOptions, InterleavedPipeline, InterleavedResponse,
};
pub use model::{LFM2Audio, ModelInfo};
pub use tokenizer::LFM2Tokenizer;
pub use tts::{decode_audio_codes_standalone, TTSEvent, TTSOptions, TTSPipeline, TTSStreamOutput};

// Audio utilities
pub use audio::istft;
pub use audio::mel;

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
