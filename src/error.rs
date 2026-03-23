//! Error types for LFM2-Audio

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LFM2Error {
    #[error("ONNX error: {0}")]
    Onnx(#[from] ort::Error),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Invalid configuration: {0}")]
    Config(String),
    
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    
    #[error("Audio processing error: {0}")]
    Audio(String),
    
    #[error("Generation error: {0}")]
    Generation(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("NDArray error: {0}")]
    Ndarray(#[from] ndarray::ShapeError),
    
    #[error("Invalid audio format: {0}")]
    InvalidAudioFormat(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Embedding error: {0}")]
    Embedding(String),
}

pub type Result<T> = std::result::Result<T, LFM2Error>;

impl From<std::string::FromUtf8Error> for LFM2Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        LFM2Error::Tokenizer(format!("Invalid UTF-8: {}", e))
    }
}

impl From<hound::Error> for LFM2Error {
    fn from(e: hound::Error) -> Self {
        LFM2Error::Audio(format!("WAV error: {}", e))
    }
}