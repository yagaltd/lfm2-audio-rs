//! ONNX Runtime session management for LFM2.5 models
//! Reference: hand-voice-racer/audio-model.js:300-450 (loadOnnxWithExternalData)

use ort::session::builder::GraphOptimizationLevel;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::path::{Path, PathBuf};

use crate::config::{Device, Precision};
use crate::error::{LFM2Error, Result};

/// Container for all LFM2.5 ONNX sessions
/// Uses RefCell for interior mutability since session.run() requires &mut self
/// Audio detokenizer uses Arc<Mutex> for async decode support
pub struct LFM2Sessions {
    /// Audio encoder: mel → audio embeddings
    pub audio_encoder: RefCell<ort::session::Session>,
    /// LFM2 decoder: main autoregressive model
    pub decoder: RefCell<ort::session::Session>,
    /// Depthformer: audio codebook prediction
    pub depthformer: RefCell<ort::session::Session>,
    /// Audio detokenizer: codes → STFT → waveform (thread-safe for async decode)
    pub audio_detokenizer: Arc<Mutex<ort::session::Session>>,
    /// Audio embedding (optional - can use binary lookup instead)
    pub audio_embedding: Option<RefCell<ort::session::Session>>,
}

/// Session loader configuration
pub struct SessionLoader {
    model_dir: PathBuf,
    precision: Precision,
    device: Device,
}

impl SessionLoader {
    pub fn new<P: AsRef<Path>>(model_dir: P, precision: Precision, device: Device) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
            precision,
            device,
        }
    }

    /// Load all sessions
    pub fn load(&self) -> Result<LFM2Sessions> {
        log::info!("Loading LFM2.5 models with precision: {:?}", self.precision);

        let onnx_dir = self.model_dir.join("onnx");

        // Load each model with external data
        let audio_encoder = self.load_model(&onnx_dir, "audio_encoder")?;
        log::info!("Loaded audio_encoder");

        let decoder = self.load_decoder(&onnx_dir)?;
        log::info!("Loaded decoder");

        let depthformer = self.load_model(&onnx_dir, "vocoder_depthformer")?;
        log::info!("Loaded depthformer");

        let audio_detokenizer = self.load_model(&onnx_dir, "audio_detokenizer")?;
        log::info!("Loaded audio_detokenizer");

        // Audio embedding is optional - we can use binary lookup instead
        let audio_embedding = self.load_model(&onnx_dir, "audio_embedding").ok();
        if audio_embedding.is_some() {
            log::info!("Loaded audio_embedding");
        } else {
            log::info!("Audio embedding ONNX not found - will use binary lookup");
        }

        Ok(LFM2Sessions {
            audio_encoder: RefCell::new(audio_encoder),
            decoder: RefCell::new(decoder),
            depthformer: RefCell::new(depthformer),
            audio_detokenizer: Arc::new(Mutex::new(audio_detokenizer)),
            audio_embedding: audio_embedding.map(RefCell::new),
        })
    }

    /// Load a single model with external data support
    fn load_model(&self, onnx_dir: &Path, name: &str) -> Result<ort::session::Session> {
        let suffix = self.precision.suffix();
        let file_name = format!("{}{}.onnx", name, suffix);
        let model_path = onnx_dir.join(&file_name);

        if !model_path.exists() {
            return Err(LFM2Error::ModelNotFound(format!(
                "{} not found at {}",
                file_name,
                model_path.display()
            )));
        }

        // Build session
        let mut builder = ort::session::Session::builder()
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        // Set optimization level and memory pattern
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| LFM2Error::Onnx(e.into()))?
            .with_memory_pattern(true)
            .map_err(|e| LFM2Error::Onnx(e.into()))?
            .with_intra_threads(8)
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        // Set execution providers
        let eps = self.device.execution_providers();
        if !eps.is_empty() {
            builder = builder
                .with_execution_providers(eps)
                .map_err(|e| LFM2Error::Onnx(e.into()))?;
        }

        // Create session
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        log::debug!("Loaded {}", name);

        Ok(session)
    }

    /// Load decoder with special handling for output locations (WebGPU support)
    fn load_decoder(&self, onnx_dir: &Path) -> Result<ort::session::Session> {
        let suffix = self.precision.suffix();
        let file_name = format!("decoder{}.onnx", suffix);
        let model_path = onnx_dir.join(&file_name);

        if !model_path.exists() {
            return Err(LFM2Error::ModelNotFound(format!(
                "Decoder not found at {}",
                model_path.display()
            )));
        }

        let mut builder = ort::session::Session::builder()
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| LFM2Error::Onnx(e.into()))?
            .with_memory_pattern(true)
            .map_err(|e| LFM2Error::Onnx(e.into()))?
            .with_intra_threads(8)
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        let eps = self.device.execution_providers();
        if !eps.is_empty() {
            builder = builder
                .with_execution_providers(eps)
                .map_err(|e| LFM2Error::Onnx(e.into()))?;
        }

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| LFM2Error::Onnx(e.into()))?;

        Ok(session)
    }
}

/// Model file paths for reference
pub struct ModelPaths {
    pub audio_encoder: PathBuf,
    pub decoder: PathBuf,
    pub depthformer: PathBuf,
    pub audio_detokenizer: PathBuf,
    pub audio_embedding: PathBuf,
}

impl ModelPaths {
    pub fn from_dir<P: AsRef<Path>>(dir: P, precision: Precision) -> Self {
        let onnx_dir = dir.as_ref().join("onnx");
        let suffix = precision.suffix();

        Self {
            audio_encoder: onnx_dir.join(format!("audio_encoder{}.onnx", suffix)),
            decoder: onnx_dir.join(format!("decoder{}.onnx", suffix)),
            depthformer: onnx_dir.join(format!("vocoder_depthformer{}.onnx", suffix)),
            audio_detokenizer: onnx_dir.join(format!("audio_detokenizer{}.onnx", suffix)),
            audio_embedding: onnx_dir.join(format!("audio_embedding{}.onnx", suffix)),
        }
    }

    pub fn all_exist(&self) -> Result<()> {
        let required = [
            (&self.audio_encoder, "audio_encoder"),
            (&self.decoder, "decoder"),
            (&self.depthformer, "depthformer"),
            (&self.audio_detokenizer, "audio_detokenizer"),
        ];

        for (path, name) in &required {
            if !path.exists() {
                return Err(LFM2Error::ModelNotFound(format!(
                    "Required model {} not found at {}",
                    name,
                    path.display()
                )));
            }
        }

        Ok(())
    }
}
