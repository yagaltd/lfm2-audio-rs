//! Main LFM2Audio model orchestration
//!
//! This module provides the high-level API for LFM2.5-Audio inference,
//! coordinating ONNX sessions, embeddings, tokenization, and caching.

use ndarray::{Array2, Array3};
use std::path::Path;

use crate::cache::GenerationCache;
use crate::config::{Device, ModelConfig, Precision, PreprocessorConfig};
use crate::embeddings::{AudioEmbedding, EmbedTokens};
use crate::error::{LFM2Error, Result};
use crate::sessions::{LFM2Sessions, SessionLoader};
use crate::tokenizer::LFM2Tokenizer;
use crate::tokenizer::CODEBOOK_VOCAB;

/// Main LFM2Audio model
pub struct LFM2Audio {
    /// ONNX sessions
    pub(crate) sessions: LFM2Sessions,
    /// Model configuration
    pub(crate) config: ModelConfig,
    /// Tokenizer
    pub(crate) tokenizer: LFM2Tokenizer,
    /// Text embeddings (from binary file)
    pub(crate) embed_tokens: EmbedTokens,
    /// Audio embeddings (from binary file, optional)
    pub(crate) audio_embedding: Option<AudioEmbedding>,
    /// Preprocessor configuration
    pub(crate) preprocessor_config: PreprocessorConfig,
}

impl LFM2Audio {
    /// Get a reference to the sessions (for async decode)
    pub fn sessions(&self) -> &LFM2Sessions {
        &self.sessions
    }

    /// Load model from directory
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        precision: Precision,
        device: Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        log::info!("Loading LFM2.5 model from {}", model_dir.display());

        // Load configuration
        let config = ModelConfig::from_file(model_dir.join("config.json"))?;
        log::info!(
            "Loaded config: {} layers, {} codebooks",
            config.lfm.num_hidden_layers,
            config.codebooks
        );

        // Load tokenizer
        let tokenizer = LFM2Tokenizer::from_dir(model_dir)?;
        log::info!(
            "Loaded tokenizer with vocab size {}",
            tokenizer.vocab_size()
        );

        // Load embeddings
        let embed_tokens = EmbedTokens::from_dir(model_dir.join("onnx"))?;
        log::info!(
            "Loaded text embeddings: {} x {}",
            embed_tokens.vocab_size(),
            embed_tokens.hidden_size()
        );

        let audio_embedding = AudioEmbedding::from_dir(model_dir.join("onnx")).ok();
        if let Some(ref ae) = audio_embedding {
            log::info!(
                "Loaded audio embeddings: {} codebooks x {} vocab",
                ae.codebooks(),
                ae.codebook_vocab()
            );
        }

        // Load ONNX sessions
        let loader = SessionLoader::new(model_dir, precision, device);
        let sessions = loader.load()?;

        // Extract preprocessor config
        let preprocessor_config = config.preprocessor.clone();

        Ok(Self {
            sessions,
            config,
            tokenizer,
            embed_tokens,
            audio_embedding,
            preprocessor_config,
        })
    }

    /// Get text embeddings for a sequence of token IDs
    /// Returns [1, seq_len, hidden_size]
    pub fn get_text_embeddings(&self, token_ids: &[u32]) -> Array3<f32> {
        self.embed_tokens.embed_sequence_array(token_ids)
    }

    /// Get audio embeddings for a frame of audio codes
    /// codes: [codebook_0, codebook_1, ..., codebook_7]
    /// Returns [1, 1, hidden_size]
    pub fn get_audio_embeddings(&self, codes: &[u16; 8]) -> Result<Array3<f32>> {
        if let Some(ref ae) = self.audio_embedding {
            // Use binary embedding lookup
            let emb = ae.lookup_codes(codes);
            Ok(Array3::from_shape_vec((1, 1, emb.len()), emb)?)
        } else if let Some(ref session) = self.sessions.audio_embedding {
            // Use ONNX session
            let input = ndarray::Array2::from_shape_vec(
                (1, 8),
                codes
                    .iter()
                    .enumerate()
                    .map(|(idx, &c)| (idx * CODEBOOK_VOCAB + c as usize) as i64)
                    .collect(),
            )?;

            // Ensure contiguous layout
            let input_contig = input.as_standard_layout().to_owned();
            let t_input = ort::value::Value::from_array(input_contig)?;
            let mut session = session.borrow_mut();
            let outputs = session.run(ort::inputs! {
                "audio_codes" => t_input,
            })?;

            // Extract and average across codebooks if needed
            let output = outputs
                .get("audio_embeds")
                .ok_or_else(|| LFM2Error::Generation("audio_embeds not found".to_string()))?;

            let view = output.try_extract_array::<f32>()?;
            let shape = view.shape();

            // Shape should be [1, 8, hidden_size] - sum across codebook dim
            if shape.len() == 3 && shape[1] == 8 {
                let hidden_size = shape[2];
                let mut result = vec![0.0f32; hidden_size];
                for i in 0..hidden_size {
                    for cb in 0..8 {
                        result[i] += view[[0, cb, i]];
                    }
                }
                Ok(Array3::from_shape_vec((1, 1, hidden_size), result)?)
            } else {
                // Assume already averaged or different shape
                let flat: Vec<f32> = view.iter().copied().collect();
                Ok(Array3::from_shape_vec((1, 1, flat.len()), flat)?)
            }
        } else {
            Err(LFM2Error::Embedding(
                "No audio embedding available".to_string(),
            ))
        }
    }

    /// Get audio embeddings for multiple frames
    pub fn get_audio_embeddings_batch(&self, frames: &[[u16; 8]]) -> Result<Array3<f32>> {
        if let Some(ref ae) = self.audio_embedding {
            Ok(ae.lookup_frames_3d(frames))
        } else {
            // Process frame by frame using ONNX
            let mut all_embs = Vec::new();
            for frame in frames {
                let emb = self.get_audio_embeddings(frame)?;
                all_embs.push(emb);
            }
            // Concatenate along sequence dimension
            let seq_len = all_embs.len();
            let hidden_size = all_embs[0].shape()[2];
            let flat: Vec<f32> = all_embs
                .into_iter()
                .flat_map(|a| {
                    let (raw, _offset) = a.into_raw_vec_and_offset();
                    raw
                })
                .collect();
            Ok(Array3::from_shape_vec((1, seq_len, hidden_size), flat)?)
        }
    }

    /// Run audio encoder to get audio embeddings from mel spectrogram
    pub fn encode_audio(&self, mel: &Array2<f32>) -> Result<Array3<f32>> {
        let num_frames = mel.shape()[0];

        // Prepare inputs: [1, num_frames, 128]
        let mel_3d = mel.clone().insert_axis(ndarray::Axis(0));

        // Mel lengths
        let mel_lengths = ndarray::Array1::from_vec(vec![num_frames as i64]);

        // Ensure contiguous layout
        let mel_contig = mel_3d.as_standard_layout().to_owned();
        let lengths_contig = mel_lengths.as_standard_layout().to_owned();
        let t_mel = ort::value::Value::from_array(mel_contig)?;
        let t_lengths = ort::value::Value::from_array(lengths_contig)?;

        let mut encoder = self.sessions.audio_encoder.borrow_mut();
        let outputs = encoder.run(ort::inputs! {
            "mel_spectrogram" => t_mel,
            "mel_lengths" => t_lengths,
        })?;

        let audio_embeds = outputs
            .get("audio_embeddings")
            .ok_or_else(|| LFM2Error::Generation("audio_embeddings not found".to_string()))?;

        let view = audio_embeds.try_extract_array::<f32>()?;
        let shape = view.shape();

        if shape.len() != 3 {
            return Err(LFM2Error::Generation(format!(
                "Expected 3D audio embeddings, got {}D",
                shape.len()
            )));
        }

        let flat: Vec<f32> = view.iter().copied().collect();
        Ok(Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            flat,
        )?)
    }

    /// Initialize a new generation cache
    pub fn init_cache(&self) -> Result<GenerationCache> {
        GenerationCache::new(&self.config.lfm)
    }

    /// Access ASR pipeline
    pub fn asr(&self) -> crate::asr::ASRPipeline<'_> {
        crate::asr::ASRPipeline::new(self)
    }

    /// Access TTS pipeline
    pub fn tts(&self) -> crate::tts::TTSPipeline<'_> {
        crate::tts::TTSPipeline::new(self)
    }

    /// Access interleaved pipeline
    pub fn interleaved(&self) -> crate::interleaved::InterleavedPipeline<'_> {
        crate::interleaved::InterleavedPipeline::new(self)
    }

    /// Start a chat session
    pub fn chat(&self) -> crate::chat::ChatSession<'_> {
        crate::chat::ChatSession::new(self)
    }

    /// Start a chat session with custom interleaved options
    pub fn chat_with_options(
        &self,
        options: crate::interleaved::InterleavedOptions,
    ) -> crate::chat::ChatSession<'_> {
        crate::chat::ChatSession::new_with_options(self, options)
    }

    /// Get model info
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            hidden_size: self.config.lfm.hidden_size,
            vocab_size: self.config.lfm.vocab_size,
            num_layers: self.config.lfm.num_hidden_layers,
            num_codebooks: self.config.codebooks,
        }
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_codebooks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        // Would need real model for full test
        // For now just verify the info struct works
        let info = ModelInfo {
            hidden_size: 2048,
            vocab_size: 65536,
            num_layers: 16,
            num_codebooks: 8,
        };
        assert_eq!(info.hidden_size, 2048);
    }
}
