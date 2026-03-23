//! Binary embedding loaders for text and audio tokens
//!
//! Loads embed_tokens.bin and audio_embedding.bin for direct lookup
//! Reference: hand-voice-racer/audio-model.js:400-500

use ndarray::{Array2, Array3};
use std::fs;
use std::path::Path;

use crate::error::{LFM2Error, Result};

/// Text token embeddings loaded from embed_tokens.bin
#[derive(Debug, Clone)]
pub struct EmbedTokens {
    /// Flattened weight matrix [vocab_size * hidden_size]
    weight: Vec<f32>,
    hidden_size: usize,
    vocab_size: usize,
}

/// Metadata for embed_tokens
#[derive(Debug, Clone, serde::Deserialize)]
struct EmbedTokensMeta {
    #[serde(rename = "vocab_size")]
    vocab_size: usize,
    #[serde(rename = "hidden_size")]
    hidden_size: usize,
}

impl EmbedTokens {
    /// Load from directory containing embed_tokens.bin and embed_tokens.json
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let bin_path = dir.join("embed_tokens.bin");
        let json_path = dir.join("embed_tokens.json");
        
        if !bin_path.exists() {
            return Err(LFM2Error::Embedding(format!(
                "embed_tokens.bin not found in {}",
                dir.display()
            )));
        }
        
        // Load metadata
        let meta: EmbedTokensMeta = if json_path.exists() {
            let content = fs::read_to_string(&json_path)?;
            serde_json::from_str(&content)?
        } else {
            // Infer from binary size if no metadata
            return Err(LFM2Error::Embedding(
                "embed_tokens.json metadata not found".to_string()
            ));
        };
        
        // Load binary weights
        let buf = fs::read(&bin_path)?;
        let expected_bytes = meta.vocab_size * meta.hidden_size * 4;
        
        if buf.len() != expected_bytes {
            return Err(LFM2Error::Embedding(format!(
                "embed_tokens.bin size mismatch: expected {} bytes, got {}",
                expected_bytes,
                buf.len()
            )));
        }
        
        // Convert bytes to f32 (little-endian)
        let weight: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok(Self {
            weight,
            hidden_size: meta.hidden_size,
            vocab_size: meta.vocab_size,
        })
    }
    
    /// Get embedding for a single token
    /// Returns slice of length hidden_size
    pub fn lookup(&self, token_id: u32) -> &[f32] {
        let offset = token_id as usize * self.hidden_size;
        &self.weight[offset..offset + self.hidden_size]
    }
    
    /// Get embeddings for a sequence of tokens
    /// Returns [seq_len, hidden_size] as flat Vec
    pub fn embed_sequence(&self, token_ids: &[u32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(token_ids.len() * self.hidden_size);
        for &id in token_ids {
            result.extend_from_slice(self.lookup(id));
        }
        result
    }
    
    /// Get embeddings as 3D array [1, seq_len, hidden_size]
    pub fn embed_sequence_array(&self, token_ids: &[u32]) -> Array3<f32> {
        let flat = self.embed_sequence(token_ids);
        Array3::from_shape_vec(
            (1, token_ids.len(), self.hidden_size),
            flat
        ).expect("Shape should match")
    }
    
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Audio code embeddings loaded from audio_embedding.bin
#[derive(Debug, Clone)]
pub struct AudioEmbedding {
    /// Flattened weight matrix [num_codebooks * codebook_vocab * hidden_size]
    weight: Vec<f32>,
    codebooks: usize,
    codebook_vocab: usize,
    hidden_size: usize,
}

/// Metadata for audio_embedding
#[derive(Debug, Clone, serde::Deserialize)]
struct AudioEmbeddingMeta {
    #[serde(rename = "codebooks")]
    codebooks: usize,
    #[serde(rename = "codebook_vocab")]
    codebook_vocab: usize,
    #[serde(rename = "hidden_size")]
    hidden_size: usize,
}

impl AudioEmbedding {
    /// Load from directory containing audio_embedding.bin and audio_embedding.json
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let bin_path = dir.join("audio_embedding.bin");
        let json_path = dir.join("audio_embedding.json");
        
        if !bin_path.exists() {
            return Err(LFM2Error::Embedding(format!(
                "audio_embedding.bin not found in {}",
                dir.display()
            )));
        }
        
        // Load metadata
        let meta: AudioEmbeddingMeta = if json_path.exists() {
            let content = fs::read_to_string(&json_path)?;
            serde_json::from_str(&content)?
        } else {
            return Err(LFM2Error::Embedding(
                "audio_embedding.json metadata not found".to_string()
            ));
        };
        
        // Load binary weights
        let buf = fs::read(&bin_path)?;
        let expected_bytes = meta.codebooks * meta.codebook_vocab * meta.hidden_size * 4;
        
        if buf.len() != expected_bytes {
            return Err(LFM2Error::Embedding(format!(
                "audio_embedding.bin size mismatch: expected {} bytes, got {}",
                expected_bytes,
                buf.len()
            )));
        }
        
        let weight: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok(Self {
            weight,
            codebooks: meta.codebooks,
            codebook_vocab: meta.codebook_vocab,
            hidden_size: meta.hidden_size,
        })
    }
    
    /// Lookup embedding for a single codebook token
    /// codebook_idx: 0-7, token_id: 0-2048
    fn lookup_single(&self, codebook_idx: usize, token_id: u16) -> &[f32] {
        let idx = codebook_idx * self.codebook_vocab + token_id as usize;
        let offset = idx * self.hidden_size;
        &self.weight[offset..offset + self.hidden_size]
    }
    
    /// Get averaged embedding across all codebooks for a frame of codes
    /// codes: [codebook_0, codebook_1, ..., codebook_7]
    /// Returns: averaged embedding of length hidden_size
    pub fn lookup_codes(&self, codes: &[u16; 8]) -> Vec<f32> {
        let mut result = vec![0.0f32; self.hidden_size];
        
        // Average embeddings across codebooks
        for (cb_idx, &token_id) in codes.iter().enumerate() {
            let emb = self.lookup_single(cb_idx, token_id);
            for i in 0..self.hidden_size {
                result[i] += emb[i];
            }
        }
        
        // Divide by number of codebooks
        for i in 0..self.hidden_size {
            result[i] /= self.codebooks as f32;
        }
        
        result
    }
    
    /// Get embeddings for multiple frames
    /// Returns [num_frames, hidden_size]
    pub fn lookup_frames(&self, frames: &[[u16; 8]]) -> Array2<f32> {
        let num_frames = frames.len();
        let mut result = Vec::with_capacity(num_frames * self.hidden_size);
        
        for frame in frames {
            result.extend_from_slice(&self.lookup_codes(frame));
        }
        
        Array2::from_shape_vec((num_frames, self.hidden_size), result)
            .expect("Shape should match")
    }
    
    /// Get embeddings as 3D array [1, num_frames, hidden_size]
    pub fn lookup_frames_3d(&self, frames: &[[u16; 8]]) -> Array3<f32> {
        let num_frames = frames.len();
        let mut result = Vec::with_capacity(num_frames * self.hidden_size);
        
        for frame in frames {
            result.extend_from_slice(&self.lookup_codes(frame));
        }
        
        Array3::from_shape_vec((1, num_frames, self.hidden_size), result)
            .expect("Shape should match")
    }
    
    pub fn codebooks(&self) -> usize {
        self.codebooks
    }
    
    pub fn codebook_vocab(&self) -> usize {
        self.codebook_vocab
    }
    
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}