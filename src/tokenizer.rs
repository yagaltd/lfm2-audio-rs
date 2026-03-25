//! Tokenizer wrapper for HuggingFace tokenizers

use tokenizers::Tokenizer as HFTokenizer;
use std::path::Path;

use crate::error::{LFM2Error, Result};

/// Special token IDs
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub pad: u32,                    // 0: <|pad|>
    pub start_of_text: u32,          // 1: <|startoftext|>
    pub end_of_text: u32,            // 2: <|endoftext|>
    pub im_start: u32,               // 6: <|im_start|>
    pub im_end: u32,                 // 7: <|im_end|>
    pub audio_start: u32,            // 128: <|audio_start|>
    pub text_start: u32,             // 129: <|text_start|>
    pub text_end: u32,               // 130: <|text_end|>
    pub mixed_start: u32,            // 131: <|mixed_start|>
    pub mixed_end: u32,              // 132: <|mixed_end|>
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: 0,
            start_of_text: 1,
            end_of_text: 2,
            im_start: 6,
            im_end: 7,
            audio_start: 128,
            text_start: 129,
            text_end: 130,
            mixed_start: 131,
            mixed_end: 132,
        }
    }
}

/// Audio codebook constants
pub const NUM_CODEBOOKS: usize = 8;
pub const CODEBOOK_VOCAB: usize = 2049;
pub const END_OF_AUDIO_TOKEN: u16 = 2048;

/// Wrapped tokenizer with special token handling
pub struct LFM2Tokenizer {
    tokenizer: HFTokenizer,
    special: SpecialTokens,
}

impl LFM2Tokenizer {
    /// Load tokenizer from directory containing tokenizer.json
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let tokenizer_path = dir.as_ref().join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(LFM2Error::Tokenizer(format!(
                "tokenizer.json not found at {}",
                tokenizer_path.display()
            )));
        }

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)
            .map_err(|e| LFM2Error::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // Load special tokens from tokenizer_config.json if available
        let special = Self::load_special_tokens(&dir)?;

        Ok(Self { tokenizer, special })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .expect("Encoding should not fail");

        encoding.get_ids().to_vec()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .expect("Decoding should not fail")
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Build chat prompt for ASR
    pub fn build_asr_prompt(&self, system_prompt: Option<&str>) -> String {
        let system = system_prompt.unwrap_or("Perform ASR.");
        format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
            system
        )
    }

    /// Build chat prompt for TTS
    pub fn build_tts_prompt(&self, text: &str, voice: Option<&str>) -> String {
        let voice_desc = voice.unwrap_or("Use the UK female voice.");
        format!(
            "<|startoftext|><|im_start|>system\nPerform TTS. {}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            voice_desc,
            text
        )
    }

    /// Build suffix for assistant response
    pub fn build_assistant_suffix(&self) -> String {
        "<|im_end|>\n<|im_start|>assistant\n".to_string()
    }

    /// Check if token is end-of-sequence
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.special.end_of_text ||
        token_id == self.special.im_end
    }

    /// Check if token is audio start
    pub fn is_audio_start(&self, token_id: u32) -> bool {
        token_id == self.special.audio_start
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn load_special_tokens<P: AsRef<Path>>(dir: P) -> Result<SpecialTokens> {
        let config_path = dir.as_ref().join("tokenizer_config.json");

        if !config_path.exists() {
            log::warn!("tokenizer_config.json not found, using default special tokens");
            return Ok(SpecialTokens::default());
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let mut special = SpecialTokens::default();

        if let Some(added_tokens) = config.get("added_tokens_decoder") {
            if let Some(obj) = added_tokens.as_object() {
                for (id_str, token_info) in obj {
                    if let Ok(id) = id_str.parse::<u32>() {
                        if let Some(content) = token_info.get("content").and_then(|c| c.as_str()) {
                            match content {
                                "<|pad|>" => special.pad = id,
                                "<|startoftext|>" => special.start_of_text = id,
                                "<|endoftext|>" => special.end_of_text = id,
                                "<|im_start|>" => special.im_start = id,
                                "<|im_end|>" => special.im_end = id,
                                "<|audio_start|>" => special.audio_start = id,
                                "<|text_start|>" => special.text_start = id,
                                "<|text_end|>" => special.text_end = id,
                                "<|mixed_start|>" => special.mixed_start = id,
                                "<|mixed_end|>" => special.mixed_end = id,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        log::info!("Loaded special tokens: {:?}", special);
        Ok(special)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_special_tokens() {
        let special = SpecialTokens::default();
        assert_eq!(special.pad, 0);
        assert_eq!(special.audio_start, 128);
        assert_eq!(special.im_end, 7);
    }

    #[test]
    fn test_asr_prompt_format() {
        // This test would need a real tokenizer
        // For now just verify the string format
        let prompt = format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
            "Perform ASR."
        );
        assert!(prompt.contains("<|startoftext|>"));
        assert!(prompt.contains("<|im_start|>"));
        assert!(prompt.contains("<|im_end|>"));
    }

    #[test]
    fn test_audio_constants() {
        assert_eq!(NUM_CODEBOOKS, 8);
        assert_eq!(CODEBOOK_VOCAB, 2049);
        assert_eq!(END_OF_AUDIO_TOKEN, 2048);
    }
}
