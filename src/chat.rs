//! Multi-turn chat session with persistent cache

use crate::cache::GenerationCache;
use crate::error::Result;
use crate::model::LFM2Audio;

/// A turn in the conversation
#[derive(Debug, Clone)]
pub enum Turn {
    UserAudio(Vec<f32>),
    UserText(String),
    Assistant {
        text: String,
        audio: Option<Vec<f32>>,
    },
}

/// Assistant response
#[derive(Debug, Clone)]
pub struct AssistantResponse {
    pub text: String,
    pub audio: Option<Vec<f32>>,
}

/// Chat session with persistent state
pub struct ChatSession<'a> {
    model: &'a LFM2Audio,
    cache: GenerationCache,
    turns: Vec<Turn>,
}

impl<'a> ChatSession<'a> {
    pub fn new(model: &'a LFM2Audio) -> Self {
        let cache = model.init_cache().expect("Failed to init cache");
        Self {
            model,
            cache,
            turns: Vec::new(),
        }
    }

    /// Add user audio turn
    pub fn add_user_audio(&mut self, audio: &[f32]) -> Result<()> {
        self.turns.push(Turn::UserAudio(audio.to_vec()));
        log::info!("Chat: Added user audio ({} samples)", audio.len());
        Ok(())
    }

    /// Add user text turn
    pub fn add_user_text(&mut self, text: &str) {
        self.turns.push(Turn::UserText(text.to_string()));
        log::info!("Chat: Added user text: '{}'", text.chars().take(50).collect::<String>());
    }

    /// Generate assistant response
    pub fn generate(&mut self) -> Result<AssistantResponse> {
        log::info!("Chat: Generating response for {} turns", self.turns.len());

        // TODO: Implement full chat generation with persistent cache

        Ok(AssistantResponse {
            text: "Not yet implemented".to_string(),
            audio: None,
        })
    }

    /// Get conversation history
    pub fn history(&self) -> &[Turn] {
        &self.turns
    }

    /// Reset session (clear cache and history)
    pub fn reset(&mut self) -> Result<()> {
        self.cache = self.model.init_cache()?;
        self.turns.clear();
        log::info!("Chat: Session reset");
        Ok(())
    }
}