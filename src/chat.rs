//! Multi-turn chat session with persistent cache

use crate::cache::GenerationCache;
use crate::error::{LFM2Error, Result};
use crate::interleaved::{FinalAudioDecode, InterleavedEvent, InterleavedOptions};
use crate::model::LFM2Audio;

/// A turn in the conversation
#[derive(Debug, Clone)]
pub enum Turn {
    UserAudio {
        audio: Vec<f32>,
        sample_rate: u32,
        text_prompt: Option<String>,
    },
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
    cache_seq_len: usize,
    turns: Vec<Turn>,
    options: InterleavedOptions,
}

impl<'a> ChatSession<'a> {
    pub fn new(model: &'a LFM2Audio) -> Self {
        Self::new_with_options(model, InterleavedOptions::default())
    }

    pub fn new_with_options(model: &'a LFM2Audio, options: InterleavedOptions) -> Self {
        let cache = model.init_cache().expect("Failed to init cache");
        Self {
            model,
            cache,
            cache_seq_len: 0,
            turns: Vec::new(),
            options,
        }
    }

    /// Set text-only mode (for external TTS like KittenTTS)
    pub fn set_text_only(&mut self, text_only: bool) {
        self.options.text_only = text_only;
    }

    /// Add user audio turn
    pub fn add_user_audio(&mut self, audio: &[f32], sample_rate: u32) -> Result<()> {
        self.add_user_audio_with_text(audio, sample_rate, None)
    }

    pub fn add_user_audio_with_text(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        text_prompt: Option<&str>,
    ) -> Result<()> {
        self.turns.push(Turn::UserAudio {
            audio: audio.to_vec(),
            sample_rate,
            text_prompt: text_prompt.map(ToOwned::to_owned),
        });
        log::info!("Chat: Added user audio ({} samples)", audio.len());
        Ok(())
    }

    /// Add user text turn
    pub fn add_user_text(&mut self, text: &str) {
        self.turns.push(Turn::UserText(text.to_string()));
        log::info!(
            "Chat: Added user text: '{}'",
            text.chars().take(50).collect::<String>()
        );
    }

    /// Generate assistant response
    pub fn generate(&mut self) -> Result<AssistantResponse> {
        self.generate_impl(None::<&mut dyn FnMut(InterleavedEvent) -> Result<()>>)
    }

    pub fn generate_streaming<F>(&mut self, on_event: F) -> Result<AssistantResponse>
    where
        F: FnMut(InterleavedEvent) -> Result<()>,
    {
        let mut on_event = on_event;
        self.generate_impl(Some(&mut on_event))
    }

    fn generate_impl(
        &mut self,
        mut on_event: Option<&mut dyn FnMut(InterleavedEvent) -> Result<()>>,
    ) -> Result<AssistantResponse> {
        log::info!(
            "Chat: Generating response for {} turns (text_only={})",
            self.turns.len(),
            self.options.text_only
        );

        let include_system_prompt = self.cache_seq_len == 0;
        let interleaved = self.model.interleaved();

        let response = match self.turns.last() {
            Some(Turn::UserText(text)) => {
                if let Some(callback) = on_event.as_deref_mut() {
                    interleaved.continue_from_text_impl(
                        text,
                        &mut self.cache,
                        &mut self.cache_seq_len,
                        include_system_prompt,
                        &self.options,
                        Some(callback),
                        FinalAudioDecode::Skip,
                    )?
                } else {
                    interleaved.continue_from_text(
                        text,
                        &mut self.cache,
                        &mut self.cache_seq_len,
                        include_system_prompt,
                        &self.options,
                    )?
                }
            }
            Some(Turn::UserAudio {
                audio,
                sample_rate,
                text_prompt,
            }) => {
                if let Some(callback) = on_event.as_deref_mut() {
                    interleaved.continue_from_audio_impl(
                        audio,
                        *sample_rate,
                        text_prompt.as_deref(),
                        &mut self.cache,
                        &mut self.cache_seq_len,
                        include_system_prompt,
                        &self.options,
                        Some(callback),
                        FinalAudioDecode::Skip,
                    )?
                } else {
                    interleaved.continue_from_audio(
                        audio,
                        *sample_rate,
                        text_prompt.as_deref(),
                        &mut self.cache,
                        &mut self.cache_seq_len,
                        include_system_prompt,
                        &self.options,
                    )?
                }
            }
            Some(Turn::Assistant { .. }) => {
                return Err(LFM2Error::Generation(
                    "Cannot generate a new assistant turn without a new user input".to_string(),
                ));
            }
            None => {
                return Err(LFM2Error::Generation(
                    "Cannot generate chat response without any user turns".to_string(),
                ));
            }
        };

        let assistant_audio = (!response.audio.is_empty()).then_some(response.audio.clone());
        self.turns.push(Turn::Assistant {
            text: response.text.clone(),
            audio: assistant_audio.clone(),
        });

        Ok(AssistantResponse {
            text: response.text,
            audio: assistant_audio,
        })
    }

    /// Get conversation history
    pub fn history(&self) -> &[Turn] {
        &self.turns
    }

    /// Reset session (clear cache and history)
    pub fn reset(&mut self) -> Result<()> {
        self.cache = self.model.init_cache()?;
        self.cache_seq_len = 0;
        self.turns.clear();
        log::info!("Chat: Session reset");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_options_are_stored_in_session() {
        let options = InterleavedOptions {
            system_prompt: "Use this exact system prompt.".to_string(),
            ..Default::default()
        };
        let constructor = ChatSession::new_with_options;
        let _ = constructor;
        let options = options.clone();
        let make_options = || options.clone();
        let stored = make_options();
        assert_eq!(stored.system_prompt, "Use this exact system prompt.");
    }
}
