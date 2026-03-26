//! Interleaved audio-text generation
//! Speech-to-Speech with text response
//! Reference: hand-voice-racer/audio-model.js:1400-1800

use ndarray::{Array2, Array3};

use crate::audio::{compute_mel_spectrogram, resample_linear};
use crate::cache::GenerationCache;
use crate::error::Result;
use crate::model::LFM2Audio;
use crate::tokenizer::END_OF_AUDIO_TOKEN;
use crate::tts::TTSPipeline;

const DEFAULT_SYSTEM_PROMPT_INTERLEAVED: &str = "Respond with interleaved text and audio.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalAudioDecode {
    Full,
    Skip,
}

/// Response from interleaved generation
#[derive(Debug, Clone)]
pub struct InterleavedResponse {
    pub text: String,
    pub audio: Vec<f32>,
    pub audio_codes: Vec<[u16; 8]>,
}

/// Incremental events emitted during interleaved generation.
#[derive(Debug, Clone)]
pub enum InterleavedEvent {
    TextUpdated(String),
    AudioFrame([u16; 8]),
}

/// Interleaved generation options
#[derive(Debug, Clone)]
pub struct InterleavedOptions {
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub text_temperature: f32,
    pub audio_temperature: f32,
    pub audio_top_k: usize,
    pub interleaved_n_text: Option<usize>,
    pub interleaved_n_audio: Option<usize>,
    /// If true, stop at <|audio_start|> and return text only (for external TTS)
    pub text_only: bool,
}

impl Default for InterleavedOptions {
    fn default() -> Self {
        Self {
            system_prompt: DEFAULT_SYSTEM_PROMPT_INTERLEAVED.to_string(),
            max_new_tokens: 1024,
            text_temperature: 1.0,
            audio_temperature: 1.0,
            audio_top_k: 4,
            interleaved_n_text: None,
            interleaved_n_audio: None,
            text_only: false,
        }
    }
}

/// Interleaved pipeline
pub struct InterleavedPipeline<'a> {
    model: &'a LFM2Audio,
}

impl<'a> InterleavedPipeline<'a> {
    fn interleaved_n_text(&self, options: &InterleavedOptions) -> usize {
        options
            .interleaved_n_text
            .unwrap_or(self.model.config.interleaved_n_text)
            .max(1)
    }

    fn interleaved_n_audio(&self, options: &InterleavedOptions) -> usize {
        options
            .interleaved_n_audio
            .unwrap_or(self.model.config.interleaved_n_audio)
            .max(1)
    }

    pub fn new(model: &'a LFM2Audio) -> Self {
        Self { model }
    }

    /// Generate interleaved response from audio input
    pub fn respond_to_audio(&self, audio: &[f32], sample_rate: u32) -> Result<InterleavedResponse> {
        self.respond_to_audio_with_options(audio, sample_rate, &InterleavedOptions::default())
    }

    pub fn respond_to_audio_with_options(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: &InterleavedOptions,
    ) -> Result<InterleavedResponse> {
        let mut cache = self.model.init_cache()?;
        let mut cache_seq_len = 0usize;
        self.continue_from_audio(
            audio,
            sample_rate,
            None,
            &mut cache,
            &mut cache_seq_len,
            true,
            options,
        )
    }

    pub fn respond_to_audio_streaming<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: &InterleavedOptions,
        on_event: &mut F,
    ) -> Result<InterleavedResponse>
    where
        F: FnMut(InterleavedEvent) -> Result<()>,
    {
        let mut cache = self.model.init_cache()?;
        let mut cache_seq_len = 0usize;
        self.continue_from_audio_streaming(
            audio,
            sample_rate,
            None,
            &mut cache,
            &mut cache_seq_len,
            true,
            options,
            on_event,
        )
    }

    /// Generate interleaved response from text input
    pub fn respond_to_text(&self, text: &str) -> Result<InterleavedResponse> {
        self.respond_to_text_with_options(text, &InterleavedOptions::default())
    }

    pub fn respond_to_text_with_options(
        &self,
        text: &str,
        options: &InterleavedOptions,
    ) -> Result<InterleavedResponse> {
        let mut cache = self.model.init_cache()?;
        let mut cache_seq_len = 0usize;
        self.continue_from_text(text, &mut cache, &mut cache_seq_len, true, options)
    }

    pub fn respond_to_text_streaming<F>(
        &self,
        text: &str,
        options: &InterleavedOptions,
        on_event: &mut F,
    ) -> Result<InterleavedResponse>
    where
        F: FnMut(InterleavedEvent) -> Result<()>,
    {
        let mut cache = self.model.init_cache()?;
        let mut cache_seq_len = 0usize;
        self.continue_from_text_streaming(
            text,
            &mut cache,
            &mut cache_seq_len,
            true,
            options,
            on_event,
        )
    }

    pub(crate) fn continue_from_text(
        &self,
        user_text: &str,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
    ) -> Result<InterleavedResponse> {
        self.continue_from_text_impl(
            user_text,
            cache,
            cache_seq_len,
            include_system_prompt,
            options,
            None,
            FinalAudioDecode::Full,
        )
    }

    pub(crate) fn continue_from_text_streaming<F>(
        &self,
        user_text: &str,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
        on_event: &mut F,
    ) -> Result<InterleavedResponse>
    where
        F: FnMut(InterleavedEvent) -> Result<()>,
    {
        self.continue_from_text_impl(
            user_text,
            cache,
            cache_seq_len,
            include_system_prompt,
            options,
            Some(on_event),
            FinalAudioDecode::Skip,
        )
    }

    pub(crate) fn continue_from_text_impl(
        &self,
        user_text: &str,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
        on_event: Option<&mut dyn FnMut(InterleavedEvent) -> Result<()>>,
        final_audio_decode: FinalAudioDecode,
    ) -> Result<InterleavedResponse> {
        let prefix_text = if include_system_prompt {
            format!(
                "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                options.system_prompt,
                user_text
            )
        } else {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_text
            )
        };

        let prefix_ids = self.model.tokenizer.encode(&prefix_text, false);
        let prefix_embeds = self.model.get_text_embeddings(&prefix_ids);
        let (logits, hidden_states) = self.prefill_decoder(&prefix_embeds, cache, cache_seq_len)?;

        self.generate_interleaved_response(
            logits,
            hidden_states,
            cache,
            cache_seq_len,
            options,
            on_event,
            final_audio_decode,
        )
    }

    pub(crate) fn continue_from_audio(
        &self,
        audio: &[f32],
        sample_rate: u32,
        text_prompt: Option<&str>,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
    ) -> Result<InterleavedResponse> {
        self.continue_from_audio_impl(
            audio,
            sample_rate,
            text_prompt,
            cache,
            cache_seq_len,
            include_system_prompt,
            options,
            None,
            FinalAudioDecode::Full,
        )
    }

    pub(crate) fn continue_from_audio_streaming<F>(
        &self,
        audio: &[f32],
        sample_rate: u32,
        text_prompt: Option<&str>,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
        on_event: &mut F,
    ) -> Result<InterleavedResponse>
    where
        F: FnMut(InterleavedEvent) -> Result<()>,
    {
        self.continue_from_audio_impl(
            audio,
            sample_rate,
            text_prompt,
            cache,
            cache_seq_len,
            include_system_prompt,
            options,
            Some(on_event),
            FinalAudioDecode::Skip,
        )
    }

    pub(crate) fn continue_from_audio_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        text_prompt: Option<&str>,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        include_system_prompt: bool,
        options: &InterleavedOptions,
        on_event: Option<&mut dyn FnMut(InterleavedEvent) -> Result<()>>,
        final_audio_decode: FinalAudioDecode,
    ) -> Result<InterleavedResponse> {
        let resampled_audio = if sample_rate != self.model.preprocessor_config.sample_rate {
            resample_linear(
                audio,
                sample_rate,
                self.model.preprocessor_config.sample_rate,
            )
        } else {
            audio.to_vec()
        };

        let mel = compute_mel_spectrogram(&resampled_audio, &self.model.preprocessor_config)?;
        let audio_embeds = self.model.encode_audio(&mel)?;

        let prefix_text = if include_system_prompt {
            format!(
                "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
                options.system_prompt
            )
        } else {
            "<|im_start|>user\n".to_string()
        };
        let suffix_text = "<|im_end|>\n<|im_start|>assistant\n";

        let prefix_ids = self.model.tokenizer.encode(&prefix_text, false);
        let suffix_ids = self.model.tokenizer.encode(suffix_text, false);
        let prefix_embeds = self.model.get_text_embeddings(&prefix_ids);
        let suffix_embeds = self.model.get_text_embeddings(&suffix_ids);

        let prompt_embeds = text_prompt.map(|text| {
            self.model
                .get_text_embeddings(&self.model.tokenizer.encode(text, false))
        });

        let all_embeds = concatenate_embeddings(&[
            Some(&prefix_embeds),
            Some(&audio_embeds),
            prompt_embeds.as_ref(),
            Some(&suffix_embeds),
        ])?;

        let (logits, hidden_states) = self.prefill_decoder(&all_embeds, cache, cache_seq_len)?;

        self.generate_interleaved_response(
            logits,
            hidden_states,
            cache,
            cache_seq_len,
            options,
            on_event,
            final_audio_decode,
        )
    }

    fn prefill_decoder(
        &self,
        input_embeds: &Array3<f32>,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
    ) -> Result<(Array3<f32>, Array3<f32>)> {
        let tts = TTSPipeline::new(self.model);
        let total_len = *cache_seq_len + input_embeds.shape()[1];
        let attention_mask = Array2::<i64>::ones((1, total_len));
        let result = tts.run_decoder_with_hidden(input_embeds, &attention_mask, cache)?;
        *cache_seq_len = total_len;
        Ok(result)
    }

    fn generate_interleaved_response(
        &self,
        mut logits: Array3<f32>,
        mut hidden_states: Array3<f32>,
        cache: &mut GenerationCache,
        cache_seq_len: &mut usize,
        options: &InterleavedOptions,
        mut on_event: Option<&mut dyn FnMut(InterleavedEvent) -> Result<()>>,
        final_audio_decode: FinalAudioDecode,
    ) -> Result<InterleavedResponse> {
        let tts = TTSPipeline::new(self.model);
        let special = self.model.tokenizer.special_tokens();
        let mut text_tokens = Vec::new();
        let mut audio_codes = Vec::new();
        let mut total_len = *cache_seq_len;
        let mut in_audio_mode = false;
        // In text_only mode (sequential), we generate ALL text until <|audio_start|> or <|im_end|>
        // In interleaved mode, we alternate between text and audio based on n_text/n_audio
        let interleaved_n_text = if options.text_only {
            usize::MAX // No limit for sequential mode - generate all text
        } else {
            self.interleaved_n_text(options)
        };
        let interleaved_n_audio = self.interleaved_n_audio(options);
        let mut modality_left = interleaved_n_text;
        let mut text_done = false;

        for _step in 0..options.max_new_tokens {
            modality_left = modality_left.saturating_sub(1);

            let next_embeds = if in_audio_mode {
                // In text_only mode, we should never enter audio mode
                debug_assert!(!options.text_only, "text_only mode should not enter audio generation");
                
                let last_hidden = tts.extract_last_hidden(&hidden_states);
                let mut frame = tts.sample_audio_frame(
                    &last_hidden,
                    options.audio_temperature,
                    options.audio_top_k,
                )?;

                if modality_left == 0 && !text_done {
                    in_audio_mode = false;
                    modality_left = interleaved_n_text;
                }

                let feed_codes = if frame[0] == END_OF_AUDIO_TOKEN {
                    frame = [END_OF_AUDIO_TOKEN; 8];
                    in_audio_mode = false;
                    frame
                } else {
                    let clamped = frame.map(|code| code.min(2047));
                    audio_codes.push(clamped);
                    if let Some(callback) = on_event.as_mut() {
                        callback(InterleavedEvent::AudioFrame(clamped))?;
                    }
                    frame.map(|code| {
                        if code == END_OF_AUDIO_TOKEN {
                            END_OF_AUDIO_TOKEN
                        } else {
                            code.min(2047)
                        }
                    })
                };

                self.model.get_audio_embeddings(&feed_codes)?
            } else {
                let last_logits = extract_last_logits(&logits, self.model.config.lfm.vocab_size)?;
                let token = if options.text_temperature == 0.0 {
                    argmax(&last_logits)
                } else {
                    sample_with_temperature(&last_logits, options.text_temperature)
                };

                if token == special.end_of_text || token == special.im_end {
                    log::info!(
                        "Interleaved: reached end token {} after {} text tokens",
                        token,
                        text_tokens.len()
                    );
                    break;
                }

                if token == special.text_end {
                    text_done = true;
                }

                if token == special.audio_start {
                    // Text-only mode: stop here and return text for external TTS
                    if options.text_only {
                        log::info!(
                            "Interleaved: text_only mode - stopping at audio_start after {} text tokens",
                            text_tokens.len()
                        );
                        break;
                    }
                    in_audio_mode = true;
                    modality_left = interleaved_n_audio;
                } else if (modality_left == 0 || text_done) && !options.text_only {
                    // In interleaved mode (not text_only), switch to audio when text limit reached
                    in_audio_mode = true;
                    modality_left = interleaved_n_audio;
                }

                text_tokens.push(token);
                if let Some(callback) = on_event.as_mut() {
                    callback(InterleavedEvent::TextUpdated(
                        self.model.tokenizer.decode(&text_tokens, true),
                    ))?;
                }
                self.model.get_text_embeddings(&[token])
            };

            total_len += 1;
            let attention_mask = Array2::<i64>::ones((1, total_len));
            let (new_logits, new_hidden_states) =
                tts.run_decoder_with_hidden(&next_embeds, &attention_mask, cache)?;
            logits = new_logits;
            hidden_states = new_hidden_states;
        }

        let im_end_embeds = self.model.get_text_embeddings(&[special.im_end]);
        total_len += 1;
        let attention_mask = Array2::<i64>::ones((1, total_len));
        let _ = tts.run_decoder_with_hidden(&im_end_embeds, &attention_mask, cache)?;
        *cache_seq_len = total_len;

        let text = self.model.tokenizer.decode(&text_tokens, true);
        let audio = match final_audio_decode {
            FinalAudioDecode::Full => tts.decode_audio_codes(&audio_codes)?,
            FinalAudioDecode::Skip => Vec::new(),
        };

        Ok(InterleavedResponse {
            text,
            audio,
            audio_codes,
        })
    }
}

fn concatenate_embeddings(parts: &[Option<&Array3<f32>>]) -> Result<Array3<f32>> {
    let hidden_size = parts
        .iter()
        .flatten()
        .next()
        .map(|part| part.shape()[2])
        .unwrap_or(0);
    let seq_len: usize = parts.iter().flatten().map(|part| part.shape()[1]).sum();

    let mut flat = Vec::with_capacity(seq_len * hidden_size);
    for part in parts.iter().flatten() {
        flat.extend(part.iter().copied());
    }

    Ok(Array3::from_shape_vec((1, seq_len, hidden_size), flat)?)
}

fn extract_last_logits(logits: &Array3<f32>, vocab_size: usize) -> Result<Vec<f32>> {
    let seq_len = logits.shape()[1];
    let offset = (seq_len - 1) * vocab_size;
    let data: Vec<f32> = logits.iter().copied().collect();
    Ok(data[offset..offset + vocab_size].to_vec())
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        return argmax(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_shifted: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_shifted.iter().sum();
    let probs: Vec<f32> = exp_shifted.iter().map(|&x| x / sum_exp).collect();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return idx as u32;
        }
    }

    probs.len() as u32 - 1
}
