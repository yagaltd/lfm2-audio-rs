//! TTS (Text-to-Speech) pipeline
//! Text → Audio
//! Reference: hand-voice-racer/audio-model.js:1000-1400

use crate::cache::GenerationCache;
use crate::error::{LFM2Error, Result};
use crate::model::LFM2Audio;
use crate::tokenizer::{CODEBOOK_VOCAB, END_OF_AUDIO_TOKEN, NUM_CODEBOOKS};

/// TTS pipeline
pub struct TTSPipeline<'a> {
    model: &'a LFM2Audio,
}

/// TTS generation options
#[derive(Debug, Clone)]
pub struct TTSOptions {
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub text_temperature: f32,
    pub audio_temperature: f32,
    pub audio_top_k: usize,
}

impl Default for TTSOptions {
    fn default() -> Self {
        Self {
            system_prompt: "Perform TTS. Use the UK female voice.".to_string(),
            max_new_tokens: 1024,
            text_temperature: 1.0,
            audio_temperature: 0.8,
            audio_top_k: 64,
        }
    }
}

/// Debug information for one TTS generation run.
#[derive(Debug, Clone)]
pub struct TTSDebugOutput {
    pub text_tokens: Vec<u32>,
    pub audio_codes: Vec<[u16; 8]>,
    pub audio: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum TTSEvent {
    TextUpdated(String),
    AudioFrame([u16; 8]),
}

#[derive(Debug, Clone)]
pub struct TTSStreamOutput {
    pub text: String,
    pub audio_codes: Vec<[u16; 8]>,
}

#[derive(Debug, Clone)]
pub struct TTSLogitCandidate {
    pub token: u32,
    pub logit: f32,
}

#[derive(Debug, Clone)]
pub struct TTSCodebookTrace {
    pub chosen_token: u16,
    pub argmax_token: u32,
    pub argmax_logit: f32,
    pub top_candidates: Vec<TTSLogitCandidate>,
}

#[derive(Debug, Clone)]
pub struct TTSFrameTrace {
    pub frame_index: usize,
    pub hidden_prefix: Vec<f32>,
    pub codebook_traces: Vec<TTSCodebookTrace>,
}

#[derive(Debug, Clone)]
pub struct TTSTraceOutput {
    pub text_tokens: Vec<u32>,
    pub audio_codes: Vec<[u16; 8]>,
    pub frame_traces: Vec<TTSFrameTrace>,
}

impl TTSOptions {
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = prompt.to_string();
        self
    }
}

impl<'a> TTSPipeline<'a> {
    pub fn new(model: &'a LFM2Audio) -> Self {
        Self { model }
    }

    /// Synthesize text to speech
    pub fn synthesize(&self, text: &str, options: &TTSOptions) -> Result<Vec<f32>> {
        Ok(self.synthesize_debug(text, options)?.audio)
    }

    pub fn synthesize_streaming<F>(
        &self,
        text: &str,
        options: &TTSOptions,
        on_event: &mut F,
    ) -> Result<TTSStreamOutput>
    where
        F: FnMut(TTSEvent) -> Result<()>,
    {
        let (text_tokens, mut cache, mut hidden_states, mut current_len) =
            self.generate_text_streaming(text, options, on_event)?;
        let audio_codes = self.generate_audio_codes_streaming(
            &mut cache,
            &mut hidden_states,
            &mut current_len,
            options,
            on_event,
        )?;
        Ok(TTSStreamOutput {
            text: self.model.tokenizer.decode(&text_tokens, true),
            audio_codes,
        })
    }

    /// Synthesize text to speech and return intermediate generation artifacts.
    pub fn synthesize_debug(&self, text: &str, options: &TTSOptions) -> Result<TTSDebugOutput> {
        log::info!(
            "TTS: Synthesizing '{}'",
            text.chars().take(50).collect::<String>()
        );

        // Phase 1: Generate text tokens until <|audio_start|>
        let (text_tokens, mut cache, mut hidden_states, mut current_len) =
            self.generate_text(text, options)?;

        log::info!(
            "TTS: Text phase complete, {} tokens, entering audio mode",
            text_tokens.len()
        );

        // Phase 2: Generate audio codes autoregressively
        let audio_codes =
            self.generate_audio_codes(&mut cache, &mut hidden_states, &mut current_len, options)?;

        log::info!("TTS: Generated {} audio frames", audio_codes.len());

        // Phase 3: Decode audio codes to waveform
        let audio = self.decode_audio_codes(&audio_codes)?;

        log::info!("TTS: Generated {} samples at 24kHz", audio.len());

        Ok(TTSDebugOutput {
            text_tokens,
            audio_codes,
            audio,
        })
    }

    /// Trace intermediate decoder/depthformer state for the first N audio frames.
    pub fn synthesize_trace(
        &self,
        text: &str,
        options: &TTSOptions,
        trace_frames: usize,
    ) -> Result<TTSTraceOutput> {
        let (text_tokens, mut cache, mut hidden_states, mut current_len) =
            self.generate_text(text, options)?;

        let mut audio_codes = Vec::new();
        let mut frame_traces = Vec::new();
        let mut last_hidden = self.extract_last_hidden(&hidden_states);

        for frame_idx in 0..options.max_new_tokens {
            let trace = if frame_idx < trace_frames {
                Some(TTSFrameTrace {
                    frame_index: frame_idx,
                    hidden_prefix: last_hidden.iter().take(16).copied().collect(),
                    codebook_traces: Vec::new(),
                })
            } else {
                None
            };

            let (frame_codes, trace) = self.sample_audio_codes_with_trace(
                &last_hidden,
                options.audio_temperature,
                options.audio_top_k,
                trace,
            )?;

            if frame_codes[0] == END_OF_AUDIO_TOKEN {
                break;
            }

            if let Some(trace) = trace {
                frame_traces.push(trace);
            }

            audio_codes.push(frame_codes);

            let clamped_codes = frame_codes.map(|c| c.min(2047));
            let audio_embeds = self.model.get_audio_embeddings(&clamped_codes)?;
            current_len += 1;
            let attention_mask = ndarray::Array2::<i64>::ones((1, current_len));
            let (_logits, new_hidden_states) =
                self.run_decoder_with_hidden(&audio_embeds, &attention_mask, &mut cache)?;
            hidden_states = new_hidden_states;
            last_hidden = self.extract_last_hidden(&hidden_states);
        }

        Ok(TTSTraceOutput {
            text_tokens,
            audio_codes,
            frame_traces,
        })
    }

    /// Generate text portion until <|audio_start|> or max tokens
    fn generate_text(
        &self,
        text: &str,
        options: &TTSOptions,
    ) -> Result<(Vec<u32>, GenerationCache, ndarray::Array3<f32>, usize)> {
        // Build prompt
        let prompt = format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            options.system_prompt,
            text
        );

        let input_ids = self.model.tokenizer.encode(&prompt, false);
        let input_embeds = self.model.get_text_embeddings(&input_ids);
        let mut current_len = input_ids.len();

        // Initialize cache
        let mut cache = self.model.init_cache()?;

        // Create attention mask
        let attention_mask = ndarray::Array2::<i64>::ones((1, current_len));

        // Prefill
        let (mut logits, mut hidden_states) =
            self.run_decoder_with_hidden(&input_embeds, &attention_mask, &mut cache)?;

        let mut text_tokens = Vec::new();
        let vocab_size = self.model.config.lfm.vocab_size;
        let audio_start_token = self.model.tokenizer.special_tokens().audio_start;

        // Generate text tokens
        for _ in 0..options.max_new_tokens / 2 {
            // Reserve half for audio
            let last_logits = extract_last_logits(&logits, vocab_size)?;
            let next_token = if options.text_temperature == 0.0 {
                argmax(&last_logits)
            } else {
                sample_with_temperature(&last_logits, options.text_temperature)
            };

            // Check for audio start
            if next_token == audio_start_token {
                log::debug!("TTS: Audio start token reached");
                break;
            }

            // Check for EOS
            if self.model.tokenizer.is_eos(next_token) {
                log::warn!("TTS: EOS before audio start, forcing audio");
                break;
            }

            text_tokens.push(next_token);

            // Next step
            let next_embeds = self.model.get_text_embeddings(&[next_token]);
            current_len += 1;
            let next_mask = ndarray::Array2::<i64>::ones((1, current_len));

            let (new_logits, new_hidden) =
                self.run_decoder_with_hidden(&next_embeds, &next_mask, &mut cache)?;
            logits = new_logits;

            // Update hidden states
            hidden_states = new_hidden;
        }

        // If we didn't hit audio_start, force it
        if text_tokens.last() != Some(&audio_start_token) {
            let audio_embeds = self.model.get_text_embeddings(&[audio_start_token]);
            current_len += 1;
            let audio_mask = ndarray::Array2::<i64>::ones((1, current_len));

            let (_logits, hidden_states) =
                self.run_decoder_with_hidden(&audio_embeds, &audio_mask, &mut cache)?;

            return Ok((text_tokens, cache, hidden_states, current_len));
        }

        Ok((text_tokens, cache, hidden_states, current_len))
    }

    fn generate_text_streaming<F>(
        &self,
        text: &str,
        options: &TTSOptions,
        on_event: &mut F,
    ) -> Result<(Vec<u32>, GenerationCache, ndarray::Array3<f32>, usize)>
    where
        F: FnMut(TTSEvent) -> Result<()>,
    {
        let prompt = format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            options.system_prompt,
            text
        );

        let input_ids = self.model.tokenizer.encode(&prompt, false);
        let input_embeds = self.model.get_text_embeddings(&input_ids);
        let mut current_len = input_ids.len();
        let mut cache = self.model.init_cache()?;
        let attention_mask = ndarray::Array2::<i64>::ones((1, current_len));
        let (mut logits, _hidden_states) =
            self.run_decoder_with_hidden(&input_embeds, &attention_mask, &mut cache)?;

        let mut text_tokens = Vec::new();
        let vocab_size = self.model.config.lfm.vocab_size;
        let audio_start_token = self.model.tokenizer.special_tokens().audio_start;

        for _ in 0..options.max_new_tokens / 2 {
            let last_logits = extract_last_logits(&logits, vocab_size)?;
            let next_token = if options.text_temperature == 0.0 {
                argmax(&last_logits)
            } else {
                sample_with_temperature(&last_logits, options.text_temperature)
            };

            if next_token == audio_start_token {
                log::debug!("TTS: Audio start token reached");
                break;
            }

            if self.model.tokenizer.is_eos(next_token) {
                log::warn!("TTS: EOS before audio start, forcing audio");
                break;
            }

            text_tokens.push(next_token);
            on_event(TTSEvent::TextUpdated(
                self.model.tokenizer.decode(&text_tokens, true),
            ))?;

            let next_embeds = self.model.get_text_embeddings(&[next_token]);
            current_len += 1;
            let next_mask = ndarray::Array2::<i64>::ones((1, current_len));
            let (new_logits, _new_hidden) =
                self.run_decoder_with_hidden(&next_embeds, &next_mask, &mut cache)?;
            logits = new_logits;
        }

        let audio_embeds = self.model.get_text_embeddings(&[audio_start_token]);
        current_len += 1;
        let audio_mask = ndarray::Array2::<i64>::ones((1, current_len));
        let (_logits, hidden_states) =
            self.run_decoder_with_hidden(&audio_embeds, &audio_mask, &mut cache)?;

        Ok((text_tokens, cache, hidden_states, current_len))
    }

    /// Generate audio codes autoregressively using depthformer
    fn generate_audio_codes(
        &self,
        cache: &mut crate::cache::GenerationCache,
        hidden_states: &mut ndarray::Array3<f32>,
        current_len: &mut usize,
        options: &TTSOptions,
    ) -> Result<Vec<[u16; 8]>> {
        let mut audio_codes = Vec::new();

        // Extract initial hidden state from the last position
        let mut last_hidden = self.extract_last_hidden(hidden_states);

        for frame_idx in 0..options.max_new_tokens {
            // Sample audio codes for this frame using depthformer
            let frame_codes = self.sample_audio_codes(
                &last_hidden,
                options.audio_temperature,
                options.audio_top_k,
            )?;

            // Check for end of audio
            if frame_codes[0] == END_OF_AUDIO_TOKEN {
                log::debug!("TTS: End of audio at frame {}", frame_idx);
                break;
            }

            audio_codes.push(frame_codes);

            // Get audio embeddings for next decoder input
            let clamped_codes = frame_codes.map(|c| c.min(2047));
            let audio_embeds = self.model.get_audio_embeddings(&clamped_codes)?;
            *current_len += 1;
            let attention_mask = ndarray::Array2::<i64>::ones((1, *current_len));

            // Run decoder step and get new hidden states
            let (_logits, new_hidden_states) =
                self.run_decoder_with_hidden(&audio_embeds, &attention_mask, cache)?;

            // Update hidden states and extract last position for next iteration
            *hidden_states = new_hidden_states;
            last_hidden = self.extract_last_hidden(hidden_states);
        }

        Ok(audio_codes)
    }

    fn generate_audio_codes_streaming<F>(
        &self,
        cache: &mut crate::cache::GenerationCache,
        hidden_states: &mut ndarray::Array3<f32>,
        current_len: &mut usize,
        options: &TTSOptions,
        on_event: &mut F,
    ) -> Result<Vec<[u16; 8]>>
    where
        F: FnMut(TTSEvent) -> Result<()>,
    {
        let mut audio_codes = Vec::new();
        let mut last_hidden = self.extract_last_hidden(hidden_states);

        for frame_idx in 0..options.max_new_tokens {
            let frame_codes = self.sample_audio_codes(
                &last_hidden,
                options.audio_temperature,
                options.audio_top_k,
            )?;

            if frame_codes[0] == END_OF_AUDIO_TOKEN {
                log::debug!("TTS: End of audio at frame {}", frame_idx);
                break;
            }

            on_event(TTSEvent::AudioFrame(frame_codes))?;
            audio_codes.push(frame_codes);

            let clamped_codes = frame_codes.map(|c| c.min(2047));
            let audio_embeds = self.model.get_audio_embeddings(&clamped_codes)?;
            *current_len += 1;
            let attention_mask = ndarray::Array2::<i64>::ones((1, *current_len));

            let (_logits, new_hidden_states) =
                self.run_decoder_with_hidden(&audio_embeds, &attention_mask, cache)?;

            *hidden_states = new_hidden_states;
            last_hidden = self.extract_last_hidden(hidden_states);
        }

        Ok(audio_codes)
    }

    /// Extract the last hidden state from decoder output
    /// Input: hidden_states [batch, seq_len, hidden_size]
    /// Output: Vec<f32> of length hidden_size
    pub(crate) fn extract_last_hidden(&self, hidden_states: &ndarray::Array3<f32>) -> Vec<f32> {
        let seq_len = hidden_states.shape()[1];
        if seq_len == 0 {
            return vec![0.0f32; self.model.config.lfm.hidden_size];
        }

        hidden_states
            .slice(ndarray::s![0, seq_len - 1, ..])
            .iter()
            .copied()
            .collect()
    }

    /// Sample audio codes using depthformer (autoregressive across codebooks)
    fn sample_audio_codes(
        &self,
        hidden_state: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[u16; 8]> {
        let (codes, _trace) =
            self.sample_audio_codes_with_trace(hidden_state, temperature, top_k, None)?;
        Ok(codes)
    }

    pub(crate) fn sample_audio_frame(
        &self,
        hidden_state: &[f32],
        temperature: f32,
        top_k: usize,
    ) -> Result<[u16; 8]> {
        self.sample_audio_codes(hidden_state, temperature, top_k)
    }

    fn sample_audio_codes_with_trace(
        &self,
        hidden_state: &[f32],
        temperature: f32,
        top_k: usize,
        mut trace: Option<TTSFrameTrace>,
    ) -> Result<([u16; 8], Option<TTSFrameTrace>)> {
        let depthformer_config = &self.model.config.depthformer;
        let num_layers = depthformer_config.layers;
        let num_kv_heads = 8;
        let head_dim = 32;

        self.sample_audio_codes_inner(
            hidden_state,
            temperature,
            top_k,
            num_layers,
            num_kv_heads,
            head_dim,
            &mut trace,
        )
    }

    fn sample_audio_codes_inner(
        &self,
        hidden_state: &[f32],
        temperature: f32,
        top_k: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        trace: &mut Option<TTSFrameTrace>,
    ) -> Result<([u16; 8], Option<TTSFrameTrace>)> {
        let mut codes = [0u16; 8];
        let mut prev_token = 0i64;

        let hidden_array =
            ndarray::Array2::from_shape_vec((1, hidden_state.len()), hidden_state.to_vec())?;

        // Depthformer expects packed cache tensors (not per-layer past_key_values.* inputs)
        // Shapes follow the ONNX graph:
        // - past_keys/past_values: [layers, batch, num_kv_heads, past_len, head_dim]
        // - depth_slices_in: [batch, 8, depth_dim]
        let mut past_keys =
            ndarray::Array5::<f32>::zeros((num_layers, 1, num_kv_heads, 0, head_dim));
        let mut past_values =
            ndarray::Array5::<f32>::zeros((num_layers, 1, num_kv_heads, 0, head_dim));
        let mut depth_slices_in =
            ndarray::Array3::<f32>::zeros((1, NUM_CODEBOOKS, self.model.config.depthformer.dim));

        // Generate each codebook sequentially
        for codebook_idx in 0..NUM_CODEBOOKS {
            // Prepare depthformer inputs
            let step_idx = ndarray::arr0(codebook_idx as i64);
            let prev_token_array = ndarray::Array1::from_vec(vec![prev_token]);
            let seqlens_k = ndarray::Array1::from_vec(vec![codebook_idx as i32]);
            let total_seq_len = ndarray::arr0((codebook_idx + 1) as i32);

            // Ensure contiguous layout
            let hidden_contig = hidden_array.as_standard_layout().to_owned();
            let depth_slices_contig = depth_slices_in.as_standard_layout().to_owned();
            let step_contig = step_idx.as_standard_layout().to_owned();
            let prev_contig = prev_token_array.as_standard_layout().to_owned();
            let past_keys_contig = past_keys.as_standard_layout().to_owned();
            let past_values_contig = past_values.as_standard_layout().to_owned();
            let seqlens_k_contig = seqlens_k.as_standard_layout().to_owned();
            let total_seq_len_contig = total_seq_len.as_standard_layout().to_owned();

            let t_hidden = ort::value::Value::from_array(hidden_contig)?;
            let t_depth_slices = ort::value::Value::from_array(depth_slices_contig)?;
            let t_step = ort::value::Value::from_array(step_contig)?;
            let t_prev = ort::value::Value::from_array(prev_contig)?;
            let t_past_keys = ort::value::Value::from_array(past_keys_contig)?;
            let t_past_values = ort::value::Value::from_array(past_values_contig)?;
            let t_seqlens_k = ort::value::Value::from_array(seqlens_k_contig)?;
            let t_total_seq_len = ort::value::Value::from_array(total_seq_len_contig)?;

            let mut depthformer = self.model.sessions.depthformer.borrow_mut();
            let outputs = depthformer.run(vec![
                ("hidden_states".to_string(), t_hidden.into_dyn()),
                ("depth_slices_in".to_string(), t_depth_slices.into_dyn()),
                ("step_idx".to_string(), t_step.into_dyn()),
                ("prev_token".to_string(), t_prev.into_dyn()),
                ("past_keys".to_string(), t_past_keys.into_dyn()),
                ("past_values".to_string(), t_past_values.into_dyn()),
                ("seqlens_k".to_string(), t_seqlens_k.into_dyn()),
                ("total_seq_len".to_string(), t_total_seq_len.into_dyn()),
            ])?;

            // Extract logits
            let logits_output = outputs
                .get("logits")
                .ok_or_else(|| LFM2Error::Generation("depthformer logits not found".to_string()))?;
            let (_, logits_data) = logits_output.try_extract_tensor::<f32>()?;
            let logits: Vec<f32> = logits_data.to_vec();

            let argmax_token = argmax(&logits[..CODEBOOK_VOCAB.min(logits.len())]);
            let argmax_logit = logits[argmax_token as usize];

            // Sample token
            let token = if temperature == 0.0 {
                argmax_token
            } else {
                sample_top_k(
                    &logits[..CODEBOOK_VOCAB.min(logits.len())],
                    temperature,
                    top_k,
                )
            };

            codes[codebook_idx] = token as u16;
            prev_token = token as i64;

            if let Some(trace) = trace.as_mut() {
                trace.codebook_traces.push(TTSCodebookTrace {
                    chosen_token: token as u16,
                    argmax_token,
                    argmax_logit,
                    top_candidates: top_candidates(&logits[..CODEBOOK_VOCAB.min(logits.len())], 5),
                });
            }

            // Feed depthformer recurrent outputs back as next-step inputs.
            let depth_slices = outputs
                .get("depth_slices")
                .ok_or_else(|| {
                    LFM2Error::Generation("depthformer depth_slices not found".to_string())
                })?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| {
                    LFM2Error::Generation(format!("Invalid depth_slices shape: {:?}", e))
                })?;
            depth_slices_in = depth_slices;

            let new_keys = outputs
                .get("new_keys")
                .ok_or_else(|| LFM2Error::Generation("depthformer new_keys not found".to_string()))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix5>()
                .map_err(|e| LFM2Error::Generation(format!("Invalid new_keys shape: {:?}", e)))?;
            past_keys = new_keys;

            let new_values = outputs
                .get("new_values")
                .ok_or_else(|| {
                    LFM2Error::Generation("depthformer new_values not found".to_string())
                })?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix5>()
                .map_err(|e| LFM2Error::Generation(format!("Invalid new_values shape: {:?}", e)))?;
            past_values = new_values;
        }

        Ok((codes, trace.take()))
    }

    /// Decode audio codes to waveform using audio detokenizer
    /// The detokenizer outputs 1282-channel features: [log_magnitude (641) | angle (641)]
    /// We apply exp() to log_magnitude, combine with angle into complex spectrogram, then ISTFT
    pub fn decode_audio_codes(&self, codes: &[[u16; 8]]) -> Result<Vec<f32>> {
        self.decode_audio_codes_impl(codes, true)
    }

    pub fn decode_audio_codes_raw(&self, codes: &[[u16; 8]]) -> Result<Vec<f32>> {
        self.decode_audio_codes_impl(codes, false)
    }

    fn decode_audio_codes_impl(&self, codes: &[[u16; 8]], normalize: bool) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }

        use crate::audio::istft::ISTFT;

        // Convert codes to tensor for detokenizer.
        // Model expects [batch, 8, time].
        let num_frames = codes.len();
        let mut codes_array = ndarray::Array3::<i64>::zeros((1, 8, num_frames));
        for (t, frame) in codes.iter().enumerate() {
            for (cb, &code) in frame.iter().enumerate() {
                // Clamp to valid audio code range for detokenizer (matches reference).
                codes_array[[0, cb, t]] = code.min(2047) as i64;
            }
        }

        // Ensure contiguous layout
        let codes_contig = codes_array.as_standard_layout().to_owned();
        let t_codes = ort::value::Value::from_array(codes_contig)?;

        // Run audio detokenizer
        let mut detokenizer = self.model.sessions.audio_detokenizer.lock().unwrap();
        let outputs = detokenizer.run(ort::inputs! {
            "audio_codes" => t_codes,
        })?;

        // The detokenizer outputs features of shape [1, num_frames, 1282]
        // 1282 = 641 (log_magnitude) + 641 (angle)
        // n_fft = 1280, so n_freqs = n_fft / 2 + 1 = 641
        let features_output = outputs.get("stft_features")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("waveform"))
            .or_else(|| outputs.get("stft"))
            .ok_or_else(|| LFM2Error::Generation(
                "detokenizer output not found. Expected 'stft_features', 'output', 'waveform', or 'stft'".to_string()
            ))?;

        let view = features_output.try_extract_array::<f32>()?;
        let shape = view.shape();

        // Expected shape: [1, num_frames, 1282] or [num_frames, 1282]
        if shape.len() < 2 {
            return Err(LFM2Error::Generation(format!(
                "Expected 2D or 3D output, got {:?}",
                shape
            )));
        }

        let (out_frames, feature_dim) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else {
            (shape[0], shape[1])
        };

        if feature_dim != 1282 {
            log::warn!(
                "TTS: Unexpected feature dimension {}, expected 1282",
                feature_dim
            );
        }

        // Extract features and split into log_abs and angle
        let flat: Vec<f32> = view.iter().copied().collect();
        let n_freqs = 641; // feature_dim / 2 = 1282 / 2

        let mut log_abs_data = Vec::with_capacity(out_frames * n_freqs);
        let mut angle_data = Vec::with_capacity(out_frames * n_freqs);

        for frame in 0..out_frames {
            let frame_offset = frame * feature_dim;
            // First half is log_abs
            for i in 0..n_freqs {
                log_abs_data.push(flat[frame_offset + i]);
            }
            // Second half is angle
            for i in 0..n_freqs {
                angle_data.push(flat[frame_offset + n_freqs + i]);
            }
        }

        // Detokenizer output is frame-major [time, feature]. Build [time, freq] first,
        // then transpose to the [freq, time] layout expected by ISTFT.
        let log_abs =
            ndarray::Array2::from_shape_vec((out_frames, n_freqs), log_abs_data)?.reversed_axes();
        let angle =
            ndarray::Array2::from_shape_vec((out_frames, n_freqs), angle_data)?.reversed_axes();

        // Apply ISTFT
        // n_fft=1280, hop_length=320, win_length=1280 (from Python reference)
        let istft = ISTFT::new(1280, 320, 1280);
        let mut waveform = istft.inverse_from_log_polar(&log_abs, &angle)?;

        if normalize {
            // Normalize to a stable output range.
            let mut max_abs = 0.0f32;
            for &v in &waveform {
                max_abs = max_abs.max(v.abs());
            }
            if max_abs > 0.0 {
                let scale = 0.9f32 / max_abs;
                for v in &mut waveform {
                    *v *= scale;
                }
            }
        }

        log::info!(
            "TTS: Decoded {} frames -> {} audio samples",
            out_frames,
            waveform.len()
        );

        Ok(waveform)
    }

    pub(crate) fn run_decoder_with_hidden(
        &self,
        inputs_embeds: &ndarray::Array3<f32>,
        attention_mask: &ndarray::Array2<i64>,
        cache: &mut GenerationCache,
    ) -> Result<(ndarray::Array3<f32>, ndarray::Array3<f32>)> {
        // Ensure contiguous layout for inputs
        let inputs_contig = inputs_embeds.as_standard_layout().to_owned();
        let mask_contig = attention_mask.as_standard_layout().to_owned();
        let t_inputs = ort::value::Value::from_array(inputs_contig)?;
        let t_mask = ort::value::Value::from_array(mask_contig)?;

        // Get cache inputs (now returns DynValue directly)
        let cache_inputs = cache.prepare_cache_inputs();

        log::debug!(
            "Cache inputs: {:?}",
            cache_inputs.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );

        // Build input list
        let mut inputs_list: Vec<(String, ort::value::DynValue)> = vec![
            ("inputs_embeds".to_string(), t_inputs.into_dyn()),
            ("attention_mask".to_string(), t_mask.into_dyn()),
        ];

        // Add cache inputs (already DynValue)
        for (name, value) in cache_inputs {
            inputs_list.push((name, value));
        }

        log::debug!("Running decoder with {} inputs", inputs_list.len());

        // Run decoder
        let mut decoder = self.model.sessions.decoder.borrow_mut();
        let outputs = decoder.run(inputs_list)?;

        // Extract logits
        let logits_output = outputs
            .get("logits")
            .ok_or_else(|| LFM2Error::Generation("logits not found".to_string()))?;

        let view = logits_output.try_extract_array::<f32>()?;
        let shape = view.shape();

        if shape.len() != 3 {
            return Err(LFM2Error::Generation(format!(
                "Expected 3D logits, got {}D",
                shape.len()
            )));
        }

        let flat: Vec<f32> = view.iter().copied().collect();
        let logits = ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), flat)?;

        // Extract hidden_states if available
        let hidden_size = self.model.config.lfm.hidden_size;
        let hidden_states = if let Some(hidden_output) = outputs.get("hidden_states") {
            let view = hidden_output.try_extract_array::<f32>()?;
            let shape = view.shape();
            if shape.len() == 3 {
                let flat: Vec<f32> = view.iter().copied().collect();
                ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), flat)?
            } else {
                ndarray::Array3::zeros((1, 1, hidden_size))
            }
        } else {
            ndarray::Array3::zeros((1, 1, hidden_size))
        };

        // Update cache from outputs
        cache.update(&outputs)?;

        Ok((logits, hidden_states))
    }
}

// Helper functions
fn extract_last_logits(logits: &ndarray::Array3<f32>, vocab_size: usize) -> Result<Vec<f32>> {
    let seq_len = logits.shape()[1];
    let offset = (seq_len - 1) * vocab_size;

    let data: Vec<f32> = logits.iter().copied().collect();
    if offset + vocab_size > data.len() {
        return Err(LFM2Error::Generation("Invalid logits shape".to_string()));
    }

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

fn sample_top_k(logits: &[f32], temperature: f32, k: usize) -> u32 {
    // Get top k logits and indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = indexed.into_iter().take(k).collect::<Vec<_>>();

    // Apply temperature and softmax
    let scaled: Vec<f32> = top_k.iter().map(|(_, x)| x / temperature).collect();
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_shifted: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_shifted.iter().sum();
    let probs: Vec<f32> = exp_shifted.iter().map(|&x| x / sum_exp).collect();

    // Sample
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return top_k[i].0 as u32;
        }
    }
    top_k.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

fn top_candidates(logits: &[f32], n: usize) -> Vec<TTSLogitCandidate> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .into_iter()
        .take(n)
        .map(|(token, logit)| TTSLogitCandidate {
            token: token as u32,
            logit,
        })
        .collect()
}

/// Standalone audio decode function for async/background thread use.
/// Takes the detokenizer session directly, avoiding the need for full LFM2Audio.
/// This enables thread-safe decode operations.
pub fn decode_audio_codes_standalone(
    detokenizer: &mut ort::session::Session,
    codes: &[[u16; 8]],
) -> Result<Vec<f32>> {
    use crate::audio::istft::ISTFT;

    if codes.is_empty() {
        return Ok(Vec::new());
    }

    // Convert codes to tensor
    let num_frames = codes.len();
    let mut codes_array = ndarray::Array3::<i64>::zeros((1, 8, num_frames));
    for (t, frame) in codes.iter().enumerate() {
        for (cb, &code) in frame.iter().enumerate() {
            codes_array[[0, cb, t]] = code.min(2047) as i64;
        }
    }

    let codes_contig = codes_array.as_standard_layout().to_owned();
    let t_codes = ort::value::Value::from_array(codes_contig)?;

    // Run audio detokenizer
    let outputs = detokenizer
        .run(ort::inputs! {
            "audio_codes" => t_codes,
        })
        .map_err(LFM2Error::Onnx)?;

    // Extract features
    let features_output = outputs
        .get("stft_features")
        .or_else(|| outputs.get("output"))
        .or_else(|| outputs.get("waveform"))
        .or_else(|| outputs.get("stft"))
        .ok_or_else(|| LFM2Error::Generation("detokenizer output not found".to_string()))?;

    let view = features_output
        .try_extract_array::<f32>()
        .map_err(LFM2Error::Onnx)?;
    let shape = view.shape();

    let (out_frames, feature_dim) = if shape.len() == 3 {
        (shape[1], shape[2])
    } else {
        (shape[0], shape[1])
    };

    let n_freqs = 641; // feature_dim / 2
    let flat: Vec<f32> = view.iter().copied().collect();

    // Extract spectrogram
    let mut log_abs_data = Vec::with_capacity(out_frames * n_freqs);
    let mut angle_data = Vec::with_capacity(out_frames * n_freqs);

    for frame in 0..out_frames {
        let frame_offset = frame * feature_dim;
        for i in 0..n_freqs {
            log_abs_data.push(flat[frame_offset + i]);
        }
        for i in 0..n_freqs {
            angle_data.push(flat[frame_offset + n_freqs + i]);
        }
    }

    let log_abs =
        ndarray::Array2::from_shape_vec((out_frames, n_freqs), log_abs_data)?.reversed_axes();
    let angle = ndarray::Array2::from_shape_vec((out_frames, n_freqs), angle_data)?.reversed_axes();

    // ISTFT
    let istft = ISTFT::new(1280, 320, 1280);
    istft.inverse_from_log_polar(&log_abs, &angle)
}
