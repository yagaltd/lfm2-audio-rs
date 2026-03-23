//! TTS (Text-to-Speech) pipeline
//! Text → Audio
//! Reference: hand-voice-racer/audio-model.js:1000-1400

use ort::value::TensorRef;

use crate::error::{LFM2Error, Result};
use crate::model::LFM2Audio;
use crate::tokenizer::{END_OF_AUDIO_TOKEN, NUM_CODEBOOKS, CODEBOOK_VOCAB};

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
        log::info!("TTS: Synthesizing '{}'", text.chars().take(50).collect::<String>());

        // Phase 1: Generate text tokens until <|audio_start|>
        let (text_tokens, mut cache, mut hidden_states, mut current_len) = 
            self.generate_text(text, options)?;

        log::info!("TTS: Text phase complete, {} tokens, entering audio mode", text_tokens.len());

        // Phase 2: Generate audio codes autoregressively
        let audio_codes = self.generate_audio_codes(
            &mut cache,
            &mut hidden_states,
            &mut current_len,
            options,
        )?;

        log::info!("TTS: Generated {} audio frames", audio_codes.len());

        // Phase 3: Decode audio codes to waveform
        let audio = self.decode_audio_codes(&audio_codes)?;

        log::info!("TTS: Generated {} samples at 24kHz", audio.len());

        Ok(audio)
    }

    /// Generate text portion until <|audio_start|> or max tokens
    fn generate_text(
        &self,
        text: &str,
        options: &TTSOptions,
    ) -> Result<(Vec<u32>, crate::cache::GenerationCache, ndarray::Array3<f32>, usize)> {
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
        let (mut logits, _hidden_states) = 
            self.run_decoder_with_hidden(&input_embeds, &attention_mask, &cache)?;
        // TODO: Update cache from outputs
        // cache.update(&outputs)?;

        let mut text_tokens = Vec::new();
        let vocab_size = self.model.config.lfm.vocab_size;
        let audio_start_token = self.model.tokenizer.special_tokens().audio_start;

        // Generate text tokens
        for _ in 0..options.max_new_tokens / 2 { // Reserve half for audio
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

            let (new_logits, _new_hidden) = 
                self.run_decoder_with_hidden(&next_embeds, &next_mask, &cache)?;
            logits = new_logits;
            // TODO: Update cache from outputs
            // cache.update(&outputs)?;

            // Update hidden states (we need the last one for audio generation)
            // hidden_states = new_hidden;
        }

        // If we didn't hit audio_start, force it
        if text_tokens.last() != Some(&audio_start_token) {
            let audio_embeds = self.model.get_text_embeddings(&[audio_start_token]);
            current_len += 1;
            let audio_mask = ndarray::Array2::<i64>::ones((1, current_len));

            let (logits, hidden_states) = 
                self.run_decoder_with_hidden(&audio_embeds, &audio_mask, &cache)?;
            // TODO: Update cache from outputs
            // cache.update(&outputs)?;

            return Ok((text_tokens, cache, hidden_states, current_len));
        }

        // Return final hidden states
        // For now, return zeros - in full implementation extract from last step
        let hidden_size = self.model.config.lfm.hidden_size;
        let hidden_states = ndarray::Array3::zeros((1, 1, hidden_size));

        Ok((text_tokens, cache, hidden_states, current_len))
    }

    /// Generate audio codes autoregressively using depthformer
    fn generate_audio_codes(
        &self,
        cache: &mut crate::cache::GenerationCache,
        _hidden_states: &mut ndarray::Array3<f32>,
        current_len: &mut usize,
        options: &TTSOptions,
    ) -> Result<Vec<[u16; 8]>> {
        let mut audio_codes = Vec::new();
        let hidden_size = self.model.config.lfm.hidden_size;

        // Initialize depthformer cache
        let depthformer_config = &self.model.config.depthformer;
        let num_layers = depthformer_config.layers; // 6
        let num_kv_heads = 8; // From config
        let head_dim = 32;   // depthformer_dim / num_heads = 1024 / 32 = 32

        for frame_idx in 0..options.max_new_tokens {
            // Get last hidden state from decoder
            // In full implementation, this comes from decoder outputs
            let last_hidden = vec![0.0f32; hidden_size]; // Placeholder

            // Sample audio codes for this frame using depthformer
            let frame_codes = self.sample_audio_codes(
                &last_hidden,
                options.audio_temperature,
                options.audio_top_k,
                num_layers,
                num_kv_heads,
                head_dim,
            )?;

            // Check for end of audio
            if frame_codes[0] == END_OF_AUDIO_TOKEN {
                log::debug!("TTS: End of audio at frame {}", frame_idx);
                break;
            }

            audio_codes.push(frame_codes);

            // Get audio embeddings for next decoder input
            let audio_embeds = self.model.get_audio_embeddings(&frame_codes)?;
            *current_len += 1;
            let attention_mask = ndarray::Array2::<i64>::ones((1, *current_len));

            // Run decoder step
            let (_logits, _hidden) = 
                self.run_decoder_with_hidden(&audio_embeds, &attention_mask, cache)?;
            // TODO: Update cache from outputs
            // cache.update(&outputs)?;

            // Update hidden states
            // _hidden_states = _hidden;
        }

        Ok(audio_codes)
    }

    /// Sample audio codes using depthformer (autoregressive across codebooks)
    fn sample_audio_codes(
        &self,
        hidden_state: &[f32],
        temperature: f32,
        top_k: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<[u16; 8]> {
        let mut codes = [0u16; 8];
        let mut prev_token = 0i64;

        // Initialize depthformer KV cache
        let mut depthformer_cache = Vec::new();
        for _ in 0..num_layers {
            depthformer_cache.push((
                ndarray::Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)),
                ndarray::Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)),
            ));
        }

        let hidden_array = ndarray::Array2::from_shape_vec(
            (1, hidden_state.len()),
            hidden_state.to_vec(),
        )?;

        // Generate each codebook sequentially
        for codebook_idx in 0..NUM_CODEBOOKS {
            // Prepare depthformer inputs
            let step_idx = ndarray::Array1::from_vec(vec![codebook_idx as i64]);
            let prev_token_array = ndarray::Array1::from_vec(vec![prev_token]);

            // Build cache inputs
            let mut feeds: std::collections::HashMap<String, ort::value::Value> = std::collections::HashMap::new();
            feeds.insert("hidden_states".to_string(), ort::value::Value::from_array(hidden_array.clone())?.into());
            feeds.insert("step_idx".to_string(), ort::value::Value::from_array(step_idx.clone())?.into());
            feeds.insert("prev_token".to_string(), ort::value::Value::from_array(prev_token_array.clone())?.into());

            // Add KV cache
            for (layer_idx, (k, v)) in depthformer_cache.iter().enumerate() {
                feeds.insert(format!("past_key_values.{}.key", layer_idx), ort::value::Value::from_array(k.clone())?.into());
                feeds.insert(format!("past_key_values.{}.value", layer_idx), ort::value::Value::from_array(v.clone())?.into());
            }

            // Run depthformer
            let mut depthformer = self.model.sessions.depthformer.borrow_mut();
            let t_hidden = TensorRef::from_array_view(hidden_array.view())?;
            let t_step = TensorRef::from_array_view(step_idx.view())?;
            let t_prev = TensorRef::from_array_view(prev_token_array.view())?;
            let outputs = depthformer.run(ort::inputs! {
                "hidden_states" => t_hidden,
                "step_idx" => t_step,
                "prev_token" => t_prev,
            })?;

            // Extract logits
            let logits_output = outputs.get("logits")
                .ok_or_else(|| LFM2Error::Generation("depthformer logits not found".to_string()))?;
            let (_, logits_data) = logits_output.try_extract_tensor::<f32>()?;
            let logits: Vec<f32> = logits_data.to_vec();

            // Sample token
            let token = if temperature == 0.0 {
                argmax(&logits[..CODEBOOK_VOCAB.min(logits.len())])
            } else {
                sample_top_k(&logits[..CODEBOOK_VOCAB.min(logits.len())], temperature, top_k)
            };

            codes[codebook_idx] = token as u16;
            prev_token = token as i64;

            // Update depthformer cache (simplified - in full implementation extract from outputs)
        }

        Ok(codes)
    }

    /// Decode audio codes to waveform
    fn decode_audio_codes(&self, codes: &[[u16; 8]]) -> Result<Vec<f32>> {
        // Run audio detokenizer
        // This produces log-magnitude and phase for ISTFT
        // For now, return silence
        let sample_rate = 24000;
        let duration_secs = codes.len() as f32 * 0.08; // 80ms per frame
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        // Placeholder: return silence
        // In full implementation:
        // 1. Run audio_detokenizer.onnx on codes
        // 2. Split output into log_magnitude and phase
        // 3. Apply ISTFT

        Ok(vec![0.0f32; num_samples])
    }

    fn run_decoder_with_hidden(
        &self,
        inputs_embeds: &ndarray::Array3<f32>,
        attention_mask: &ndarray::Array2<i64>,
        _cache: &crate::cache::GenerationCache,
    ) -> Result<(ndarray::Array3<f32>, ndarray::Array3<f32>)> {
        use ort::value::TensorRef;
        
        // Similar to ASR but also extract hidden_states
        let t_inputs = TensorRef::from_array_view(inputs_embeds.view())?;
        let t_mask = TensorRef::from_array_view(attention_mask.view())?;
        
        let mut decoder = self.model.sessions.decoder.borrow_mut();
        let outputs = decoder.run(ort::inputs! {
            "inputs_embeds" => t_inputs,
            "attention_mask" => t_mask,
        })?;

        let logits_output = outputs.get("logits")
            .ok_or_else(|| LFM2Error::Generation("logits not found".to_string()))?;
        let (shape, data) = logits_output.try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let logits = ndarray::Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            data.to_vec(),
        )?;

        // Try to extract hidden_states if available
        let hidden_size = self.model.config.lfm.hidden_size;
        let _hidden_states = if let Some(hidden_output) = outputs.get("hidden_states") {
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

        // TODO: Update cache from outputs
        // For now, just return logits without outputs (which references local decoder borrow)
        Ok((logits, _hidden_states))
    }
}

// Helper functions
fn extract_last_logits(logits: &ndarray::Array3<f32>, vocab_size: usize) -> Result<Vec<f32>> {
    let seq_len = logits.shape()[1];
    let offset = (seq_len - 1) * vocab_size;
    let data = logits.as_slice().ok_or_else(||
        LFM2Error::Generation("Failed to get logits slice".to_string())
    )?;
    Ok(data[offset..offset + vocab_size].to_vec())
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
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