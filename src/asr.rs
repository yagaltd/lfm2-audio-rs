//! ASR (Automatic Speech Recognition) pipeline
//! Audio → Text
//! Reference: hand-voice-racer/audio-model.js:800-1000

use ndarray::Array2;

use crate::audio::{compute_mel_spectrogram, resample_linear};
use crate::cache::GenerationCache;
use crate::error::{LFM2Error, Result};
use crate::model::LFM2Audio;

/// ASR pipeline
pub struct ASRPipeline<'a> {
    model: &'a LFM2Audio,
}

/// ASR generation options
#[derive(Debug, Clone)]
pub struct ASROptions {
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
}

impl Default for ASROptions {
    fn default() -> Self {
        Self {
            system_prompt: "Perform ASR.".to_string(),
            max_new_tokens: 256,
            temperature: 0.0,
        }
    }
}

impl<'a> ASRPipeline<'a> {
    pub fn new(model: &'a LFM2Audio) -> Self {
        Self { model }
    }

    /// Transcribe audio to text
    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: &ASROptions,
    ) -> Result<String> {
        log::info!(
            "ASR: Transcribing {} samples at {} Hz",
            audio.len(),
            sample_rate
        );

        let prepared_audio = prepare_audio(
            audio,
            sample_rate,
            self.model.preprocessor_config.sample_rate,
        );

        // 1. Compute mel spectrogram
        let mel = compute_mel_spectrogram(&prepared_audio, &self.model.preprocessor_config)?;
        log::debug!("Mel spectrogram shape: {:?}", mel.shape());

        // 2. Encode audio to embeddings
        let audio_embeds = self.model.encode_audio(&mel)?;
        log::debug!("Audio embeddings shape: {:?}", audio_embeds.shape());

        // 3. Build text prompt
        let prefix_text = format!(
            "<|startoftext|><|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
            options.system_prompt
        );
        let suffix_text = "<|im_end|>\n<|im_start|>assistant\n";

        let prefix_ids = self.model.tokenizer.encode(&prefix_text, false);
        let suffix_ids = self.model.tokenizer.encode(suffix_text, false);

        log::debug!(
            "Prefix tokens: {}, Suffix tokens: {}",
            prefix_ids.len(),
            suffix_ids.len()
        );

        // 4. Get text embeddings
        let prefix_embeds = self.model.get_text_embeddings(&prefix_ids);
        let suffix_embeds = self.model.get_text_embeddings(&suffix_ids);

        // 5. Concatenate embeddings: prefix + audio + suffix
        let audio_len = audio_embeds.shape()[1];
        let prefix_len = prefix_ids.len();
        let suffix_len = suffix_ids.len();
        let total_len = prefix_len + audio_len + suffix_len;

        let hidden_size = self.model.config.lfm.hidden_size;
        let mut all_embeds = vec![0.0f32; total_len * hidden_size];

        // Copy prefix
        let prefix_flat: Vec<f32> = prefix_embeds.iter().copied().collect();
        all_embeds[..prefix_len * hidden_size].copy_from_slice(&prefix_flat);

        // Copy audio embeddings
        let audio_flat: Vec<f32> = audio_embeds.iter().copied().collect();
        all_embeds[prefix_len * hidden_size..(prefix_len + audio_len) * hidden_size]
            .copy_from_slice(&audio_flat);

        // Copy suffix
        let suffix_flat: Vec<f32> = suffix_embeds.iter().copied().collect();
        all_embeds[(prefix_len + audio_len) * hidden_size..].copy_from_slice(&suffix_flat);

        let input_embeds =
            ndarray::Array3::from_shape_vec((1, total_len, hidden_size), all_embeds)?;

        // 6. Create attention mask (all ones)
        let attention_mask = Array2::<i64>::ones((1, total_len));

        // 7. Initialize cache
        let mut cache = self.model.init_cache()?;

        // 8. Prefill decoder
        let mut generated_tokens = Vec::new();
        let mut current_len = total_len;

        log::info!("ASR: Starting generation with {} prefill tokens", total_len);

        // 9. First forward pass (prefill)
        let mut logits = self.run_decoder(&input_embeds, &attention_mask, &mut cache)?;

        // 10. Autoregressive generation
        for step in 0..options.max_new_tokens {
            // Get logits for last position
            let vocab_size = self.model.config.lfm.vocab_size;
            let last_logits = extract_last_logits(&logits, vocab_size)?;

            // Sample next token
            let next_token = if options.temperature == 0.0 {
                argmax(&last_logits)
            } else {
                sample_with_temperature(&last_logits, options.temperature)
            };

            // Check for end of sequence
            if self.model.tokenizer.is_eos(next_token) {
                log::info!("ASR: EOS token reached at step {}", step);
                break;
            }

            generated_tokens.push(next_token);

            // Prepare next input (single token embedding)
            let next_embeds = self.model.get_text_embeddings(&[next_token]);
            current_len += 1;

            let next_mask = Array2::<i64>::ones((1, current_len));

            // Run decoder step
            logits = self.run_decoder(&next_embeds, &next_mask, &mut cache)?;
        }

        // 11. Decode tokens to text
        let text = self.model.tokenizer.decode(&generated_tokens, true);
        log::info!(
            "ASR: Generated {} tokens -> {} chars",
            generated_tokens.len(),
            text.len()
        );

        Ok(text)
    }

    fn run_decoder(
        &self,
        inputs_embeds: &ndarray::Array3<f32>,
        attention_mask: &Array2<i64>,
        cache: &mut GenerationCache,
    ) -> Result<ndarray::Array3<f32>> {
        use ort::value::Value;

        // Ensure arrays are contiguous by cloning if necessary
        let inputs_contig = inputs_embeds.as_standard_layout().to_owned();
        let mask_contig = attention_mask.as_standard_layout().to_owned();

        // Create values from owned arrays
        let t_inputs = Value::from_array(inputs_contig)?;
        let t_mask = Value::from_array(mask_contig)?;

        // Get cache inputs (now returns DynValue directly)
        let cache_inputs = cache.prepare_cache_inputs();

        // Build input list: start with required inputs
        let mut inputs_list: Vec<(String, ort::value::DynValue)> = vec![
            ("inputs_embeds".to_string(), t_inputs.into_dyn()),
            ("attention_mask".to_string(), t_mask.into_dyn()),
        ];

        // Add cache inputs (already DynValue)
        for (name, value) in cache_inputs {
            inputs_list.push((name, value));
        }

        // Run decoder with cache
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

        // Update cache from decoder outputs (present_* -> past_*)
        cache.update(&outputs)?;

        Ok(logits)
    }
}

fn prepare_audio(audio: &[f32], sample_rate: u32, target_sample_rate: u32) -> Vec<f32> {
    if sample_rate == target_sample_rate {
        audio.to_vec()
    } else {
        resample_linear(audio, sample_rate, target_sample_rate)
    }
}

/// Extract logits for the last position
fn extract_last_logits(logits: &ndarray::Array3<f32>, vocab_size: usize) -> Result<Vec<f32>> {
    let seq_len = logits.shape()[1];
    let offset = (seq_len - 1) * vocab_size;

    let data: Vec<f32> = logits.iter().copied().collect();
    if offset + vocab_size > data.len() {
        return Err(LFM2Error::Generation("Invalid logits shape".to_string()));
    }

    Ok(data[offset..offset + vocab_size].to_vec())
}

/// Argmax sampling (greedy)
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Temperature sampling
fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    if temperature == 0.0 {
        return argmax(logits);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax
    let max_logit = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_shifted: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_shifted.iter().sum();
    let probs: Vec<f32> = exp_shifted.iter().map(|&x| x / sum_exp).collect();

    // Sample
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

#[cfg(test)]
mod tests {
    use super::{prepare_audio, ASROptions};

    #[test]
    fn asr_defaults_to_greedy_decoding() {
        assert_eq!(ASROptions::default().temperature, 0.0);
    }

    #[test]
    fn prepare_audio_resamples_when_sample_rate_differs() {
        let audio = vec![0.0f32; 48_000];
        let prepared = prepare_audio(&audio, 48_000, 16_000);
        assert_eq!(prepared.len(), 16_000);
    }

    #[test]
    fn prepare_audio_leaves_matching_sample_rate_unchanged() {
        let audio = vec![0.1f32, -0.2, 0.3];
        let prepared = prepare_audio(&audio, 16_000, 16_000);
        assert_eq!(prepared, audio);
    }
}
