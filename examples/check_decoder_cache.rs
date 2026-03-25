use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use lfm2_audio::cache::GenerationCache;
use lfm2_audio::config::{Device, ModelConfig, Precision};
use lfm2_audio::embeddings::EmbedTokens;
use lfm2_audio::sessions::SessionLoader;
use lfm2_audio::LFM2Tokenizer;
use ndarray::{Array2, Array3};
use ort::value::Value;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX")]
    model: PathBuf,

    #[arg(long, default_value = "fp16")]
    precision: String,
}

fn parse_precision(value: &str) -> Result<Precision> {
    Ok(match value {
        "fp32" => Precision::FP32,
        "fp16" => Precision::FP16,
        "q4" => Precision::Q4,
        "q8" => Precision::Q8,
        other => anyhow::bail!("unsupported precision: {}", other),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let precision = parse_precision(&args.precision)?;

    let config = ModelConfig::from_file(args.model.join("config.json"))?;
    let tokenizer = LFM2Tokenizer::from_dir(&args.model)?;
    let embed_tokens = EmbedTokens::from_dir(args.model.join("onnx"))?;
    let loader = SessionLoader::new(&args.model, precision, Device::CPU);
    let sessions = loader.load()?;

    let prompt = "<|startoftext|><|im_start|>system\nPerform TTS. Use the UK male voice.<|im_end|>\n<|im_start|>user\nWhat is this obsession people have with books?<|im_end|>\n<|im_start|>assistant\n";
    let input_ids = tokenizer.encode(prompt, false);
    let seq_len = input_ids.len();

    let full_embeds = embed_tokens.embed_sequence_array(&input_ids);
    let full_mask = Array2::<i64>::ones((1, seq_len));
    let mut full_cache = GenerationCache::new(&config.lfm)?;
    let (full_logits, full_hidden) = {
        let mut decoder = sessions.decoder.borrow_mut();
        run_decoder(&mut decoder, &full_embeds, &full_mask, &mut full_cache)?
    };
    let full_last_logits = extract_last(&full_logits);
    let full_last_hidden = extract_last(&full_hidden);

    let mut step_cache = GenerationCache::new(&config.lfm)?;
    let mut step_logits_last = Vec::new();
    let mut step_hidden_last = Vec::new();
    for end in 0..seq_len {
        let token = [input_ids[end]];
        let embeds = embed_tokens.embed_sequence_array(&token);
        let mask = Array2::<i64>::ones((1, end + 1));
        let (logits, hidden) = {
            let mut decoder = sessions.decoder.borrow_mut();
            run_decoder(&mut decoder, &embeds, &mask, &mut step_cache)?
        };
        step_logits_last = extract_last(&logits);
        step_hidden_last = extract_last(&hidden);
    }

    let (logits_max, logits_mean) = diff_stats(&full_last_logits, &step_logits_last);
    let (hidden_max, hidden_mean) = diff_stats(&full_last_hidden, &step_hidden_last);

    println!("seq_len={seq_len}");
    println!("logits_max_abs_diff={logits_max:.8}");
    println!("logits_mean_abs_diff={logits_mean:.8}");
    println!("hidden_max_abs_diff={hidden_max:.8}");
    println!("hidden_mean_abs_diff={hidden_mean:.8}");

    Ok(())
}

fn run_decoder(
    decoder: &mut ort::session::Session,
    inputs_embeds: &Array3<f32>,
    attention_mask: &Array2<i64>,
    cache: &mut GenerationCache,
) -> Result<(Array3<f32>, Array3<f32>)> {
    let t_inputs = Value::from_array(inputs_embeds.as_standard_layout().to_owned())?;
    let t_mask = Value::from_array(attention_mask.as_standard_layout().to_owned())?;

    let mut feeds: Vec<(String, ort::value::DynValue)> = vec![
        ("inputs_embeds".to_string(), t_inputs.into_dyn()),
        ("attention_mask".to_string(), t_mask.into_dyn()),
    ];
    feeds.extend(cache.prepare_cache_inputs());

    let outputs = decoder.run(feeds)?;

    let logits = outputs
        .get("logits")
        .ok_or_else(|| anyhow::anyhow!("logits output missing"))?
        .try_extract_array::<f32>()?
        .to_owned()
        .into_dimensionality::<ndarray::Ix3>()?;

    let hidden = outputs
        .get("hidden_states")
        .ok_or_else(|| anyhow::anyhow!("hidden_states output missing"))?
        .try_extract_array::<f32>()?
        .to_owned()
        .into_dimensionality::<ndarray::Ix3>()?;

    cache.update(&outputs)?;

    Ok((logits, hidden))
}

fn extract_last(arr: &Array3<f32>) -> Vec<f32> {
    let seq_len = arr.shape()[1];
    arr.slice(ndarray::s![0, seq_len - 1, ..]).iter().copied().collect()
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        let diff = (lhs - rhs).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff;
    }
    mean_abs /= a.len().max(1) as f32;
    (max_abs, mean_abs)
}
