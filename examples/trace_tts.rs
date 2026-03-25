use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use lfm2_audio::{Device, LFM2Audio, Precision, TTSOptions};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX")]
    model: PathBuf,

    #[arg(long, default_value = "q4")]
    precision: String,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value = "Perform TTS. Use the UK female voice.")]
    voice: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    text_temperature: f32,

    #[arg(long, default_value_t = 0.0)]
    audio_temperature: f32,

    #[arg(long, default_value_t = 1)]
    audio_top_k: usize,

    #[arg(long, default_value_t = 4)]
    trace_frames: usize,
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

fn parse_device(value: &str) -> Result<Device> {
    Ok(match value {
        "cpu" => Device::CPU,
        "cuda" => Device::Cuda,
        "coreml" => Device::CoreML,
        "directml" => Device::DirectML,
        "tensorrt" => Device::TensorRT,
        other => anyhow::bail!("unsupported device: {}", other),
    })
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let precision = parse_precision(&args.precision)?;
    let device = parse_device(&args.device)?;

    let model = LFM2Audio::from_pretrained(&args.model, precision, device)?;
    let options = TTSOptions {
        system_prompt: args.voice,
        max_new_tokens: args.max_new_tokens,
        text_temperature: args.text_temperature,
        audio_temperature: args.audio_temperature,
        audio_top_k: args.audio_top_k,
    };

    let trace = model
        .tts()
        .synthesize_trace(&args.prompt, &options, args.trace_frames)?;

    println!("text_tokens={:?}", trace.text_tokens);
    println!("audio_frames={}", trace.audio_codes.len());
    for frame in &trace.frame_traces {
        println!("trace_frame={}", frame.frame_index);
        println!("hidden_prefix={:?}", frame.hidden_prefix);
        for (codebook_idx, codebook) in frame.codebook_traces.iter().enumerate() {
            println!(
                "codebook[{:02}] chosen={} argmax={} argmax_logit={:.8}",
                codebook_idx, codebook.chosen_token, codebook.argmax_token, codebook.argmax_logit
            );
            println!("top_candidates={:?}", codebook.top_candidates);
        }
    }

    Ok(())
}
