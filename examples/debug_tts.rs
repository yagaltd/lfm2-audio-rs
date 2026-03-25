use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use lfm2_audio::{Device, LFM2Audio, Precision, TTSOptions};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX")]
    model: PathBuf,

    #[arg(long, default_value = "fp16")]
    precision: String,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value = "Perform TTS. Use the US male voice.")]
    voice: String,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    text_temperature: f32,

    #[arg(long, default_value_t = 0.0)]
    audio_temperature: f32,

    #[arg(long, default_value_t = 1)]
    audio_top_k: usize,

    #[arg(long, default_value_t = false)]
    show_all_frames: bool,
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

    let debug = model.tts().synthesize_debug(&args.prompt, &options)?;

    println!("text_tokens={:?}", debug.text_tokens);
    println!("audio_frames={}", debug.audio_codes.len());
    let frames_to_show = if args.show_all_frames {
        debug.audio_codes.len()
    } else {
        debug.audio_codes.len().min(12)
    };
    for (idx, frame) in debug.audio_codes.iter().take(frames_to_show).enumerate() {
        println!("frame[{idx:02}]={frame:?}");
    }

    let eos_like = debug
        .audio_codes
        .iter()
        .filter(|frame| frame[0] >= 2048)
        .count();
    println!("frames_with_eos_like_codebook0={}", eos_like);

    let max_abs = debug.audio.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let mean_abs = if debug.audio.is_empty() {
        0.0
    } else {
        debug.audio.iter().map(|v| v.abs()).sum::<f32>() / debug.audio.len() as f32
    };
    println!("audio_samples={}", debug.audio.len());
    println!("audio_max_abs={:.6}", max_abs);
    println!("audio_mean_abs={:.6}", mean_abs);

    Ok(())
}
