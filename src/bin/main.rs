//! LFM2-Audio CLI

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use lfm2_audio::{ASROptions, Device, LFM2Audio, Precision, TTSOptions};

#[derive(Parser)]
#[command(name = "lfm2-audio")]
#[command(about = "LFM2.5-Audio: Multimodal speech model (ASR, TTS, Interleaved)")]
#[command(version)]
struct Cli {
    /// Path to model directory
    #[arg(short, long, default_value = "./LFM2.5-Audio-1.5B-ONNX")]
    model: PathBuf,

    /// Model precision
    #[arg(short, long, value_enum, default_value = "q4")]
    precision: PrecisionArg,

    /// Device for inference
    #[arg(short, long, value_enum, default_value = "cpu")]
    device: DeviceArg,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum PrecisionArg {
    Fp32,
    Fp16,
    Q4,
    Q8,
}

impl From<PrecisionArg> for Precision {
    fn from(arg: PrecisionArg) -> Self {
        match arg {
            PrecisionArg::Fp32 => Precision::FP32,
            PrecisionArg::Fp16 => Precision::FP16,
            PrecisionArg::Q4 => Precision::Q4,
            PrecisionArg::Q8 => Precision::Q8,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
    CoreML,
    DirectML,
    TensorRT,
}

impl From<DeviceArg> for Device {
    fn from(arg: DeviceArg) -> Self {
        match arg {
            DeviceArg::Cpu => Device::CPU,
            DeviceArg::Cuda => Device::Cuda,
            DeviceArg::CoreML => Device::CoreML,
            DeviceArg::DirectML => Device::DirectML,
            DeviceArg::TensorRT => Device::TensorRT,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe audio to text (ASR)
    Asr {
        /// Input audio file (WAV, 16kHz)
        input: PathBuf,
        /// Output text file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// System prompt
        #[arg(short, long)]
        system_prompt: Option<String>,
        /// Max tokens to generate
        #[arg(short, long, default_value = "256")]
        max_tokens: usize,
        /// Temperature (0 = greedy)
        #[arg(short, long, default_value = "1.0")]
        temperature: f32,
    },
    /// Synthesize speech from text (TTS)
    Tts {
        /// Input text
        text: String,
        /// Output audio file (WAV)
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,
        /// Voice description
        #[arg(short, long)]
        voice: Option<String>,
        /// Max tokens
        #[arg(short, long, default_value = "1024")]
        max_tokens: usize,
        /// Text temperature
        #[arg(long, default_value = "1.0")]
        text_temp: f32,
        /// Audio temperature
        #[arg(long, default_value = "0.8")]
        audio_temp: f32,
    },
    /// Chat with interleaved audio/text
    Chat,
    /// Show model info
    Info,
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Asr { input, output, system_prompt, max_tokens, temperature } => {
            cmd_asr(&cli.model, cli.precision.into(), cli.device.into(), input, output, system_prompt, max_tokens, temperature)
        }
        Commands::Tts { text, output, voice, max_tokens, text_temp, audio_temp } => {
            cmd_tts(&cli.model, cli.precision.into(), cli.device.into(), text, output, voice, max_tokens, text_temp, audio_temp)
        }
        Commands::Chat => {
            cmd_chat(&cli.model, cli.precision.into(), cli.device.into())
        }
        Commands::Info => {
            cmd_info(&cli.model, cli.precision.into(), cli.device.into())
        }
    }
}

fn cmd_asr(
    model_path: &PathBuf,
    precision: Precision,
    device: Device,
    input: PathBuf,
    output: Option<PathBuf>,
    system_prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
) -> Result<()> {
    eprintln!("Loading model from {}...", model_path.display());
    let model = LFM2Audio::from_pretrained(model_path, precision, device)?;
    eprintln!("Model loaded: {:?}", model.info());

    eprintln!("Loading audio from {}...", input.display());
    let (audio, spec) = lfm2_audio::load_audio(&input)?;
    eprintln!("Audio: {} samples at {} Hz", audio.len(), spec.sample_rate);

    let options = ASROptions {
        system_prompt: system_prompt.unwrap_or_else(|| "Perform ASR.".to_string()),
        max_new_tokens: max_tokens,
        temperature,
    };

    eprintln!("Transcribing...");
    let start = std::time::Instant::now();
    let text = model.asr().transcribe(&audio, spec.sample_rate, &options)?;
    let elapsed = start.elapsed();

    let audio_duration = audio.len() as f32 / spec.sample_rate as f32;
    let rtf = elapsed.as_secs_f32() / audio_duration;

    eprintln!("Done! RTF: {:.2}x", rtf);

    if let Some(output_path) = output {
        std::fs::write(&output_path, &text)?;
        eprintln!("Saved to {}", output_path.display());
    } else {
        println!("{}", text);
    }

    Ok(())
}

fn cmd_tts(
    model_path: &PathBuf,
    precision: Precision,
    device: Device,
    text: String,
    output: PathBuf,
    voice: Option<String>,
    max_tokens: usize,
    text_temp: f32,
    audio_temp: f32,
) -> Result<()> {
    eprintln!("Loading model from {}...", model_path.display());
    let model = LFM2Audio::from_pretrained(model_path, precision, device)?;
    eprintln!("Model loaded: {:?}", model.info());

    let options = TTSOptions {
        system_prompt: voice.unwrap_or_else(|| "Perform TTS. Use the UK female voice.".to_string()),
        max_new_tokens: max_tokens,
        text_temperature: text_temp,
        audio_temperature: audio_temp,
        ..Default::default()
    };

    eprintln!("Synthesizing: '{}'", text.chars().take(50).collect::<String>());
    let start = std::time::Instant::now();
    let audio = model.tts().synthesize(&text, &options)?;
    let elapsed = start.elapsed();

    let audio_duration = audio.len() as f32 / 24000.0;
    let rtf = elapsed.as_secs_f32() / audio_duration;

    eprintln!("Done! Generated {} samples ({:.2}s), RTF: {:.2}x", audio.len(), audio_duration, rtf);

    lfm2_audio::save_audio(&output, &audio, 24000)?;
    eprintln!("Saved to {}", output.display());

    Ok(())
}

fn cmd_chat(
    _model_path: &PathBuf,
    _precision: Precision,
    _device: Device,
) -> Result<()> {
    eprintln!("Interactive chat mode not yet implemented");
    eprintln!("Use 'asr' or 'tts' subcommands instead");
    Ok(())
}

fn cmd_info(
    model_path: &PathBuf,
    precision: Precision,
    device: Device,
) -> Result<()> {
    eprintln!("Loading model from {}...", model_path.display());
    let model = LFM2Audio::from_pretrained(model_path, precision, device)?;
    
    let info = model.info();
    println!("LFM2.5-Audio Model Info:");
    println!("  Hidden size: {}", info.hidden_size);
    println!("  Vocab size: {}", info.vocab_size);
    println!("  Num layers: {}", info.num_layers);
    println!("  Num codebooks: {}", info.num_codebooks);
    println!("  Precision: {:?}", precision);
    println!("  Device: {:?}", device);

    Ok(())
}