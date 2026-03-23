# LFM2-Audio-RS

Rust implementation of [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX) multimodal speech model supporting ASR, TTS, and interleaved audio-text generation.

## Features

- **ASR (Automatic Speech Recognition)**: Audio → Text
- **TTS (Text-to-Speech)**: Text → Audio with voice control
- **Interleaved**: Audio ↔ Text + Audio (speech-to-speech)
- **Multi-turn Chat**: Persistent conversation with KV-cache

## Model Requirements

Download the ONNX model from HuggingFace:

```bash
mkdir -p models/LFM2.5-Audio-1.5B-ONNX
cd models/LFM2.5-Audio-1.5B-ONNX

# Download config and tokenizer
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/config.json
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/tokenizer.json
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/tokenizer_config.json

# Download ONNX models (Q4 recommended for most uses)
mkdir -p onnx
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/audio_encoder_q4.onnx
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/decoder_q4.onnx
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/vocoder_depthformer_q4.onnx
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/audio_detokenizer_q4.onnx

# Download embedding binaries
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/embed_tokens.bin
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/embed_tokens.json
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/audio_embedding.bin
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main/onnx/audio_embedding.json

# Download external data files if they exist (for models > 2GB)
# Check the HuggingFace repo for .onnx_data files
```

## Installation

### From Source

```bash
git clone https://github.com/yourusername/lfm2-audio-rs.git
cd lfm2-audio-rs

# CPU only
cargo build --release

# With CUDA
cargo build --release --features cuda

# With CoreML (macOS)
cargo build --release --features coreml
```

Binaries will be in `target/release/`.

## Usage

### CLI

#### ASR (Transcription)

```bash
# Basic transcription
./lfm2-audio --model ./models/LFM2.5-Audio-1.5B-ONNX asr input.wav

# With options
./lfm2-audio \
  --model ./models/LFM2.5-Audio-1.5B-ONNX \
  --precision q4 \
  --device cpu \
  asr input.wav \
  --output transcript.txt \
  --system-prompt "Transcribe this audio accurately."
```

#### TTS (Synthesis)

```bash
# Basic synthesis
./lfm2-audio --model ./models/LFM2.5-Audio-1.5B-ONNX tts "Hello, world!"

# With voice selection
./lfm2-audio \
  --model ./models/LFM2.5-Audio-1.5B-ONNX \
  tts "Hello, world!" \
  --voice "Use the UK female voice." \
  --output hello.wav \
  --audio-temp 0.8
```

#### Show Model Info

```bash
./lfm2-audio --model ./models/LFM2.5-Audio-1.5B-ONNX info
```

### Library

```rust
use lfm2_audio::{LFM2Audio, Precision, Device, ASROptions, TTSOptions};

fn main() -> anyhow::Result<()> {
    // Load model
    let model = LFM2Audio::from_pretrained(
        "./models/LFM2.5-Audio-1.5B-ONNX",
        Precision::Q4,
        Device::CPU,
    )?;

    // ASR
    let (audio, spec) = lfm2_audio::load_audio("input.wav")?;
    let text = model.asr().transcribe(&audio, spec.sample_rate, &ASROptions::default())?;
    println!("Transcription: {}", text);

    // TTS
    let options = TTSOptions::default()
        .with_system_prompt("Use the UK female voice.");
    let speech = model.tts().synthesize("Hello, world!", &options)?;
    lfm2_audio::save_audio("output.wav", &speech, 24000)?;

    Ok(())
}
```

### API Server

```bash
# Start server
./lfm2-server ./models/LFM2.5-Audio-1.5B-ONNX --port 8080

# Transcribe
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{
    "file": "<base64-audio>",
    "model": "lfm2.5-audio"
  }'

# Synthesize
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2.5-audio",
    "input": "Hello, world!",
    "voice": "alloy"
  }' \
  --output speech.wav
```

## Architecture

```
LFM2.5-Audio Model
├── Audio Encoder (Conformer)
│   └── Mel Spectrogram → Audio Embeddings
├── LFM2 Decoder (1.2B params, 16 layers)
│   ├── Conv layers with cache
│   └── Attention layers with KV-cache
├── Depthformer (6 layers)
│   └── Autoregressive codebook prediction
└── Audio Detokenizer
    └── Codes → STFT → Waveform (24kHz)
```

## Performance

Expected performance on modern hardware (Q4 precision):

| Task | RTF (lower is better) | Notes |
|------|----------------------|-------|
| ASR | ~0.5-1.0x | Real-time capable |
| TTS | ~1.0-2.0x | Depends on length |

RTF = Real-Time Factor (processing time / audio duration)

## Project Structure

```
src/
├── lib.rs           # Public API
├── model.rs         # Main LFM2Audio struct
├── asr.rs           # ASR pipeline
├── tts.rs           # TTS pipeline
├── interleaved.rs   # Speech-to-speech
├── chat.rs          # Multi-turn chat
├── cache.rs         # KV-cache management
├── sessions.rs      # ONNX session loading
├── embeddings.rs    # Binary embedding loaders
├── tokenizer.rs     # Tokenizer wrapper
├── config.rs        # Model configuration
├── error.rs         # Error types
└── audio/
    ├── mod.rs       # Audio I/O
    ├── mel.rs       # Mel spectrogram
    └── istft.rs     # Inverse STFT
```

## Implementation Status

- [x] Model loading (ONNX sessions + binary embeddings)
- [x] Mel spectrogram computation
- [x] ISTFT for audio detokenization
- [x] KV-cache management
- [x] ASR pipeline (partial)
- [ ] ASR pipeline (complete with full cache handling)
- [ ] TTS pipeline (complete)
- [ ] Depthformer integration
- [ ] Audio detokenizer integration
- [ ] Interleaved mode
- [ ] Chat sessions
- [ ] API server

## References

- [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX) - Original model
- [Liquid AI Cookbook](https://github.com/Liquid4All/onnx-export) - Python reference implementation
- [ONNX Runtime](https://onnxruntime.ai/) - Inference engine

## License

MIT OR Apache-2.0
