# LFM2-Audio-RS

Rust bindings and demo server for Liquid AI's [`LFM2.5-Audio-1.5B-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX) model, using [ONNX Runtime](https://onnxruntime.ai/) through the Rust [`ort`] crate.

This repo is built around the ONNX export, not the original PyTorch checkpoints. In practice the default path here is:

- model: `LFM2.5-Audio-1.5B-ONNX`
- runtime: ONNX Runtime via Rust `ort`
- precision: `q4`
- default CPU path: sequential ASR and sequential TTS

## What Works Well

- ASR: audio to text
- TTS: text to audio
- Interleaved chat: text/audio input with text+audio output
- Browser demo server for websocket interleaved chat and HTTP ASR/TTS

## Practical Guidance

- Use `q4` unless you have a strong reason not to.
- On CPU, sequential ASR and sequential TTS are the recommended paths. TTS is not smooth.
- Interleaved mode works, but CPU realtime speech-to-speech is still not smooth enough. It can sound choppy under load.
- On GPU, interleaved mode is the path that makes the most sense to push further.

## Model Files

Download the Liquid AI ONNX package and place it under a model directory such as:

```text
models/LFM2.5-Audio-1.5B-ONNX/
  config.json
  tokenizer.json
  tokenizer_config.json
  onnx/
    audio_encoder_q4.onnx
    decoder_q4.onnx
    vocoder_depthformer_q4.onnx
    audio_detokenizer_q4.onnx
    embed_tokens.bin
    embed_tokens.json
    audio_embedding.bin
    audio_embedding.json
    *.onnx_data
```

Important:

- `.onnx` contains the graph.
- `.onnx_data` contains external weights for large ONNX models.
- This crate expects the Liquid AI ONNX export layout, including the embedding binaries.

## Build

```bash
cargo build --release
```

Optional accelerators:

```bash
cargo build --release --features cuda
cargo build --release --features coreml
```

## Sequential Mode

Sequential mode means using the dedicated ASR and TTS pipelines directly instead of the interleaved chat pipeline.

### Library: ASR

```rust
use lfm2_audio::{ASROptions, Device, LFM2Audio, Precision};

fn main() -> anyhow::Result<()> {
    let model = LFM2Audio::from_pretrained(
        "./models/LFM2.5-Audio-1.5B-ONNX",
        Precision::Q4,
        Device::CPU,
    )?;

    let (audio, spec) = lfm2_audio::load_audio("input.wav")?;
    let text = model
        .asr()
        .transcribe(&audio, spec.sample_rate, &ASROptions::default())?;

    println!("{}", text);
    Ok(())
}
```

### Library: TTS

```rust
use lfm2_audio::{Device, LFM2Audio, Precision, TTSOptions};

fn main() -> anyhow::Result<()> {
    let model = LFM2Audio::from_pretrained(
        "./models/LFM2.5-Audio-1.5B-ONNX",
        Precision::Q4,
        Device::CPU,
    )?;

    let options = TTSOptions::default()
        .with_system_prompt("Use the UK female voice.");
    let audio = model.tts().synthesize("Hello from Rust.", &options)?;
    lfm2_audio::save_audio("output.wav", &audio, 24_000)?;
    Ok(())
}
```

### Server: ASR

```bash
./target/release/lfm2-server ./models/LFM2.5-Audio-1.5B-ONNX --port 8080

curl -X POST http://127.0.0.1:8080/api/asr \
  -H "Content-Type: audio/wav" \
  --data-binary @input.wav
```

### Server: TTS

```bash
curl -X POST http://127.0.0.1:8080/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from the sequential TTS route.",
    "voice": "Use the UK female voice."
  }' \
  --output speech.wav
```

## Interleaved Mode

Interleaved mode keeps a persistent chat session with decoder KV cache and emits text updates plus audio chunks over the websocket route:

```text
/ws/interleaved
```

This is the speech-to-speech / multimodal chat path. It is the most expensive mode in the repo.

Current guidance:

- CPU: functional, but not smooth enough for true realtime speech-to-speech
- GPU: preferred if you want to keep pushing interleaved UX

## Lag Debugging

The demo server can write an opt-in NDJSON trace for interleaved sessions:

```bash
./target/release/lfm2-server \
  ./models/LFM2.5-Audio-1.5B-ONNX \
  --interleaved-log ./logs/interleaved.ndjson
```

Or with env:

```bash
LFM2_INTERLEAVED_LOG_PATH=./logs/interleaved.ndjson ./target/release/lfm2-server ./models/LFM2.5-Audio-1.5B-ONNX
```

The trace captures:

- session start/reset/close
- user text turns
- user audio turn metadata
- final assistant text
- interleaved stream timing summary:
  first frame
  first decode
  detokenizer wait
  frame-gap spikes
  queue wait

This is meant to explain where interleaved lag comes from without logging raw PCM.

## Browser Demo

The built-in page now exposes three paths:

- interleaved websocket chat
- sequential ASR file transcription
- sequential TTS from typed text

That split is intentional. Sequential ASR/TTS are the more realistic CPU workflows today.

## Architecture

```text
Audio encoder          : audio -> embeddings
Decoder                : main autoregressive LFM2 backbone
Depthformer            : autoregressive audio codebook prediction
Audio detokenizer      : audio codes -> waveform
Binary embeddings      : text/audio embedding lookup when available
ONNX Runtime sessions  : execution backend through Rust ort
```

## API Surface

- `LFM2Audio::asr()` for sequential transcription
- `LFM2Audio::tts()` for sequential synthesis
- `LFM2Audio::interleaved()` and chat session APIs for multimodal/interleaved generation
- `POST /api/asr` and `POST /v1/audio/transcriptions`
- `POST /api/tts` and `POST /v1/audio/speech`
- `GET /ws/interleaved`

## Notes

- The server keeps model sessions resident.
- Interleaved chat sessions preserve history through the persistent decoder cache until reset or socket close.
- There is currently no automatic session history trimming in interleaved chat.

## License

MIT
