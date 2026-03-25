# LFM2 Audio Streaming Gap Analysis

Date: 2026-03-24

## Purpose

This document captures the current state of streaming audio in `lfm2-audio-rs`, the exact gaps versus the official `liquid-audio` realtime path, and the evidence collected so far. The goal is traceability before any attempt to change ONNX export, switch runtimes, or add new hardware backends.

## Scope

This analysis covers:

- current Rust server streaming path in `lfm2-audio-rs`
- current `onnx-export` audio detokenizer contract
- official `liquid-audio` Python streaming path
- ONNX Runtime hardware/provider support relevant to AMD GPU/NPU
- measured evidence for where the bottleneck actually is

This analysis does not propose a final implementation. It documents what exists, what is missing, and what the evidence rules out.

## Current Rust Streaming Path

### High-level flow

For interleaved audio replies, the current Rust server does:

1. generate interleaved text/audio events incrementally
2. collect audio code frames from the model
3. periodically decode recent audio codes to PCM in `StreamingAudioDecoder`
4. send PCM chunks over WebSocket
5. play them in the browser using an `AudioWorklet`

Relevant code:

- streaming decode configuration and CLI/env parsing:
  - `src/bin/server.rs`
- streaming detokenizer implementation:
  - `src/bin/server.rs`
- worker-side interleaved streaming:
  - `src/bin/server.rs`
- WebSocket send path:
  - `src/bin/server.rs`

### Exact decoder behavior

`StreamingAudioDecoder` is currently a bounded recent-context decoder, not a truly stateful decoder.

For each flush it does:

1. decode the current recent window with `tts.decode_audio_codes_raw(window_codes)`
2. decode the already-flushed context with `tts.decode_audio_codes_raw(context_codes)`
3. subtract the context waveform prefix
4. emit the remaining waveform tail as PCM

This is implemented in:

- `src/bin/server.rs`

That means each streamed chunk still requires two detokenizer runs. This is the most important current cost center.

## Measured Evidence

### What was instrumented

The server now logs:

- stream decode config (`batch_frames`, `context_frames`)
- per-flush detokenizer timing:
  - `full_decode_ms`
  - `context_decode_ms`
  - `total_decode_ms`
- chunk-ready time (`elapsed_ms`)
- queue lag from worker to socket (`queue_wait_ms`)
- WebSocket send time (`ws_send_ms`)

Relevant code:

- `src/bin/server.rs`

### Controlled reproduction

Environment used:

- `q4`
- CPU
- `2` workers
- same audio input on each run:
  - `liquid-audio/assets/question.wav`

#### Baseline: `batch=4`, `context=16`

Observed:

- first audio chunk at about `1116 ms`
- each chunk size: `15360 bytes`
- that is `7680` samples at `24 kHz`
- so each chunk contains about `320 ms` of audio

Steady-state detokenizer timing:

- `full_decode_ms ~ 1611`
- `context_decode_ms ~ 1290`
- `total_decode_ms ~ 2900`

Interpretation:

- about `320 ms` of audio is produced every `~2900 ms`
- sustained throughput is about `0.11x realtime`

#### Tuned: `batch=16`, `context=8`

Observed:

- first audio chunk at about `3663 ms`
- each chunk size: `61440 bytes`
- that is `30720` samples at `24 kHz`
- so each chunk contains about `1280 ms` of audio

Steady-state detokenizer timing:

- `full_decode_ms ~ 1935`
- `context_decode_ms ~ 650`
- `total_decode_ms ~ 2580-2610`

Interpretation:

- about `1280 ms` of audio is produced every `~4170 ms` end to end
- sustained throughput improves to about `0.3x realtime`
- this is better than baseline, but still far below realtime playback

### What this rules out

The same instrumentation showed:

- `queue_wait_ms = 0`
- `ws_send_ms = 0`

for streamed audio chunks in the local test.

That means:

- the worker-to-socket queue is not backing up
- WebSocket send cost is negligible on the tested local path
- the dominant bottleneck is upstream of the socket layer

Conclusion:

- the main bottleneck is the streaming decode path, not WebSocket transport

## Why The Browser Underruns

The browser consumes PCM at realtime. If the server produces PCM slower than realtime, the `AudioWorklet` queue empties and reports underruns.

Given the measured throughput:

- baseline path: about `0.11x realtime`
- tuned path: about `0.3x realtime`

underruns are expected even with buffering. Buffering can hide startup latency, but it cannot fix a producer that is slower than playback on average.

## Official `liquid-audio` Streaming Path

The official Python demo does not use the same audio decode path as the ONNX export used in Rust.

In `liquid-audio`:

- generation runs under `mimi.streaming(1)`
- each audio token frame is decoded immediately with `mimi.decode(...)`
- the demo yields `24 kHz` chunks directly as they become available

Relevant code:

- `liquid-audio/src/liquid_audio/demo/chat.py`

The critical lines are:

- `with torch.no_grad(), mimi.streaming(1):`
- `wav_chunk = mimi.decode(t[None, :, None])[0]`

This is a true stateful streaming decoder path.

## Current ONNX Export Path

### Exported audio detokenizer interface

The exported audio detokenizer ONNX graph accepts:

- `audio_codes`

and returns:

- `stft_features`

Relevant code:

- `onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py`

Important observation:

- there are no cache inputs
- there are no cache outputs
- the ONNX contract is stateless from the caller’s point of view

### Underlying PyTorch detokenizer behavior

The PyTorch detokenizer behind this export also does not use cache:

- `self.lfm(inputs_embeds=x, attention_mask=mask, use_cache=False)`

Relevant code:

- `liquid-audio/src/liquid_audio/detokenizer.py`

So the export is not “missing an obvious cache that already exists in the PyTorch detokenizer.” The current detokenizer path itself is non-cached.

### What `onnx-export` does today

`onnx-export` uses the ONNX detokenizer as a batch decode path:

- audio codes `[1, 8, T]`
- detokenizer output
- ISTFT reconstruction

This is conceptually aligned with what `lfm2-audio-rs` is doing today for streaming, except Rust repeats the bounded prefix/window decode to simulate incremental output.

## Mapping: Current ONNX Path vs Mimi Path

### What both paths share

- same high-level interleaved generation concept
- same model emits audio code frames incrementally
- same need to turn audio code frames into waveform output

### What differs

#### Current Rust + ONNX path

- exported component: `audio_detokenizer.onnx`
- contract: stateless `audio_codes -> stft_features`
- caller strategy: re-run recent decode window repeatedly
- output chunking: synthetic, produced by repeated detokenizer invocation

#### Official Liquid streaming path

- runtime component: `mimi`
- contract: stateful streaming decoder
- caller strategy: feed one audio frame at a time into the decoder state
- output chunking: native, decoder emits waveform incrementally

### The actual gap

The main missing piece is not “Rust forgot to cache something in the current ONNX graph.”

The main gap is:

- the current ONNX export uses a stateless STFT detokenizer path
- the official realtime demo uses a stateful Mimi decoder path

That is a decoder architecture gap, not only a transport or frontend gap.

## Hardware / Provider Gap

### What the repo supports today

`lfm2-audio-rs` currently exposes only:

- `CPU`
- `Cuda`
- `CoreML`
- `DirectML`
- `TensorRT`

Relevant code:

- `src/config.rs`

`--device auto` currently only considers:

- NVIDIA CUDA
- Apple CoreML
- Windows DirectML

Relevant code:

- `src/bin/server.rs`

So this repo is not currently wired for AMD GPU or Ryzen AI NPU execution.

### What upstream `ort` supports

The `ort` crate version already present in this environment supports more execution providers, including:

- `migraphx`
- `rocm`
- `qnn`
- `vitis`

This was verified from the installed `ort-2.0.0-rc.12` crate metadata.

So upstream ORT support is broader than what `lfm2-audio-rs` currently exposes.

### AMD Ryzen AI docs

Official Ryzen AI docs indicate:

- NPU flow uses Ryzen AI Software plus NPU drivers
- Windows-only installation path is documented
- NPU quicktest uses `VitisAIExecutionProvider`
- hybrid LLM flow uses OnnxRuntime GenAI (OGA), not plain `ort` inference
- AMD GPU flow on Ryzen AI docs uses DirectML on Windows

Sources:

- <https://ryzenai.docs.amd.com/en/latest/inst.html>
- <https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html>
- <https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html>

Implication:

- AMD NPU is not “already available automatically” to this Rust app
- it would require both the right platform stack and explicit provider wiring
- OGA is a separate integration path from the current plain ORT session model

## What We Can Say With High Confidence

1. The current browser underruns are not mainly caused by WebSocket overhead.
2. The current server-side streaming decode path is the dominant bottleneck.
3. The current ONNX export path is stateless at the audio detokenizer boundary.
4. The official Liquid realtime demo uses a different, stateful decoder path (`mimi.streaming(1)`).
5. Current Rust hardware support is narrower than upstream ORT capability and does not currently expose AMD GPU/NPU paths.

## What Is Still Unknown

1. Whether a different ONNX export could expose a genuinely stateful audio decoder path suitable for Rust + ORT.
2. Whether a Rust binding or alternative integration of Mimi is practical in this project.
3. How much performance improvement AMD GPU or Ryzen AI NPU could provide for this specific model family and graph split.
4. Whether a different batching/window strategy can improve the current ONNX path enough for acceptable pseudo-streaming, even if not true realtime.

## Decisions This Analysis Supports

### Safe conclusions

- do not spend more time blaming WebSocket transport
- do not treat frontend buffering as the main solution
- do not assume “the current ONNX export just needs a small cache patch”

### Recommended next investigations

1. Hardware acceleration track
   - map feasible AMD execution provider paths for this Rust app:
     - DirectML on Windows
     - ROCm/MIGraphX on Linux, if applicable
     - Vitis / Ryzen AI NPU only if the platform/runtime stack can actually be integrated from Rust

2. Decoder architecture track
   - compare the current ONNX detokenizer path against the Mimi path in more detail:
     - model artifact ownership
     - runtime dependencies
     - whether Mimi can be exported or wrapped for Rust

3. Export track
   - assess whether a new export should target:
     - a stateful streaming decoder graph
     - a Mimi-compatible runtime path
     - or a different ORT/OGA split

## References

### Local code

- `src/bin/server.rs`
- `src/config.rs`
- `liquid-audio/src/liquid_audio/demo/chat.py`
- `liquid-audio/src/liquid_audio/detokenizer.py`
- `onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py`

### External docs

- ONNX Runtime install / EP overview:
  - <https://onnxruntime.ai/docs/install/>
- Ryzen AI installation:
  - <https://ryzenai.docs.amd.com/en/latest/inst.html>
- Ryzen AI GPU DirectML flow:
  - <https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html>
- Ryzen AI OGA hybrid flow:
  - <https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html>
