# RustyMimi and GGUF Traceability Note

Date: 2026-03-24

## Purpose

This note records two follow-up investigations:

1. what can be established about `rustymimi` and the underlying Rust Mimi contract
2. what can be established about the local `LFM2.5-Audio-1.5B-GGUF` package, especially whether its audio path is stateful or stateless

This note complements:

- [2026-03-24-streaming-gap-analysis.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-streaming-gap-analysis.md)
- [2026-03-24-decoder-architecture-track.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-decoder-architecture-track.md)
- [2026-03-24-mimi-ecosystem-traceability.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-mimi-ecosystem-traceability.md)
- [2026-03-24-lfm2-vs-pocket-mimi-contracts.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-lfm2-vs-pocket-mimi-contracts.md)

## RustyMimi Investigation

### What is directly confirmed

Kyutai publicly states that:

- `rustymimi` exists
- it is a Rust implementation of Mimi
- it has Python bindings
- it can be built from `rust/mimi-pyo3/Cargo.toml`

Sources:

- https://github.com/kyutai-labs/moshi
- https://docs.rs/crate/moshi-db/latest
- https://pypi.org/project/rustymimi/

The `moshi` README also states that Rust provides both Moshi and Mimi, and that Mimi is the streaming audio codec used by the overall system.

### What the binding source now confirms

The actual `mimi-pyo3` source is available in the Kyutai repo and it exposes the Python module:

- `rustymimi`

with two classes:

- `Tokenizer`
- `StreamTokenizer`

and one helper function:

- `write_wav`

Evidence:

- https://raw.githubusercontent.com/kyutai-labs/moshi/main/rust/mimi-pyo3/src/lib.rs
- https://raw.githubusercontent.com/kyutai-labs/moshi/main/rust/mimi-pyo3/Cargo.toml

The exact API surface visible in the binding source is:

- `Tokenizer(path, *, num_codebooks=8, dtype=\"f32\", max_seq_len=None)`
- `Tokenizer.encode(pcm_data)`
- `Tokenizer.encode_step(pcm_data)`
- `Tokenizer.decode(codes)`
- `Tokenizer.decode_step(codes)`
- `Tokenizer.reset()`

and for the streaming helper:

- `StreamTokenizer(path, *, num_codebooks=8, dtype=\"f32\", max_seq_len=None)`
- `StreamTokenizer.encode(pcm_chunk)`
- `StreamTokenizer.decode(codes)`
- `StreamTokenizer.get_encoded()`
- `StreamTokenizer.get_decoded()`

Most importantly, the binding source shows the decode tensor shapes explicitly:

- `Tokenizer.decode(codes)` accepts a 3D NumPy array and forwards it to `self.mimi.decode(&codes)`
- `Tokenizer.decode_step(codes)` also accepts a 3D NumPy array and forwards it to `self.mimi.decode_step(&codes.into(), &().into())`
- `StreamTokenizer.decode(codes)` takes a 2D NumPy array, converts it to nested vectors, and sends one frame worth of codebooks to the worker thread
- the worker reconstructs a tensor and calls:
  - `d_mimi.decode_step(&codes.into())`

The worker uses:

- `unsqueeze(2)` on the 2D codes

which means the streaming step path expects one time-step at a time, with codebooks provided explicitly.

The reset behavior is also explicit in the source:

- `Tokenizer.reset()` calls `self.mimi.reset_state()`

This is the strongest direct evidence yet that `rustymimi` is a true stateful streaming decode surface, not only a batch wrapper.

### What the Rust crate structure confirms

The public Rust docs for `moshi_db` show that the Rust backend contains:

- `mimi`
- `streaming`
- `kv_cache`
- `tts_streaming`

Evidence:

- https://docs.rs/moshi-db/latest/moshi_db/
- https://docs.rs/moshi/latest/moshi/all.html

This is strong evidence that the Rust Mimi implementation is not a thin wrapper around a stateless batch decoder. It lives inside a broader streaming-oriented Rust/Candle model stack.

### What the shared code path suggests

From the local Liquid/Kyutai-compatible Mimi implementation, the important contract is:

- Mimi decode consumes discrete code tensors
- state is carried on the Mimi object via streaming mode
- state is reset through explicit streaming lifecycle calls

Evidence from the local code:

- `decode(codes)` in [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:406)
- latent decode in [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:431)
- streaming lifecycle in [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:131)
- state reset in [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:139)
- server-side Mimi reset/use in [server.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/server.py:134) and [server.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/server.py:141)

The earlier uncertainty about the binding surface is now resolved:

- `rustymimi` does expose both batch decode and step decode
- it also exposes explicit reset behavior
- and it has a streaming helper class that internally keeps encoder/decoder workers alive

### Direct compatibility test with LFM2 output frames

A direct local experiment was run against real LFM2 output codes saved from:

- [/home/aurel/Documents/vibe/STT-rust/tts_output_codes.npy](/home/aurel/Documents/vibe/STT-rust/tts_output_codes.npy)

Those codes had shape:

- `[T, 8] = [44, 8]`

The important compatibility findings were:

1. `rustymimi` does accept LFM2's 8-code audio frames directly.
2. The accepted dtype is:
   - `uint32`
3. The accepted shapes are:
   - batch decode: `[1, 8, T]`
   - step decode: `[1, 8, 1]` for one frame

Observed results from the local run:

- `Tokenizer.decode([1, 8, 44] uint32)` returned:
  - `(1, 1, 84480)` `float32`
- `Tokenizer.decode_step([1, 8, 1] uint32)` returned:
  - `(1, 1, 1920)` `float32` per frame
- concatenating 44 `decode_step` outputs produced:
  - `84480` samples total
- batch decode vs step decode comparison:
  - correlation: `0.9999999999996678`
  - mean absolute error: `1.9234621007058195e-08`
  - sample-count difference: `0`

This is strong direct evidence that:

- LFM2's emitted 8-code frames can be fed directly into `rustymimi`
- `rustymimi` step decode follows the expected state lifecycle
- the per-frame output size matches the expected `1920` samples = `80 ms` at `24 kHz`

There was one caveat in the same experiment:

- `StreamTokenizer.decode(...)` did not behave as a drop-in streaming path for this test setup
- `Tokenizer.decode_step(...)` was the path that worked correctly and matched batch decode

The concrete audio outputs from the experiment were saved under:

- [/home/aurel/Documents/vibe/STT-rust/rustymimi-test/batch.wav](/home/aurel/Documents/vibe/STT-rust/rustymimi-test/batch.wav)
- [/home/aurel/Documents/vibe/STT-rust/rustymimi-test/step.wav](/home/aurel/Documents/vibe/STT-rust/rustymimi-test/step.wav)
- [/home/aurel/Documents/vibe/STT-rust/rustymimi-test/stream.wav](/home/aurel/Documents/vibe/STT-rust/rustymimi-test/stream.wav)

### Practical conclusion for `lfm2-audio-rs`

`rustymimi` is realistic as a candidate integration path because:

- it is the right decoder family
- it is Rust-based
- it is associated with a streaming Mimi implementation

The remaining practical question is narrower now:

- not whether `rustymimi` can decode LFM2-style code frames
- but how best to integrate it in Rust without reintroducing Python into the serving path

The direct experiment established that the exact LFM2 frame contract is already compatible with `rustymimi` step decode once the frames are cast to `uint32`.

## GGUF Investigation

### Local package layout

The local `LiquidAI__LFM2.5-Audio-1.5B-GGUF` package contains:

- main model:
  - `LFM2.5-Audio-1.5B-Q8_0.gguf`
- multimodal projector:
  - `mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf`
- tokenizer/speaker artifact:
  - `tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf`
- vocoder artifact:
  - `vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf`
- bundled runtime:
  - `llama-liquid-audio-ubuntu-x64.zip`

Evidence:

- [/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF](/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF)

### Runner interface

The bundled CLI expects all audio-related pieces separately:

- `--model`
- `--mmproj`
- `--model-vocoder`
- `--tts-speaker-file`

Evidence from the bundled CLI help:

- `Usage: ... -m <model.gguf> --mmproj <mmproj.gguf> -mv <vocoder.gguf> --tts-speaker-file <tokenizer.gguf> ...`
- `--model-vocoder`
- `--tts-speaker-file`

This was read from:

- `/tmp/llama-liquid-audio-ubuntu-x64/llama-liquid-audio-cli`

### What the main GGUF reveals

Readable metadata in the main model GGUF includes:

- `lfm2-audio`
- `liquid-audio`
- `audio-to-audio`
- `lfm2.shortconv.l_cache`
- tokenizer metadata

Evidence:

- strings from [/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF/LFM2.5-Audio-1.5B-Q8_0.gguf](/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF/LFM2.5-Audio-1.5B-Q8_0.gguf)

`lfm2.shortconv.l_cache` is a strong sign that the main GGUF model itself includes internal caching-oriented architecture metadata.

### What the vocoder GGUF reveals

Readable strings in the vocoder GGUF include:

- `this model cannot be used as LLM, use it via --model-vocoder in TTS examples`
- `audio_embedding.embedding.weight`
- `audio_embedding.embedding_norm.weight`
- `audio_embedding.to_logits.weight`
- `depthformer.layers.*`

Evidence:

- strings from [/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF/vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf](/home/aurel/.cache/liquid-runner/LiquidAI__LFM2.5-Audio-1.5B-GGUF/vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf)

This matches the audio-generation side we already know from the ONNX path:

- audio embedding
- depthformer

Notably, the readable strings do not suggest a separately exported `audio_detokenizer`-style artifact inside the GGUF package.

### What the runtime library reveals

Strings in `libliquid-audio.so` show several important symbols:

- `liquid::audio::Decoder::embed_for_detokenizer`
- `mtmd_audio_streaming_istft::process_frame`
- `mtmd_audio_streaming_istft::reset`
- `istft_state`
- `depthformer_n_layer`
- `depthformer_n_embd`
- `token.size() == config.n_codebook`

Evidence:

- strings from `/tmp/llama-liquid-audio-ubuntu-x64/libliquid-audio.so`

This is the strongest GGUF-side clue in the whole investigation.

It implies:

1. the runtime has a dedicated decoder implementation in native code
2. ISTFT is handled in a streaming way inside the runtime
3. decoder state exists internally (`istft_state`)
4. the runtime is aware of codebook-sized token frames

### Stateful or stateless?

At the user-facing artifact boundary, the GGUF package is opaque:

- you do not get an explicit graph contract like ONNX
- you do not see state tensors directly

But at the runtime level, the evidence strongly suggests the audio path is stateful internally.

Why:

- the CLI exposes generic KV cache controls for the main model
- the library contains `mtmd_audio_streaming_istft::process_frame` and `reset`
- the library contains `istft_state`
- the library contains codebook-aware decoder logic

So the best traceable conclusion is:

- the GGUF runtime is not behaving like a stateless `audio_codes -> full waveform` batch artifact
- state almost certainly exists, but it is encapsulated inside the native runner/library, not exposed as user-managed tensors

## Comparison Against ONNX

### ONNX

- state visibility: explicit if exported, otherwise absent
- current LFM2 audio decoder artifact: stateless `audio_detokenizer`
- caller sees exact tensor contract

### GGUF

- state visibility: hidden inside runner/runtime
- audio runtime appears to contain streaming ISTFT and decoder internals
- caller sees CLI/runtime options, not graph tensors

This means:

- ONNX is easier to inspect formally
- GGUF may already have the behavior we want, but that behavior is hidden in the native runtime rather than packaged as an inspectable graph

## Practical Conclusions

1. `rustymimi` remains a plausible decoder integration path.
   - There is strong evidence it belongs to the correct streaming Mimi family.

2. `rustymimi` binding API and basic LFM2 compatibility are now directly confirmed.
   - The open question is integration strategy, not raw compatibility.

3. The GGUF package strongly suggests a stateful internal decoder/runtime.
   - This is not merely a stateless detokenizer artifact in another file format.

4. The GGUF runtime likely already solves part of the streaming problem by hiding decoder state internally.
   - especially via native streaming ISTFT and codebook-aware decoder logic

5. That makes the architecture split even clearer:
   - ONNX path: explicit but currently stateless decoder artifact
   - GGUF path: opaque but likely stateful runtime

## Recommended Follow-up

The next useful investigation should be one of:

1. inspect the actual `rustymimi` binding source to confirm its decode API surface
2. run the bundled GGUF CLI/server on a controlled prompt and compare time-to-first-audio and chunk cadence against `lfm2-audio-rs`
3. inspect whether `libliquid-audio.so` exposes a reusable library API that could be wrapped instead of rebuilding the decoder stack from scratch
