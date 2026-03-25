# Decoder Architecture Track: ONNX Detokenizer vs Mimi

Date: 2026-03-24

## Purpose

This document records the decoder-side architecture gap in one place, with enough traceability to answer three practical questions:

1. Which model artifacts belong to the current ONNX path versus the Mimi path?
2. What runtime stack does each path require?
3. Can the Mimi path be exported or wrapped for Rust, and what would that actually mean?

This is not an implementation plan. It is an evidence-backed architecture note.

## Summary

The current Rust server and the official ONNX export use the same decoder family:

- `audio_detokenizer`
- stateless invocation
- audio codes in, STFT or waveform out

The official realtime `liquid-audio` demo uses a different decoder family:

- `mimi`
- stateful streaming invocation
- one audio frame in, one waveform chunk out

The most important conclusion is:

- the gap is not just “Rust forgot to cache more tensors”
- the gap is that the shipped ONNX artifact is a stateless detokenizer, while the official realtime path is a separate stateful Mimi decoder

## Artifact Ownership

### Base model repo: `LiquidAI/LFM2.5-Audio-1.5B`

The Python `liquid-audio` processor expects the base model repo to contain both kinds of decoder artifacts:

- Mimi weights:
  - `tokenizer-e351c8d8-checkpoint125.safetensors`
- LFM detokenizer weights:
  - `audio_detokenizer/config.json`
  - `audio_detokenizer/model.safetensors`

Evidence:

- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:67)
- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:70)
- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:107)
- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:128)

The processor explicitly says one path may exist without the other:

- if Mimi weights are missing, use `decode(...)` instead
- if LFM detokenizer weights are missing, use `mimi` instead

Evidence:

- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:107)
- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:128)

### ONNX repo: `LiquidAI/LFM2.5-Audio-1.5B-ONNX`

The ONNX repo ships:

- `decoder*.onnx`
- `audio_encoder*.onnx`
- `audio_embedding*.onnx`
- `audio_detokenizer*.onnx`
- `vocoder_depthformer*.onnx`

It does not ship Mimi weights or a Mimi ONNX graph.

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX/README.md:55)
- local artifact listing from [/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX](/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX)

### Export repo: `onnx-export`

`onnx-export` exports the LFM detokenizer, not Mimi.

Evidence:

- [export.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/export.py:12)
- [export.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/export.py:617)
- [detokenizer_builder.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py:647)

There is no Mimi builder in `onnx-export`.

Evidence:

- repo search over `/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx`

### Important ownership conclusion

There are two decoder artifacts in the broader Liquid ecosystem:

1. `audio_detokenizer`
   - owned by the LFM2 audio model package and exported to ONNX
2. `mimi`
   - owned by the base model package and loaded by the Python processor
   - not exported in the current ONNX repo

## Runtime Dependencies

### Current ONNX detokenizer path

The ONNX path is designed to run without PyTorch in inference.

Evidence:

- [infer.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/infer.py:1495)

The practical stack is:

- ONNX Runtime session
- NumPy
- custom ISTFT reconstruction

This is the path used by:

- `lfm2-audio-rs`
- `transformers-js`
- `onnx-export` inference tools

### Mimi path

The `liquid-audio` package depends on a full PyTorch stack:

- `torch`
- `torchaudio`
- `torchcodec`
- `transformers`
- `accelerate`
- `einops`
- `librosa`
- `sentencepiece`

Evidence:

- [pyproject.toml](/home/aurel/Documents/vibe/STT-rust/liquid-audio/pyproject.toml:1)

The realtime demo also uses:

- `gradio`
- `fastrtc`

for the interactive browser demo.

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/liquid-audio/README.md:25)
- [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:1)

### Mimi runtime internals

Mimi is not a tiny helper. It is a full streaming compression model stack:

- encoder
- decoder
- quantizer
- optional transformer blocks
- streaming state machinery

Evidence:

- [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:105)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:54)

The streaming state is explicit and reusable:

- `streaming(batch_size)`
- `reset_streaming(...)`
- `get_streaming_state()`
- `set_streaming_state(...)`

Evidence:

- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:131)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:139)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:158)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:168)

That statefulness is the main runtime difference versus the ONNX detokenizer path.

## Architecture Comparison

### Current ONNX detokenizer path

The exported ONNX detokenizer contract is:

- input:
  - `audio_codes [B, 8, T]`
- output:
  - `stft_features`

Evidence:

- [detokenizer_builder.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py:68)
- [detokenizer_builder.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py:75)

There are no cache inputs and no cache outputs in the exported graph.

The PyTorch LFM detokenizer behind it:

- fuses 8 codebooks into embeddings
- upsamples by `6x`
- applies causal/sliding attention
- calls `Lfm2Model(..., use_cache=False)`
- projects to STFT features or waveform

Evidence:

- [detokenizer.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/detokenizer.py:8)
- [detokenizer.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/detokenizer.py:106)
- [detokenizer.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/detokenizer.py:117)
- [detokenizer.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/detokenizer.py:130)

Important consequence:

- even the reference PyTorch detokenizer is non-cached
- the ONNX export is not “missing an obvious cache that already exists internally”

### Mimi path

The official realtime demo uses:

- `mimi.streaming(1)`
- `mimi.decode(token[None, :, None])`

for each audio token frame emitted by `generate_interleaved(...)`.

Evidence:

- [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:21)
- [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:34)

Mimi’s decode path:

- decodes discrete codes to latent space
- upsamples back to encoder frame rate
- runs decoder transformer if present
- runs decoder to waveform

Evidence:

- [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:406)
- [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:432)

Unlike the ONNX detokenizer path, Mimi decode is designed to run inside a persistent streaming context.

### Direct comparison

| Dimension | ONNX detokenizer path | Mimi path |
| --- | --- | --- |
| Primary artifact | `audio_detokenizer.onnx` | `tokenizer-e351c8d8-checkpoint125.safetensors` |
| Repository ownership | ONNX export repo / base repo weights | base repo weights + `liquid_audio.moshi` code |
| Runtime | ORT + ISTFT | PyTorch + streaming state |
| Input granularity | full code sequence or re-decoded window | per-frame incremental decode |
| State contract | stateless graph | explicit streaming state |
| Output form | STFT features then waveform | waveform chunk directly |
| Current Rust parity | yes | no |
| Current ONNX export parity | yes | no |

## What `onnx-export` Itself Says

The export repo already treats these as two different decoder backends.

Its internal comparison script supports:

- `--decoder mimi`
- `--decoder detokenizer`

and documents them as:

- `mimi`: official demo style
- `detokenizer`: ONNX-compatible

Evidence:

- [interleaved_liquidaudio.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/scripts/interleaved_liquidaudio.py:1)
- [interleaved_liquidaudio.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/scripts/interleaved_liquidaudio.py:33)
- [interleaved_liquidaudio.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/scripts/interleaved_liquidaudio.py:132)
- [interleaved_liquidaudio.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/scripts/interleaved_liquidaudio.py:253)

That is strong evidence that the repo already understands these as distinct decoding strategies, not interchangeable wrappers around the same artifact.

## Can Mimi Be Wrapped For Rust?

### Option 1: Python sidecar

Yes. This is the lowest-risk wrapping strategy.

Shape:

- Rust keeps the app server, VAD, session control, and possibly the decoder/depthformer path
- a local Python service owns:
  - `liquid-audio`
  - `proc.mimi`
  - streaming decode
- Rust sends audio code frames to Python and receives PCM chunks back

Why it is feasible:

- the Python demo already does exactly the needed streaming pattern
- the decoder boundary is narrow:
  - input: one 8-code frame
  - output: one waveform chunk

Costs:

- extra process
- Python runtime deployment
- serialization boundary

Risk:

- low technical risk
- medium operational complexity

### Option 2: Embed Python in Rust

Also feasible in principle, for example through a Python embedding layer.

Why it is less attractive:

- more complex packaging than a sidecar
- GIL and Python runtime lifecycle inside the Rust server
- harder crash isolation

Risk:

- medium technical risk
- high deployment complexity

### Option 3: Call a native library from Rust

Not currently available from the inspected codebase.

What is missing:

- no standalone C or C++ Mimi library in this repo
- no Rust crate wrapping Mimi
- no GGUF-style native runtime for Mimi in this tree

Conclusion:

- not available without new upstream work

## Can Mimi Be Exported To ONNX?

### Short answer

Probably yes in principle, but not with the current export repo, and not as a trivial patch.

### Why “yes in principle”

The Mimi path is implemented as ordinary PyTorch modules:

- `MimiModel`
- quantizer
- decoder transformer
- decoder
- streaming state in Python objects

There is no evidence of a hard dependency on custom CUDA kernels for correctness.

The `CUDAGraphed` pieces are performance wrappers, not the defining model contract.

Evidence:

- [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:219)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:54)

So a new export path could, in theory, export a decode-only graph.

### Why “not trivial”

The current streaming state lives in Python objects, not in an existing exported tensor contract.

To make Mimi usable from ORT, a new export would need an explicit graph contract such as:

- inputs:
  - `codes_step`
  - decoder state tensors
  - transformer state tensors
  - convolution caches
- outputs:
  - `pcm_chunk`
  - updated state tensors

That contract does not exist today.

There is also no Mimi export builder in `onnx-export`, and no local evidence of prior ONNX export support in the `moshi` code.

Evidence:

- repo search over `/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi`
- [interleaved_liquidaudio.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/scripts/interleaved_liquidaudio.py:33)

### Practical export assessment

The likely export candidates are:

1. decode-only Mimi step graph
   - most relevant for streaming Rust
   - still requires new explicit state tensors
2. full Mimi batch decoder graph
   - easier than stateful step export
   - less useful for realtime streaming
3. full encode+decode export
   - largest scope
   - not necessary for the current streaming bottleneck

The best ONNX-compatible target would be:

- a decode-only step graph with explicit caller-managed state

## Can Mimi Be Ported To Rust?

Yes in principle, but this is the most expensive option.

Why:

- Mimi is a full model stack, not a small helper
- it includes:
  - streaming module semantics
  - quantizer decode
  - convolutional decoder
  - optional transformer layers
  - state reset and masking behavior

Evidence:

- [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:105)
- [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:54)

What would have to be ported:

- model definition
- weight loading
- quantizer decode logic
- streaming state semantics
- exact waveform equivalence tests

This is technically possible but much larger than:

- wrapping Python
- or exporting a dedicated Mimi step graph

## Traceable Conclusions

1. The current ONNX path is not a “Mimi path with some missing Rust cache.”
   - It is a different decoder artifact and a different runtime contract.

2. The official realtime demo’s smoothness comes from `mimi.streaming(1)`, not from WebSocket behavior or browser buffering.
   - Evidence is in [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:21).

3. The current ONNX export does not contain the artifact needed for true Mimi-style streaming parity.
   - It exports `audio_detokenizer`, not Mimi.

4. If Rust must remain the main server, the realistic decoder-side options are:
   - wrap Python Mimi as a sidecar
   - create a new Mimi ONNX export with explicit state tensors
   - port Mimi to Rust

5. Of those three, the best near-term path for correctness is wrapping Python Mimi.
   - The best long-term ORT-native path is a new Mimi export, not more client buffering.

## Recommended Next Analysis

Before implementation, the next decoder-track document should answer:

1. What exact Mimi state tensors would need to become explicit ONNX inputs/outputs?
2. Can a decode-only Mimi step graph reproduce the Python chunk boundaries exactly?
3. What is the smallest Rust/Python bridge needed to validate that architecture before any full rewrite?
