# LFM2 vs Pocket TTS Mimi Contract Comparison

Date: 2026-03-24

## Purpose

This note answers one precise question:

- can the Pocket TTS ONNX Mimi decoder be reused directly for `lfm2-audio-rs`?

The answer depends on the actual tensor contracts, not on the shared use of the word “Mimi”.

## Short Answer

No, not directly.

Pocket TTS’s ONNX Mimi decoder is a stateful graph that consumes latent frames:

- `latent [1, seq_len, 32]`

and returns:

- `audio_frame`
- `56` updated state tensors

LFM2’s current ONNX audio output path uses a different contract:

- `audio_codes [batch, 8, time]`
- stateless `audio_detokenizer`
- output `stft_features`

So Pocket TTS’s exported Mimi decoder is a useful reference architecture, but not a drop-in decoder for LFM2.

## LFM2 Audio Output Contract

### Token-level model output

In the official Liquid/LFM2 interleaved path:

- audio output is emitted as 8-token frames
- each token frame corresponds to one audio frame
- end-of-audio uses token `2048`

Evidence:

- [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:31)
- [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:34)
- [processor.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/processor.py:170)
- [infer.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/infer.py:290)
- [infer.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/infer.py:292)
- [infer.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/infer.py:426)

The Rust implementation matches this shape:

- `Vec<[u16; 8]>`

Evidence:

- [tts.rs](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/src/tts.rs:41)
- [interleaved.rs](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/src/interleaved.rs:27)

### Current ONNX decoder-side contract

The exported LFM2 ONNX detokenizer takes:

- `audio_codes INT64 [batch_size, 8, time]`

and returns:

- `stft_features FLOAT [batch_size, time, 1282]`

This was verified both from the builder code and from the actual local ONNX model metadata.

Evidence:

- [detokenizer_builder.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py:68)
- [detokenizer_builder.py](/home/aurel/Documents/vibe/STT-rust/onnx-export/src/liquidonnx/lfm2_audio/builder/detokenizer_builder.py:75)
- local model file: [/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX/onnx/audio_detokenizer.onnx](/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX/onnx/audio_detokenizer.onnx)

Important implication:

- LFM2’s shipped ONNX decoder-side artifact is token-driven and stateless
- it does not expose explicit recurrent decoder state

## Pocket TTS Mimi Contract

### Upstream generation representation

Pocket TTS does not produce Mimi code tokens in the way LFM2 does.

Its generation loop yields latent frames from the flow model:

- each step produces `latent = x.reshape(1, 1, 32)`

Evidence:

- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:297)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:386)

So the decoder boundary in Pocket TTS ONNX is:

- flow latent frames
- not discrete 8-codebook token frames

### Mimi decoder ONNX signature

The actual local `mimi_decoder.onnx` takes:

- `latent FLOAT [1, seq_len, 32]`
- `state_0 ... state_55`

and returns:

- `audio_frame FLOAT [1, 1, T]`
- `out_state_0 ... out_state_55`

This was verified from the local ONNX graph and ORT session metadata for:

- [mimi_decoder.onnx](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/onnx/mimi_decoder.onnx)

The wrapper code uses that stateful contract directly:

- initialize decoder state
- call `self.mimi_decoder.run(None, {"latent": chunk, **state})`
- replace state with returned `out_state_*`

Evidence:

- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:183)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:390)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:411)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:526)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:564)

### Pocket TTS encoder-side note

Pocket TTS also ships:

- `mimi_encoder.onnx`

Its verified signature is:

- input:
  - `audio FLOAT [1, 1, audio_len]`
- output:
  - `latents FLOAT [1, T, 1024]`

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/README.md:128)
- local model file: [/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/onnx/mimi_encoder.onnx](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/onnx/mimi_encoder.onnx)

That confirms Pocket TTS’s internal representation differs from LFM2’s token output path at multiple boundaries.

## Direct Contract Comparison

| Dimension | LFM2 current ONNX path | Pocket TTS ONNX Mimi path |
| --- | --- | --- |
| Upstream generated unit | 8 discrete audio code tokens per frame | 32-dim latent frame |
| Token/cardinality info | 8 codebooks, 2049 vocab including EOA | no codebook token interface at decoder boundary |
| Decoder input | `audio_codes [B, 8, T]` | `latent [1, seq, 32]` |
| Decoder output | `stft_features [B, T, 1282]` | `audio_frame [1, 1, T]` |
| Decoder state | none in graph contract | 56 explicit state tensors |
| Reconstruction step | external ISTFT after detokenizer | waveform chunk returned directly |
| Streaming shape | simulated by repeated re-decode | native graph-level state progression |

## What This Proves

### What cannot be reused directly

We cannot simply plug Pocket TTS’s `mimi_decoder.onnx` into `lfm2-audio-rs` because:

1. LFM2 produces discrete code tokens, not 32-dim Pocket TTS latents
2. Pocket TTS’s Mimi decoder expects a state bundle that LFM2 does not currently produce
3. LFM2’s shipped ONNX decoder path reconstructs via STFT features, not direct waveform chunks

So the model artifact is not interchangeable.

### What can be reused conceptually

Pocket TTS ONNX is still highly relevant as a reference because it proves:

1. a stateful decoder can be exported to ONNX
2. ORT can drive that decoder incrementally with explicit caller-managed state
3. streaming audio can be produced without repeated full-prefix detokenization

That matters for LFM2 because it moves one question from “unknown” to “yes, in principle”:

- yes, ORT can support a Mimi-like streaming decoder architecture

But it does not prove:

- that the current LFM2 audio outputs can feed the Pocket TTS decoder
- or that the current LFM2 export already contains the right stateful decoder artifact

## Realistic Reuse Paths

### Path 1: direct artifact reuse

Not realistic.

Reason:

- contracts do not match

### Path 2: reuse the export pattern

Realistic.

This means:

- study Pocket TTS’s stateful ONNX Mimi export pattern
- design an LFM2-compatible stateful decoder contract
- export a new graph with explicit state inputs/outputs

This is the strongest “yes” supported by the current evidence.

### Path 3: wrap an external Mimi runtime

Also realistic.

This means:

- use `rustymimi` / Kyutai Rust Mimi, or a Python sidecar
- convert LFM2’s emitted 8-code frames into the runtime expected by that decoder

This path still requires one more answer:

- does Kyutai/Liquid Mimi decode accept exactly the same 8-code token semantics emitted by LFM2 in a reusable way outside the Python demo path?

The demo strongly suggests yes, but the exact reusable boundary still needs a contract-level inspection.

## Conclusion

Pocket TTS ONNX is a proof-of-architecture reference, not a drop-in component for `lfm2-audio-rs`.

The critical mismatch is:

- Pocket TTS ONNX Mimi decoder:
  - stateful
  - latent-driven
  - waveform out
- LFM2 current ONNX path:
  - stateless
  - token-driven
  - STFT out

So the next useful question is not:

- “can we reuse Pocket TTS mini directly?”

It is:

- “can we create an LFM2-compatible stateful decoder contract, using Pocket TTS ONNX as a reference for how to expose decoder state to ORT?”

## Recommended Follow-up

The next investigation should compare:

1. `rustymimi` / Kyutai Rust Mimi decode input contract
2. the exact code-to-wave contract used by Liquid’s `mimi.decode(token[None, :, None])`
3. whether LFM2’s emitted audio frame format is directly valid for that runtime outside the current Python demo

That would answer whether the best next step is:

- wrap `rustymimi`
- export a new stateful ONNX Mimi-style decoder for LFM2
- or both
