# Mimi Ecosystem Traceability Note

Date: 2026-03-24

## Purpose

This note captures the follow-up investigation around Mimi itself:

1. what Mimi is in the Liquid/Kyutai stack
2. whether Rust bindings exist
3. which local Rust projects in this workspace use Mimi
4. how `pocket-tts-onnx` uses Mimi
5. whether `T-Mimi` currently has a public code or model release

This is intended to complement the earlier decoder gap note:

- [2026-03-24-streaming-gap-analysis.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-streaming-gap-analysis.md)
- [2026-03-24-decoder-architecture-track.md](/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/docs/2026-03-24-decoder-architecture-track.md)

## What Mimi Is

Mimi is not just a decoder. It is a full neural audio codec.

In the Kyutai/Liquid ecosystem it includes:

- an audio encoder
- a quantizer
- a decoder
- explicit streaming state

Evidence from the local `liquid-audio` code:

- `MimiModel` is defined as a full compression model in [compression.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/models/compression.py:105)
- streaming state is defined in [streaming.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/moshi/modules/streaming.py:54)
- the official interleaved realtime demo uses Mimi in streaming decode mode in [chat.py](/home/aurel/Documents/vibe/STT-rust/liquid-audio/src/liquid_audio/demo/chat.py:21)

Important practical distinction:

- for realtime audio output, the path we care about is Mimi decode
- but Mimi itself is a broader codec model, not only the decoder half

## Rust Bindings / Rust Mimi Availability

There is public evidence that Rust-based Mimi exists.

Kyutai’s own `moshi` repo states:

- `pip install rustymimi`
- `rustymimi` is a Rust implementation of Mimi with Python bindings

Evidence:

- https://github.com/kyutai-labs/moshi
- https://docs.rs/crate/moshi-db/latest
- https://pypi.org/project/rustymimi/

The same repo also states that the Rust backend provides both Moshi and Mimi, and that `rustymimi` can be built from the repo with `maturin`.

This means:

- yes, Rust Mimi exists
- yes, Python bindings exist
- but this is part of the Kyutai Moshi ecosystem, not part of `lfm2-audio-rs`

## Local Workspace Survey

### `parakeet-rs`

`parakeet-rs` does not use Mimi.

It is an ONNX Runtime ASR and diarization library centered on NVIDIA Parakeet and related streaming ASR/diarization models.

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/parakeet-rs/README.md:5)
- [Cargo.toml](/home/aurel/Documents/vibe/STT-rust/parakeet-rs/Cargo.toml:6)
- [Cargo.toml](/home/aurel/Documents/vibe/STT-rust/parakeet-rs/Cargo.toml:37)

### `transcribe-rs`

`transcribe-rs` does not use Mimi.

It is a multi-engine STT library. Its ONNX engines are:

- Parakeet
- Canary
- Moonshine
- SenseVoice
- GigaAM

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/transcribe-rs/README.md:3)
- [README.md](/home/aurel/Documents/vibe/STT-rust/transcribe-rs/README.md:27)
- [Cargo.toml](/home/aurel/Documents/vibe/STT-rust/transcribe-rs/Cargo.toml:15)

### `kitten_tts_rs`

`kitten_tts_rs` does not use Mimi.

It is an ONNX Runtime Rust port of KittenTTS.

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/kitten_tts_rs/README.md:3)
- [Cargo.toml](/home/aurel/Documents/vibe/STT-rust/kitten_tts_rs/Cargo.toml:5)
- [Cargo.toml](/home/aurel/Documents/vibe/STT-rust/kitten_tts_rs/Cargo.toml:25)

### `pocket-tts-onnx`

`pocket-tts-onnx` does use Mimi.

But it uses Mimi through ONNX-exported model files, not through `rustymimi`.

Its exported ONNX files include:

- `mimi_encoder.onnx`
- `mimi_decoder.onnx`

Evidence:

- [README.md](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/README.md:128)

Its runtime wrapper builds ONNX Runtime sessions for:

- `mimi_encoder`
- `mimi_decoder`
- flow model pieces
- text conditioner

Evidence:

- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:149)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:166)

More importantly, it manages Mimi decoder state explicitly across chunked inference:

- initialize state for the decoder session
- call `mimi_decoder.run(...)`
- update returned state tensors
- continue decoding further chunks

Evidence:

- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:183)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:404)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:426)
- [pocket_tts_onnx.py](/home/aurel/Documents/vibe/STT-rust/pocket-tts-onnx/pocket_tts_onnx.py:526)

That matters for our LFM2 investigation because it proves an important point:

- a Mimi-style stateful decoder can be exported to ONNX
- caller-managed state tensors are a workable ORT pattern

This is different from the current `LFM2.5-Audio-1.5B-ONNX` export, which only ships a stateless `audio_detokenizer` path.

## `babybirdprd/pocket-tts`

The Kyutai Pocket TTS model card lists a separate community project:

- `pocket-tts` by `@babybirdprd`
- described there as a Candle Rust implementation with WebAssembly and PyO3 bindings

Evidence:

- https://huggingface.co/kyutai/pocket-tts

From the sources inspected in this workspace, I do not have evidence that this project uses `rustymimi` specifically.

So the safe statement is:

- `babybirdprd/pocket-tts` exists as a Rust/Candle Pocket TTS implementation
- I have not verified that it uses `rustymimi`
- the verified Mimi-on-ORT example in this workspace is `pocket-tts-onnx`, not `babybirdprd/pocket-tts`

## `T-Mimi`

I found public evidence for the `T-Mimi` research work itself:

- paper: `T-Mimi: A Transformer-based Mimi Decoder for Real-Time On-Phone TTS`
- publication date visible on arXiv / indexing pages

Sources:

- https://arxiv.org/abs/2601.20094
- https://www.catalyzex.com/author/Julian%20Chan

What I did not find as of 2026-03-24:

- official GitHub repository for `T-Mimi`
- official Hugging Face model release for `T-Mimi`
- official ONNX export or inference package for `T-Mimi`

So the current traceable status is:

- paper found
- public code/model release not found

## Practical Conclusions

1. Mimi is a codec model, not merely a decoder.
   - For our realtime output problem, the decode side is the relevant half.

2. Rust Mimi exists.
   - `rustymimi` is real and publicly referenced by Kyutai.

3. None of the local ASR-focused Rust projects use Mimi.
   - `parakeet-rs` and `transcribe-rs` are unrelated to this decoder problem.

4. `kitten_tts_rs` is also unrelated to Mimi.
   - It is ONNX TTS, but not Mimi-based.

5. `pocket-tts-onnx` is strong evidence that stateful Mimi-style decoding can exist in ONNX Runtime.
   - It exports `mimi_decoder.onnx`
   - it keeps decoder state explicitly across chunks

6. That makes `pocket-tts-onnx` an important reference point for the next decoder-track question:
   - not “can ORT do stateful Mimi-style decoding at all?”
   - but “can LFM2’s audio output path be reshaped into a similarly explicit stateful decoder contract?”

7. `T-Mimi` is not yet a usable implementation reference from the evidence currently available.
   - paper yes
   - public code/model no

## Recommended Follow-up

The next useful architecture note should compare:

1. `pocket-tts-onnx` Mimi decoder state contract
2. `rustymimi` / Kyutai Moshi Rust backend contract
3. current `LFM2.5-Audio-1.5B-ONNX` `audio_detokenizer` contract

That comparison would tell us whether the next realistic path is:

- wrapping `rustymimi`
- designing a new Mimi-like ONNX export for LFM2 audio output
- or accepting that LFM2’s current detokenizer family is fundamentally different
