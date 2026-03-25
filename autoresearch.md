# Autoresearch: Smooth Realtime TTS Streaming

## Objective
Optimize the interleaved TTS streaming pipeline to eliminate choppy audio during continuous speech generation. The server generates audio frames while the client plays them - we need smooth, gap-free playback even during long responses.

## Root Cause Analysis
The choppiness likely stems from one or more of:
1. **Frame generation latency** - decoder takes too long to generate next audio frame
2. **Audio decoding latency** - converting codes to waveform is too slow
3. **Batching inefficiency** - `stream_decode_batch_frames=4` may not be optimal
4. **WebSocket overhead** - network/serialization delays
5. **Client buffer underruns** - playback consumes audio faster than we can decode it

## Metrics
- **Primary**: `max_frame_gap_ms` (lower is better) — the worst-case gap between consecutive audio frames. Gaps >50ms cause audible choppiness.
- **Secondary**:
  - `avg_frame_gap_ms` — typical inter-frame latency
  - `decode_ms_per_frame` — audio detokenization cost
  - `rtf` — Real-Time Factor (processing_time / audio_duration)
  - `total_ms` — end-to-end response time

## Target
- `max_frame_gap_ms` < 30ms (imperceptible gaps)
- `avg_frame_gap_ms` < 15ms
- `rtf` < 1.0 for TTS generation

## How to Run
```bash
./autoresearch.sh
```

Outputs structured `METRIC` lines to stdout. The script:
1. Starts the server with timing instrumentation
2. Sends two queries via WebSocket: "hello" (short) and "tell me a joke" (long)
3. Parses server logs to extract frame timing metrics
4. Reports median values across both queries

## Files in Scope
All source files may be modified:
- `src/bin/server.rs` — WebSocket server, streaming pipeline, audio decoder orchestration
- `src/interleaved.rs` — Interleaved text+audio generation loop
- `src/stream_decode.rs` — Mimi streaming decoder wrapper
- `src/tts.rs` — TTS pipeline, audio code decoding
- `src/cache.rs` — KV-cache management
- `src/audio/mel.rs` — Mel spectrogram (relevant for ASR side)
- `src/audio/istft.rs` — Inverse STFT for audio detokenizer
- `src/model.rs` — Model loading and session management
- `src/sessions.rs` — ONNX session handling

## Off Limits
- `Cargo.toml` — No new dependencies allowed (must use existing ORT crate and ONNX models)
- `tests/` — Must remain passing (correctness validation)

## Constraints
1. All existing tests must pass (`cargo test`)
2. No new external dependencies
3. Must maintain API compatibility with existing client
4. Must work with existing ONNX model files (Q4 quantization)
5. Server must remain stable under load

## Baseline Metrics
(To be filled after first run)

## What's Been Tried
(Update as experiments accumulate)

### Current Architecture
- Streaming decoder batches 4 frames before decoding (`stream_decode_batch_frames=4`)
- Uses sliding window of 16 context frames for quality (`stream_decode_context_frames=16`)
- Each frame is ~80ms of audio (1920 samples at 24kHz)
- Decoding re-processes context frames every batch (potential optimization target)

### Potential Optimizations
1. **Reduce batch size** - smaller batches = lower latency but more overhead
2. **Parallel decode** - decode next batch while sending current
3. **Context caching** - avoid re-decoding the same context frames
4. **WebSocket batching** - send multiple small chunks together
5. **Pre-buffering** - generate ahead during text-only phases
6. **ONNX session optimization** - graph optimizations, memory patterns
7. **Frame interpolation** - smooth over small gaps client-side
