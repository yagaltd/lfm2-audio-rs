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
- Initial (before optimization): `max_frame_gap_ms` ~1500ms
- After Moshi removal + context tracking: ~1500ms (same, but eliminated redundant context decode)
- After batch_frames=1: ~1200ms
- After context_frames=0: ~519ms
- **Current (batch_frames=1, context_frames=0)**: `max_frame_gap_ms` **305ms**

## What's Been Tried

### Completed Optimizations
1. **Removed Moshi/Candle dependency** - Eliminated redundant audio decode pipeline
2. **Track last_emitted_samples** - Avoid re-decoding context frames to find slice position
3. **Set batch_frames=1** - Decode each frame immediately, reducing batch latency
4. **Set context_frames=0** - Eliminate window growth, decode only pending frames
5. **Fixed tests** - Removed Mimi-specific tests after dependency removal

### Results Summary
| Config | Max Gap | RTF | Notes |
|--------|----------|-----|-------|
| Baseline (batch=4, ctx=16) | ~1500ms | ? | Window grows to 17 frames |
| batch=1 | ~1200ms | ? | Marginally better |
| ctx=0 | ~519ms | ? | Major improvement |
| **batch=1, ctx=0** | **305ms** | **0.986** | **80% improvement** |

### Remaining Bottleneck
- Decode latency ~69ms per frame, but frames arrive every ~50ms
- Max gap 305ms still above 30ms target
- Need parallel decode/generate pipeline or smaller frame sizes

### Potential Future Optimizations
1. **Parallel decode** - decode next batch while sending current (requires async pipeline)
2. **Pre-buffering** - start audio decode during text generation phase
3. **ONNX thread optimization** - tune intra_op_num_threads for parallel execution
4. **Frame interpolation** - smooth over small gaps client-side
5. **Smaller frame sizes** - if model supports finer granularity
