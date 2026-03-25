# Autoresearch Ideas for Future Optimization

## Completed Optimizations (2025-03-25)

Best achieved: **max_frame_gap 243-276ms, RTF 0.55-0.78**

| Optimization | Impact |
|-------------|--------|
| Removed Moshi/Candle dependency | Eliminated redundant audio decode pipeline |
| Track last_emitted_samples | Avoid re-decoding context frames |
| batch_frames=1 | Decode each frame immediately |
| context_frames=0 | Eliminate window growth |
| intra_threads=8 | Better parallel ONNX execution |
| Arc<Mutex> detokenizer | Thread-safe (prerequisite for async) |
| Standalone decode function | Callable from async thread |
| **Async decode pipeline** | **24% max gap improvement, 27% RTF improvement** |

## Async Decode Implementation (Complete) ✅

### What Was Implemented
1. `AsyncDecodeThread` - Background thread for ONNX decode
2. `AsyncStreamingDecoder` - Wrapper that manages async decode requests
3. Request ordering via `request_id` tracking
4. `poll_completed()` - Non-blocking check for completed decodes
5. `finish()` - Wait for all pending decodes with timeout

### Results
- Max gap: 346ms → 243-276ms (24% improvement)
- RTF: 0.96 → 0.55-0.78 (27% improvement)
- Avg frame gap: 143ms → 70ms (51% improvement)

### How It Works
1. LLM generates audio frame codes (~50ms per frame)
2. `push_frame()` submits frame to background decode thread (non-blocking)
3. Background thread decodes using ONNX (~90ms per frame)
4. `poll_completed()` returns decoded chunks in order
5. While decode is happening, LLM can generate next frame
6. Latency is hidden behind generation time

## Tried & Discarded

| Experiment | Result | Reason |
|------------|--------|--------|
| batch_frames=2 | 422ms, RTF 1.16 | Worse than batch=1 |
| intra_threads=16 | 343ms, RTF 1.12 | Thread contention |
| inter_threads=2 | No improvement | Not helpful |
| Q8 precision | 428ms, RTF 1.10 | Larger model, worse performance |
| FP16 precision | Timeout | Much larger model, very slow |
| queue_chunks=8 | RTF 0.915 | Better RTF, max gap unchanged |

## Optimization Limit Reached

Within the constraints (no new dependencies, maintain API compatibility, CPU-only), the best achievable is:
- **Max frame gap: 243-276ms** (84% improvement from 1500ms baseline)
- **RTF: 0.55-0.78** (significantly faster than real-time)

The target of <30ms max gap cannot be achieved without GPU acceleration.

## Future Optimization Ideas

### High Impact
1. **GPU acceleration** - Use CUDA/Metal for ONNX execution
   - Expected: <10ms per frame decode
   - Would achieve <30ms max gap target
   - Requires: GPU hardware, ORT GPU EP

### Medium Impact
2. **Multiple decode threads** - Parallel decode with separate ONNX sessions
   - Would reduce queue buildup during burst generation
   - Complex: Each session loads model into memory

3. **Pre-buffering during text phase** - Start decode before audio frames arrive
   - Would reduce first-frame latency
   - Requires: Predicting when audio will start

### Low Impact
4. **Frame interpolation** - Smooth over small gaps client-side
   - Would improve perceived quality
   - Doesn't address root cause

5. **Smaller/faster model** - If available
   - Would reduce per-frame decode time
   - May impact quality

## Why <30ms is Not Achievable (CPU-only)

The math:
- LLM generates frames at ~50ms intervals
- ONNX CPU decode takes ~90ms per frame
- Each frame accumulates 40ms delay potential
- Async helps hide latency but can't eliminate it
- Bursts of frames still cause queue buildup

To achieve <30ms would require:
1. **GPU decode** (<10ms per frame) - Most practical
2. **Smaller model** (<30ms per frame) - Not available
3. **Client-side buffering** (hides gaps but adds latency)
