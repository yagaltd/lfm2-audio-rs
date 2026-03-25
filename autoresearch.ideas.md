# Autoresearch Ideas for Future Optimization

## Completed Optimizations (2025-03-25)

Best achieved: **max_frame_gap 324-346ms, RTF ~0.92-1.0**

| Optimization | Impact |
|-------------|--------|
| Removed Moshi/Candle dependency | Eliminated redundant audio decode pipeline |
| Track last_emitted_samples | Avoid re-decoding context frames |
| batch_frames=1 | Decode each frame immediately |
| context_frames=0 | Eliminate window growth |
| intra_threads=8 | Better parallel ONNX execution |

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
- **Max frame gap: 324-346ms** (78% improvement from 1500ms baseline)
- **RTF: 0.92-1.0** (at or near real-time)

The target of <30ms max gap cannot be achieved without architectural changes.

## Why <30ms is Not Achievable

The math:
- LLM generates frames at ~50ms intervals
- ONNX decode takes ~72ms per frame
- Each frame accumulates 22ms delay
- After 15 frames: 15 × 22 = 330ms (matches observed max gap)

To achieve <30ms would require either:
1. **GPU decode** (<10ms per frame)
2. **Async decode pipeline** (hide latency behind generation)
3. **Smaller/faster model** (not available)

## Async Decode Requirements

Analysis showed async decode would require:
1. Change `RefCell<Session>` to `Mutex<Session>` in sessions.rs
2. Create decode thread pool
3. Use channels for async communication
4. Maintain decode ordering

This is significant but achievable refactoring (no new dependencies).