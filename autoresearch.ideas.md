# Autoresearch Ideas for Future Optimization

## Completed Optimizations (2025-03-25)

Best achieved: **max_frame_gap 324-346ms, RTF ~1.0**

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

## Optimization Limit Reached

Within the constraints (no new dependencies, maintain API compatibility, CPU-only), the best achievable is:
- **Max frame gap: 324-346ms** (78% improvement from 1500ms baseline)
- **RTF: 0.97-1.02** (at or near real-time)

The target of <30ms max gap cannot be achieved without:
1. **Async decode pipeline** (architectural change)
2. **GPU acceleration** (hardware requirement)

## Future Ideas (Require Constraints Change)

### High Impact (Architectural Changes)

1. **Async Decode Pipeline**
   - Run ONNX inference in blocking thread pool
   - Decode next batch while sending current audio
   - Would require tokio::task::spawn_blocking
   - Expected improvement: 50-70% reduction in max gap
   - Effort: High (significant refactoring)
   - **No new dependencies required** - tokio already used

2. **GPU Acceleration**
   - Use CUDA/CoreML/DirectML execution providers
   - ONNX Runtime supports GPU inference
   - Would require CUDA-enabled hardware
   - Expected improvement: 5-10x faster decode
   - Effort: Medium (add feature flag, hardware dependency)

### Medium Impact (Requires Dependencies)

3. **Thread-affinity for ONNX Threads**
   - Pin ONNX intra threads to specific cores
   - Reduce context switch overhead
   - Would require `core_affinity` crate
   - **Blocked by: No new dependencies constraint**

### Low Impact (Already Optimized)

4. **ONNX Graph Optimization**
   - Already using GraphOptimizationLevel::Level3
   - No further improvement possible

5. **Memory Pre-allocation**
   - Minimal impact compared to ONNX inference time

## Constraints

- No new dependencies allowed
- Must maintain API compatibility
- Tests must pass
- Must work with existing ONNX models (Q4 quantization is optimal)

## Hardware Requirements for <30ms Target

Current: 72ms decode per frame on CPU
Target: <30ms max gap

Options:
1. **CUDA GPU** - Would likely achieve <10ms decode
2. **Apple Silicon (CoreML)** - Could achieve ~20-30ms decode
3. **Async pipeline** - Would hide decode latency behind transmission

## Recommendation

The most practical path forward is implementing an **async decode pipeline**:
- No new dependencies (uses existing tokio)
- Could reduce max gap by 50-70%
- Would allow audio streaming while next batch decodes
- Implementation: Use `tokio::task::spawn_blocking` for ONNX inference