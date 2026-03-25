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

## Future Ideas

### High Impact (Requires Architectural Changes)

1. **Async Decode Pipeline**
   - Run ONNX inference in separate thread
   - Decode next batch while sending current audio
   - Would require tokio channel integration
   - Expected improvement: 50-70% reduction in max gap
   - Effort: High (weeks)

2. **GPU Acceleration**
   - Use CUDA/CoreML/DirectML execution providers
   - ONNX Runtime supports GPU inference
   - Would require CUDA-enabled hardware
   - Expected improvement: 5-10x faster decode
   - Effort: Medium (days)

3. **Pre-buffering During Text Generation**
   - Start decoding first audio frames during text-only phase
   - Fill client buffer before audio streaming starts
   - Would help initial latency, not steady-state gaps
   - Effort: Medium

### Medium Impact

4. **ONNX Session Pooling**
   - Pre-warm multiple ONNX sessions
   - Could allow parallel frame decoding
   - Limited by thread safety of sessions

5. **Thread-affinity for ONNX Threads**
   - Pin ONNX intra threads to specific cores
   - Reduce context switch overhead
   - Linux-specific optimization

6. **Memory Pre-allocation**
   - Pre-allocate output buffers in hot path
   - Reduce allocation overhead in decode loop

### Low Impact / Exploration

7. **Try Q8 Precision**
   - Q8 model might have slightly better accuracy/performance tradeoff
   - File: audio_detokenizer_q8.onnx (76MB vs 56MB for Q4)

8. **ONNX Graph Optimization**
   - Explore ONNX Runtime graph optimization flags
   - Could reduce inference overhead

9. **Client-side Frame Interpolation**
   - Smooth over gaps in audio playback
   - Would require client modifications
   - Not a server-side fix

## Constraints

- No new dependencies allowed
- Must maintain API compatibility
- Tests must pass
- Must work with existing ONNX models

## Hardware Requirements for <30ms Target

Current: 72ms decode per frame on CPU
Target: <30ms max gap

Options:
1. **CUDA GPU** - Would likely achieve <10ms decode
2. **Apple Silicon (CoreML)** - Could achieve ~20-30ms decode
3. **Async pipeline** - Would hide decode latency behind transmission
4. **Smaller model** - Would need model retraining/export

## Next Steps

1. Profile ONNX inference to identify bottlenecks
2. Add CUDA support to Cargo.toml (feature flag)
3. Implement async decode pipeline for streaming
4. Consider WebSocket ping/pong for latency measurement