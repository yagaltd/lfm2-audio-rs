# Autoresearch: Realtime TTS Streaming Without Choppiness

## Objective
Improve realtime interleaved TTS playback quality on the real websocket path. The benchmark must reward genuine user-visible improvements: lower time to first playable audio, no replayed/overlapping streamed audio, and enough steady-state slack that chunk delivery does not fall behind playout.

## Metrics
- **Primary**: `realtime_penalty_ms` (ms, lower is better) — median across the tune prompt suite of:
  - `first_playable_audio_ms`
  - `overlap_ms`
  - `max_chunk_deficit_ms`
  This intentionally makes replay/choppiness dominate until fixed, then lets startup latency and starvation risk drive further work.
- **Secondary**:
  - `first_playable_audio_ms`
  - `overlap_ms`
  - `max_chunk_deficit_ms`
  - `max_binary_gap_ms`
  - `server_first_frame_ms`
  - `server_first_decode_ms`
  - `server_first_queue_wait_ms`
  - `holdout_realtime_penalty_ms`
  - `holdout_first_playable_audio_ms`
  - `holdout_overlap_ms`
  - `holdout_max_chunk_deficit_ms`

## How to Run
`./autoresearch.sh`

The script:
1. incrementally builds the release websocket server,
2. starts it on a local port,
3. drives the real websocket streaming path with a small prompt suite plus a holdout prompt,
4. outputs structured `METRIC` lines.

## Files in Scope
- `src/bin/server.rs` — streaming decode bookkeeping, pacing, websocket metadata, tests
- `static/app.js` — timing logs for manual validation
- `static/assistant-player-worklet.js` — playback-buffer instrumentation
- `scripts/bench_realtime_ws.mjs` — websocket benchmark harness
- `autoresearch.sh`, `autoresearch.checks.sh`, `autoresearch.ideas.md`

## Off Limits
- `Cargo.toml` — no new dependencies
- model files under `/home/aurel/Documents/vibe/STT-rust/LFM2.5-Audio-1.5B-ONNX`
- benchmark prompt semantics must stay fixed once established

## Constraints
- Existing server/unit tests must pass.
- Do not optimize only one canned prompt.
- Do not hide regressions behind larger buffers alone.
- Prefer simpler fixes when gains are comparable.
- If a change improves the primary metric but obviously worsens holdout behavior or correctness, discard it and note why.

## Benchmark Notes
### Tune prompts
1. `Say hello in one short sentence and speak it.`
2. `Count to three quickly and speak it.`
3. `Tell me one short fun fact and speak it.`

### Holdout prompt
- `In two short sentences, explain why the sky looks blue and speak it.`

## What’s Been Tried
- Previous branch work optimized frame-gap timing directly, but the current code inspection and websocket reproduction showed a more basic likely issue: streamed chunks were replaying overlapping waveform windows because server-side tail accounting was wrong.
- Baseline on the honest websocket benchmark: `realtime_penalty_ms=15376`, `first_playable_audio_ms=1478`, `overlap_ms=13440`, holdout overlap `58240`. Replay completely dominated the user-visible penalty.
- **Keep:** update `last_emitted_samples` immediately after each decode and slice new waveform tails from that tracked position. Result: `realtime_penalty_ms=2842`, `overlap_ms=0`, holdout overlap `0`. This fixed the correctness bug and removed the huge replay penalty.
- **Keep:** reduce the default streaming decode batch from 4 frames to 2 frames. Result: `realtime_penalty_ms=2201`, `first_playable_audio_ms=910`, `max_chunk_deficit_ms=1332`, holdout penalty `2140`. This improved startup and chunk deficit without bringing overlap back, but total turn time jumped to ~43s and chunk count to 38.
- **Keep:** reduce default streaming decode context from 16 frames to 0 frames. Result: `realtime_penalty_ms=1047`, `first_playable_audio_ms=769`, `max_chunk_deficit_ms=285`, `total_turn_ms=13324`, holdout penalty `1057`. This recovered throughput while keeping overlap at zero, making it the best result so far.
- **Crash/Discard insight:** a 1-frame batch improved startup further (~0.6s first playable) but exploded chunk count enough that the tune suite stopped completing reliably under the harness.
- After the tail fix, context retention turned out to be the biggest remaining CPU cost. With context=0 and batch=2, the system is much closer to tolerable realtime but still not smooth enough.
- Current pacing still uses chunk-count startup gating, which is suspicious because chunk duration varies dramatically when decode windows grow.

## Current Hypotheses
1. The biggest remaining gain is now likely in pacing, not overlap or decode-window bookkeeping.
2. Millisecond-based pacing should reduce startup latency and may smooth delivery better than chunk-count startup gating.
3. If pacing yields only marginal gains, the next structural option is decoupling generation from detokenization so decode work is hidden behind frame generation again.
