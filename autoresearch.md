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
- After the tail fix, the dominant bottleneck is now starvation/throughput rather than overlap: `max_chunk_deficit_ms` increased to ~1.3–1.5s and first playable audio stayed around 1.54s.
- Current pacing still uses chunk-count startup gating, which is suspicious because chunk duration varies dramatically when decode windows grow.

## Current Hypotheses
1. The biggest remaining gain is now in pacing and decode granularity, not correctness: we need more buffered-audio-aware scheduling and/or smaller/faster chunk production.
2. Millisecond-based pacing should reduce startup latency and may smooth delivery better than chunk-count startup gating.
3. If pacing alone is not enough, tuning `stream_batch_frames` / `stream_context_frames` should attack starvation directly before any larger architectural changes.
