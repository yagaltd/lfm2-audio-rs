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
- Previous branch work optimized frame-gap timing directly, but the current code inspection and websocket reproduction show a more basic likely issue: streamed chunks appear to replay overlapping waveform windows because server-side tail accounting is wrong.
- Reproduction on the live websocket path showed first playable audio around 1.4–1.5s after request send, while streamed audio duration greatly exceeded unique audio duration implied by frame count. This suggests a correctness bug plus a thin realtime margin.
- Current pacing also uses chunk-count startup gating, which is suspicious because chunk duration varies dramatically when decode windows grow.

## Current Hypotheses
1. Fixing streamed tail accounting in `OnnxStreamingAudioDecoder` will sharply reduce overlap/replay and improve user-perceived smoothness more than any buffer tweak.
2. After correctness is fixed, millisecond-based pacing should reduce startup latency without reintroducing starvation.
3. Only after those two fixes should we tune decode batch/context sizes or attempt structural producer/consumer decoupling.
