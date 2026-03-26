# Realtime Streaming Audio Design

## Goal
Improve user-perceived smoothness of interleaved TTS streaming without cheating on the benchmark. The system should emit non-overlapping audio chunks, reach first playable audio quickly, and maintain enough steady-state slack to avoid choppy playback.

## Problem Statement
Current streaming behavior appears choppy because the server is balancing three competing concerns at once:
1. correctness of streamed tail extraction,
2. startup latency to first audible audio,
3. steady-state chunk production vs playback consumption.

The current code already logs useful timing data, but the signals are fragmented across server logs and browser logs, which makes it easy to tune one phase while regressing another. The design therefore treats correctness as a hard prerequisite and latency tuning as a second phase.

## Constraints
- No benchmark cheating: do not optimize a single canned prompt, hide regressions behind larger buffers, or ignore correctness to lower latency.
- No new external dependencies.
- Existing websocket/API behavior must remain compatible.
- Existing tests must keep passing.
- Improvements should generalize across short, medium, and longer spoken responses.

## Proposed Architecture
### 1. Trustworthy measurement harness
Create a benchmark harness that starts the local server, drives it through websocket text turns, and records per-turn realtime metrics from the actual streaming protocol. Use a small fixed prompt suite plus a holdout prompt. The harness should compute:
- first playable audio latency,
- overlap/replay amount,
- per-chunk receive gaps,
- chunk deficit vs playout duration,
- server-side first-frame/decode/queue timings.

The primary optimization metric should reflect real user pain: replay/choppiness first, then startup latency and starvation risk. Secondary metrics should expose the breakdown so we do not overfit a composite blindly.

### 2. Correctness-first streaming decoder
The first code change should target streamed tail emission in `OnnxStreamingAudioDecoder`. Each flush must emit only newly produced waveform samples. Context retention may still be used to stabilize decode quality, but retained context must not be replayed to the client. Add a regression test for cumulative emitted samples so later latency experiments cannot silently reintroduce overlap.

### 3. Better visibility into latency phases
Promote existing ad hoc timing logs into structured metrics. The server should expose when the first audio frame was generated, when the first chunk was decoded, and how long each chunk sat in the send queue. The client-side harness should separately record when the first binary payload arrived and how large each playout chunk is.

### 4. Pacing and chunk-size tuning
Once streamed tails are correct, tune the output queue and decode batch/context settings. The current chunk-count startup gate is unstable because chunk duration grows with decode window size. Replace or supplement it with a buffered-audio-duration target. Then experiment with decode batch and context sizes to increase steady-state slack without exploding websocket overhead.

### 5. Structural work only after signal says it is needed
Only if correctness and pacing changes are not enough should the autoresearch loop attempt deeper architectural changes, such as decoupling model generation from audio detokenization. Those changes are promising but riskier and should be deferred until measurements prove the simpler fixes are exhausted.

## Files Expected to Change
- `src/bin/server.rs` — streaming decoder, pacing, websocket metadata, tests
- `static/app.js` — optional richer client-side timing logs for manual validation
- `static/assistant-player-worklet.js` — optional playback-buffer instrumentation
- `scripts/bench_realtime_ws.mjs` — benchmark harness
- `autoresearch.md`, `autoresearch.sh`, `autoresearch.checks.sh`, `autoresearch.ideas.md` — session control

## Success Criteria
1. Overlap/replay metric goes to ~0 on the benchmark suite.
2. First playable audio latency improves materially.
3. Chunk deficit / starvation risk improves or stays stable.
4. Holdout prompt metrics do not regress catastrophically.
5. Subjective listening on a few prompts matches the benchmark direction.

## Risks and Mitigations
- **Risk:** composite metric hides which subsystem regressed.
  - **Mitigation:** always log the full breakdown as secondary metrics.
- **Risk:** larger startup buffers appear to help by hiding starvation while making latency worse.
  - **Mitigation:** keep first playable audio as an explicit reported metric and reject false wins.
- **Risk:** smaller decode batches reduce latency but hurt decode quality or throughput.
  - **Mitigation:** tune in small steps and use holdout prompts.
- **Risk:** context trimming breaks continuity.
  - **Mitigation:** add regression tests around streamed tail accounting before tuning context sizes.
