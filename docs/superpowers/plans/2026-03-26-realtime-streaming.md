# Realtime Streaming Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make interleaved TTS streaming sound smooth in realtime by fixing overlap/replay, improving observability, and then tuning pacing and decode granularity against an honest websocket benchmark.

**Architecture:** Build a measurement harness around the real websocket path, fix streamed tail correctness in the server-side decoder, then optimize pacing and decode settings using measured first-playable latency and starvation risk instead of a single synthetic microbenchmark.

**Tech Stack:** Rust, tokio, axum websockets, Node.js `ws`, shell scripts, cargo tests

---

## File Map
- `src/bin/server.rs` — streaming decoder state, websocket chunk metadata, pacing policy, server unit tests
- `static/app.js` — browser timing logs for manual validation
- `static/assistant-player-worklet.js` — playback queue state and underrun reporting
- `scripts/bench_realtime_ws.mjs` — websocket benchmark driver
- `autoresearch.md` — benchmark contract and scope
- `autoresearch.sh` — reproducible benchmark entrypoint
- `autoresearch.checks.sh` — correctness backpressure checks
- `autoresearch.ideas.md` — deferred ideas backlog

### Task 1: Install the autoresearch harness

**Files:**
- Create: `scripts/bench_realtime_ws.mjs`
- Create: `autoresearch.md`
- Create: `autoresearch.sh`
- Create: `autoresearch.checks.sh`
- Create: `autoresearch.ideas.md`

- [ ] **Step 1: Write the benchmark harness**
- [ ] **Step 2: Make `autoresearch.sh` build the server, start it, run the harness, and emit `METRIC` lines**
- [ ] **Step 3: Add checks that run fast server-focused tests after a passing benchmark**
- [ ] **Step 4: Verify the harness fails clearly when no audio arrives**
- [ ] **Step 5: Commit via autoresearch baseline keep**

### Task 2: Lock in the current baseline

**Files:**
- Modify: `autoresearch.md`
- Inspect: `src/bin/server.rs`

- [ ] **Step 1: Run the baseline benchmark without code changes**
- [ ] **Step 2: Record primary and secondary metrics in `autoresearch.jsonl`**
- [ ] **Step 3: Update `autoresearch.md` with baseline observations and likely root causes**

### Task 3: Fix streamed tail correctness with regression coverage

**Files:**
- Modify: `src/bin/server.rs`
- Test: `src/bin/server.rs` server unit tests

- [ ] **Step 1: Write a failing regression test for overlapping streamed output**
- [ ] **Step 2: Run the targeted test and confirm failure**
- [ ] **Step 3: Update `OnnxStreamingAudioDecoder::decode_pending()` / context bookkeeping so only new samples are emitted**
- [ ] **Step 4: Run targeted tests and the benchmark**
- [ ] **Step 5: Keep only if replay/overlap drops meaningfully without breaking checks**

### Task 4: Improve timing visibility

**Files:**
- Modify: `src/bin/server.rs`
- Modify: `static/app.js`
- Modify: `static/assistant-player-worklet.js`
- Modify: `autoresearch.md`

- [ ] **Step 1: Add structured per-turn fields for first chunk ready/sent, chunk duration, and queue wait**
- [ ] **Step 2: Extend the client/browser logs with first-playback and queued-audio state where useful**
- [ ] **Step 3: Feed the additional signals into the benchmark harness or manual validation notes**
- [ ] **Step 4: Re-run benchmark and keep only if signal quality improves without changing semantics**

### Task 5: Tune pacing policy honestly

**Files:**
- Modify: `src/bin/server.rs`
- Test: `src/bin/server.rs` pacing tests

- [ ] **Step 1: Add or extend tests around startup buffering behavior**
- [ ] **Step 2: Replace chunk-count-only startup gating with buffered-audio-duration logic or a hybrid policy**
- [ ] **Step 3: Benchmark first-playable latency and starvation risk across the prompt suite and holdout**
- [ ] **Step 4: Keep only if first-playable latency improves without introducing deficits/underruns**

### Task 6: Tune decode batch/context settings

**Files:**
- Modify: `src/bin/server.rs`
- Modify: `autoresearch.md`

- [ ] **Step 1: Experiment with smaller `stream_batch_frames` values**
- [ ] **Step 2: Experiment with context retention only after tail correctness is stable**
- [ ] **Step 3: Compare steady-state chunk slack, overlap, and first-playable latency**
- [ ] **Step 4: Keep only broadly positive settings; discard prompt-specific wins**

### Task 7: Explore structural changes only if simpler paths stall

**Files:**
- Modify: `src/bin/server.rs`
- Possibly modify: `src/tts.rs`, related decode plumbing
- Update: `autoresearch.ideas.md`

- [ ] **Step 1: If pacing and settings are exhausted, prototype generation/decode decoupling**
- [ ] **Step 2: Measure whether the structural change increases slack or lowers first-playable latency**
- [ ] **Step 3: Keep only if the improvement is real and the added complexity is justified**
