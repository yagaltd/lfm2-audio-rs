import WebSocket from 'ws';

const STREAM_SAMPLE_RATE = 24_000;
const FRAME_MS = 80;
const DEFAULT_TIMEOUT_MS = Number(process.env.BENCH_TIMEOUT_MS || 90_000);
const WS_URL = process.env.BENCH_WS_URL || 'ws://127.0.0.1:18080/ws/interleaved';
const SYSTEM_PROMPT = 'Respond with interleaved text and audio.';

const TUNE_PROMPTS = [
  'Say hello in one short sentence and speak it.',
  'Count to three quickly and speak it.',
  'Tell me one short fun fact and speak it.',
];

const HOLDOUT_PROMPTS = [
  'In two short sentences, explain why the sky looks blue and speak it.',
];

function median(values) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
    : sorted[mid];
}

function maxOf(values) {
  return values.length === 0 ? 0 : Math.max(...values);
}

function round(value) {
  return Math.round(value);
}

function audioMsFromBytes(bytes) {
  return (bytes / 2 / STREAM_SAMPLE_RATE) * 1000;
}

async function runTurn(prompt, suite, index) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(WS_URL);
    const openedAt = Date.now();
    let turnStartedAt = null;
    let pendingMeta = null;
    let firstBinaryAt = null;
    let lastBinaryAt = null;
    let lastChunkAudioMs = null;
    let totalBinaryBytes = 0;
    let chunkCount = 0;
    let maxBinaryGapMs = 0;
    let maxChunkDeficitMs = 0;
    let lastFrameIndex = 0;
    let firstMeta = null;
    let finished = false;
    let timeout = null;

    const finish = (err, result) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      try {
        ws.close();
      } catch {}
      if (err) reject(err);
      else resolve(result);
    };

    timeout = setTimeout(() => {
      finish(new Error(`timeout waiting for prompt ${suite}:${index + 1}`));
    }, DEFAULT_TIMEOUT_MS);

    ws.on('open', () => {
      ws.send(JSON.stringify({ type: 'session.start', system_prompt: SYSTEM_PROMPT }));
    });

    ws.on('message', (data, isBinary) => {
      if (!isBinary) {
        let message;
        try {
          message = JSON.parse(data.toString());
        } catch (error) {
          finish(error);
          return;
        }

        if (message.type === 'error') {
          finish(new Error(message.message || 'server error'));
          return;
        }

        if (message.type === 'session.started') {
          turnStartedAt = Date.now();
          ws.send(JSON.stringify({ type: 'user.text', text: prompt }));
          return;
        }

        if (message.type === 'assistant.audio.chunk') {
          pendingMeta = message;
          if (!firstMeta) {
            firstMeta = message;
          }
          if (typeof message.frame_index === 'number') {
            lastFrameIndex = message.frame_index;
          }
          return;
        }

        if (message.type === 'assistant.audio.end') {
          if (turnStartedAt == null) {
            finish(new Error(`turn never started for ${suite}:${index + 1}`));
            return;
          }
          if (chunkCount === 0) {
            finish(new Error(`no audio chunks for ${suite}:${index + 1}`));
            return;
          }

          const totalTurnMs = Date.now() - turnStartedAt;
          const streamedAudioMs = audioMsFromBytes(totalBinaryBytes);
          const uniqueAudioMs = lastFrameIndex * FRAME_MS;
          const overlapMs = Math.max(0, round(streamedAudioMs - uniqueAudioMs));
          const firstPlayableAudioMs = firstBinaryAt == null ? 0 : firstBinaryAt - turnStartedAt;
          const realtimePenaltyMs = firstPlayableAudioMs + overlapMs + maxChunkDeficitMs;

          finish(null, {
            suite,
            prompt,
            opened_ms: Date.now() - openedAt,
            first_playable_audio_ms: firstPlayableAudioMs,
            overlap_ms: overlapMs,
            max_chunk_deficit_ms: maxChunkDeficitMs,
            max_binary_gap_ms: maxBinaryGapMs,
            streamed_audio_ms: round(streamedAudioMs),
            unique_audio_ms: uniqueAudioMs,
            total_turn_ms: totalTurnMs,
            chunk_count: chunkCount,
            last_frame_index: lastFrameIndex,
            realtime_penalty_ms: realtimePenaltyMs,
            server_first_frame_ms: firstMeta?.frame_elapsed_ms ?? 0,
            server_first_decode_ms: firstMeta?.decode_elapsed_ms ?? 0,
            server_first_queue_wait_ms: firstMeta?.queue_wait_ms ?? 0,
          });
        }
        return;
      }

      const now = Date.now();
      const chunkAudioMs = audioMsFromBytes(data.length);
      chunkCount += 1;
      totalBinaryBytes += data.length;

      if (firstBinaryAt == null) {
        firstBinaryAt = now;
      }
      if (lastBinaryAt != null) {
        const binaryGapMs = now - lastBinaryAt;
        maxBinaryGapMs = Math.max(maxBinaryGapMs, binaryGapMs);
        if (lastChunkAudioMs != null) {
          maxChunkDeficitMs = Math.max(
            maxChunkDeficitMs,
            Math.max(0, round(binaryGapMs - lastChunkAudioMs)),
          );
        }
      }

      lastBinaryAt = now;
      lastChunkAudioMs = chunkAudioMs;
      pendingMeta = null;
    });

    ws.on('error', (error) => finish(error));
    ws.on('close', () => {
      if (!finished) {
        finish(new Error(`socket closed before assistant.audio.end for ${suite}:${index + 1}`));
      }
    });
  });
}

async function runSuite(prompts, suite) {
  const results = [];
  for (let idx = 0; idx < prompts.length; idx += 1) {
    const result = await runTurn(prompts[idx], suite, idx);
    results.push(result);
    console.error(
      `[bench] ${suite}:${idx + 1} first=${result.first_playable_audio_ms}ms overlap=${result.overlap_ms}ms deficit=${result.max_chunk_deficit_ms}ms gap=${result.max_binary_gap_ms}ms chunks=${result.chunk_count}`,
    );
  }
  return results;
}

function aggregate(results) {
  return {
    realtime_penalty_ms: median(results.map((r) => r.realtime_penalty_ms)),
    first_playable_audio_ms: median(results.map((r) => r.first_playable_audio_ms)),
    overlap_ms: median(results.map((r) => r.overlap_ms)),
    max_chunk_deficit_ms: maxOf(results.map((r) => r.max_chunk_deficit_ms)),
    max_binary_gap_ms: maxOf(results.map((r) => r.max_binary_gap_ms)),
    total_turn_ms: median(results.map((r) => r.total_turn_ms)),
    chunk_count: median(results.map((r) => r.chunk_count)),
    server_first_frame_ms: median(results.map((r) => r.server_first_frame_ms)),
    server_first_decode_ms: median(results.map((r) => r.server_first_decode_ms)),
    server_first_queue_wait_ms: median(results.map((r) => r.server_first_queue_wait_ms)),
  };
}

const tune = await runSuite(TUNE_PROMPTS, 'tune');
const holdout = await runSuite(HOLDOUT_PROMPTS, 'holdout');
const tuneAgg = aggregate(tune);
const holdoutAgg = aggregate(holdout);

for (const result of [...tune, ...holdout]) {
  console.error(`PROMPT ${JSON.stringify(result)}`);
}

console.log(`METRIC realtime_penalty_ms=${tuneAgg.realtime_penalty_ms}`);
console.log(`METRIC first_playable_audio_ms=${tuneAgg.first_playable_audio_ms}`);
console.log(`METRIC overlap_ms=${tuneAgg.overlap_ms}`);
console.log(`METRIC max_chunk_deficit_ms=${tuneAgg.max_chunk_deficit_ms}`);
console.log(`METRIC max_binary_gap_ms=${tuneAgg.max_binary_gap_ms}`);
console.log(`METRIC total_turn_ms=${tuneAgg.total_turn_ms}`);
console.log(`METRIC chunk_count=${tuneAgg.chunk_count}`);
console.log(`METRIC server_first_frame_ms=${tuneAgg.server_first_frame_ms}`);
console.log(`METRIC server_first_decode_ms=${tuneAgg.server_first_decode_ms}`);
console.log(`METRIC server_first_queue_wait_ms=${tuneAgg.server_first_queue_wait_ms}`);
console.log(`METRIC holdout_realtime_penalty_ms=${holdoutAgg.realtime_penalty_ms}`);
console.log(`METRIC holdout_first_playable_audio_ms=${holdoutAgg.first_playable_audio_ms}`);
console.log(`METRIC holdout_overlap_ms=${holdoutAgg.overlap_ms}`);
console.log(`METRIC holdout_max_chunk_deficit_ms=${holdoutAgg.max_chunk_deficit_ms}`);
