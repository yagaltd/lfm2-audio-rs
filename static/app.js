import createVADModule from "/static/lib/Web/ten_vad.js";
import { acceptAssistantAudioChunk } from "/static/assistant-stream-guard.js";

const SAMPLE_RATE = 16000;
const HOP_SIZE = 256;
const FRAME_DURATION_MS = (HOP_SIZE / SAMPLE_RATE) * 1000;
const PRE_ROLL_FRAMES = 4;
const SPEECH_START_FRAMES = 3;
const NOISE_FLOOR_RMS = 0.012;
const ASSISTANT_SAMPLE_RATE = 24000;
const STATIC_ASSET_VERSION = "20260324-timing";

const connectButton = document.getElementById("connect");
const disconnectButton = document.getElementById("disconnect");
const resetSessionButton = document.getElementById("resetSession");
const startMicButton = document.getElementById("startMic");
const stopMicButton = document.getElementById("stopMic");
const sendTextButton = document.getElementById("sendText");
const textInput = document.getElementById("textInput");
const systemPromptInput = document.getElementById("systemPrompt");
const vadThresholdInput = document.getElementById("vadThreshold");
const vadThresholdValue = document.getElementById("vadThresholdValue");
const silenceTimeoutInput = document.getElementById("silenceTimeout");
const minSpeechMsInput = document.getElementById("minSpeechMs");
const showTranscriptInput = document.getElementById("showTranscript");
const fastTtsModeInput = document.getElementById("fastTtsMode");
if (!fastTtsModeInput) {
  console.error("fastTtsMode checkbox not found!");
}
const connectionState = document.getElementById("connectionState");
const phaseState = document.getElementById("phaseState");
const vadMeter = document.getElementById("vadMeter");
const feed = document.getElementById("feed");
const logEl = document.getElementById("log");
const assistantAudio = document.getElementById("assistantAudio");

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let workletNode = null;
let assistantPlaybackContext = null;
let assistantPlaybackNode = null;
let vadModule = null;
let vadHandle = null;
let vadHandlePtr = null;
let isMicRunning = false;
let speechFrames = [];
let preRollFrames = [];
let speechActive = false;
let consecutiveSpeechFrames = 0;
let silenceMs = 0;
let trailingSilenceFrames = 0;
let assistantPcmChunks = [];
let assistantStreamingAudio = false;
let currentAssistantItem = null;
let pendingAssistantChunkMeta = null;
let assistantStreamMetrics = null;
let lastAssistantChunkIndex = null;

function log(message) {
  const time = new Date().toLocaleTimeString();
  logEl.textContent = `[${time}] ${message}\n${logEl.textContent}`.slice(0, 12000);
}

function samplesToMs(samples, sampleRate = ASSISTANT_SAMPLE_RATE) {
  return Math.round((samples / sampleRate) * 1000);
}

function resetAssistantMetrics() {
  assistantStreamMetrics = {
    startedAt: performance.now(),
    firstChunkAt: null,
    lastChunkAt: null,
    chunkCount: 0,
    totalSamples: 0,
    maxReceiveGapMs: 0,
    decodeBackend: null,
    underruns: 0,
  };
}

function currentVadThreshold() {
  return Number(vadThresholdInput.value);
}

function updateVadThresholdLabel() {
  vadThresholdValue.textContent = currentVadThreshold().toFixed(2);
}

function setConnectionLabel(text) {
  connectionState.textContent = text;
}

function setPhase(text) {
  phaseState.textContent = text;
}

function appendFeed(role, text) {
  const item = document.createElement("div");
  item.className = `feed-item ${role}`;
  item.innerHTML = `<span class="feed-role">${role}</span>${text}`;
  feed.prepend(item);
  return item;
}

function updateAssistantDraft(text, done = false) {
  if (!currentAssistantItem) {
    currentAssistantItem = appendFeed("assistant", "");
  }
  currentAssistantItem.innerHTML = `<span class="feed-role">assistant</span>${text}`;
  if (done) {
    currentAssistantItem = null;
  }
}

function binaryToInt16Array(buffers) {
  const totalBytes = buffers.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const combined = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of buffers) {
    combined.set(new Uint8Array(chunk), offset);
    offset += chunk.byteLength;
  }
  return new Int16Array(combined.buffer);
}

function createWavBlobFromInt16(samples, sampleRate) {
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let idx = 0; idx < value.length; idx += 1) {
      view.setUint8(offset + idx, value.charCodeAt(idx));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let idx = 0; idx < samples.length; idx += 1) {
    view.setInt16(offset, samples[idx], true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

async function ensureAssistantPlayback() {
  if (assistantPlaybackNode && assistantPlaybackContext) {
    return;
  }

  assistantPlaybackContext = new AudioContext({ sampleRate: ASSISTANT_SAMPLE_RATE });
  await assistantPlaybackContext.audioWorklet.addModule(
    `/static/assistant-player-worklet.js?v=${STATIC_ASSET_VERSION}`,
  );
  assistantPlaybackNode = new AudioWorkletNode(
    assistantPlaybackContext,
    "assistant-player-worklet",
  );
  assistantPlaybackNode.port.onmessage = (event) => {
    if (event.data?.type !== "buffer-state") {
      return;
    }

    if (event.data.reason === "playback-started") {
      log(
        `Assistant playback started (${samplesToMs(event.data.queued_samples)}ms queued)`,
      );
      return;
    }

    if (event.data.reason === "underrun") {
      if (assistantStreamMetrics) {
        assistantStreamMetrics.underruns = event.data.underruns;
      }
      log(
        `Assistant playback underrun (#${event.data.underruns}, ${samplesToMs(event.data.queued_samples)}ms queued)`,
      );
    }
  };
  assistantPlaybackNode.connect(assistantPlaybackContext.destination);
}

function resetAssistantStream() {
  assistantPcmChunks = [];
  assistantStreamingAudio = true;
  pendingAssistantChunkMeta = null;
  lastAssistantChunkIndex = null;
  resetAssistantMetrics();
  assistantPlaybackNode?.port.postMessage({ type: "reset" });
}

async function enqueueAssistantChunk(buffer) {
  await ensureAssistantPlayback();
  if (assistantPlaybackContext.state === "suspended") {
    await assistantPlaybackContext.resume();
  }

  const pcm = new Int16Array(buffer);
  assistantPcmChunks.push(new Int16Array(pcm));
  const floatChunk = new Float32Array(pcm.length);
  for (let idx = 0; idx < pcm.length; idx += 1) {
    floatChunk[idx] = pcm[idx] / 32768;
  }
  assistantPlaybackNode.port.postMessage(
    { type: "enqueue", samples: floatChunk },
    [floatChunk.buffer],
  );
}

function frameRms(frame) {
  let sumSquares = 0;
  for (let idx = 0; idx < frame.length; idx += 1) {
    const normalized = frame[idx] / 32768;
    sumSquares += normalized * normalized;
  }
  return Math.sqrt(sumSquares / frame.length);
}

function pushRollingFrame(buffer, frame, maxFrames) {
  buffer.push(new Int16Array(frame));
  while (buffer.length > maxFrames) {
    buffer.shift();
  }
}

function resetSpeechTracking() {
  speechFrames = [];
  preRollFrames = [];
  speechActive = false;
  consecutiveSpeechFrames = 0;
  silenceMs = 0;
  trailingSilenceFrames = 0;
}

async function initVAD() {
  if (vadModule) {
    return;
  }

  log("Loading VAD runtime");
  vadModule = await createVADModule();
  if (!vadModule.getValue) {
    vadModule.getValue = function getValue(ptr, type) {
      const view = new DataView(vadModule.HEAPU8.buffer);
      if (type === "i32") return view.getInt32(ptr, true);
      if (type === "float") return view.getFloat32(ptr, true);
      throw new Error(`Unsupported getValue type: ${type}`);
    };
  }
  recreateVAD();
}

function recreateVAD() {
  destroyVAD();
  vadHandlePtr = vadModule._malloc(4);
  const threshold = currentVadThreshold();
  const result = vadModule._ten_vad_create(vadHandlePtr, HOP_SIZE, threshold);
  if (result !== 0) {
    throw new Error(`ten_vad_create failed with code ${result}`);
  }
  vadHandle = vadModule.getValue(vadHandlePtr, "i32");
  log(`VAD ready (threshold=${threshold.toFixed(2)})`);
}

function destroyVAD() {
  if (vadModule && vadHandlePtr) {
    vadModule._ten_vad_destroy(vadHandlePtr);
    vadModule._free(vadHandlePtr);
  }
  vadHandle = null;
  vadHandlePtr = null;
}

function runVAD(frame) {
  const audioPtr = vadModule._malloc(HOP_SIZE * 2);
  const probPtr = vadModule._malloc(4);
  const flagPtr = vadModule._malloc(4);
  try {
    vadModule.HEAP16.set(frame, audioPtr / 2);
    const result = vadModule._ten_vad_process(vadHandle, audioPtr, HOP_SIZE, probPtr, flagPtr);
    if (result !== 0) {
      throw new Error(`ten_vad_process failed with code ${result}`);
    }
    return {
      probability: vadModule.getValue(probPtr, "float"),
      flag: vadModule.getValue(flagPtr, "i32"),
    };
  } finally {
    vadModule._free(audioPtr);
    vadModule._free(probPtr);
    vadModule._free(flagPtr);
  }
}

function updateControls() {
  const connected = ws && ws.readyState === WebSocket.OPEN;
  connectButton.disabled = connected;
  disconnectButton.disabled = !connected;
  resetSessionButton.disabled = !connected;
  startMicButton.disabled = !connected || isMicRunning;
  stopMicButton.disabled = !connected || !isMicRunning;
}

function connectSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    return;
  }

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/interleaved`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    setConnectionLabel("Connected");
    setPhase("Listening");
    updateControls();
    const ttsBackend = fastTtsModeInput.checked ? "kitten" : "lfm2";
    const msg = {
      type: "session.start",
      system_prompt: systemPromptInput.value.trim(),
      tts_backend: ttsBackend,
    };
    log(`WebSocket connected (TTS: ${ttsBackend}, checked=${fastTtsModeInput.checked})`);
    console.log("Sending session.start:", msg);
    ws.send(JSON.stringify(msg));
  });

  ws.addEventListener("close", () => {
    setConnectionLabel("Disconnected");
    setPhase("Idle");
    updateControls();
    log("WebSocket disconnected");
  });

  ws.addEventListener("error", () => {
    log("WebSocket error");
  });

  ws.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      handleServerJson(JSON.parse(event.data));
      return;
    }

    if (assistantStreamingAudio) {
      const chunkDecision = acceptAssistantAudioChunk(
        lastAssistantChunkIndex,
        pendingAssistantChunkMeta,
      );
      if (!chunkDecision.accept) {
        const rejectedChunkIndex = pendingAssistantChunkMeta?.chunk_index ?? "n/a";
        log(
          `Dropped assistant audio chunk (${chunkDecision.reason}, chunk=${rejectedChunkIndex}, last=${lastAssistantChunkIndex ?? "n/a"})`,
        );
        pendingAssistantChunkMeta = null;
        return;
      }

      lastAssistantChunkIndex = chunkDecision.nextChunkIndex;
      const receivedAt = performance.now();
      if (assistantStreamMetrics) {
        if (assistantStreamMetrics.firstChunkAt == null) {
          assistantStreamMetrics.firstChunkAt = receivedAt;
        }
        if (assistantStreamMetrics.lastChunkAt != null) {
          assistantStreamMetrics.maxReceiveGapMs = Math.max(
            assistantStreamMetrics.maxReceiveGapMs,
            Math.round(receivedAt - assistantStreamMetrics.lastChunkAt),
          );
        }
        assistantStreamMetrics.lastChunkAt = receivedAt;
      }
      if (pendingAssistantChunkMeta) {
        if (assistantStreamMetrics) {
          assistantStreamMetrics.chunkCount += 1;
          assistantStreamMetrics.totalSamples += pendingAssistantChunkMeta.chunk_samples || 0;
          assistantStreamMetrics.decodeBackend = pendingAssistantChunkMeta.backend || null;
        }
        if (pendingAssistantChunkMeta.chunk_index === 1) {
          log(
            `Assistant first chunk received (${pendingAssistantChunkMeta.backend}, frame=${pendingAssistantChunkMeta.frame_index}, server_frame=${pendingAssistantChunkMeta.frame_elapsed_ms}ms, frame_gap=${pendingAssistantChunkMeta.frame_gap_ms}ms, decode=${pendingAssistantChunkMeta.decode_elapsed_ms}ms, queue=${pendingAssistantChunkMeta.queue_wait_ms}ms)`,
          );
        }
      }
      enqueueAssistantChunk(event.data).catch((error) => {
        log(`Assistant playback failed: ${error.message}`);
      });
      pendingAssistantChunkMeta = null;
    }
  });
}

function disconnectSocket() {
  if (ws) {
    ws.close();
    ws = null;
  }
  assistantStreamingAudio = false;
  assistantPcmChunks = [];
  currentAssistantItem = null;
  pendingAssistantChunkMeta = null;
  assistantStreamMetrics = null;
  lastAssistantChunkIndex = null;
  updateControls();
}

function handleServerJson(message) {
  switch (message.type) {
    case "status":
      setPhase(message.phase);
      log(`Phase: ${message.phase}`);
      break;
    case "session.started":
      log("Session initialized");
      break;
    case "session.reset":
      log("Session reset");
      break;
    case "assistant.text.done":
      if (message.text && message.text.trim()) {
        updateAssistantDraft(message.text, true);
      } else if (currentAssistantItem) {
        currentAssistantItem = null;
      }
      break;
    case "assistant.text.delta":
      updateAssistantDraft(message.text || "", false);
      break;
    case "user.transcript":
      if (message.text && message.text.trim()) {
        appendFeed("user", message.text);
      }
      break;
    case "assistant.audio.start":
      resetAssistantStream();
      break;
    case "assistant.audio.chunk":
      pendingAssistantChunkMeta = message;
      break;
    case "assistant.audio.end":
      if (assistantStreamingAudio && assistantPcmChunks.length > 0) {
        const pcm = binaryToInt16Array(assistantPcmChunks.map((chunk) => chunk.buffer));
        const blob = createWavBlobFromInt16(pcm, ASSISTANT_SAMPLE_RATE);
        assistantAudio.src = URL.createObjectURL(blob);
        log(`Received assistant audio (${blob.size} bytes)`);
        if (assistantStreamMetrics) {
          const firstChunkLatency =
            assistantStreamMetrics.firstChunkAt == null
              ? null
              : Math.round(assistantStreamMetrics.firstChunkAt - assistantStreamMetrics.startedAt);
          log(
            `Assistant stream summary: backend=${assistantStreamMetrics.decodeBackend ?? "unknown"}, chunks=${assistantStreamMetrics.chunkCount}, audio=${samplesToMs(assistantStreamMetrics.totalSamples)}ms, first_chunk=${firstChunkLatency ?? "n/a"}ms, max_gap=${assistantStreamMetrics.maxReceiveGapMs}ms, underruns=${assistantStreamMetrics.underruns}`,
          );
        }
      }
      assistantStreamingAudio = false;
      pendingAssistantChunkMeta = null;
      lastAssistantChunkIndex = null;
      break;
    case "pong":
      break;
    case "error":
      log(`Server error: ${message.message}`);
      break;
    default:
      log(`Unhandled server message: ${message.type}`);
  }
}

async function startMic() {
  if (isMicRunning) {
    return;
  }

  await initVAD();
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioContext.audioWorklet.addModule("/static/vad-processor.js");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: SAMPLE_RATE,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "vad-processor", {
    processorOptions: { hopSize: HOP_SIZE },
  });
  workletNode.port.onmessage = (event) => {
    if (event.data?.type === "frame") {
      handleMicFrame(event.data.frame);
    }
  };

  sourceNode.connect(workletNode);
  isMicRunning = true;
  updateControls();
  log("Microphone started");
}

function stopMic() {
  if (!isMicRunning) {
    return;
  }

  finalizeSpeechSegment();

  workletNode?.disconnect();
  sourceNode?.disconnect();
  mediaStream?.getTracks().forEach((track) => track.stop());
  audioContext?.close();

  workletNode = null;
  sourceNode = null;
  mediaStream = null;
  audioContext = null;
  isMicRunning = false;
  resetSpeechTracking();
  vadMeter.style.width = "0%";
  updateControls();
  log("Microphone stopped");
}

function handleMicFrame(frameLike) {
  const frame = frameLike instanceof Int16Array ? frameLike : new Int16Array(frameLike);
  const { probability, flag } = runVAD(frame);
  const rms = frameRms(frame);
  vadMeter.style.width = `${Math.min(100, probability * 100)}%`;

  const threshold = currentVadThreshold();
  const consideredSpeech = (flag === 1 || probability >= threshold) && rms >= NOISE_FLOOR_RMS;

  if (!speechActive) {
    pushRollingFrame(preRollFrames, frame, PRE_ROLL_FRAMES);
    if (consideredSpeech) {
      consecutiveSpeechFrames += 1;
    } else {
      consecutiveSpeechFrames = 0;
    }

    if (consecutiveSpeechFrames >= SPEECH_START_FRAMES) {
      speechActive = true;
      speechFrames = preRollFrames.map((savedFrame) => new Int16Array(savedFrame));
      preRollFrames = [];
      silenceMs = 0;
      trailingSilenceFrames = 0;
      consecutiveSpeechFrames = 0;
      setPhase("Speech detected");
      log("Speech started");
    }
    return;
  }

  speechFrames.push(new Int16Array(frame));

  if (consideredSpeech) {
    silenceMs = 0;
    trailingSilenceFrames = 0;
    return;
  }

  trailingSilenceFrames += 1;
  silenceMs += FRAME_DURATION_MS;

  if (silenceMs >= Number(silenceTimeoutInput.value)) {
    finalizeSpeechSegment();
  }
}

function finalizeSpeechSegment() {
  if (!speechActive || speechFrames.length === 0 || !ws || ws.readyState !== WebSocket.OPEN) {
    resetSpeechTracking();
    return;
  }

  const keptFrames = trailingSilenceFrames > 0
    ? speechFrames.slice(0, Math.max(0, speechFrames.length - trailingSilenceFrames))
    : speechFrames.slice();

  if (keptFrames.length === 0) {
    resetSpeechTracking();
    setPhase("Listening");
    return;
  }

  const pcm = binaryToInt16Array(keptFrames.map((frame) => frame.buffer));
  const durationMs = (pcm.length / SAMPLE_RATE) * 1000;
  if (durationMs < Number(minSpeechMsInput.value)) {
    log(`Discarded short utterance (${durationMs.toFixed(0)}ms)`);
    resetSpeechTracking();
    setPhase("Listening");
    return;
  }

  ws.send(JSON.stringify({
    type: "user.audio.start",
    sample_rate: SAMPLE_RATE,
    format: "pcm_s16le",
    channels: 1,
  }));
  ws.send(pcm.buffer.slice(0));
  ws.send(JSON.stringify({
    type: "user.audio.end",
    include_transcript: showTranscriptInput.checked,
  }));
  log(`Sent trimmed voice segment (${durationMs.toFixed(0)}ms, ${pcm.byteLength} bytes)`);

  resetSpeechTracking();
  setPhase("Processing");
}

function sendTextTurn() {
  const text = textInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  ws.send(JSON.stringify({ type: "user.text", text }));
  appendFeed("user", text);
  textInput.value = "";
  setPhase("Processing");
}

connectButton.addEventListener("click", connectSocket);
disconnectButton.addEventListener("click", () => {
  stopMic();
  disconnectSocket();
});
resetSessionButton.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "session.reset" }));
    feed.innerHTML = "";
    currentAssistantItem = null;
  }
});
startMicButton.addEventListener("click", async () => {
  try {
    await startMic();
  } catch (error) {
    log(`Failed to start microphone: ${error.message}`);
  }
});
stopMicButton.addEventListener("click", stopMic);
sendTextButton.addEventListener("click", sendTextTurn);
textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendTextTurn();
  }
});
vadThresholdInput.addEventListener("input", () => {
  updateVadThresholdLabel();
  if (vadModule) {
    recreateVAD();
  }
});

// === Standalone TTS ===
const synthesizeBtn = document.getElementById("synthesizeBtn");
const ttsInput = document.getElementById("ttsInput");
const ttsBackend = document.getElementById("ttsBackend");
const ttsVoice = document.getElementById("ttsVoice");
const ttsStatus = document.getElementById("ttsStatus");
const ttsAudio = document.getElementById("ttsAudio");

async function synthesizeTTS() {
  const text = ttsInput.value.trim();
  if (!text) {
    ttsStatus.textContent = "Enter text first";
    return;
  }

  const backend = ttsBackend.value;
  const voice = ttsVoice.value;

  synthesizeBtn.disabled = true;
  ttsStatus.textContent = `Synthesizing with ${backend}...`;

  const startTime = performance.now();

  try {
    const response = await fetch("/v1/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        backend: backend,
        voice: voice,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }

    const blob = await response.blob();
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    const audioUrl = URL.createObjectURL(blob);

    ttsAudio.src = audioUrl;
    ttsAudio.play();

    ttsStatus.textContent = `Done in ${elapsed}s (${backend})`;
  } catch (error) {
    ttsStatus.textContent = `Error: ${error.message}`;
    console.error("TTS error:", error);
  } finally {
    synthesizeBtn.disabled = false;
  }
}

synthesizeBtn.addEventListener("click", synthesizeTTS);
ttsInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    synthesizeTTS();
  }
});

// Update voice selector state based on backend
ttsBackend.addEventListener("change", () => {
  const isKitten = ttsBackend.value === "kitten";
  ttsVoice.disabled = !isKitten;
  if (!isKitten) {
    ttsVoice.value = "Bella";
  }
});
ttsVoice.disabled = ttsBackend.value !== "kitten";

// Fast TTS mode toggle - requires reconnect to take effect
fastTtsModeInput.addEventListener("change", () => {
  const mode = fastTtsModeInput.checked ? "KittenTTS (~50ms/frame)" : "LFM2 (~90ms/frame)";
  log(`TTS mode changed to ${mode}. ${ws && ws.readyState === WebSocket.OPEN ? "Reconnect to apply." : "Will apply on next connection."}`);
});

updateVadThresholdLabel();
updateControls();
