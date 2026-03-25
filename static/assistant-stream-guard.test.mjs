import test from "node:test";
import assert from "node:assert/strict";

import { acceptAssistantAudioChunk } from "./assistant-stream-guard.js";

test("accepts the first assistant audio chunk", () => {
  assert.deepEqual(acceptAssistantAudioChunk(null, { chunk_index: 1 }), {
    accept: true,
    nextChunkIndex: 1,
    reason: null,
  });
});

test("rejects duplicate assistant audio chunks", () => {
  assert.deepEqual(acceptAssistantAudioChunk(7, { chunk_index: 7 }), {
    accept: false,
    nextChunkIndex: 7,
    reason: "duplicate",
  });
});

test("rejects non-monotonic assistant audio chunks", () => {
  assert.deepEqual(acceptAssistantAudioChunk(7, { chunk_index: 6 }), {
    accept: false,
    nextChunkIndex: 7,
    reason: "non-monotonic",
  });
});

test("rejects binary audio payloads without chunk metadata", () => {
  assert.deepEqual(acceptAssistantAudioChunk(7, null), {
    accept: false,
    nextChunkIndex: 7,
    reason: "missing-meta",
  });
});
