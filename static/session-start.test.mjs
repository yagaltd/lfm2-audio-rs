import test from "node:test";
import assert from "node:assert/strict";

import { buildSessionStartMessage } from "./session-start.js";

test("session start payload only carries the system prompt", () => {
  assert.deepEqual(buildSessionStartMessage("Respond with interleaved text and audio."), {
    type: "session.start",
    system_prompt: "Respond with interleaved text and audio.",
  });
});
