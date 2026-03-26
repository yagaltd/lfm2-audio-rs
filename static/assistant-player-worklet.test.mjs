import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const WORKLET_PATH =
  "/home/aurel/Documents/vibe/STT-rust/lfm2-audio-rs/static/assistant-player-worklet.js";

function loadProcessorClass() {
  let registeredProcessor = null;

  class MockPort {
    constructor() {
      this.messages = [];
      this.onmessage = null;
    }

    postMessage(message) {
      this.messages.push(message);
    }
  }

  class MockAudioWorkletProcessor {
    constructor() {
      this.port = new MockPort();
    }
  }

  const context = vm.createContext({
    AudioWorkletProcessor: MockAudioWorkletProcessor,
    Float32Array,
    registerProcessor(name, processorClass) {
      registeredProcessor = { name, processorClass };
    },
  });

  const source = fs.readFileSync(WORKLET_PATH, "utf8");
  new vm.Script(source, { filename: WORKLET_PATH }).runInContext(context);

  assert.ok(registeredProcessor, "worklet should register a processor");
  return registeredProcessor.processorClass;
}

function enqueue(processor, sampleCount) {
  processor.port.onmessage({
    data: { type: "enqueue", samples: new Float32Array(sampleCount) },
  });
}

test("assistant player starts playback as soon as the first chunk arrives", () => {
  const Processor = loadProcessorClass();
  const processor = new Processor();

  enqueue(processor, 1920);

  assert.equal(processor.started, true);
  assert.equal(processor.port.messages.length, 1);
  assert.equal(processor.port.messages[0].reason, "playback-started");
  assert.equal(processor.port.messages[0].queued_samples, 1920);
});

test("assistant player buffers a too-small first chunk until threshold is met", () => {
  const Processor = loadProcessorClass();
  const processor = new Processor();

  enqueue(processor, 64);

  assert.equal(processor.started, false);
  assert.equal(processor.port.messages.length, 0);

  enqueue(processor, 64);

  assert.equal(processor.started, true);
  assert.equal(processor.port.messages.length, 1);
  assert.equal(processor.port.messages[0].reason, "playback-started");
  assert.equal(processor.port.messages[0].queued_samples, 128);
});

test("assistant player waits for a small refill before resuming after underrun", () => {
  const Processor = loadProcessorClass();
  const processor = new Processor();
  const output = [new Float32Array(128)];

  enqueue(processor, 128);
  assert.equal(processor.started, true);

  processor.process([], [output], {});
  processor.process([], [output], {});

  assert.equal(processor.underruns, 1);
  assert.equal(processor.inUnderrun, true);

  enqueue(processor, 64);
  assert.equal(processor.started, false);
  assert.equal(processor.inUnderrun, true);

  enqueue(processor, 64);
  assert.equal(processor.started, true);
  assert.equal(processor.inUnderrun, false);

  const resumeMessage = processor.port.messages.at(-1);
  assert.equal(resumeMessage.reason, "playback-resumed");
  assert.equal(resumeMessage.queued_samples, 128);
});
