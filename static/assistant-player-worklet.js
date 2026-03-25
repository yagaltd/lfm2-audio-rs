class AssistantPlayerWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.chunkOffset = 0;
    this.queuedSamples = 0;
    this.started = false;
    this.underruns = 0;
    this.inUnderrun = false;

    this.port.onmessage = (event) => {
      if (event.data?.type === "reset") {
        this.queue = [];
        this.chunkOffset = 0;
        this.queuedSamples = 0;
        this.started = false;
        this.underruns = 0;
        this.inUnderrun = false;
        return;
      }

      if (event.data?.type === "enqueue" && event.data.samples) {
        const chunk = new Float32Array(event.data.samples);
        this.queue.push(chunk);
        this.queuedSamples += chunk.length;
        this.inUnderrun = false;
        if (!this.started) {
          this.started = true;
          this.reportState("playback-started");
        }
      }
    };
  }

  reportState(reason) {
    this.port.postMessage({
      type: "buffer-state",
      reason,
      queued_samples: this.queuedSamples,
      started: this.started,
      underruns: this.underruns,
    });
  }

  process(_, outputs) {
    const output = outputs[0]?.[0];
    if (!output) {
      return true;
    }

    output.fill(0);
    if (!this.started) {
      return true;
    }

    let written = 0;

    while (written < output.length && this.queue.length > 0) {
      const current = this.queue[0];
      const remaining = current.length - this.chunkOffset;
      const toCopy = Math.min(output.length - written, remaining);

      output.set(
        current.subarray(this.chunkOffset, this.chunkOffset + toCopy),
        written,
      );

      written += toCopy;
      this.chunkOffset += toCopy;
      this.queuedSamples -= toCopy;

      if (this.chunkOffset >= current.length) {
        this.queue.shift();
        this.chunkOffset = 0;
      }
    }

    if (written < output.length) {
      if (!this.inUnderrun) {
        this.inUnderrun = true;
        this.underruns += 1;
        this.reportState("underrun");
      }
    }

    return true;
  }
}

registerProcessor("assistant-player-worklet", AssistantPlayerWorklet);
