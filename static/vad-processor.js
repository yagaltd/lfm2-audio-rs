class VADProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.hopSize = options.processorOptions?.hopSize || 256;
    this.buffer = new Int16Array(this.hopSize);
    this.bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input.length === 0) {
      return true;
    }

    const channel = input[0];
    for (let idx = 0; idx < channel.length; idx += 1) {
      const clamped = Math.max(-1, Math.min(1, channel[idx]));
      this.buffer[this.bufferIndex] = clamped * 32767;
      this.bufferIndex += 1;

      if (this.bufferIndex >= this.hopSize) {
        this.port.postMessage({
          type: "frame",
          frame: new Int16Array(this.buffer),
        });
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("vad-processor", VADProcessor);
