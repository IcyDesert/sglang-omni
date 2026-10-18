class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.sourceRate = config.sourceRate || sampleRate;
    this.targetRate = config.targetRate || 16000;
    this.frameSamples = config.frameSamples || 1280;
    this.input = [];
    this.sourcePosition = 0;
    this.frame = new Int16Array(this.frameSamples);
    this.frameOffset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let index = 0; index < channel.length; index += 1) this.input.push(channel[index]);
    const step = this.sourceRate / this.targetRate;
    while (this.sourcePosition <= this.input.length - 2) {
      const left = Math.floor(this.sourcePosition);
      const weight = this.sourcePosition - left;
      const sample = this.input[left] * (1 - weight) + this.input[left + 1] * weight;
      const clipped = Math.max(-1, Math.min(1, sample));
      this.frame[this.frameOffset] = clipped < 0 ? clipped * 32768 : clipped * 32767;
      this.frameOffset += 1;
      this.sourcePosition += step;
      if (this.frameOffset === this.frameSamples) {
        let energy = 0;
        for (const value of this.frame) energy += (value / 32768) ** 2;
        const frame = this.frame;
        this.port.postMessage({ type: "frame", frame, rms: Math.sqrt(energy / frame.length) }, [frame.buffer]);
        this.frame = new Int16Array(this.frameSamples);
        this.frameOffset = 0;
      }
    }
    const discard = Math.max(0, Math.floor(this.sourcePosition) - 1);
    if (discard) {
      this.input.splice(0, discard);
      this.sourcePosition -= discard;
    }
    return true;
  }
}

// Plays replies at exactly 1.0x: playback starts once initialSamples are queued (or the
// reply has ended), an underrun plays silence until the buffer refills, and queued audio
// is dropped only when a reply is interrupted (clear / cancel) or the session resets.
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.initialSamples = config.initialSamples || Math.round(sampleRate * 0.3);
    this.queue = [];
    this.headOffset = 0;
    this.queuedSamples = 0;
    this.started = false;
    this.lastSample = 0;
    this.fadeRemaining = 0;
    this.fadeLength = Math.max(1, Math.round(sampleRate * 0.01));
    this.port.onmessage = ({ data }) => {
      if (data.type === "push" && data.samples) {
        // Each chunk remembers its response so a cancel drops only that reply.
        data.samples.responseId = data.responseId;
        this.queue.push(data.samples);
        this.queuedSamples += data.samples.length;
      } else if (data.type === "flush") {
        // The reply has ended: play what is left even if it is shorter than the buffer target.
        if (this.queuedSamples > 0) this.started = true;
      } else if (data.type === "cancel") {
        this.cancel(data.responseId);
      } else if (data.type === "clear" || data.type === "reset") {
        this.drop();
      }
    };
  }

  drop() {
    this.queue = [];
    this.headOffset = 0;
    this.queuedSamples = 0;
    this.started = false;
    this.fadeRemaining = this.fadeLength;
  }

  // Drops every queued sample of one response; audio of a later response keeps its place.
  cancel(responseId) {
    if (this.queue.length && this.queue[0].responseId === responseId) {
      this.headOffset = 0;
      this.fadeRemaining = this.fadeLength;
    }
    this.queue = this.queue.filter((chunk) => chunk.responseId !== responseId);
    this.queuedSamples = this.queue.reduce((total, chunk) => total + chunk.length, 0) - this.headOffset;
    if (this.queuedSamples === 0) this.started = false;
  }

  takeSample() {
    if (this.queuedSamples === 0) return null;
    const chunk = this.queue[0];
    const value = chunk[this.headOffset];
    this.headOffset += 1;
    this.queuedSamples -= 1;
    if (this.headOffset === chunk.length) {
      this.queue.shift();
      this.headOffset = 0;
    }
    return value;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!this.started && this.queuedSamples >= this.initialSamples) this.started = true;
    for (let index = 0; index < output.length; index += 1) {
      let value = 0;
      if (this.fadeRemaining > 0) {
        value = this.lastSample * (this.fadeRemaining / this.fadeLength);
        this.fadeRemaining -= 1;
      } else if (this.started) {
        const next = this.takeSample();
        if (next === null) this.started = false;
        else value = next;
      }
      output[index] = value;
      this.lastSample = value;
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
