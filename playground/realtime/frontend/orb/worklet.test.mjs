import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const workletSource = readFileSync(new URL("./worklet.js", import.meta.url), "utf8");

function createPlayback(sampleRate) {
  const statuses = [];
  const processors = new Map();
  vm.runInNewContext(workletSource, {
    sampleRate,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { postMessage: (status) => statuses.push(status) };
      }
    },
    registerProcessor: (name, processor) => processors.set(name, processor),
  });
  const processor = new (processors.get("playback-processor"))({});
  return {
    statuses,
    send: (message) => processor.port.onmessage({ data: message }),
    render: (sampleCount = 128) => {
      const output = new Float32Array(sampleCount);
      processor.process([], [[output]]);
      return output;
    },
  };
}

function replayCadence(sampleRate, intervalSeconds = 1) {
  const playback = createPlayback(sampleRate);
  playback.send({ type: "config", nativeUnitSamples: sampleRate });
  playback.send({ type: "open" });
  const gaps = [];
  let nextChunk = 0;
  let sounded = false;
  let gapSamples = 0;
  let audibleSamples = 0;
  for (let offset = 0; offset < sampleRate * 12; offset += 128) {
    if (offset >= nextChunk) {
      playback.send({ type: "push", samples: new Float32Array(sampleRate).fill(0.2), responseId: "reply" });
      nextChunk += sampleRate * intervalSeconds;
    }
    for (const sample of playback.render()) {
      if (Math.abs(sample) > 0.001) {
        audibleSamples += 1;
        sounded = true;
        if (gapSamples) gaps.push(gapSamples);
        gapSamples = 0;
      } else if (sounded) gapSamples += 1;
    }
  }
  return { gaps, audibleSamples, statuses: playback.statuses };
}

for (const sampleRate of [24000, 48000]) {
  test(`one-second native chunks stay continuous at ${sampleRate} Hz`, () => {
    const { gaps, audibleSamples, statuses } = replayCadence(sampleRate);
    const gapMilliseconds = gaps.map((samples) => Math.round(samples / sampleRate * 1000));
    assert.ok(gaps.every((samples) => samples <= 128), `interior gaps in milliseconds: ${JSON.stringify(gapMilliseconds)}`);
    assert.ok(audibleSamples >= sampleRate * 12 - 128 * 12, `only ${audibleSamples / sampleRate} audible seconds from 12 seconds of PCM`);
    assert.ok(statuses.every((status) => status.rate === 1), "nominal chunks must not trigger catch-up");
    assert.equal(statuses.at(-1).droppedMs, 0);
  });
}
