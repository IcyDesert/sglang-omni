# SPDX-License-Identifier: Apache-2.0
"""Standard stream vocabulary and the ports a Session publishes."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.proto.session import OutputChunk, TimedChunk

AUDIO = "audio"
IMAGE = "image"
TEXT = "text"
TRANSCRIPT = "transcript"
CONTROL = "control"
TOOL = "tool"

PCM16 = "pcm16"  # bytes, signed 16-bit little-endian mono at the port's rate
PCM_F32 = "pcm_f32le"  # bytes, float32 little-endian mono at the port's rate
AUDIO_FORMATS = frozenset({PCM16, PCM_F32})
PCM16_BYTES_PER_SAMPLE = 2
SESSION_INPUT_SAMPLE_RATE_HZ = 16000
SESSION_UNIT_MS = 200
TEXT_DELTA = "text_delta"  # {"text": str}, one increment of a reply
MESSAGES = "messages"  # {"messages": [{"role", "content"}, ...]}
TRANSCRIPT_FORMAT = "transcript"  # {"text": str, "is_final": bool}
# {"calls": [{"id": str, "name": str, "arguments": object}, ...]}
TOOL_CALL = "tool_call"
# {"responses": [{"id": str, "name": str, "response": object}, ...]}
TOOL_RESPONSE = "tool_response"
SESSION_CONTROL = "session"  # {"type": <control type>, ...}

COMMIT = "commit"  # the utterance ends here
CONFIGURE = "configure"  # settings change; answered with CONFIGURED
INTERRUPT = "interrupt"  # stop the reply of the chunk's turn, or the open reply
# {"t_ms": float}; the listener heard the interrupted reply up to here
TRUNCATE = "truncate"
SPEECH_STARTED = "speech.started"  # {"t_ms": float}
SPEECH_STOPPED = "speech.stopped"  # {"t_ms": float}
TURN_QUEUED = "turn.queued"  # an utterance waits for its reply
TURN_STARTED = "turn.started"  # a reply begins
TURN_DONE = "turn.done"  # the reply and its audio are complete
TURN_EMPTY = "turn.empty"  # the utterance carried no words; no reply
TURN_INTERRUPTED = "turn.interrupted"  # the reply stopped before its end
CONFIGURED = "configured"  # {"nodes": {name: kind}}


def control_type(chunk: TimedChunk | OutputChunk) -> str | None:
    """The control type of a SESSION_CONTROL chunk; None for any other chunk and for an input confirmation, which carries no payload."""
    if chunk.format != SESSION_CONTROL or chunk.payload is None:
        return None
    elif isinstance(chunk.payload, dict) and isinstance(chunk.payload.get("type"), str):
        return chunk.payload["type"]
    else:
        raise ValueError(
            f"a {SESSION_CONTROL!r} chunk carries a string type: {chunk!r:.300}"
        )


def edge_event(chunk: TimedChunk) -> str | None:
    """The event a control edge takes a node's chunk by: the control type of a SESSION_CONTROL chunk, the format of any other, such as `tool_response`."""
    if chunk.format == SESSION_CONTROL:
        return control_type(chunk)
    else:
        return chunk.format


def is_interrupted_reply(chunk: TimedChunk | OutputChunk, turn: int | None) -> bool:
    """A chunk of the interrupted reply other than the turn.interrupted that ends it."""
    return chunk.turn == turn and control_type(chunk) != TURN_INTERRUPTED


@dataclass(frozen=True)
class AudioPort:
    format: str
    sample_rate_hz: int


@dataclass(frozen=True)
class SessionPorts:
    """What a pipeline's Session accepts and produces; `input` is a tuple containing one audio port; hot_settings are the realtime session fields a CONFIGURE control may change while it is open."""

    input: tuple[AudioPort, ...]
    output: AudioPort | None
    unit_ms: int
    controls: frozenset[str] = frozenset()
    hot_settings: frozenset[str] = frozenset()
    sampling_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        audio_ports = [port for port in self.input if isinstance(port, AudioPort)]
        if len(audio_ports) != 1:
            raise ValueError(
                f"a Session takes one audio input port, got {len(audio_ports)}"
            )
        else:
            pass

    @property
    def audio_input(self) -> AudioPort:
        return next(port for port in self.input if isinstance(port, AudioPort))

    @property
    def input_modalities(self) -> tuple[str, ...]:
        return (AUDIO,)
