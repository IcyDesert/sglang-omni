# SPDX-License-Identifier: Apache-2.0
"""Generation clients, detached Session work and a harness for QwenSessionExecution tests."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator

from sglang_omni.client.types import CompletionStreamChunk, GenerateRequest
from sglang_omni.models.qwen3_omni.session_execution import (
    TRANSCRIPTION_PROMPT,
    QwenSessionExecution,
    QwenSessionLimits,
)
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import AUDIO, INTERRUPT, PCM16, TURN_DONE, control_type
from tests.unit_test.fixtures.session_outputs import RecordedOutputs, control_chunk

TEST_TIMEOUT_S = 5


class DetachedSessionWork:
    """Session work for execution tests whose generation client never touches stages."""

    def __init__(self) -> None:
        self.session_id = "qwen-test-session"
        self.operation_timeout_s = TEST_TIMEOUT_S
        self.released = False

    def stream(
        self,
        request_id: str,
        request: OmniRequest,
        terminal_stages: set[str] | None = None,
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        raise AssertionError("detached work does not start pipeline requests")

    def owns(self, request_id: str) -> bool:
        return False

    async def abort(self, request_id: str) -> bool:
        return False

    async def release_all(self, timeout_s: float) -> None:
        self.released = True


class BlockingGenerationClient:
    """Transcribes every utterance as "user words" and replies "Hello" plus audio, then blocks until aborted."""

    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []
        self.audio_formats: list[str] = []
        self.aborted: list[str] = []
        self.generation_started = asyncio.Event()
        self.abort_started = asyncio.Event()
        self.transcription_started = asyncio.Event()
        self.release_transcription = asyncio.Event()
        self.release_transcription.set()

    async def completion_stream(
        self, request: GenerateRequest, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.requests.append(request)
        self.audio_formats.append(audio_format)
        if request.messages[0].content == TRANSCRIPTION_PROMPT:
            self.transcription_started.set()
            await self.release_transcription.wait()
            yield CompletionStreamChunk(request_id=request_id, text="user ")
            yield CompletionStreamChunk(
                request_id=request_id, text="words", finish_reason="stop"
            )
        else:
            async for reply in self.reply(request, request_id):
                yield reply

    async def reply(
        self, request: GenerateRequest, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        yield CompletionStreamChunk(request_id=request_id, text="Hello")
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="audio",
            audio_b64=base64.b64encode(bytes(20)).decode(),
        )
        self.generation_started.set()
        await asyncio.Event().wait()

    def reply_request(self) -> GenerateRequest:
        return next(
            r for r in self.requests if r.messages[0].content != TRANSCRIPTION_PROMPT
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        self.abort_started.set()


class FailedGenerationClient(BlockingGenerationClient):
    def reply(
        self, request: GenerateRequest, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        raise RuntimeError("backend generation failed")


class CompletedGenerationClient(BlockingGenerationClient):
    def __init__(self, *, has_audio: bool = True) -> None:
        super().__init__()
        self.has_audio = has_audio

    async def reply(
        self, request: GenerateRequest, request_id: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        yield CompletionStreamChunk(
            request_id=request_id, text="assistant words", finish_reason="stop"
        )
        if self.has_audio and "audio" in request.output_modalities:
            yield CompletionStreamChunk(
                request_id=request_id,
                modality="audio",
                audio_b64=base64.b64encode(bytes(96)).decode(),
                finish_reason="stop",
            )
        else:
            pass


class Harness(RecordedOutputs):
    """Drives one QwenSessionExecution and records its outputs; like the Session once a reply's audio has left, it returns each turn.done to the execution."""

    def __init__(
        self,
        client: BlockingGenerationClient,
        *,
        output_modalities: tuple[str, ...] = ("text", "audio"),
        limits: QwenSessionLimits = QwenSessionLimits(cleanup_timeout_s=1),
        returns_turn_done: bool = True,
    ) -> None:
        super().__init__()
        self.returns_turn_done = returns_turn_done
        self.work = DetachedSessionWork()
        self.execution = QwenSessionExecution(
            client,
            "qwen-test",
            self.work,
            output_modalities=output_modalities,
            limits=limits,
        )
        self.seq = 0
        self.t_ms = 0.0
        self.error: BaseException | None = None

    async def open(self, params: dict | None = None) -> None:
        await self.execution.open(OmniRequest(None, params=params or {}), self)

    async def emit_output(self, chunk: TimedChunk) -> None:
        self.outputs.append(chunk)
        if self.returns_turn_done and control_type(chunk) == TURN_DONE:
            await self.execution.control(chunk)
        else:
            pass

    def fail(self, error: BaseException) -> None:
        self.error = self.error or error

    async def append(self, chunk: TimedChunk) -> None:
        await self.execution.append(chunk)
        self.seq += 1

    async def audio(self, payload: bytes = b"\x00\x10" * 3200, *, eos=False) -> None:
        """Append `payload` (one 200 ms unit by default) as the next pcm16 input."""
        duration_ms = len(payload) / 32
        await self.append(
            TimedChunk(
                AUDIO, self.t_ms, duration_ms, self.seq, payload, format=PCM16, eos=eos
            )
        )
        self.t_ms += duration_ms

    async def control(self, kind: str, **fields: object) -> None:
        await self.append(control_chunk(kind, self.t_ms, seq=self.seq, **fields))

    async def interrupt(self, turn: int | None = None) -> None:
        await self.execution.control(control_chunk(INTERRUPT, self.t_ms, turn=turn))

    async def settle(self) -> None:
        """Wait for every running turn task."""
        await asyncio.wait_for(
            asyncio.gather(*self.execution.turn_tasks), TEST_TIMEOUT_S
        )

    async def until(self, kind: str, *, count: int = 1) -> None:
        async def reached() -> None:
            while self.labels().count(kind) < count:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(reached(), TEST_TIMEOUT_S)
