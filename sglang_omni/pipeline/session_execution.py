# SPDX-License-Identifier: Apache-2.0
"""Execution components owned by coordinator sessions."""

from collections.abc import AsyncIterator
from typing import Protocol

from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from sglang_omni.proto.session import TimedChunk


class AsyncChunkEmitter(Protocol):
    async def __call__(self, chunk: TimedChunk) -> None: ...


class SessionEmitter(Protocol):
    """Session output channel."""

    async def emit_output(self, chunk: TimedChunk) -> None: ...

    def fail(self, error: BaseException) -> None:
        """End the session with `error`; its output stream raises it."""
        ...


class SessionWork(Protocol):
    """Pipeline work the session owns: stage ownership and release stay in the coordinator."""

    session_id: str
    operation_timeout_s: float

    def stream(
        self,
        request_id: str,
        request: OmniRequest,
        terminal_stages: set[str] | None = None,
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        """Run one request; it ends at `terminal_stages` when given, else where the pipeline resolves it to end."""
        ...

    async def abort(self, request_id: str) -> bool: ...

    async def release_all(self, timeout_s: float) -> None: ...


class SessionExecution(Protocol):
    async def open(self, request: OmniRequest, emitter: SessionEmitter) -> None: ...

    async def append(self, chunk: TimedChunk) -> None: ...

    async def control(self, event: TimedChunk) -> None: ...

    async def close(self) -> None:
        """Stop and release all owned work; an error raised once that work is released fails this session alone."""
        ...


class SessionExecutionFactory(Protocol):
    def __call__(self, work: SessionWork) -> SessionExecution: ...
