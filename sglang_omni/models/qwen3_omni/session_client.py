# SPDX-License-Identifier: Apache-2.0
"""Qwen requests constrained to their coordinator session and fixed owners."""

from collections.abc import AsyncIterator
from contextlib import aclosing

from sglang_omni.client.client import Client
from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    GenerateChunk,
    GenerateRequest,
)
from sglang_omni.pipeline.session_execution import SessionWork
from sglang_omni.proto.messages import StreamMessage


class QwenSessionClient(Client):
    def __init__(self, client: Client, work: SessionWork) -> None:
        super().__init__(
            client.coordinator, client.result_builder, client.stream_builder
        )
        self.work = work

    async def generate(
        self, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        assert request_id is not None
        iterator = self.work.stream(request_id, self.build_omni_request(request))
        async with aclosing(iterator):
            async for message in iterator:
                if isinstance(message, StreamMessage):
                    yield self.stream_builder(request_id, message)
                else:
                    yield self.result_builder(request_id, message.result)

    async def abort(
        self, request_id: str, level: AbortLevel = AbortLevel.SOFT
    ) -> AbortResult:
        return AbortResult(
            success=await self.work.abort(request_id), level_applied=level
        )
