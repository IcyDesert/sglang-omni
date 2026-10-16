# SPDX-License-Identifier: Apache-2.0
"""Generator kinds: reply text from the conversation."""

from collections.abc import AsyncIterator
from typing import Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    ComponentConfig,
    Replier,
    Replies,
    bearer_headers,
    control_chunk,
    message_dict,
    messages_of,
    replying,
    text_chunk,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import MESSAGES, TURN_DONE
from sglang_omni.scheduling.session import ChunkEmitter

# The request fields this kind sets itself; `extra_body` adds others.
CHAT_REQUEST_FIELDS = frozenset(
    {"model", "messages", "stream", "max_tokens", "temperature"}
)


class OpenAIChatGeneratorConfig(ComponentConfig):
    kind: Literal["openai_chat"] = "openai_chat"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    max_tokens: int = Field(default=512, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    api_key_env: str = "OPENAI_API_KEY"
    extra_body: dict[str, object] = Field(default_factory=dict)

    @field_validator("extra_body")
    @classmethod
    def check_extra_body(cls, extra_body: dict[str, object]) -> dict[str, object]:
        if CHAT_REQUEST_FIELDS & set(extra_body):
            raise ValueError(
                f"extra_body cannot set {sorted(CHAT_REQUEST_FIELDS & set(extra_body))}"
            )
        else:
            return extra_body


class ChatDelta(BaseModel):
    content: str | None = None


class ChatChoice(BaseModel):
    delta: ChatDelta
    finish_reason: str | None = None


class ChatEvent(BaseModel):
    choices: list[ChatChoice]


class OpenAIChatGenerator(Replier):
    """One streamed completion per messages chunk, closed by turn.done."""

    def __init__(self, config: OpenAIChatGeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:

        async def handle(chunk: TimedChunk, replies: Replies) -> None:
            if chunk.format == MESSAGES:
                await replies.finish()
                replies.start(
                    chunk.turn, self.reply(messages_of(chunk), chunk.turn, emit)
                )
            else:
                pass

        await replying(inputs, handle, self.replies)

    async def reply(
        self, messages: list[ChatMessage], turn: int, emit: ChunkEmitter
    ) -> None:
        async for delta in self.complete(messages):
            if delta.content:
                emit(text_chunk(delta.content, turn))
            else:
                pass
        emit(control_chunk(TURN_DONE, turn=turn))

    def request_body(self, messages: list[ChatMessage]) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.config.model,
            "messages": [message_dict(message) for message in messages],
            "stream": True,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        else:
            pass
        return {**body, **self.config.extra_body}

    async def complete(self, messages: list[ChatMessage]) -> AsyncIterator[ChatDelta]:
        finished = False
        async with self.client.stream(
            "POST",
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            json=self.request_body(messages),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                else:
                    pass
                encoded = line[5:].strip()
                if encoded == "[DONE]":
                    if not finished:
                        raise RuntimeError("LLM ended without a stop finish reason")
                    else:
                        break
                else:
                    pass
                event = ChatEvent.model_validate_json(encoded)
                for choice in event.choices:
                    if choice.finish_reason is None:
                        pass
                    elif choice.finish_reason not in ("stop", "length"):
                        raise RuntimeError(f"LLM stopped with {choice.finish_reason}")
                    else:
                        finished = True
                    yield choice.delta
            if not finished:
                raise RuntimeError("LLM stream ended before completion")
            else:
                pass

    async def close(self) -> None:
        await self.client.aclose()
