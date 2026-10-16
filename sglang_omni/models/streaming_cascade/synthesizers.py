# SPDX-License-Identifier: Apache-2.0
"""Synthesizer kinds: reply audio from the reply text."""

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import numpy as np
from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    Replier,
    Replies,
    audio_chunk,
    bearer_headers,
    control_chunk,
    iterate,
    replying,
    sentences,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    REPLY_PROGRESS,
    TEXT_DELTA,
    TURN_DONE,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter


class OpenAISpeechSynthesizerConfig(ComponentConfig):
    """Sentence-chunked `/v1/audio/speech` streaming PCM, one request per sentence."""

    kind: Literal["openai_speech"] = "openai_speech"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    voice: str = "default"
    reference_audio_path: str = ""
    reference_text: str = ""
    language: str = ""
    max_sentence_words: int = Field(default=24, gt=0)
    api_key_env: str = "OPENAI_API_KEY"


class TurnSpeaker(Replier):
    """Speaks each reply's text deltas as they stream and ends the turn after its audio.

    `speak_reply` turns one reply's deltas into emitted audio; an `interrupt`
    control of that turn cancels it and the rest of the reply's text is dropped.
    """

    async def speak_reply(
        self, deltas: AsyncIterator[TimedChunk], turn: int, emit: ChunkEmitter
    ) -> None:
        raise NotImplementedError

    def progress(
        self, emit: ChunkEmitter, turn: int, words: int, audio_ms: float
    ) -> None:
        """Tell the dialog how much of the reply has audio, so a truncate can cut the history where the listener stopped."""
        emit(control_chunk(REPLY_PROGRESS, turn=turn, words=words, audio_ms=audio_ms))

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        deltas: asyncio.Queue[TimedChunk | None] = asyncio.Queue()

        async def speak_then_end(turn: int) -> None:
            await self.speak_reply(iterate(deltas), turn, emit)
            emit(control_chunk(TURN_DONE, turn=turn))

        async def handle(chunk: TimedChunk, replies: Replies) -> None:
            nonlocal deltas
            if chunk.format == TEXT_DELTA and chunk.turn in replies.interrupted:
                # Note (Junnan Li): The generator may emit a few deltas after the interrupt before it stops.
                pass
            elif chunk.format == TEXT_DELTA:
                if not replies.is_running(chunk.turn):
                    await replies.finish()
                    deltas = asyncio.Queue()
                    replies.start(chunk.turn, speak_then_end(chunk.turn))
                else:
                    pass
                deltas.put_nowait(chunk)
            elif control_type(chunk) == TURN_DONE:
                if replies.is_running(chunk.turn):
                    deltas.put_nowait(None)
                elif chunk.turn in replies.interrupted:
                    replies.interrupted.discard(chunk.turn)
                else:
                    emit(control_chunk(TURN_DONE, turn=chunk.turn))
            else:
                pass

        await replying(inputs, handle, self.replies)


class OpenAISpeechSynthesizer(TurnSpeaker):
    """Speaks each reply sentence by sentence; turn.done follows the reply's audio."""

    sample_rate_hz = SPEECH_SAMPLE_RATE_HZ

    def __init__(self, config: OpenAISpeechSynthesizerConfig) -> None:
        super().__init__()
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )

    async def speak(self, sentence: str) -> AsyncIterator[np.ndarray]:
        body: dict[str, object] = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": sentence,
            "response_format": "pcm",
            "stream": True,
        }
        if self.config.language:
            # Note (Junnan Li): A sentence with CJK characters is spoken as Chinese whatever the configured language, so a reply in the user's language keeps the right tag.
            body["language"] = (
                "Chinese"
                if any("\u4e00" <= char <= "\u9fff" for char in sentence)
                else self.config.language
            )
        else:
            pass
        if self.config.reference_audio_path:
            body["references"] = [
                {
                    "audio_path": self.config.reference_audio_path,
                    **(
                        {"text": self.config.reference_text}
                        if self.config.reference_text
                        else {}
                    ),
                }
            ]
        else:
            pass
        async with self.client.stream(
            "POST", f"{self.config.base_url.rstrip('/')}/audio/speech", json=body
        ) as response:
            response.raise_for_status()
            rate = response.headers.get("X-Sample-Rate")
            if rate is None:
                raise ValueError(
                    "speech service sent no X-Sample-Rate header, so its PCM rate is unknown"
                )
            elif int(rate) != self.sample_rate_hz:
                raise ValueError(
                    f"speech service streams {rate} Hz, expected {self.sample_rate_hz}"
                )
            else:
                pass
            carry = b""
            async for data in response.aiter_bytes():
                data = carry + data
                usable = len(data) - len(data) % 2
                carry = data[usable:]
                if usable:
                    yield np.frombuffer(data[:usable], dtype="<i2").astype(
                        "<f4"
                    ) / 32768
                else:
                    pass

    async def speak_reply(
        self, deltas: AsyncIterator[TimedChunk], turn: int, emit: ChunkEmitter
    ) -> None:
        words_spoken = 0
        audio_ms = 0.0
        async for sentence in sentences(deltas, self.config.max_sentence_words):
            async for samples in self.speak(sentence):
                audio_ms += len(samples) * 1000 / self.sample_rate_hz
                emit(audio_chunk(samples, self.sample_rate_hz, turn))
            words_spoken += len(sentence.split())
            self.progress(emit, turn, words_spoken, audio_ms)

    async def close(self) -> None:
        await self.client.aclose()
