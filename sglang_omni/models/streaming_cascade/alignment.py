# SPDX-License-Identifier: Apache-2.0
"""The alignment kind: where in the reply text the reply audio is, as `reply.progress` controls."""

import asyncio
import base64
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sglang_omni.models.streaming_cascade.base import (
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    iterate,
    probe_models,
    stop_running,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    PCM_F32,
    REPLY_PROGRESS,
    TEXT_DELTA,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)

MAX_ALIGNMENT_AUDIO_BYTES = 32 * 1024 * 1024


class AlignmentBoundary(BaseModel):
    """The first `text_end_chars` characters of the request text end `audio_end_samples` into its audio."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text_end_chars: int = Field(gt=0)
    audio_end_samples: int = Field(ge=0)


class AlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    boundaries: list[AlignmentBoundary]


class AlignmentConfig(ComponentConfig):
    """Reply progress for truncating an interrupted reply.

    `sentence` passes the synthesizer's sentence ends through; `forced` also
    asks the worker at `alignment_url` for the word ends of each sentence
    once its audio is complete.
    """

    kind: Literal["alignment"] = "alignment"
    method: Literal["sentence", "forced"] = "sentence"
    alignment_url: str = Field(default="", pattern=r"^(https?://[^/]+.*)?$")
    timeout_s: float = Field(default=2, gt=0)

    @model_validator(mode="after")
    def check_worker(self) -> "AlignmentConfig":
        if self.method == "forced" and not self.alignment_url:
            raise ValueError("the forced method needs alignment_url")
        elif self.method != "forced" and self.alignment_url:
            raise ValueError("alignment_url belongs to the forced method")
        else:
            return self


async def probe_alignment_worker(config: AlignmentConfig) -> None:
    if config.method == "forced":
        await probe_models(
            f"{config.alignment_url.rstrip('/')}/health", config.timeout_s, ""
        )
    else:
        pass


@dataclass
class SpokenReply:
    """One reply's text and audio; `start` and `start_ms` open the sentence being spoken."""

    turn: int
    text: str = ""
    start: int = 0
    start_ms: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)


@dataclass(frozen=True)
class SentenceAudio:
    """A spoken sentence of `turn` for the forced aligner: `text` begins at character `start` of the reply and its audio at `start_ms`."""

    turn: int
    text: str
    start: int
    start_ms: float
    samples: np.ndarray


class Alignment(StreamComponent):
    """Turns the reply text, the reply audio and the synthesizer's sentence ends into `reply.progress` controls.

    The synthesizer's `reply.progress` ends a sentence. A sentence cut off by
    an interruption never ends, so it reports nothing and counts as unheard.
    """

    def __init__(self, config: AlignmentConfig) -> None:
        self.config = config
        self.reply: SpokenReply | None = None
        self.client = httpx.AsyncClient(timeout=config.timeout_s)

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Alignment):
            self.reply = previous.reply
        else:
            pass

    def state_bytes(self) -> int:
        if self.reply is None:
            return 0
        else:
            return len(self.reply.text) * 4 + sum(
                samples.nbytes for samples in self.reply.samples
            )

    def reply_of(self, turn: int) -> SpokenReply:
        if self.reply is None or self.reply.turn != turn:
            self.reply = SpokenReply(turn)
        else:
            pass
        return self.reply

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        sentences: asyncio.Queue[SentenceAudio | None] = asyncio.Queue()
        worker = (
            asyncio.ensure_future(self.align_sentences(iterate(sentences), emit))
            if self.config.method == "forced"
            else None
        )
        try:
            async for chunk in inputs:
                self.handle(chunk, emit, sentences)
            if worker is not None:
                sentences.put_nowait(None)
                await worker
            else:
                pass
        finally:
            await stop_running([worker])

    def handle(
        self,
        chunk: TimedChunk,
        emit: ChunkEmitter,
        sentences: asyncio.Queue[SentenceAudio | None],
    ) -> None:
        event = control_type(chunk)
        if chunk.format == TEXT_DELTA:
            self.reply_of(chunk.turn).text += str(chunk.payload["text"])
        elif chunk.format == PCM_F32 and self.config.method == "forced":
            self.reply_of(chunk.turn).samples.append(
                np.frombuffer(chunk.payload, dtype="<f4")
            )
        elif event == REPLY_PROGRESS:
            self.end_sentence(
                self.reply_of(chunk.turn),
                int(chunk.payload["text_end"]),
                float(chunk.payload["audio_ms"]),
                emit,
                sentences,
            )
        else:
            pass

    def end_sentence(
        self,
        reply: SpokenReply,
        text_end: int,
        audio_ms: float,
        emit: ChunkEmitter,
        sentences: asyncio.Queue[SentenceAudio | None],
    ) -> None:
        span = reply.text[reply.start : text_end]
        text = span.strip()
        progress(emit, reply.turn, text_end, audio_ms)
        if self.config.method == "forced" and text and reply.samples:
            sentences.put_nowait(
                SentenceAudio(
                    reply.turn,
                    text,
                    reply.start + len(span) - len(span.lstrip()),
                    reply.start_ms,
                    np.concatenate(reply.samples),
                )
            )
        else:
            pass
        reply.start, reply.start_ms, reply.samples = text_end, audio_ms, []

    async def align_sentences(
        self, sentences: AsyncIterator[SentenceAudio], emit: ChunkEmitter
    ) -> None:
        async for sentence in sentences:
            boundaries = await self.align(sentence)
            for text_end, audio_ms in boundaries:
                progress(emit, sentence.turn, text_end, audio_ms)

    async def align(self, sentence: SentenceAudio) -> list[tuple[int, float]]:
        """The word ends the worker finds in the sentence, as reply offsets; none when it fails, which leaves the sentence's own end."""
        pcm16 = (np.clip(sentence.samples, -1, 1) * 32767).astype("<i2").tobytes()
        if len(pcm16) > MAX_ALIGNMENT_AUDIO_BYTES:
            logger.warning("sentence audio exceeds the alignment worker's limit")
            return []
        else:
            pass
        body = {
            "text": sentence.text,
            "audio": base64.b64encode(pcm16).decode(),
            "sample_rate_hz": SPEECH_SAMPLE_RATE_HZ,
            "audio_format": "pcm16",
        }
        try:
            response = await self.client.post(
                f"{self.config.alignment_url.rstrip('/')}/align", json=body
            )
        except httpx.TransportError as error:
            logger.warning("alignment worker failed: %r", error)
            return []
        if response.status_code != 200:
            logger.warning("alignment worker answered %s", response.status_code)
            return []
        else:
            pass
        try:
            result = AlignmentResult.model_validate_json(response.content)
        except ValidationError as error:
            logger.warning("alignment worker sent an invalid result: %s", error)
            return []
        ends = [boundary.text_end_chars for boundary in result.boundaries]
        samples = [boundary.audio_end_samples for boundary in result.boundaries]
        if (
            ends != sorted(ends)
            or samples != sorted(samples)
            or any(end > len(sentence.text) for end in ends)
            or any(sample > len(sentence.samples) for sample in samples)
        ):
            logger.warning("alignment worker sent boundaries outside the sentence")
            return []
        else:
            return [
                (
                    sentence.start + end,
                    sentence.start_ms + sample * 1000 / SPEECH_SAMPLE_RATE_HZ,
                )
                for end, sample in zip(ends, samples)
            ]

    async def close(self) -> None:
        await self.client.aclose()


def progress(emit: ChunkEmitter, turn: int, text_end: int, audio_ms: float) -> None:
    emit(control_chunk(REPLY_PROGRESS, turn=turn, text_end=text_end, audio_ms=audio_ms))
