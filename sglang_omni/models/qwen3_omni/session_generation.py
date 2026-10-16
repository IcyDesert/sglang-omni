# SPDX-License-Identifier: Apache-2.0
"""Stream one Qwen reply as text deltas and audio with cancellation."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.qwen3_omni.session_node import OUTPUT_SAMPLE_RATE_HZ
from sglang_omni.pipeline.session_execution import AsyncChunkEmitter
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    PCM16,
    PCM16_BYTES_PER_SAMPLE,
    TEXT,
    TEXT_DELTA,
)


@dataclass(kw_only=True)
class Reply:
    """One reply in progress: its turn, its pipeline request and the text streamed so far.

    The reply stays open after its turn.done until the Session has delivered
    its audio and returns that turn.done, which sets `delivery_event`.
    """

    turn: int
    request_id: str
    text_chunks: list[str] = field(default_factory=list)
    is_cancelled: bool = False
    is_abort_scheduled: bool = False
    cancellation_event: asyncio.Event = field(default_factory=asyncio.Event)
    delivery_event: asyncio.Event = field(default_factory=asyncio.Event)


async def generate_response(
    client: Client,
    request: GenerateRequest,
    emit: AsyncChunkEmitter,
    reply: Reply,
    closing_event: asyncio.Event,
) -> None:
    should_emit_audio = "audio" in request.output_modalities
    has_audio_output = False
    async for generation_chunk in client.completion_stream(
        request, request_id=reply.request_id, audio_format="pcm"
    ):
        if reply.is_cancelled or closing_event.is_set():
            break
        else:
            pass
        if generation_chunk.text and (
            generation_chunk.modality == "text" or not reply.text_chunks
        ):
            reply.text_chunks.append(generation_chunk.text)
            await emit(
                TimedChunk(
                    TEXT,
                    0,
                    0,
                    0,
                    {"text": generation_chunk.text},
                    format=TEXT_DELTA,
                    turn=reply.turn,
                )
            )
        else:
            pass
        if (
            should_emit_audio
            and generation_chunk.modality == "audio"
            and generation_chunk.audio_b64
        ):
            has_audio_output = True
            pcm = base64.b64decode(generation_chunk.audio_b64)
            await emit(
                TimedChunk(
                    AUDIO,
                    0,
                    len(pcm) * 1000 / PCM16_BYTES_PER_SAMPLE / OUTPUT_SAMPLE_RATE_HZ,
                    0,
                    pcm,
                    format=PCM16,
                    turn=reply.turn,
                )
            )
        else:
            pass
    if reply.is_cancelled or closing_event.is_set():
        return
    elif should_emit_audio and not has_audio_output:
        raise RuntimeError("the pipeline completed the reply without audio output")
    else:
        pass
