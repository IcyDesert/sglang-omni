# SPDX-License-Identifier: Apache-2.0
"""Session-owned Qwen3-Omni turns: utterances, generation, transcription, history and cleanup."""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.audio.segmentation import (
    EMPTY_UTTERANCE,
    FINISH_UTTERANCE,
    OPEN_UTTERANCE,
    UtteranceSegmenter,
)
from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.qwen3_omni.session_client import QwenSessionClient
from sglang_omni.models.qwen3_omni.session_generation import Reply, generate_response
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.session_execution import (
    SessionEmitter,
    SessionExecutionFactory,
    SessionWork,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONTROL,
    IMAGE_FRAME,
    INTERRUPT,
    MAX_HISTORY_CHARS,
    PCM16,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TRANSCRIPT,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)

INPUT_BYTES_PER_SECOND = PCM16_BYTES_PER_SAMPLE * SESSION_INPUT_SAMPLE_RATE_HZ
DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime voice assistant. Respond conversationally."
)
# Transcription must be verbatim regardless of the session instructions.
TRANSCRIPTION_PROMPT = (
    "You are a speech-to-text engine. Transcribe the user's spoken audio "
    "verbatim into the same language they spoke. Output ONLY the transcript "
    "— no descriptions, no refusals, no explanations."
)
TRANSCRIPTION_INSTRUCTION = "Transcribe the spoken audio."
RESPONSE_INSTRUCTION = "Listen to the spoken audio above and respond to it."
SEEING_RESPONSE_INSTRUCTION = (
    "The images are frames from the user's camera while they spoke. "
    "Listen to the spoken audio above and respond to it."
)


@dataclass(frozen=True)
class QwenSessionLimits:
    max_utterance_s: float = 60
    max_history_chars: int = MAX_HISTORY_CHARS
    max_queued_turns: int = 8
    cleanup_timeout_s: float = 30
    # The newest camera frames an utterance keeps; MiniCPM-o accepts as many per second of audio.
    max_utterance_frames: int = 4


@dataclass
class ConversationItem:
    role: str
    text: str
    turn: int


def wav_payload(pcm: bytes) -> str:
    wav = pcm16_wav(pcm, SESSION_INPUT_SAMPLE_RATE_HZ)
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def control_chunk(control_type: str, turn: int) -> TimedChunk:
    return TimedChunk(
        CONTROL, 0, 0, 0, {"type": control_type}, format=SESSION_CONTROL, turn=turn
    )


def transcript_chunk(text: str, is_final: bool) -> TimedChunk:
    return TimedChunk(
        TRANSCRIPT,
        0,
        0,
        0,
        {"text": text, "is_final": is_final},
        format=TRANSCRIPT_FORMAT,
    )


async def cancel_tasks(tasks: Iterable[asyncio.Task[None]], timeout_s: float) -> None:
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=timeout_s)
    else:
        pass


class QwenSessionExecution:
    def __init__(
        self,
        client: Client,
        model_name: str,
        work: SessionWork,
        *,
        output_modalities: tuple[str, ...],
        limits: QwenSessionLimits = QwenSessionLimits(),
    ) -> None:
        self.client: Client = client
        self.model_name: str = model_name
        self.work: SessionWork = work
        self.session_id: str = work.session_id
        self.output_modalities = output_modalities
        self.limits = limits
        self.instructions = DEFAULT_INSTRUCTIONS
        self.sampling: dict[str, object] = {}
        self.conversation: list[ConversationItem] = []
        self.segmenter = UtteranceSegmenter()
        self.utterance: bytearray | None = None
        self.utterance_start_ms: float | None = None
        # (t_ms, frame), the newest within the utterance bound.
        self.frames: deque[tuple[float, bytes]] = deque(
            maxlen=limits.max_utterance_frames
        )
        self.waiting: deque[tuple[int, bytes, tuple[bytes, ...]]] = deque()
        self.turns = 0
        self.reply: Reply | None = None
        self.interrupted_turn: int | None = None
        # A truncate may arrive before its interrupted turn is recorded; recording then leaves the reply out.
        self.truncated_turn: int | None = None
        self.response_tasks: dict[str, asyncio.Task[None]] = {}
        self.transcription_request_ids: set[str] = set()
        self.turn_tasks: set[asyncio.Task[None]] = set()
        self.abort_tasks: set[asyncio.Task[None]] = set()
        self.cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self.closing_event: asyncio.Event = asyncio.Event()
        self.session_emitter: SessionEmitter | None = None
        self.input_seq = 0
        self.input_end_ms = 0.0
        self.output_end_ms = 0.0
        self.failure: BaseException | None = None

    async def open(self, request: OmniRequest, emitter: SessionEmitter) -> None:
        self.session_emitter = emitter
        self.configure(request.params)

    def configure(self, settings: Mapping[str, object]) -> None:
        instructions = settings.get("instructions")
        sampling = settings.get("sampling")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError(
                f"instructions must be a string, got {type(instructions).__name__}"
            )
        elif sampling is not None and not isinstance(sampling, Mapping):
            raise ValueError(
                f"sampling must map parameter names to values, got {type(sampling).__name__}"
            )
        else:
            pass
        if instructions is not None:
            self.instructions = instructions
        else:
            pass
        if sampling is not None:
            self.sampling = dict(sampling)
        else:
            pass

    async def emit_output(self, chunk: TimedChunk) -> None:
        """Place a chunk on the session timelines: reply audio runs on the output timeline, everything else at the input heard so far."""
        assert self.session_emitter is not None
        if chunk.modality == AUDIO:
            t_start_ms = self.output_end_ms
            self.output_end_ms += chunk.duration_ms
        elif control_type(chunk) == TURN_STARTED:
            self.output_end_ms = max(self.output_end_ms, self.input_end_ms)
            t_start_ms = self.output_end_ms
        else:
            t_start_ms = self.input_end_ms
        await self.session_emitter.emit_output(
            TimedChunk(
                chunk.modality,
                t_start_ms,
                chunk.duration_ms,
                self.input_seq,
                chunk.payload,
                format=chunk.format,
                turn=chunk.turn,
            )
        )

    async def append(self, chunk: TimedChunk) -> None:
        if self.failure is not None:
            raise self.failure
        else:
            pass
        self.input_seq = chunk.seq
        chunk_control_type = control_type(chunk)
        if chunk.modality == AUDIO:
            if chunk.format != PCM16 or not isinstance(chunk.payload, bytes):
                raise ValueError("audio must be 16 kHz mono pcm16 bytes")
            else:
                pass
            self.input_end_ms = chunk.t_start_ms + chunk.duration_ms
            for action in self.segmenter.step(chunk):
                await self.act(action)
        elif chunk.format == IMAGE_FRAME:
            if not isinstance(chunk.payload, bytes):
                raise ValueError("a camera frame must be JPEG or PNG bytes")
            else:
                self.frames.append((chunk.t_start_ms, chunk.payload))
        elif chunk_control_type in (SPEECH_STARTED, SPEECH_STOPPED, COMMIT):
            if chunk_control_type == SPEECH_STARTED and self.reply is not None:
                self.interrupt(self.reply)
            else:
                pass
            for action in self.segmenter.step(chunk):
                await self.act(action)
        elif chunk_control_type == CONFIGURE:
            self.configure({k: v for k, v in chunk.payload.items() if k != "type"})
        elif chunk_control_type == TRUNCATE:
            self.truncate()
        elif chunk_control_type is None:
            raise ValueError(
                "expected pcm16 audio, a camera frame or a session control"
            )
        else:
            pass
        if chunk.eos:
            for action in self.segmenter.finish():
                await self.act(action)
            await self.drain()
        else:
            pass

    async def control(self, event: TimedChunk) -> None:
        if self.reply is None:
            pass
        elif control_type(event) == INTERRUPT and event.turn in (None, self.reply.turn):
            self.interrupt(self.reply)
        elif control_type(event) == TURN_DONE and event.turn == self.reply.turn:
            self.reply.delivery_event.set()
        else:
            pass

    async def act(self, action: TimedChunk | str) -> None:
        if action == OPEN_UTTERANCE:
            self.utterance = bytearray()
            self.utterance_start_ms = None
        elif action == EMPTY_UTTERANCE:
            self.turns += 1
            await self.emit_output(transcript_chunk("", True))
            await self.emit_output(control_chunk(TURN_QUEUED, turn=self.turns))
            await self.emit_output(control_chunk(TURN_EMPTY, turn=self.turns))
        elif self.utterance is None:
            raise RuntimeError(f"qwen_omni received {action!r} outside an utterance")
        elif action == FINISH_UTTERANCE:
            utterance = bytes(self.utterance)
            self.utterance = None
            frames = tuple(
                frame
                for t_ms, frame in self.frames
                if self.utterance_start_ms is not None
                and t_ms >= self.utterance_start_ms
            )
            self.frames.clear()
            if len(self.waiting) >= self.limits.max_queued_turns:
                raise ValueError("queued turn limit exhausted")
            else:
                pass
            self.turns += 1
            await self.emit_output(control_chunk(TURN_QUEUED, turn=self.turns))
            self.waiting.append((self.turns, utterance, frames))
            self.start_next_turn()
        else:
            assert isinstance(action.payload, bytes)
            if self.utterance_start_ms is None:
                self.utterance_start_ms = action.t_start_ms
            else:
                pass
            self.utterance.extend(action.payload)
            if (
                len(self.utterance)
                > self.limits.max_utterance_s * INPUT_BYTES_PER_SECOND
            ):
                raise ValueError("utterance exceeds max_utterance_s")
            else:
                pass

    def start_next_turn(self) -> None:
        """Start the oldest waiting utterance once the previous turn has been recorded."""
        if self.reply is not None or not self.waiting or self.closing_event.is_set():
            return
        else:
            pass
        turn, utterance, frames = self.waiting.popleft()
        self.reply = Reply(
            turn=turn, request_id=f"rt-{self.session_id}-{uuid.uuid4().hex}"
        )
        turn_task = asyncio.create_task(self.run_turn(self.reply, utterance, frames))
        self.turn_tasks.add(turn_task)

        def turn_exited(task: asyncio.Task[None]) -> None:
            self.turn_tasks.discard(task)
            if task.cancelled():
                pass
            elif task.exception() is not None:
                assert self.session_emitter is not None
                self.session_emitter.fail(task.exception())
            else:
                pass

        turn_task.add_done_callback(turn_exited)

    def build_response_request(
        self,
        audio_payload: str,
        frames: tuple[bytes, ...],
        modalities: tuple[str, ...],
    ) -> GenerateRequest:
        """The reply sees the utterance's camera frames as images before its audio."""
        messages = [Message(role="system", content=self.instructions)]
        messages.extend(
            Message(role=entry.role, content=entry.text) for entry in self.conversation
        )
        messages.append(
            Message(
                role="user",
                content=SEEING_RESPONSE_INSTRUCTION if frames else RESPONSE_INSTRUCTION,
            )
        )
        if frames:
            metadata = {
                "audios": [audio_payload],
                "images": [
                    "data:"
                    + ("image/png" if frame.startswith(b"\x89PNG") else "image/jpeg")
                    + ";base64,"
                    + base64.b64encode(frame).decode("ascii")
                    for frame in frames
                ],
            }
        else:
            metadata = {"audios": [audio_payload]}
        return GenerateRequest(
            model=self.model_name,
            messages=messages,
            sampling=self.build_sampling_parameters(),
            stream=True,
            output_modalities=list(modalities),
            metadata=metadata,
        )

    def build_transcription_request(self, audio_payload: str) -> GenerateRequest:
        """The instruction precedes the audio and decoding is greedy, so the model transcribes instead of answering."""
        return GenerateRequest(
            model=self.model_name,
            messages=[
                Message(role="system", content=TRANSCRIPTION_PROMPT),
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": TRANSCRIPTION_INSTRUCTION},
                        {"type": "audio_url", "audio_url": {"url": audio_payload}},
                    ],
                ),
            ],
            sampling=replace(self.build_sampling_parameters(), temperature=0.0),
            stream=True,
            output_modalities=["text"],
        )

    def build_sampling_parameters(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.sampling.get("temperature", 0.8),
            top_p=1.0,
            max_new_tokens=self.sampling.get("max_new_tokens_per_unit"),
        )

    async def run_turn(
        self, reply: Reply, utterance: bytes, frames: tuple[bytes, ...]
    ) -> None:
        """Answer one utterance from its audio while transcribing it; turn.done follows both, and the reply stays interruptible until the Session returns that turn.done once its audio has left."""
        audio_payload = wav_payload(utterance)
        await self.emit_output(control_chunk(TURN_STARTED, turn=reply.turn))
        transcription_task = asyncio.create_task(self.transcribe(audio_payload))
        generation_task = asyncio.create_task(
            generate_response(
                self.client,
                self.build_response_request(
                    audio_payload, frames, self.output_modalities
                ),
                self.emit_output,
                reply,
                self.closing_event,
            )
        )
        self.response_tasks[reply.request_id] = generation_task

        def generation_exited(task: asyncio.Task[None]) -> None:
            self.response_tasks.pop(reply.request_id, None)
            if not task.cancelled():
                task.exception()
            else:
                pass

        generation_task.add_done_callback(generation_exited)
        cancellation_task = asyncio.create_task(reply.cancellation_event.wait())
        delivery_task = asyncio.create_task(reply.delivery_event.wait())
        try:
            await asyncio.wait(
                [generation_task, cancellation_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not reply.is_cancelled:
                generation_task.result()
                await transcription_task
                await self.emit_output(control_chunk(TURN_DONE, turn=reply.turn))
                await asyncio.wait(
                    [delivery_task, cancellation_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                pass
            if reply.is_cancelled and not reply.delivery_event.is_set():
                self.interrupted_turn = reply.turn
                await self.emit_output(control_chunk(TURN_INTERRUPTED, turn=reply.turn))
            else:
                pass
            self.record_turn(reply, await transcription_task)
        except BaseException:
            await cancel_tasks(
                [generation_task, transcription_task], self.limits.cleanup_timeout_s
            )
            raise
        finally:
            cancellation_task.cancel()
            delivery_task.cancel()
            await asyncio.wait({cancellation_task, delivery_task})
            self.reply = None
            self.start_next_turn()

    def record_turn(self, reply: Reply, transcript: str) -> None:
        """Both sides of a finished turn enter the history, within its limit."""
        assistant_text = "".join(reply.text_chunks)
        if transcript:
            self.conversation.append(ConversationItem("user", transcript, reply.turn))
        else:
            pass
        if assistant_text and reply.turn != self.truncated_turn:
            self.conversation.append(
                ConversationItem("assistant", assistant_text, reply.turn)
            )
        else:
            pass
        if (
            sum(len(entry.text) for entry in self.conversation)
            > self.limits.max_history_chars
        ):
            raise ContextExhaustedError(
                f"{ContextExhaustedError.CODE}: conversation history limit exhausted"
            )
        else:
            pass

    async def transcribe(self, audio_payload: str) -> str:
        request_id = f"rt-{self.session_id}-{uuid.uuid4().hex}"
        self.transcription_request_ids.add(request_id)
        transcription_text_chunks: list[str] = []
        transcription_character_count = 0
        try:
            async for chunk in self.client.completion_stream(
                self.build_transcription_request(audio_payload), request_id=request_id
            ):
                if chunk.modality == "text" and chunk.text:
                    transcription_character_count += len(chunk.text)
                    if transcription_character_count > self.limits.max_history_chars:
                        self.schedule_abort(request_id)
                        raise ContextExhaustedError(
                            f"{ContextExhaustedError.CODE}: user transcription limit exhausted"
                        )
                    else:
                        pass
                    transcription_text_chunks.append(chunk.text)
                    await self.emit_output(transcript_chunk(chunk.text, False))
                else:
                    pass
                if chunk.finish_reason is not None:
                    break
                else:
                    pass
            transcript = "".join(transcription_text_chunks).strip()
            await self.emit_output(transcript_chunk(transcript, True))
            return transcript
        finally:
            self.transcription_request_ids.discard(request_id)

    def truncate(self) -> None:
        """Forget what the interrupted reply said; the listener did not hear it to the end."""
        if self.interrupted_turn is None:
            return
        else:
            pass
        self.truncated_turn = self.interrupted_turn
        self.conversation[:] = [
            entry
            for entry in self.conversation
            if entry.turn != self.interrupted_turn or entry.role != "assistant"
        ]
        self.interrupted_turn = None

    def schedule_abort(self, request_id: str) -> None:
        if request_id in self.cleanup_tasks:
            return
        elif len(self.cleanup_tasks) >= self.limits.max_queued_turns + 1:
            raise RuntimeError("pending backend abort limit exhausted")
        else:
            pass
        abort_task = asyncio.create_task(self.abort(request_id))
        self.abort_tasks.add(abort_task)

        def abort_exited(task: asyncio.Task[None]) -> None:
            self.abort_tasks.discard(task)
            if not task.cancelled():
                task.exception()
            else:
                pass

        abort_task.add_done_callback(abort_exited)
        cleanup_task = asyncio.create_task(
            self.supervise_cleanup(
                request_id,
                abort_task,
                asyncio.get_running_loop().time() + self.limits.cleanup_timeout_s,
            )
        )
        self.cleanup_tasks[request_id] = cleanup_task

        def cleanup_exited(task: asyncio.Task[None]) -> None:
            self.cleanup_tasks.pop(request_id, None)
            if not task.cancelled():
                task.exception()
            else:
                pass

        cleanup_task.add_done_callback(cleanup_exited)

    async def abort(self, request_id: str) -> None:
        await self.client.abort(request_id)

    async def supervise_cleanup(
        self,
        request_id: str,
        abort_task: asyncio.Task[None],
        cleanup_deadline_s: float,
    ) -> None:
        """An abort that fails or outlives its deadline leaves backend work behind; that fails the session."""
        reader_task = self.response_tasks.get(request_id)
        cleanup_tasks = [abort_task] + (
            [reader_task] if reader_task is not None else []
        )
        try:
            finished_tasks, unfinished_tasks = await asyncio.wait(
                cleanup_tasks,
                timeout=max(0, cleanup_deadline_s - asyncio.get_running_loop().time()),
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in finished_tasks:
                if not task.cancelled():
                    task.result()
                else:
                    pass
            if unfinished_tasks:
                raise TimeoutError("backend response cleanup timed out")
            else:
                pass
        except Exception as exc:
            self.failure = self.failure or exc

    def interrupt(self, reply: Reply) -> None:
        if reply.is_abort_scheduled:
            return
        else:
            reply.is_abort_scheduled = True
        reply.is_cancelled = True
        generation_task = self.response_tasks.get(reply.request_id)
        if generation_task is not None and not generation_task.done():
            generation_task.cancel()
        else:
            pass
        self.schedule_abort(reply.request_id)
        reply.cancellation_event.set()

    async def drain(self) -> None:
        """Stay in the append that ended the input until every reply is out."""
        deadline_s = asyncio.get_running_loop().time() + self.work.operation_timeout_s
        while self.turn_tasks:
            _, unfinished = await asyncio.wait(
                list(self.turn_tasks),
                timeout=max(0, deadline_s - asyncio.get_running_loop().time()),
            )
            if unfinished:
                raise TimeoutError("the replies did not finish after the input ended")
            else:
                pass
        if self.failure is not None:
            raise self.failure
        else:
            pass

    async def close(self) -> None:
        self.closing_event.set()
        cleanup_deadline_s = (
            asyncio.get_running_loop().time() + self.limits.cleanup_timeout_s
        )
        for request_id, generation_task in list(self.response_tasks.items()):
            generation_task.cancel()
            self.schedule_abort(request_id)
        for request_id in list(self.transcription_request_ids):
            self.schedule_abort(request_id)
        for turn_task in list(self.turn_tasks):
            turn_task.cancel()
        cleanup_timed_out = False
        if self.cleanup_tasks:
            _, unfinished_cleanup_tasks = await asyncio.wait(
                list(self.cleanup_tasks.values()),
                timeout=max(0, cleanup_deadline_s - asyncio.get_running_loop().time()),
            )
            cleanup_timed_out = bool(unfinished_cleanup_tasks)
        else:
            pass
        await cancel_tasks(
            [
                *self.turn_tasks,
                *self.response_tasks.values(),
                *self.abort_tasks,
                *self.cleanup_tasks.values(),
            ],
            max(0, cleanup_deadline_s - asyncio.get_running_loop().time()),
        )
        self.turn_tasks.clear()
        self.response_tasks.clear()
        self.abort_tasks.clear()
        self.cleanup_tasks.clear()
        await self.work.release_all(
            max(0, cleanup_deadline_s - asyncio.get_running_loop().time())
        )
        if cleanup_timed_out:
            raise TimeoutError("backend response cleanup timed out during close")
        else:
            pass


def create_session_execution_factory(
    coordinator: Coordinator, config: PipelineConfig
) -> SessionExecutionFactory:
    """The Qwen3-Omni node of a Session graph; the pipeline's stages run its requests."""
    client = Client(coordinator)
    output_modalities = (
        ("text", "audio") if type(config).code2wav_stage() is not None else ("text",)
    )

    def factory(work: SessionWork) -> QwenSessionExecution:
        return QwenSessionExecution(
            QwenSessionClient(client, work),
            config.name,
            work,
            output_modalities=output_modalities,
        )

    return factory
