# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

from collections import deque
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    messages_chunk,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    INTERRUPT,
    MAX_HISTORY_CHARS,
    REPLY_PROGRESS,
    SPEECH_STARTED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter


class DialogConfig(ComponentConfig):
    """Conversation history; one messages chunk per answered utterance."""

    kind: Literal["dialog"] = "dialog"
    system_prompt: str = "You are a helpful voice assistant. Answer concisely."
    max_history_chars: int = Field(default=MAX_HISTORY_CHARS, gt=0)
    max_text_chars: int = Field(default=16000, gt=0)


class Dialog(StreamComponent):
    """Turns final transcripts into prompts and records the replies.

    Each final transcript opens the next turn with turn.queued. An utterance
    that arrives while a reply is running waits for the turn.done of that
    reply's turn, so history always alternates user and assistant. Speech
    over a reply, or an `interrupt` control, ends the reply: the dialog
    emits `interrupt` for the reply's turn to the nodes that take it as a
    control and publishes `turn.interrupted`; `reply.progress` controls from
    the synthesizer tell it how much of the reply was spoken. A replacement
    adopts the history, the open reply and the waiting utterances, so new
    settings apply at once and reach the next prompt.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        self.waiting: deque[tuple[int, str]] = deque()
        self.reply: list[str] | None = None
        self.reply_turn = 0
        self.turns = 0
        # Reply audio produced so far as (words spoken, audio ms), for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # Note (Junnan Li): The ended reply a truncate may still cut, as (turn, history index), until the next reply begins.
        self.truncatable: tuple[int, int] | None = None

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.waiting = previous.waiting
            self.reply = previous.reply
            self.reply_turn = previous.reply_turn
            self.turns = previous.turns
            self.progress = previous.progress
            self.truncatable = previous.truncatable
        else:
            pass

    def state_bytes(self) -> int:
        chars = sum(len(message.content) for message in self.messages)
        chars += sum(len(text) for _, text in self.waiting)
        chars += sum(len(part) for part in self.reply or ())
        return chars * 4

    def is_reply(self, chunk: TimedChunk) -> bool:
        return self.reply is not None and chunk.turn == self.reply_turn

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        async for chunk in inputs:
            self.handle(chunk)
            self.respond_to_waiting()

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT and self.reply is not None:
            self.interrupt_reply()
            self.respond_to_waiting()
        elif control_type(event) == REPLY_PROGRESS and self.is_reply(event):
            self.progress.append(
                (int(event.payload["words"]), float(event.payload["audio_ms"]))
            )
        else:
            pass

    def handle(self, chunk: TimedChunk) -> None:
        assert self.emit is not None
        if chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
            self.turns += 1
            self.waiting.append((self.turns, str(chunk.payload["text"]).strip()))
            self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
        elif chunk.format == TEXT_DELTA and self.is_reply(chunk):
            self.reply.append(str(chunk.payload["text"]))
            if sum(len(part) for part in self.reply) > self.config.max_text_chars:
                raise ValueError("reply exceeds max_text_chars")
            else:
                pass
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            reply = "".join(self.reply)
            self.messages.append(ChatMessage(role="assistant", content=reply))
            self.truncatable = (self.reply_turn, len(self.messages) - 1)
            self.reply = None
        elif control_type(chunk) == SPEECH_STARTED and self.reply is not None:
            self.interrupt_reply()
        elif control_type(chunk) == TRUNCATE and self.truncatable is not None:
            self.truncate(float(chunk.payload["t_ms"]))
        else:
            pass

    def interrupt_reply(self) -> None:
        """End the open reply; the history keeps what was generated until a truncate says what was heard."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        self.messages.append(ChatMessage(role="assistant", content="".join(self.reply)))
        self.truncatable = (turn, len(self.messages) - 1)
        self.reply = None
        self.emit(control_chunk(INTERRUPT, turn=turn))
        self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))

    def respond_to_waiting(self) -> None:
        assert self.emit is not None
        while self.reply is None and self.waiting:
            for output in self.respond(*self.waiting.popleft()):
                self.emit(output)

    def truncate(self, heard_ms: float) -> None:
        """Cut the ended reply in the history to the words whose audio the listener heard."""
        assert self.truncatable is not None
        _, index = self.truncatable
        words = self.messages[index].content.split()
        heard_words = max(
            (count for count, audio_ms in self.progress if audio_ms <= heard_ms),
            default=0,
        )
        self.messages[index] = ChatMessage(
            role="assistant", content=" ".join(words[:heard_words])
        )

    def respond(self, turn: int, text: str) -> list[TimedChunk]:
        history_chars = sum(len(message.content) for message in self.messages)
        if not text:
            return [control_chunk(TURN_EMPTY, turn=turn)]
        elif len(text) > self.config.max_text_chars:
            raise ValueError("transcript exceeds max_text_chars")
        elif (
            history_chars + len(text) + self.config.max_text_chars
            > self.config.max_history_chars
        ):
            raise ValueError(
                "conversation exceeds max_history_chars; open a new session"
            )
        else:
            self.messages.append(ChatMessage(role="user", content=text))
            self.reply = []
            self.reply_turn = turn
            self.progress = []
            self.truncatable = None
            return [
                control_chunk(TURN_STARTED, turn=turn),
                messages_chunk(self.messages, turn),
            ]
