# SPDX-License-Identifier: Apache-2.0
"""Session graph edges: bounds, order, interruption fencing and control delivery, over in-process execution nodes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.session_execution import SessionEmitter
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    ControlEdge,
    DataEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    SessionNode,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    INTERRUPT,
    TOOL,
    TOOL_CALL,
    TOOL_RESPONSE,
    control_type,
)
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import pipeline, wait_until

OUTPUT_TIMEOUT_S = 5
APPEND_RETRY_S = 0.01
# Every edge holds one chunk and every node session one pending append, so a held node backs its edges up at once.
LIMITS = SessionLimits(max_pending_chunks=1)
SPEECH_STARTED = {"text": "speech_started"}
# The coordinator hands a chunk through an inbox, a node session and its pump; this outlasts those hops.
SETTLE_S = 0.2


class RecordingNode:
    """Emits name(text) for each input; push_gate and control_gate hold an append or a control open."""

    def __init__(self, *, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.push_gate = asyncio.Event()
        self.push_gate.set()
        self.control_gate = asyncio.Event()
        self.control_gate.set()
        self.pushed: list[str] = []
        self.interrupt_count = 0
        self.should_fail_push = False
        self.emitter: SessionEmitter | None = None

    async def open(self, request: OmniRequest, emitter: SessionEmitter) -> None:
        self.events.append(f"open {self.name}")
        self.emitter = emitter

    async def append(self, chunk: TimedChunk) -> None:
        assert self.emitter is not None and isinstance(chunk.payload, dict)
        text = chunk.payload["text"]
        self.pushed.append(f"{chunk.format}:{text}")
        interrupt_count = self.interrupt_count
        await self.push_gate.wait()
        if self.should_fail_push:
            raise RuntimeError(f"{self.name} failed")
        elif interrupt_count != self.interrupt_count:
            pass
        else:
            await self.emitter.emit_output(
                replace(chunk, payload={"text": f"{self.name}({text})"})
            )

    async def control(self, event: TimedChunk) -> None:
        self.events.append(f"control {self.name}")
        await self.control_gate.wait()
        if control_type(event) == INTERRUPT:
            self.interrupt_count += 1
        else:
            pass

    async def close(self) -> None:
        self.events.append(f"close {self.name}")


class VadNode:
    """Emits an interrupt of the chunk's turn when it hears SPEECH_STARTED."""

    def __init__(self) -> None:
        self.emitter: SessionEmitter | None = None

    async def open(self, request: OmniRequest, emitter: SessionEmitter) -> None:
        self.emitter = emitter

    async def append(self, chunk: TimedChunk) -> None:
        assert self.emitter is not None
        if chunk.payload == SPEECH_STARTED:
            await self.emitter.emit_output(
                control_chunk(
                    INTERRUPT, chunk.t_start_ms, seq=chunk.seq, turn=chunk.turn
                )
            )
        else:
            pass

    async def control(self, event: TimedChunk) -> None:
        raise AssertionError(f"vad takes no control events, got {event!r}")

    async def close(self) -> None:
        pass


class ToolNode:
    """Answers each input with an empty tool_response of the input's turn."""

    def __init__(self) -> None:
        self.emitter: SessionEmitter | None = None

    async def open(self, request: OmniRequest, emitter: SessionEmitter) -> None:
        self.emitter = emitter

    async def append(self, chunk: TimedChunk) -> None:
        assert self.emitter is not None
        await self.emitter.emit_output(
            TimedChunk(
                TOOL,
                chunk.t_start_ms,
                0,
                chunk.seq,
                {"responses": []},
                TOOL_RESPONSE,
                turn=chunk.turn,
            )
        )

    async def control(self, event: TimedChunk) -> None:
        raise AssertionError(f"the tool takes no control events, got {event!r}")

    async def close(self) -> None:
        pass


def text_chunk(text: str, seq: int, turn: int | None = None) -> TimedChunk:
    return TimedChunk("text", seq * 20, 20, seq, {"text": text}, "text", turn=turn)


def speech_chunk(seq: int, turn: int) -> TimedChunk:
    return TimedChunk("audio", seq * 20, 20, seq, SPEECH_STARTED, "pcm16", turn=turn)


async def settle() -> None:
    """Let every ready task run until it blocks."""
    await asyncio.sleep(SETTLE_S)


def cascade_graph(order: tuple[str, ...] = ("vad", "llm", "tts")) -> SessionGraph:
    interrupt = ControlEdge("vad", INTERRUPT)
    nodes = {
        "vad": SessionNode(("vad",), ("input:pcm16",), execution="vad"),
        "llm": SessionNode(
            ("llm",), ("input:text",), execution="llm", controls=(interrupt,)
        ),
        "tts": SessionNode(
            ("tts",), ("llm:text",), execution="tts", controls=(interrupt,)
        ),
    }
    return SessionGraph({name: nodes[name] for name in order}, {"tts": ("text",)})


def edge_graph(edge: DataEdge) -> SessionGraph:
    return SessionGraph(
        {
            "src": SessionNode(("src",), ("input:text",), execution="src"),
            "sink": SessionNode(("sink",), (edge,), execution="sink"),
        },
        {"sink": ("text",)},
    )


@asynccontextmanager
async def open_graph(
    tmp_path: Path,
    graph: SessionGraph,
    nodes: dict[str, RecordingNode | VadNode | ToolNode],
) -> AsyncIterator[tuple[Coordinator, SessionIdentity]]:
    async with pipeline(
        tmp_path, names=list(nodes), session_request_owners=frozenset(nodes)
    ) as (coordinator, _, _):
        for name, node in nodes.items():
            coordinator.session_execution_factories[name] = lambda work, node=node: node
        coordinator.session_graph = graph
        identity = await coordinator.open_session(OmniRequest(None), limits=LIMITS)
        yield coordinator, identity
        await coordinator.close_session(identity)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


async def append(
    coordinator: Coordinator, identity: SessionIdentity, chunk: TimedChunk
) -> None:
    """Append as a client does: a full edge rejects the append, and the client retries."""
    while True:
        try:
            await coordinator.append_session(identity, chunk)
            return
        except QueueFullError:
            await asyncio.sleep(APPEND_RETRY_S)


async def next_data(outputs: AsyncIterator[OutputChunk]) -> OutputChunk:
    while (
        output := await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
    ).kind != "data":
        pass
    return output


def recording_cascade(
    events: list[str],
) -> tuple[dict[str, RecordingNode | VadNode], RecordingNode, RecordingNode]:
    llm = RecordingNode(name="llm", events=events)
    tts = RecordingNode(name="tts", events=events)
    return {"vad": VadNode(), "llm": llm, "tts": tts}, llm, tts


@pytest.mark.asyncio
async def test_interrupt_drops_queued_stale_output_but_not_the_next_turn(tmp_path):
    """Declared downstream first, the nodes still open upstream first, take the interrupt upstream first and close downstream first."""
    events: list[str] = []
    nodes, llm, tts = recording_cascade(events)
    graph = cascade_graph(order=("tts", "llm", "vad"))
    async with open_graph(tmp_path, graph, nodes) as (coordinator, identity):
        assert events == ["open llm", "open tts"]
        outputs = coordinator.session_outputs(identity)
        tts.push_gate.clear()
        for seq in range(3):
            await append(coordinator, identity, text_chunk(f"old{seq}", seq, turn=1))
        await wait_until(lambda: len(llm.pushed) == 3)
        await settle()
        assert tts.pushed == ["text:llm(old0)"]
        await append(coordinator, identity, speech_chunk(3, turn=1))
        await wait_until(lambda: events[-2:] == ["control llm", "control tts"])
        tts.push_gate.set()
        await append(coordinator, identity, text_chunk("new", 4, turn=2))
        output = await next_data(outputs)
        assert output.payload == {"text": "tts(llm(new))"}
        assert tts.pushed == ["text:llm(old0)", "text:llm(new)"]
    assert events[-2:] == ["close tts", "close llm"]


@pytest.mark.asyncio
async def test_input_arriving_during_interrupt_waits_and_is_not_fenced(tmp_path):
    events: list[str] = []
    nodes, llm, _ = recording_cascade(events)
    async with open_graph(tmp_path, cascade_graph(), nodes) as (coordinator, identity):
        outputs = coordinator.session_outputs(identity)
        llm.control_gate.clear()
        await append(coordinator, identity, speech_chunk(0, turn=1))
        await wait_until(lambda: "control llm" in events)
        await append(coordinator, identity, text_chunk("new", 1, turn=2))
        await settle()
        assert llm.pushed == []
        llm.control_gate.set()
        output = await next_data(outputs)
        assert output.payload == {"text": "tts(llm(new))"}


@pytest.mark.asyncio
async def test_drop_oldest_edge_keeps_newest_without_blocking_producer(tmp_path):
    events: list[str] = []
    src = RecordingNode(name="src", events=events)
    sink = RecordingNode(name="sink", events=events)
    graph = edge_graph(DataEdge("src:text", capacity=2, overflow="drop_oldest"))
    async with open_graph(tmp_path, graph, {"src": src, "sink": sink}) as (
        coordinator,
        identity,
    ):
        outputs = coordinator.session_outputs(identity)
        sink.push_gate.clear()
        await append(coordinator, identity, text_chunk("0", 0))
        await wait_until(lambda: sink.pushed == ["text:src(0)"])
        for seq in range(1, 5):
            await append(coordinator, identity, text_chunk(str(seq), seq))
        receipts = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(5)
        ]
        assert receipts == [0, 1, 2, 3, 4]
        sink.push_gate.set()
        await wait_until(lambda: len(sink.pushed) == 3)
        await settle()
        assert sink.pushed == ["text:src(0)", "text:src(3)", "text:src(4)"]


@pytest.mark.asyncio
async def test_a_drop_oldest_client_edge_keeps_the_newest_input_and_never_rejects(
    tmp_path,
):
    events: list[str] = []
    node = RecordingNode(name="rt", events=events)
    graph = SessionGraph(
        {
            "rt": SessionNode(
                ("rt",),
                (DataEdge("input", capacity=2, overflow="drop_oldest"),),
                execution="rt",
            )
        },
        {"rt": ("text",)},
    )
    async with open_graph(tmp_path, graph, {"rt": node}) as (coordinator, identity):
        outputs = coordinator.session_outputs(identity)
        node.push_gate.clear()
        await coordinator.append_session(identity, text_chunk("0", 0))
        await wait_until(lambda: node.pushed == ["text:0"])
        for seq in range(1, 5):
            await coordinator.append_session(identity, text_chunk(str(seq), seq))
        dropped = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(2)
        ]
        assert dropped == [1, 2]
        node.push_gate.set()
        received = [
            await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S) for _ in range(6)
        ]
        assert node.pushed == ["text:0", "text:3", "text:4"]
        assert sorted(o.input_seq for o in received if o.kind == "input_done") == [
            0,
            3,
            4,
        ]


@pytest.mark.asyncio
async def test_a_tool_response_on_a_control_edge_passes_the_queued_audio(tmp_path):
    events: list[str] = []
    chat = RecordingNode(name="chat", events=events)
    graph = SessionGraph(
        {
            "chat": SessionNode(
                ("chat",),
                (DataEdge("input:pcm16", capacity=2),),
                execution="chat",
                controls=(ControlEdge("tool", TOOL_RESPONSE),),
            ),
            "tool": SessionNode(("tool",), (f"input:{TOOL_CALL}",), execution="tool"),
        },
        {"chat": ("pcm16",)},
    )
    async with open_graph(tmp_path, graph, {"chat": chat, "tool": ToolNode()}) as (
        coordinator,
        identity,
    ):
        chat.push_gate.clear()

        def frame(seq: int) -> TimedChunk:
            return TimedChunk("audio", seq * 20, 20, seq, {"text": f"f{seq}"}, "pcm16")

        await append(coordinator, identity, frame(0))
        await wait_until(lambda: chat.pushed == ["pcm16:f0"])
        for seq in (1, 2):
            await coordinator.append_session(identity, frame(seq))
        with pytest.raises(QueueFullError):
            await coordinator.append_session(identity, frame(3))
        call = TimedChunk("tool", 60, 0, 3, {"calls": []}, TOOL_CALL, turn=1)
        await coordinator.append_session(identity, call)
        await wait_until(lambda: "control chat" in events)
        assert chat.pushed == ["pcm16:f0"]
        chat.push_gate.set()
        await wait_until(lambda: len(chat.pushed) == 3)


@pytest.mark.asyncio
async def test_blocking_edge_holds_producer_until_consumer_drains(tmp_path):
    events: list[str] = []
    src = RecordingNode(name="src", events=events)
    sink = RecordingNode(name="sink", events=events)
    graph = edge_graph(DataEdge("src:text", capacity=1))
    async with open_graph(tmp_path, graph, {"src": src, "sink": sink}) as (
        coordinator,
        identity,
    ):
        outputs = coordinator.session_outputs(identity)
        sink.push_gate.clear()
        await append(coordinator, identity, text_chunk("0", 0))
        await wait_until(lambda: sink.pushed == ["text:src(0)"])
        for seq in (1, 2):
            await append(coordinator, identity, text_chunk(str(seq), seq))
        receipts = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(2)
        ]
        await settle()
        assert receipts == [0, 1]
        assert coordinator.graphs[identity.id].inboxes["sink"].counts["src"] == 1
        sink.push_gate.set()
        await wait_until(lambda: len(sink.pushed) == 3)
        assert sink.pushed == ["text:src(0)", "text:src(1)", "text:src(2)"]


@pytest.mark.asyncio
async def test_node_failure_fails_outputs_and_closes_the_rest_downstream_first(
    tmp_path,
):
    events: list[str] = []
    nodes, llm, _ = recording_cascade(events)
    graph = cascade_graph(order=("tts", "llm", "vad"))
    async with open_graph(tmp_path, graph, nodes) as (coordinator, identity):
        outputs = coordinator.session_outputs(identity)
        assert events == ["open llm", "open tts"]
        llm.should_fail_push = True
        await append(coordinator, identity, text_chunk("boom", 0, turn=1))
        with pytest.raises(RuntimeError, match="llm failed"):
            await next_data(outputs)
        await wait_until(lambda: not coordinator.graphs)
        assert sorted(events[2:]) == ["close llm", "close tts"]
        with pytest.raises(ValueError, match="unknown or stale session"):
            await coordinator.append_session(identity, text_chunk("late", 1, turn=1))
