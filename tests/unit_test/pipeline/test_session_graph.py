# SPDX-License-Identifier: Apache-2.0
"""Session graphs: node routes joined by format-typed edges inside the coordinator."""

from __future__ import annotations

import asyncio

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.session_graph import SessionGraphRunner
from sglang_omni.pipeline.sessions import NodeInbox, Session, SessionGraphRuntime
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
    wire_size,
)
from tests.unit_test.fixtures.session_pipeline import (
    chunk,
    event_log,
    pipeline,
    wait_until,
)

NODES = ["left", "right"]
OUTPUT_TIMEOUT_S = 5
# The client's unit in the hop test, and how long after its append the upstream node emits.
HOP_UNIT_S = 1.0
HOP_EMIT_AFTER_S = 0.1


def chain_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(stages=("right",), inputs=("left:text",)),
        },
        publish={"right": ("text",)},
    )


async def collect(outputs, count: int) -> list[OutputChunk]:
    return [
        await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S) for _ in range(count)
    ]


def test_graph_declaration_names_its_sources_and_readers():
    graph = chain_graph()
    assert graph.input_nodes() == ("left",)
    assert graph.reading_nodes("left", "text") == ("right",)
    assert graph.reading_nodes("left", "audio") == ()
    assert graph.reading_nodes("input", None) == ("left",)
    assert graph.is_published("right", "text") and not graph.is_published(
        "left", "text"
    )
    graph = control_graph()
    assert graph.control_targets("left", "interrupt") == (
        ("right", ControlEdge("left", "interrupt", should_preempt=True)),
    )
    assert graph.control_targets("input", "interrupt") == (
        ("right", ControlEdge("input", "interrupt", ("right",))),
    )
    assert graph.control_targets("left", "turn.done") == ()
    assert graph.reading_nodes("left", "session") == ()


def voicechat_tool(
    *responses: str | DataEdge, controls: tuple[ControlEdge, ...] = ()
) -> dict[str, SessionNode]:
    """A chat node and the tool node it calls, answered on `responses` or `controls`."""
    return {
        "voicechat": SessionNode(
            ("voicechat",), ("input", *responses), controls=controls
        ),
        "tool": SessionNode(("tool",), ("voicechat:tool_call",)),
    }


def after_a(
    *inputs: str | DataEdge, controls: tuple[ControlEdge, ...] = ()
) -> dict[str, SessionNode]:
    return {
        "a": SessionNode(("a",), ("input",)),
        "b": SessionNode(("b",), inputs, controls=controls),
    }


def test_a_marked_feedback_edge_or_a_control_edge_closes_a_cycle_without_ordering_it():
    feedback = DataEdge("tool:tool_response", overflow="drop_oldest", feedback=True)
    assert SessionGraph(voicechat_tool(feedback), {}).order() == ("voicechat", "tool")
    controlled = voicechat_tool(controls=(ControlEdge("tool", "tool_response"),))
    assert SessionGraph(controlled, {}).order() == ("voicechat", "tool")


@pytest.mark.parametrize(
    ("nodes", "publish", "message"),
    [
        ({"a": SessionNode(("x",), ("b",))}, {}, "unknown inputs"),
        (
            {"a": SessionNode(("x",), ("b",)), "b": SessionNode(("y",), ("a",))},
            {},
            "no node reads",
        ),
        ({"a": SessionNode(("x",), ("input",))}, {"b": ("text",)}, "publish"),
        (after_a(controls=(ControlEdge("c", "interrupt"),)), {}, "unknown inputs"),
        (
            after_a(
                "input", DataEdge("a:text", capacity=2), DataEdge("a:audio", capacity=3)
            ),
            {},
            "bounds its edge from 'a' twice",
        ),
        (after_a("input", DataEdge("a", capacity=0)), {}, "capacity of at least 1"),
        (voicechat_tool("tool:tool_response"), {}, "cycle"),
        (
            voicechat_tool(DataEdge("tool:tool_response", feedback=True)),
            {},
            "must not block",
        ),
    ],
)
def test_graph_rejects_a_miswired_declaration(nodes, publish, message):
    with pytest.raises(ValueError, match=message):
        SessionGraph(nodes, publish)


def control_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(
                stages=("right",),
                inputs=(),
                controls=(
                    ControlEdge("left", "interrupt", should_preempt=True),
                    ControlEdge("input", "interrupt", ("right",)),
                ),
            ),
        },
        publish={"left": ("session",)},
    )


@pytest.mark.asyncio
async def test_a_control_edge_delivers_the_event_as_a_control_operation(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = control_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_control": "interrupt"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 1)
        assert [item.kind for item in received] == ["input_done"]
        await wait_until(
            lambda: ("control", "right", f"{identity.id}:right", "interrupt")
            in event_log(events)
        )
        event = TimedChunk("control", 0, 0, 0, {"type": "interrupt"}, "session")
        await coordinator.control_session(identity, event)
        await wait_until(
            lambda: ("control", "right", f"{identity.id}:right", "interrupt")
            in event_log(events)
        )
        with pytest.raises(ValueError, match="control edges"):
            await coordinator.control_session(identity, event, stages=["right"])
        with pytest.raises(ValueError, match="no node takes"):
            await coordinator.control_session(
                identity, TimedChunk("control", 0, 0, 0, {"type": "nudge"}, "session")
            )
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


def test_a_pipeline_declares_its_session_as_one_route_or_one_graph():
    with pytest.raises(ValueError, match="session_route or session_graph"):
        Coordinator(
            "ipc://done",
            "ipc://abort",
            "left",
            session_route=lambda: NODES,
            session_graph=chain_graph(),
        )


@pytest.mark.asyncio
async def test_edge_chunks_reach_their_reader_and_only_published_data_reaches_the_client(
    tmp_path,
):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = chain_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        for seq in range(3):
            await coordinator.append_session(identity, chunk(seq))
        received = await collect(outputs, 6)
        receipts = [item for item in received if item.kind == "input_done"]
        data = [item for item in received if item.kind == "data"]
        assert [item.input_seq for item in receipts] == [0, 1, 2]
        assert [item.seq for item in received] == list(range(6))
        assert all(item.session_identity == identity for item in received)
        assert [item.payload["stage"] for item in data] == ["right"] * 3
        assert [item.payload["input"]["count"] for item in data] == [1, 2, 3]
        assert [item.payload["input"]["stage"] for item in data] == ["left"] * 3
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
        closes = [event[1] for event in event_log(events) if event[0] == "close"]
        assert closes == ["right", "left"]


def test_graph_outputs_count_against_max_output_bytes_until_the_client_takes_them():
    output = OutputChunk(
        SessionIdentity("graph", 1), 0, 0, "text", 0, 0, bytes(400), "text"
    )
    size = wire_size(output.to_dict())
    graph = SessionGraphRuntime(
        session_identity=SessionIdentity("graph", 1),
        graph=chain_graph(),
        nodes={},
        limits=SessionLimits(max_output_bytes=size * 2 - 1),
        inboxes={},
        outputs=asyncio.Queue(),
    )
    sessions = SessionGraphRunner()
    assert sessions.reserve_graph_output(graph, "right", output) == size
    assert graph.output_bytes == size
    with pytest.raises(RuntimeError, match="node 'right' exceeds max_output_bytes"):
        sessions.reserve_graph_output(graph, "right", output)


@pytest.mark.asyncio
async def test_input_fans_out_to_every_reader_and_input_done_waits_for_all(tmp_path):
    graph = SessionGraph(
        nodes={name: SessionNode(stages=(name,), inputs=("input",)) for name in NODES},
        publish={name: ("text",) for name in NODES},
    )
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 3)
        assert sorted(item.kind for item in received) == ["data", "data", "input_done"]
        assert {item.payload["stage"] for item in received if item.kind == "data"} == {
            "left",
            "right",
        }
        appends = [event for event in event_log(events) if event[0] == "append"]
        assert sorted(event[1] for event in appends) == NODES
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(identity, chunk(5))
        await coordinator.close_session(identity)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
        with pytest.raises(StopAsyncIteration):
            await anext(outputs)


@pytest.mark.asyncio
async def test_failed_node_open_closes_the_nodes_already_open(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = chain_graph()
        with pytest.raises(RuntimeError, match="open failed"):
            await coordinator.open_session(OmniRequest(None, {"fail_open": "right"}))
        await wait_until(lambda: not coordinator.sessions)
        assert not coordinator.graphs
        log = event_log(events)
        opened = [event[1] for event in log if event[0] == "open"]
        closed = [event[1] for event in log if event[0] == "close"]
        assert "left" not in opened or "left" in closed


@pytest.mark.asyncio
async def test_a_node_that_ends_fails_the_graph_to_its_reader(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, _, _):
        coordinator.session_graph = chain_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        await collect(outputs, 2)
        await coordinator.close_session(coordinator.graphs[identity.id].nodes["right"])
        with pytest.raises(RuntimeError, match="session ended"):
            await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_open_without_a_route_follows_the_pipeline_route(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 2)
        assert {item.kind for item in received} == {"data", "input_done"}
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)
        assert [event[1] for event in event_log(events) if event[0] == "open"] == [
            "source",
            "sink",
        ]
    async with pipeline(tmp_path, names=NODES) as (coordinator, _, _):
        with pytest.raises(ValueError, match="declares no Session"):
            await coordinator.open_session(OmniRequest(None))


def text_ends_at_text_stage(request: OmniRequest) -> list[str]:
    return ["text"] if request.params.get("text_only") else ["text", "audio"]


def hop_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(stages=("right",), inputs=("input", "left:session")),
        },
        publish={"left": ("session",)},
    )


@pytest.mark.asyncio
async def test_a_chunk_emitted_between_appends_reaches_the_reader_before_the_next_unit(
    tmp_path,
):
    """A node-to-node hop is not quantized to the client's appends: what a node emits after its append completes is forwarded at once."""
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = hop_graph()
        identity = await coordinator.open_session(
            OmniRequest(
                None,
                {"emit_control": "speech.stopped", "emit_after": HOP_EMIT_AFTER_S},
            )
        )
        outputs = coordinator.session_outputs(identity)
        log: list = []

        def right_appends() -> list[int]:
            log.extend(event_log(events))
            return [
                e[3]
                for e in log
                if e[:3] == ("append", "right", f"{identity.id}:right")
            ]

        await coordinator.append_session(identity, chunk(0))
        await wait_until(lambda: 1 in right_appends(), timeout=HOP_UNIT_S)
        received = await collect(outputs, 2)
        assert [(item.kind, item.input_seq) for item in received] == [
            ("input_done", 0),
            ("data", 0),
        ]
        assert received[1].payload == {"type": "speech.stopped"}
        await coordinator.append_session(identity, chunk(1))
        await wait_until(lambda: 2 in right_appends())
        assert right_appends() == [0, 1, 2]
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_a_node_merges_the_clocks_of_its_sources(tmp_path):
    """A control from another node may lie before the client's last control; only the client's own input keeps one timeline."""
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = SessionGraph(
            hop_graph().nodes, {"right": ("text",)}
        )
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_control": "speech.started"})
        )
        outputs = coordinator.session_outputs(identity)
        late = TimedChunk("control", 5000, 0, 0, {"type": "configure"}, "session")
        await coordinator.append_session(identity, late)
        await coordinator.append_session(identity, chunk(1))
        received = await collect(outputs, 2)
        assert [item.kind for item in received] == ["input_done", "input_done"]
        await wait_until(
            lambda: ("append", "right", f"{identity.id}:right", 3) in event_log(events)
        )
        with pytest.raises(ValueError, match="moves backwards"):
            await coordinator.append_session(
                identity,
                TimedChunk("control", 10, 0, 2, {"type": "configure"}, "session"),
            )
        await coordinator.close_session(identity)


class NodeOutputsRunner(SessionGraphRunner):
    """Graph routing over one node whose outputs are given; node controls are recorded."""

    def __init__(self, node_outputs: list[OutputChunk]) -> None:
        super().__init__()
        self.node_outputs = node_outputs
        self.controls: list[TimedChunk] = []

    async def session_outputs(self, session_identity):
        for output in self.node_outputs:
            yield output

    async def control_session(
        self, session_identity, event, *, stages=None, should_preempt=False
    ):
        self.controls.append(event)


def reply_runtime(sessions: SessionGraphRunner) -> SessionGraphRuntime:
    """A graph of one `reply` node interrupted by the client, with its node session registered."""
    node = SessionIdentity("graph:reply")
    sessions.sessions[node.id] = Session(
        session_identity=node,
        request=OmniRequest(None),
        stages=("reply",),
        bindings={},
        limits=SessionLimits(),
        is_node=True,
    )
    return SessionGraphRuntime(
        session_identity=SessionIdentity("graph"),
        graph=SessionGraph(
            nodes={
                "reply": SessionNode(
                    ("reply",),
                    ("input",),
                    controls=(ControlEdge("input", "interrupt"),),
                )
            },
            publish={"reply": ("text_delta", "session")},
        ),
        nodes={"reply": node},
        limits=SessionLimits(),
        inboxes={"reply": NodeInbox()},
        outputs=asyncio.Queue(),
        client_sequences={"reply": {}},
    )


def reply_output(output_format: str, payload: dict, turn: int) -> OutputChunk:
    modality = "control" if output_format == "session" else "text"
    return OutputChunk(
        SessionIdentity("graph:reply"),
        0,
        0,
        modality,
        0,
        0,
        payload,
        output_format,
        turn=turn,
    )


@pytest.mark.asyncio
async def test_an_interrupted_node_publishes_only_the_end_of_the_interrupted_turn():
    sessions = NodeOutputsRunner(
        [
            reply_output("text_delta", {"text": "stale"}, 1),
            reply_output("session", {"type": "turn.done"}, 1),
            reply_output("session", {"type": "turn.interrupted"}, 1),
            reply_output("text_delta", {"text": "next"}, 2),
        ]
    )
    graph = reply_runtime(sessions)
    interrupt = TimedChunk("control", 0, 0, 0, {"type": "interrupt"}, "session", turn=1)
    await sessions.control_graph(graph, interrupt, None, False)
    assert sessions.controls == [interrupt]
    await sessions.route_graph_outputs(graph, "reply")
    published = []
    while not graph.outputs.empty():
        published.append(graph.outputs.get_nowait()[0])
    assert [(item.payload, item.turn) for item in published] == [
        ({"type": "turn.interrupted"}, 1),
        ({"text": "next"}, 2),
    ]
