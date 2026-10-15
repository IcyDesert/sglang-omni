# SPDX-License-Identifier: Apache-2.0
"""The Session graph runtime: node sessions joined by bounded edges, and control edges."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import replace

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.sessions import (
    CoordinatorSessions,
    NodeInbox,
    SessionGraphRuntime,
    admit_input,
    check_timing,
    output_chunk,
)
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    GRAPH_INPUT,
    ControlEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
    wire_size,
)
from sglang_omni.proto.streams import (
    INTERRUPT,
    control_type,
    edge_event,
    is_interrupted_reply,
)

logger = logging.getLogger(__name__)

# Note (Junnan Li): A node's pending budget is a few units; a retry this short adds no audible latency.
FULL_QUEUE_RETRY_S = 0.005


class SessionGraphRunner(CoordinatorSessions):
    """Runs the coordinator's Session graphs over its node sessions."""

    async def control_graph(
        self,
        graph: SessionGraphRuntime,
        event: TimedChunk,
        stages: list[str] | None,
        should_preempt: bool,
    ) -> None:
        targets = graph.graph.control_targets(GRAPH_INPUT, control_type(event))
        if graph.is_closing:
            raise RuntimeError("session is closing")
        elif stages is not None:
            raise ValueError("a graph delivers a control along its control edges")
        elif not targets:
            raise ValueError(
                f"no node takes the control {control_type(event)!r} from {GRAPH_INPUT!r}"
            )
        else:
            pass
        for target, edge in targets:
            await self.control_edge(
                graph, target, edge, event, should_preempt=should_preempt
            )

    async def control_edge(
        self,
        graph: SessionGraphRuntime,
        target: str,
        edge: ControlEdge,
        event: TimedChunk,
        *,
        should_preempt: bool = False,
    ) -> None:
        """An interrupt naming a turn first drops what the Session still holds of that turn on its way to the target and from it.

        The target's inbox and its node session's unsent appends lose the
        turn's chunks from other nodes, and a later chunk of the turn no
        longer enters the inbox.
        The client's own input stays.
        """
        if control_type(event) == INTERRUPT and event.turn is not None:
            graph.interrupted_turns[target] = event.turn
            inbox = graph.inboxes[target]
            inbox.queued = deque(
                item
                for item in inbox.queued
                if item[0] == GRAPH_INPUT or item[1].turn != event.turn
            )
            inbox.counts = Counter(source for source, _, _ in inbox.queued)
            inbox.wake.set()
            node = self.sessions[graph.nodes[target].id]
            kept = deque(
                (chunk, size)
                for chunk, size in node.pending
                if chunk.turn != event.turn
                or chunk.seq in graph.client_sequences[target]
            )
            node.pending_count -= len(node.pending) - len(kept)
            node.pending_bytes -= sum(size for _, size in node.pending) - sum(
                size for _, size in kept
            )
            node.pending = kept
        else:
            pass
        await self.control_session(
            graph.nodes[target],
            event,
            stages=None if edge.target_stages is None else list(edge.target_stages),
            should_preempt=edge.should_preempt or should_preempt,
            is_ordered=edge.source == GRAPH_INPUT,
        )

    async def open_graph(
        self,
        request: OmniRequest,
        *,
        graph: SessionGraph,
        limits: SessionLimits | None,
        session_id: str | None,
    ) -> SessionIdentity:
        """Open one node session per node, then route between them."""
        limits = limits or SessionLimits()
        session_id = session_id or str(uuid.uuid4())
        if session_id in self.sessions or session_id in self.graphs:
            raise ValueError("session ID already reserved")
        else:
            pass
        nodes = await self.open_graph_nodes(request, graph, limits, session_id)
        runtime = SessionGraphRuntime(
            session_identity=SessionIdentity(session_id, self.next_open_index),
            graph=graph,
            nodes=nodes,
            limits=limits,
            inboxes={name: NodeInbox() for name in nodes},
            outputs=asyncio.Queue(maxsize=limits.max_output_chunks),
            client_sequences={name: {} for name in nodes},
        )
        self.next_open_index += 1
        self.graphs[session_id] = runtime
        self.start_graph_tasks(runtime)
        return runtime.session_identity

    async def open_graph_nodes(
        self,
        request: OmniRequest,
        graph: SessionGraph,
        limits: SessionLimits,
        session_id: str,
    ) -> dict[str, SessionIdentity]:
        """Open the node sessions upstream first; a failed open closes the ones already open."""
        input_nodes = graph.input_nodes()
        nodes: dict[str, SessionIdentity] = {}
        try:
            for name in graph.order():
                node = graph.nodes[name]
                nodes[name] = await self.open_route(
                    request,
                    stages=list(node.stages),
                    # Note (Junnan Li): A node fed only by other nodes may wait a whole conversation; the input nodes bound the idle time.
                    limits=(
                        limits
                        if name in input_nodes
                        else replace(limits, idle_timeout_s=math.inf)
                    ),
                    session_id=f"{session_id}:{name}",
                    is_node=True,
                )
        except BaseException:
            for identity in reversed(list(nodes.values())):
                await self.close_session(identity)
            raise
        return nodes

    def start_graph_tasks(self, graph: SessionGraphRuntime) -> None:
        """Start each node's input forwarding and output routing tasks."""
        for name in graph.nodes:
            graph.forwarders.append(
                self.graph_task(graph, name, self.forward_graph_inputs(graph, name))
            )
            graph.tasks.append(
                self.graph_task(graph, name, self.route_graph_outputs(graph, name))
            )

    def graph_task(
        self,
        graph: SessionGraphRuntime,
        name: str,
        coroutine: Coroutine[None, None, None],
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)

        def done(task: asyncio.Task[None]) -> None:
            if task.cancelled() or graph.is_closing:
                return
            else:
                error = task.exception()
                if error is None:
                    error = RuntimeError(f"graph node {name!r} session ended")
                else:
                    logger.warning(
                        f"graph {graph.session_identity.id} node {name} failed",
                        exc_info=error,
                    )
                graph.error = graph.error or error
                self.begin_graph_close(graph)

        task.add_done_callback(done)
        return task

    def append_graph(self, graph: SessionGraphRuntime, chunk: TimedChunk) -> int:
        """Hand client input to every node reading it; rejected input keeps its seq."""
        if graph.is_closing:
            raise RuntimeError("session is closing")
        elif chunk.seq != graph.next_input:
            raise ValueError("input seq must be contiguous within an open index")
        else:
            check_timing(chunk, graph.modality_end_ms)
        chunk, _ = admit_input(
            chunk, graph.limits, graph.modality_end_ms, graph.ended_modalities
        )
        reading_nodes = graph.graph.reading_nodes(GRAPH_INPUT, chunk.format)
        full = [
            name
            for name in reading_nodes
            if graph.inboxes[name].counts[GRAPH_INPUT]
            >= self.edge_capacity(graph, name, GRAPH_INPUT)
        ]
        # Note (Junnan Li): Every input the append confirms, its own or a dropped one, needs an output slot before anything changes.
        dropped_inputs = Counter(
            next(
                seq
                for source, _, seq in graph.inboxes[name].queued
                if source == GRAPH_INPUT
            )
            for name in full
        )
        confirmations = sum(
            count == graph.pending_input_done[seq]
            for seq, count in dropped_inputs.items()
        ) + (0 if reading_nodes else 1)
        if any(
            graph.graph.nodes[name].bound(GRAPH_INPUT).overflow == "block"
            for name in full
        ) or (
            graph.outputs.maxsize > 0
            and graph.outputs.qsize() + confirmations > graph.outputs.maxsize
        ):
            raise QueueFullError()
        else:
            pass
        if reading_nodes:
            graph.pending_input_done[chunk.seq] = len(reading_nodes)
        else:
            self.emit_graph_input_done(graph, GRAPH_INPUT, chunk)
        for name in full:
            inbox = graph.inboxes[name]
            index = next(
                i for i, item in enumerate(inbox.queued) if item[0] == GRAPH_INPUT
            )
            _, dropped, _ = inbox.queued[index]
            del inbox.queued[index]
            inbox.counts[GRAPH_INPUT] -= 1
            self.confirm_graph_input(graph, name, dropped)
        for name in reading_nodes:
            inbox = graph.inboxes[name]
            inbox.queued.append((GRAPH_INPUT, chunk, chunk.seq))
            inbox.counts[GRAPH_INPUT] += 1
            inbox.wake.set()
        graph.modality_end_ms[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
        if chunk.eos:
            graph.ended_modalities.add(chunk.modality)
        else:
            pass
        graph.next_input += 1
        return chunk.seq

    def edge_capacity(
        self, graph: SessionGraphRuntime, target: str, source: str
    ) -> int:
        capacity = graph.graph.nodes[target].bound(source).capacity
        if capacity is None:
            return graph.limits.max_pending_chunks
        else:
            return capacity

    def is_interrupted(
        self, graph: SessionGraphRuntime, target: str, chunk: TimedChunk
    ) -> bool:
        return (
            chunk.turn is not None and graph.interrupted_turns.get(target) == chunk.turn
        )

    async def put_graph_inbox(
        self, graph: SessionGraphRuntime, target: str, source: str, chunk: TimedChunk
    ) -> None:
        """Queue a node's chunk on its edge into `target`; a full edge holds the producer or drops its own oldest chunk, and a chunk of the turn last interrupted at `target` is dropped."""
        inbox = graph.inboxes[target]
        capacity = self.edge_capacity(graph, target, source)
        overflow = graph.graph.nodes[target].bound(source).overflow
        while (
            overflow == "block"
            and inbox.counts[source] >= capacity
            and not self.is_interrupted(graph, target, chunk)
        ):
            inbox.wake.clear()
            await inbox.wake.wait()
        if self.is_interrupted(graph, target, chunk):
            return
        elif overflow == "drop_oldest" and inbox.counts[source] >= capacity:
            del inbox.queued[
                next(i for i, item in enumerate(inbox.queued) if item[0] == source)
            ]
            inbox.counts[source] -= 1
        else:
            pass
        inbox.queued.append((source, chunk, None))
        inbox.counts[source] += 1
        inbox.wake.set()

    async def forward_graph_inputs(self, graph: SessionGraphRuntime, name: str) -> None:
        """Append one node's inbox in that node's own seq order; a chunk leaves the inbox once its node session accepts it."""
        identity = graph.nodes[name]
        inbox = graph.inboxes[name]
        loop = asyncio.get_running_loop()
        seq = 0
        deadline: float | None = None
        while True:
            if not inbox.queued:
                inbox.wake.clear()
                await inbox.wake.wait()
                continue
            else:
                source, chunk, client_seq = inbox.queued[0]
            try:
                await self.append_session(identity, replace(chunk, seq=seq))
            except QueueFullError:
                if deadline is None:
                    deadline = loop.time() + graph.limits.operation_timeout_s
                elif loop.time() >= deadline:
                    raise TimeoutError(
                        f"graph node {name!r} rejected input seq {seq} for "
                        f"{graph.limits.operation_timeout_s} s"
                    ) from None
                else:
                    pass
                await asyncio.sleep(FULL_QUEUE_RETRY_S)
            else:
                # Note (Junnan Li): The append completes without yielding, so the head is still the chunk it took.
                inbox.queued.popleft()
                inbox.counts[source] -= 1
                inbox.wake.set()
                if client_seq is not None:
                    graph.client_sequences[name][seq] = client_seq
                else:
                    pass
                seq += 1
                deadline = None

    async def route_graph_outputs(self, graph: SessionGraphRuntime, name: str) -> None:
        """Deliver one node's outputs to its readers and, when published, to the client."""
        async for output in self.session_outputs(graph.nodes[name]):
            if output.kind == "input_done":
                client_seq = graph.client_sequences[name].pop(output.input_seq, None)
                if client_seq is None:
                    pass
                else:
                    self.confirm_graph_input(
                        graph,
                        name,
                        TimedChunk(
                            output.modality,
                            output.t_start_ms,
                            output.duration_ms,
                            client_seq,
                            None,
                            format=output.format,
                            eos=output.eos,
                        ),
                    )
                continue
            else:
                pass
            chunk = output_chunk(output)
            if name in graph.interrupted_turns and is_interrupted_reply(
                chunk, graph.interrupted_turns[name]
            ):
                continue
            else:
                pass
            control_targets = graph.graph.control_targets(name, edge_event(chunk))
            if control_targets:
                for target, edge in control_targets:
                    await self.control_edge(graph, target, edge, chunk)
                continue
            else:
                pass
            is_published = graph.graph.is_published(name, output.format)
            for reader in graph.graph.reading_nodes(name, output.format):
                await self.put_graph_inbox(graph, reader, name, chunk)
            if is_published:
                await self.publish_graph_output(
                    graph, output, self.reserve_graph_output(graph, name, output)
                )
            else:
                pass

    async def publish_graph_output(
        self, graph: SessionGraphRuntime, output: OutputChunk, size: int
    ) -> None:
        """Hand a node output to the client, counted as `size` bytes of the output budget."""
        published = replace(
            output,
            session_identity=graph.session_identity,
            seq=graph.next_output,
            input_seq=max(graph.next_input - 1, 0),
        )
        graph.next_output += 1
        await graph.outputs.put((published, size))

    def reserve_graph_output(
        self, graph: SessionGraphRuntime, name: str, output: OutputChunk
    ) -> int:
        """Count an output against max_output_bytes until the client takes it; over the budget fails the graph."""
        size = wire_size(output.to_dict())
        if graph.output_bytes + size > graph.limits.max_output_bytes:
            raise RuntimeError(
                f"output of graph node {name!r} exceeds max_output_bytes "
                f"{graph.limits.max_output_bytes}; the client is not reading"
            )
        else:
            graph.output_bytes += size
        return size

    def confirm_graph_input(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> None:
        """One reader is done with the client's input `chunk`, by taking or by dropping it; the client hears of it once every reader is."""
        graph.pending_input_done[chunk.seq] -= 1
        if graph.pending_input_done[chunk.seq] == 0:
            del graph.pending_input_done[chunk.seq]
            self.emit_graph_input_done(graph, name, chunk)
        else:
            pass

    def emit_graph_input_done(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> None:
        output = OutputChunk(
            session_identity=graph.session_identity,
            seq=graph.next_output,
            input_seq=chunk.seq,
            modality=chunk.modality,
            t_start_ms=chunk.t_start_ms,
            duration_ms=chunk.duration_ms,
            payload=None,
            format=chunk.format,
            eos=chunk.eos,
            kind="input_done",
        )
        if graph.outputs.full():
            raise RuntimeError(
                f"graph node {name!r} confirmed input seq {chunk.seq} while "
                "max_output_chunks outputs wait for the client"
            )
        else:
            graph.outputs.put_nowait(
                (output, self.reserve_graph_output(graph, name, output))
            )
        graph.next_output += 1

    async def graph_outputs(
        self, graph: SessionGraphRuntime
    ) -> AsyncIterator[OutputChunk]:
        if graph.has_output_consumer:
            raise RuntimeError("session already has an output consumer")
        else:
            graph.has_output_consumer = True
        try:
            while (item := await graph.outputs.get()) is not None:
                output, size = item
                graph.output_bytes -= size
                yield output
            if graph.error is not None:
                raise graph.error
            else:
                return
        finally:
            graph.has_output_consumer = False
            await asyncio.shield(self.close_graph(graph))

    def begin_graph_close(self, graph: SessionGraphRuntime) -> None:
        if graph.close_task is None:
            graph.close_task = self.owned_session_task(self.finish_graph_close(graph))
        else:
            pass

    async def close_graph(self, graph: SessionGraphRuntime) -> None:
        self.begin_graph_close(graph)
        assert graph.close_task is not None
        await asyncio.shield(graph.close_task)

    async def stop_graph_tasks(
        self, graph: SessionGraphRuntime, tasks: list[asyncio.Task[None]]
    ) -> None:
        """Cancel the tasks and wait for them; the first that failed instead of stopping fails the graph."""
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        else:
            pass
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                graph.error = graph.error or task.exception()
            else:
                pass

    async def finish_graph_close(self, graph: SessionGraphRuntime) -> None:
        """Stop input, close the nodes downstream first (`nodes` holds them upstream first), then stop routing."""
        graph.is_closing = True
        await self.stop_graph_tasks(graph, graph.forwarders)
        unclosed: dict[str, Exception] = {}
        for name, identity in reversed(list(graph.nodes.items())):
            node = self.sessions.get(identity.id)
            try:
                await self.close_session(identity)
            except Exception as exc:
                unclosed[name] = exc
            if node is not None and node.error is not None:
                graph.error = graph.error or node.error
            else:
                pass
        if unclosed:
            graph.cleanup_error = RuntimeError(
                f"graph nodes {sorted(unclosed)} did not close: {unclosed}"
            )
            graph.cleanup_error.__cause__ = next(iter(unclosed.values()))
            graph.error = graph.error or next(iter(unclosed.values()))
        else:
            pass
        await self.stop_graph_tasks(graph, graph.tasks)
        # Note (Junnan Li): Close fences output like a route close; queued data is dropped.
        while not graph.outputs.empty():
            graph.outputs.get_nowait()
        graph.output_bytes = 0
        graph.outputs.put_nowait(None)
        self.graphs.pop(graph.session_identity.id, None)
