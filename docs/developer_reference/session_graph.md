# Session Graph

A Session is one stateful conversation on the server: the client opens it, appends input for its whole lifetime, reads what it publishes and closes it. A pipeline declares the shape of its Session in its config, and the coordinator opens it; the serving layer appends input and reads the published formats without knowing the nodes behind them. A Session is either the one stage route from the pipeline's entry stage or a graph of nodes declared in `session_nodes`.

## Nodes

Every node of the Session is an entry of `session_nodes`, and `session_publish` lists, per node, the formats that reach the client. A node with `stages` is a route through those stages of the server.

`inputs` lists the nodes a node reads; `input` is the Session's own input, its audio, its `commit` and other controls. `node:format` takes only that format from that node; two nodes never send the same format to one reader, and the audio format of `input` is the Session input's alone.

An input written `{kind: control, source, event, preempt}` is a control edge. Its `event` is a control type of the source's `session` chunks or the format of its other chunks, such as `tool_response`; the event is handed to the node as a `control` operation, outside its input queue, to every stage of its route in route order, and travels on no other edge and to no client. With `preempt` the stages cancel the node's in-flight appends first. The client's `interrupt` reaches a node through a control edge from `input`.

An input of a stage node written `{source, capacity, overflow}` bounds its edge. The inputs naming one source are one edge into the node; it holds at most `capacity` chunks the node's own session has not accepted yet, by default the Session's `max_pending_chunks`. When the edge is full, `overflow: block`, the default, holds the producing node until the node takes a chunk, and rejects the client's append as a full queue; `overflow: drop_oldest` drops the edge's oldest chunk and lets the producer go on; on the client's edge the dropped input is confirmed to the client like a taken one, so a real-time node such as a full-duplex model hears the newest audio instead of a backlog. Two inputs of one source that declare different bounds fail the declaration.

The Session opens its nodes upstream first and closes them downstream first. A cycle of data edges between nodes must name the input that closes it with `feedback: true`, and that input must not block: two nodes each held on the other's full edge never wake. An undeclared cycle fails with `form a cycle of data edges`. A control edge holds no queue, so a cycle it closes needs no mark.

## Stream vocabulary

| Format | Modality | Payload |
| --- | --- | --- |
| `pcm16` | `audio` | 16 kHz mono signed 16-bit PCM |
| `pcm_f32le` | `audio` | 24 kHz mono float32 PCM |
| `transcript` | `transcript` | `{"text", "is_final"}`; one final transcript per utterance |
| `messages` | `text` | `{"messages": [{"role", "content"}, ...]}` |
| `text_delta` | `text` | `{"text"}`, one increment of the reply |
| `tool_call` | `tool` | `{"calls": [{"id", "name", "arguments"}, ...]}`; the generator asks for these tools |
| `tool_response` | `tool` | `{"responses": [{"id", "name", "response"}, ...]}`; one response per call, in call order |
| `session` | `control` | `{"type", ...}` |

Control types sent into a Session are `commit` (the utterance ends here) and `configure` (settings change). Control types a Session emits are `speech.started` and `speech.stopped` with `t_ms` on the input timeline, `turn.queued` when an utterance waits for its reply, `turn.started` when the reply begins, `turn.done` after the reply's audio, `turn.empty` for an utterance without words, and `configured` with `{"nodes": {name: kind}}`. Every chunk of a reply carries its turn number in the chunk's `turn` field, numbered from 1 per Session in utterance order: the turn controls, the `messages`, `text_delta`, `tool_call` and `tool_response` chunks and the reply audio; every turn opens with `turn.queued` and closes with `turn.done` or `turn.empty`.

## Interruption

An `interrupt` naming a turn that reaches a node on a control edge first drops that turn's chunks from other nodes still queued for the node, on its edges and among its unsent appends, and a chunk of that turn that a node emits later no longer reaches it; the client's own input is never dropped. The node's own output of that turn, whether already buffered or emitted later, reaches neither its readers nor the client, except the `turn.interrupted` that ends the turn. Control edges are delivered upstream first, so a node's stale output is fenced before its readers hear the event. An `interrupt` without a turn, such as the client's `response.cancel`, means the open reply, which only the node knows; the Session leaves its queues to the node.

## Session API

A Session takes four operations: `open`, `append`, `control` and `close`. `Client.control_session(identity, event, stages=None, should_preempt=False)` delivers a `session` control chunk such as `interrupt` to every stage of the Session's route, or to the named stages, in the Session's arrival order with its appends; `should_preempt` cancels the stages' in-flight appends first. On a Session graph the event follows the control edges declared from `input`. A Session delivers its outputs as the nodes produce them, between appends as well as during one, so a client that has spoken may wait for the replies without appending.

A `/v1/session` client opens the Session the pipeline declares; the open frame carries `params`, and the Session's limits are the deployment's. An `append` frame carries a `chunk` and is answered with `accepted`; a `control` frame carries the event as its `chunk`, with optional `stages` and `should_preempt`, and is answered with `controlled`. Both answers carry the chunk's seq, and a full queue answers either with `rejected` and leaves the Session open.

A chunk's `turn` identifies a cancellable reply within the Session, not a tool
call or an input audio frame. All work belonging to that reply preserves the
same ID, including tool responses. Interrupted reply IDs remain fenced for the
lifetime of the graph, including tool results carried by control edges. Session
control commands such as interrupt and truncate remain deliverable. Nodes must discard any matching
work they have already buffered when they receive the interrupt.

A producer without reply boundaries leaves `turn` unset. VoiceChat's continuous
duplex stream does not currently allocate reply turns; tool call IDs identify its
individual calls. Passing through a supplied `turn` does not add native speech
interruption or reply-boundary detection to VoiceChat.
