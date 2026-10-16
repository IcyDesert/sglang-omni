# Session Graph

A Session is one stateful conversation on the server: the client opens it, appends input for its whole lifetime, reads what it publishes and closes it. A pipeline declares the shape of its Session in its config, and the coordinator opens it; the serving layer appends input and reads the published formats without knowing the nodes behind them. A Session is either the one stage route from the pipeline's entry stage or a graph of nodes declared in `session_nodes`.

## Nodes

Every node of the Session is an entry of `session_nodes`, and `session_publish` lists, per node, the formats that reach the client. A node with `stages` is a route through those stages of the server.

`inputs` lists the nodes a node reads; `input` is the Session's own input, its audio, its `commit` and other controls, and camera frames when a node of the Session takes them. `node:format` takes only that format from that node; two nodes never send the same format to one reader, and the audio format of `input` is the Session input's alone. A node that reads every format of `input` while the Session takes camera frames must take them too, or name the formats it reads, such as `input:pcm16` and `input:session`.

An input written `{kind: control, source, event, preempt}` is a control edge. Its `event` is a control type of the source's `session` chunks or the format of its other chunks, such as `tool_response`; the event is handed to the node as a `control` operation, outside its input queue, to every stage of its route in route order, and travels on no other edge and to no client. With `preempt` the stages cancel the node's in-flight appends first. The client's `interrupt` reaches a node through a control edge from `input`.

An input of a stage node written `{source, capacity, overflow}` bounds its edge. The inputs naming one source are one edge into the node; it holds at most `capacity` chunks the node's own session has not accepted yet, by default the Session's `max_pending_chunks`. When the edge is full, `overflow: block`, the default, holds the producing node until the node takes a chunk, and rejects the client's append as a full queue; `overflow: drop_oldest` drops the edge's oldest chunk and lets the producer go on; on the client's edge the dropped input is confirmed to the client like a taken one, so a real-time node such as a full-duplex model hears the newest audio instead of a backlog. Two inputs of one source that declare different bounds fail the declaration.

The Session opens its nodes upstream first and closes them downstream first. A cycle of data edges between nodes must name the input that closes it with `feedback: true`, and that input must not block: two nodes each held on the other's full edge never wake. An undeclared cycle fails with `form a cycle of data edges`. A control edge holds no queue, so a cycle it closes needs no mark.

## Stream vocabulary

| Format | Modality | Payload |
| --- | --- | --- |
| `pcm16` | `audio` | 16 kHz mono signed 16-bit PCM |
| `pcm_f32le` | `audio` | 24 kHz mono float32 PCM |
| `image_frame` | `image` | One JPEG or PNG camera frame, at the time of the audio unit it came with |
| `transcript` | `transcript` | `{"text", "is_final"}`; one final transcript per utterance |
| `messages` | `text` | `{"messages": [{"role", "content"}, ...]}` |
| `text_delta` | `text` | `{"text"}`, one increment of the reply |
| `tool_call` | `tool` | `{"calls": [{"id", "name", "arguments"}, ...]}`; the generator asks for these tools |
| `tool_response` | `tool` | `{"responses": [{"id", "name", "response"}, ...]}`; one response per call, in call order |
| `session` | `control` | `{"type", ...}` |

Control types sent into a Session are `commit` (the utterance ends here) and `configure` (settings change). Control types a Session emits are `speech.started` and `speech.stopped` with `t_ms` on the input timeline, `turn.queued` when an utterance waits for its reply, `turn.started` when the reply begins, `turn.done` after the reply's audio, `turn.empty` for an utterance without words, and `configured` with `{"nodes": {name: kind}}`. Every chunk of a reply carries its turn number in the chunk's `turn` field, numbered from 1 per Session in utterance order: the turn controls, the `messages`, `text_delta`, `tool_call` and `tool_response` chunks and the reply audio; every turn opens with `turn.queued` and closes with `turn.done` or `turn.empty`.

## VAD node

Voice activity detection is a stage node: `sglang_omni.models.vad` hosts one Silero detector per Session in a CPU stage of its own, `vad`, declared with `stages: [vad]`, reading the Session input's audio and controls and publishing `session`. It emits `speech.started` and `speech.stopped` controls whose `t_ms` is the boundary on the input audio timeline, `speech.started` reaching `prefix_padding_ms` before the first speech frame; `commit` resets the detector. Nodes that place utterances read it through a `vad:session` edge. Its detector defaults are the stage's `factory` options `threshold`, `prefix_padding_ms` and `silence_duration_ms`; a Session's `turn_detection` overrides them for that Session alone, at open and through `configure`.

## Interruption

An `interrupt` naming a turn that reaches a node on a control edge first drops that turn's chunks from other nodes still queued for the node, on its edges and among its unsent appends, and a chunk of that turn that a node emits later no longer reaches it; the client's own input is never dropped. The node's own output of that turn, whether already buffered or emitted later, reaches neither its readers nor the client, except the `turn.interrupted` that ends the turn. Control edges are delivered upstream first, so a node's stale output is fenced before its readers hear the event. An `interrupt` without a turn, such as the client's `response.cancel`, means the open reply, which only the node knows; the Session leaves its queues to the node.

## Delivery pacing

A Session delivers reply audio in real time. For every node of the Session that publishes audio, an audio chunk leaves at most `max_output_lead_ms` ahead of the wall clock since the graph's published `turn.started` for the reply left, cut at that horizon; the node's later audio and its `turn.done` wait behind it, and its other chunks, such as transcripts, reply text, speech boundaries and the confirmation of each input append, pass, so the client keeps appending while the reply plays. Once the input has ended everything leaves at once. Audio leaving together, at the input's end or when delivery catches up, is joined per turn into contiguous chunks of at most `max_output_lead_ms`, so a held 40 s reply reaches the client as about 40 chunks rather than one per generated chunk. A Session that is one stage route paces its output the same way. `max_output_lead_ms` is a field of the pipeline config (default 1000); an interruption cannot take back audio the client already holds, so it bounds how long the client may keep speaking after the user started.

When the Session releases a node's `turn.done`, it delivers it to the nodes reading it and returns it to the node itself as a `control`, so a node can keep its reply open, and interruptible, until the reply's audio has left. A `turn.interrupted` the node publishes, or an `interrupt` control naming a turn, drops what the Session still holds of that turn, its `turn.done` included. Held audio counts against the Session's `max_output_bytes`.

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


## Cascade components

The cascade config's `kind` shorthand expands each component into its own CPU stage and Session node, with the stages sharing the `cascade` process. Data and control edges use the same coordinator runtime as native model stages. Feedback edges must be declared explicitly; a native stage may read a component's output directly.

A node declares either a component `kind`, with its options beside it, or a `stages` route for a native model or the VAD. `inputs` lists the sources it reads: `input` is the client's audio and controls, `node` a node's every format, `node:format` one format, `input:session` the client's controls, and `{kind: control, source, event}` a control edge. A component accepts the formats its kind consumes; the declaration is validated when the config loads: names are unique, every input exists, every format a component requires other than `session` controls is produced by one of its inputs, exactly one reply-ending kind publishes `session`, and the kinds that open turns (`dialog`) and end them (`openai_chat` or `openai_speech`) are present once. At startup every service kind is probed once: `openai_transcription`, `openai_chat` and `openai_speech` request `{base_url}/models`, `realtime_transcription` requests `/v1/models` on the HTTP form of its `base_url`; a failed probe fails the start with the node's name.

A kind is a config class with a default `kind`, a constructor and the formats it consumes and produces, registered through `COMPONENTS.register` in `sglang_omni.models.streaming_cascade.components`; `opens_turns` marks the kind that numbers turns, `ends_turns` the one whose `turn.done` ends a reply, and `optional` the consumed formats a configuration may leave unconnected. `running_transcript` is the kind that turns the one running text of a Nemotron 3.5 ASR stage served as a node of the same Session into utterance transcripts, cut at `speech.stopped` or at a commit plus `tail_ms`.

A `configure` control reaches every node reading `input`. Its standard fields reach every node with the matching option, `instructions` as `system_prompt` and `sampling` as `temperature`, `max_tokens`; `nodes: {name: {...}}` changes one node's options, and a different `kind` replaces its component, which adopts the previous one's state. Names and inputs cannot change; a node is not replaced while it is speaking, the change waits for the reply to end. Each node answers with `configured {"nodes": {name: kind}}` on its `session` output.

An EOS append ends the input readers in graph order, after their upstream input readers finish. Its confirmation waits for pending routed input and published reply turns to drain. Background component failures fail the output read without waiting for another append.

The cascade graph enables `shared_timeline`: its published reply chunks use one coordinator-owned media clock. A reply starts no earlier than the latest client audio or the previous reply's audio end; audio advances that clock, and interruption resets it to the client audio clock. Speech boundaries retain their original input timestamps. Other graphs preserve their nodes' timestamps by default.
