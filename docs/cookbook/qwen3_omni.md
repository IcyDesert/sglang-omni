# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) is a multi-modal model
that accepts text, image, audio, and video input and can produce text-only or text + audio output.
This page covers every supported server configuration — use the generator to get the exact launch
command for your hardware, then check the tables to confirm your combination is supported.

## Prerequisites

```bash
docker pull hongccc/sglang-omni:dev
docker run -it --shm-size 32g --gpus all hongccc/sglang-omni:dev /bin/zsh
```

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.8"
```

See [Installation](../get_started/installation.md) for Docker digests and source installs.

## Server Configuration

Use the selector below to generate the exact launch command for your configuration.

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## Compatibility Matrix

Colocated topology requires `--config examples/configs/qwen3_omni_colocated_h20.yaml`
(or `qwen3_omni_colocated_h200.yaml` on H200) to set per-stage GPU memory budgets.

| Mode | Topology | Thinker TP | Precision | Status |
|---|---|---|---|---|
| Thinker-only | — | — | BF16 | ✅ |
| Thinker-only | — | — | FP8 | ✅ |
| Thinker-only | — | — | AutoRound INT4 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | BF16 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | FP8 | ✅ |
| Thinker-Talker | Disaggregated | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | Colocated | TP=1 | BF16 | ✅ |
| Thinker-Talker | Colocated | TP=1 | FP8 | ✅ |
| Thinker-Talker | Colocated | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |

## Input / Output Modalities

All input modality combinations work with both text-only and speech servers.
`modalities: ["text", "audio"]` requires a **speech-mode server** (omit `--text-only`).

| Input | Output | Speech server | Minimal request body | Notes |
|---|---|---|---|---|
| Text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]}` | — |
| Image + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]}` | — |
| Audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Image | Text | No | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from image |
| Video + text | Text | No | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]}` | — |
| Video + audio | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | content must be "" when the query is spoken |
| Video | Text | No | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]}` | content must be "" when query comes from video |
| Text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]}` | — |
| Image + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Image | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from image |
| Video + text | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| Video + audio | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when the query is spoken |
| Video | Text + Audio | **Yes** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | content must be "" when query comes from video |

### Sampling Parameters

Standard sampling parameters apply to the thinker stage. When `modalities` includes `"audio"`, the additional talker-specific parameters below control the speech generation independently.

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `temperature` | float | `0.0` (greedy) | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker; OpenAI-compatible alias for `max_tokens` |
| `stop` | str \| list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | Both |
| `audio` | dict | `null` | Speech response format config, e.g. `{"format": "wav"}` |
| `talker_temperature` | float | `0.9` | Talker (audio output only) |
| `talker_top_p` | float | `1.0` | Talker (audio output only) |
| `talker_top_k` | int | `50` | Talker (audio output only) |
| `talker_repetition_penalty` | float | `1.05` | Talker (audio output only) |
| `talker_max_new_tokens` | int | `4096` | Talker (audio output only) |
| `stage_sampling` | dict | `null` | Per-stage sampling override |
| `stage_params` | dict | `null` | Per-stage non-sampling params |
| `video_fps` | float | `null` | Frame sampling rate for video input (uses server default if unset) |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all video frames |
| `use_audio_in_video` | bool | `null` | Set to `true` to process embedded audio. Requests where all videos lack audio use video-only processing; mixing videos with and without audio is rejected. Audio decoding errors are reported. |

Non-streaming chat requests return HTTP 400 for invalid media, HTTP 503 when the request queue is full, and HTTP 500 for other
server failures. With `stream=true`, errors detected after the event stream
starts are sent as `data: {"error": {"message": "...", "type": "invalid_request_error", "code": 400}}`,
followed by `data: [DONE]`. A full request queue uses `type: "server_error"` and `code: 503`. Other server failures use `type: "server_error"` and
`code: 500`. The HTTP status remains 200 once streaming has started; clients
must check for an `error` event, including after partial output.

### Known Limitations

- **Multiple videos must have the same sampled frame rate.** The processor accepts one frame rate for video token timestamps. Requests with different sampled frame rates follow the invalid-input error contract above.

- **`modalities: ["text", "audio"]` has no effect on a text-only server.** No error is raised — the response simply contains no audio. Use a speech-mode server (without `--text-only`) to get audio output.
- **`content` must be `""` when the query is entirely in `audios`, `videos`, or `images`.** Leaving a text query in `content` alongside audio causes the model to process both, which is usually not what you want.
- **Colocated topology does not support `--thinker.tp_size 2`.** The server raises a `ValueError` at startup ("Qwen Phase 1 colocation does not support thinker TP"). Use disaggregated topology for TP=2.
- **Requests that exceed the model's context length are rejected with an error.** The preprocessor raises a `ValueError` when the prompt token count alone meets or exceeds `max_seq_len`, or when `prompt tokens + max_new_tokens ≥ max_seq_len`. Reduce input length or lower `max_tokens` to stay within the limit. Without `--preprocessing.factory.max_seq_len` and `--thinker.factory.max_seq_len`, the limit is the thinker's context in the checkpoint config (65,536 tokens for the Instruct checkpoint). Setting both to a lower value caps it. The talker's context is likewise the checkpoint's unless `--talker_ar.factory.max_seq_len` sets it. When a long spoken prompt leaves less room than `talker_max_new_tokens`, the talker generates only the frames that fit, so the audio can end before the text.

## Qwen3-Omni as one node

A node with `stages` and `execution` is a component the coordinator owns that runs the node over those stages; `execution` is the import path of its factory.

A speech-mode [Qwen3-Omni](qwen3_omni.md) hears, answers and speaks, so it stands in for the recognizer, the dialog, the generator and the synthesizer at once. A `qwen_omni` node whose `stages` are the seven Qwen3-Omni speech stages and whose `execution` is `sglang_omni.models.qwen3_omni.session_execution.create_session_execution_factory` serves it as the conversational node of this server's Session graph beside the VAD node: the pipeline adds the Qwen3-Omni stages, colocated on one GPU, and no `cascade` stage is involved. The `qwen_omni` node reads the session input's audio, controls and camera frames (`input:pcm16`, `input:session`, `input:image_frame`) and `vad:session`, and the client's `interrupt` on a control edge with `preempt: true`. It ends each utterance on `speech.stopped`, `commit` or the end of input, emits `turn.queued` and `turn.started`, answers the utterance from the audio itself with the history as text, and transcribes it in parallel: the reply streams as `text_delta` and 24 kHz `pcm16`, the transcript as `transcript`, and `turn.done` follows both the reply and the final transcript. Both sides of the turn enter the node's history once the transcript is in. Speech that starts during a reply, or an `interrupt` control operation, aborts the reply's request and publishes `turn.interrupted`; a `truncate` then drops the interrupted reply from the history. `configure` changes `instructions` and `sampling` for later turns. `examples/qwen3_omni/session.yaml` is that profile, with every Qwen3-Omni stage on one H200 as in the colocated speech config:

```bash
sgl-omni serve --config examples/qwen3_omni/session.yaml --enable-realtime --port 8765
```

The node sees as well as hears. With it in the graph, `/v1/realtime/capabilities` lists `image` among `input_modalities`, and the client sends camera frames with `sglang.input_image.append` as for [MiniCPM-o](minicpm_o.md): a base64 JPEG or PNG in `image` and its position on the audio timeline in `sglang.t_ms`, at most one frame per 200 ms unit. Each frame reaches the Session with the audio unit it falls in, and only the nodes declaring `input:image_frame` receive it. The reply request carries the newest four frames of the utterance as images before its audio, and the transcription request stays audio only. In the playground, the camera button appears once the session starts against this profile; turn the camera on during the call. Any other node reading the session input names its formats, as the VAD node does with `input:pcm16` and `input:session`; a node that reads every format of `input` without taking camera frames fails the server's startup.

The node is an execution component the coordinator owns: it submits ordinary requests to the Qwen3-Omni stages, which register each request to its session so an interruption or the Session's close aborts and releases exactly that work. The user transcript comes from a transcription request to the same model whose user message gives the instruction before the audio, decoded greedily; it can still drift into answering instead of transcribing, and a recognizer node of a cascade can replace it when the transcript must be exact. A reply stays open after its `turn.done` until the Session returns that `turn.done` once the reply's audio has left; see [Delivery pacing](../developer_reference/session_graph.md#delivery-pacing).
