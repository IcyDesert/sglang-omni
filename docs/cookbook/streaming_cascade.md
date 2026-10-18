# Streaming Cascade

The streaming cascade turns independent speech recognition, language model and speech synthesis services into one voice conversation served on `/v1/realtime`. A profile names the services; the server adds voice activity detection, the conversation history, turn taking and interruption.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then install the network clients the nodes use:

```bash
python -m pip install websockets httpx
```

The profile `qwen3-asr-qwen3.8-27b-customvoice` runs three Qwen models on two GPUs: [Qwen3-ASR](qwen3_asr.md) for recognition, Qwen3.8-27B for the answers and [Qwen3-TTS](qwen3_tts.md) CustomVoice for the speech. Install the two speech models' dependencies as their cookbooks say, then download the checkpoints (all public, no token):

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download Qwen/Qwen3.8-27B
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Server Configuration

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-customvoice/launch.yaml --enable-realtime --port 8765
```

One command starts everything: the profile's `services` are started in order and waited for, a service already answering on its port is reused, and the cascade starts once all of them answer; stopping the server stops the services it started. This profile puts Qwen3.8-27B on GPU 0 and the two speech models on GPU 1 through each service's `env`. `examples/streaming_cascade/profiles` has four profiles:

| Profile | Conversation |
| --- | --- |
| `qwen3-asr-qwen3.8-27b-customvoice` | Qwen3-ASR, Qwen3.8-27B with thinking off and Qwen3-TTS CustomVoice with the built-in speaker `Vivian`, on two GPUs |
| `qwen3-asr-qwen3-tts-customvoice` | The same with Qwen3-4B as the language model, on one GPU |
| `nemotron-qwen3-tts` | Nemotron 3.5 streaming ASR as a service on port 8001, Qwen3-4B and Qwen3-TTS Base with a cloned voice on port 8004 |
| `nemotron-stage-qwen3-tts` | The same, with Nemotron 3.5 ASR running inside this server instead of as a service |

To use other services, change a node's `base_url` and `model` in the profile and the matching `services` entry, or drop the entry for a service you run elsewhere. Any OpenAI-compatible transcription or speech endpoint works through the `openai_transcription` and `openai_speech` kinds, and the language model through `openai_chat`. A command-line override such as `session_nodes.llm.base_url=http://127.0.0.1:9000/v1` changes one field without editing the file. For voice cloning with Qwen3-TTS Base, set `reference_audio_path` to a WAV the speech server can read and `reference_text` to its transcript.

## Voice Conversation

Open the browser page to try the conversation:

```bash
python playground/cascade/app.py --api-base http://127.0.0.1:8765 --port 8090
```

Open `http://127.0.0.1:8090`, speak, and the reply plays as it is produced. Speaking while the reply plays interrupts it.

Any OpenAI Realtime client works the same way. Open a WebSocket to `/v1/realtime`, wait for `session.created`, and send the session settings:

```json
{"type": "session.update", "event_id": "cfg-1", "session": {"instructions": "Answer in one short English sentence.", "output_modalities": ["text", "audio"], "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 800}}}}}
```

Send 16 kHz mono PCM16 audio at real-time pace as `{"type": "input_audio_buffer.append", "event_id": "...", "audio": "<base64>", "sglang": {"seq": 0}}`, with `seq` counting up from 0 for the whole session. The server answers with these events:

| Event | Meaning |
| --- | --- |
| `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` | The start and the end of your utterance. |
| `conversation.item.input_audio_transcription.delta`, `.completed` | Partial and final transcript of what you said. |
| `response.created` | A reply begins. |
| `response.output_text.delta`, `response.output_audio.delta` | Reply text and reply audio, base64 PCM16 at 24 kHz mono. |
| `response.done` | The reply is complete; `response.status_details.reason` is `stop`, `empty_turn` when the utterance had no words, or `interrupted` when you spoke over it. |
| `error` | The request or the session failed. |

`response.cancel` interrupts the reply from the client side. An interrupted reply stays in the history only up to the last sentence whose audio the server had sent in full before the interruption; a reply interrupted in its first sentence leaves an empty answer. `conversation.item.truncate` with `audio_end_ms` tells the server how much of the reply was heard; the history then keeps the reply up to the last sentence whose audio ended by then, never past what the server sent, and marks the entry uncertain when the listener heard part of the next sentence.

To end turns yourself instead of by silence, set `"turn_detection": null`; the server then grants `client_commit` and `input_audio_buffer.commit` ends the utterance after the audio sent before it.

Instructions, sampling and turn detection can change at any time with another `session.update`; the change applies from the next turn.

## Session API

The same conversation runs without the realtime protocol through the Session API:

```bash
PYTHONPATH=. python examples/streaming_cascade/run.py question.wav followup.wav --output response.wav
```

Each input is a 16 kHz mono signed 16-bit PCM WAV. The command streams the files back to back through one Session of the profile named by `--config` and writes the reply audio to `response.wav`; the second file is answered with the first turn in the history. Pass `--commit` for a profile without the VAD node. The `/v1/session` protocol is described in [Session Graph](../developer_reference/session_graph.md#session-api).

## Profile Parameters

A profile is one `launch.yaml`. `services` lists the model services to start, each with a `name`, its `command`, the `port` it answers `/v1/models` on and optional `env`. Each entry of `session_nodes` is a node of the conversation, declared by its `kind` with the kind's options beside it. Copy the closest profile and change the services.

| Kind | Role | Options |
| --- | --- | --- |
| `realtime_transcription` | Streaming recognizer over an SGLang-Omni `/v1/realtime` transcription service | `base_url` (ws:// or wss://), `model`, `api_key_env`, `timeout_s` |
| `openai_transcription` | Recognizer over `/v1/audio/transcriptions` | `base_url`, `model`, `language`, `response_format` (`verbose_json`, the default, reports the language heard in each utterance; `json` for a service without it), `api_key_env`, `timeout_s` |
| `dialog` | Conversation history and turn taking | `system_prompt`, `max_history_chars`, `max_text_chars` |
| `openai_chat` | Language model over `/v1/chat/completions` | `base_url`, `model`, `max_tokens`, `temperature`, `extra_body` (request fields the service takes beyond these, such as `chat_template_kwargs`), `api_key_env`, `timeout_s` |
| `openai_speech` | Speech synthesis over `/v1/audio/speech` | `base_url`, `model`, `voice`, `reference_audio_path`, `reference_text`, `language` (the speech service's language tag for the reply; `auto` follows the user's language, see below), `max_sentence_words`, `api_key_env`, `timeout_s` |
| `running_transcript` | Cuts utterances from an ASR stage running inside the server | `tail_ms` |
| `alignment` | Reports how far into the reply text the reply audio has reached, so a truncated reply keeps what the listener heard | `method` (`sentence` or `forced`), `alignment_url`, `timeout_s` (default 2) |

`base_url` and `model` have no defaults. `api_key_env` names an environment variable whose value is sent as a bearer token. The `vad` node is declared with `stages: [vad]`; its `silence_duration_ms` (default 500) sets how long a pause ends an utterance.

`language` of `openai_speech` takes `auto`, a language name or nothing. With `auto`, every sentence is tagged with the language the recognizer last heard in an answered user turn, so a reply of digits or Latin words in a Chinese conversation is still spoken as Chinese; before the recognizer has named a language, or with a recognizer that reports none (`realtime_transcription`, `running_transcript`), a sentence with Chinese characters is tagged Chinese and any other English. A language name such as `English` tags every sentence with it except those with Chinese characters, which are tagged Chinese. Left empty, no tag is sent. The tag is the name the recognizer reports, such as `Chinese` from Qwen3-ASR, so the speech service must support that language; Qwen3-TTS speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.

The dialog cuts an interrupted reply at the points `reply.progress` reports, up to the audio the graph sent to the client before the interruption, which the graph reports to the readers of the `tts` node's `session` output as `reply.delivered`. Without an `alignment` node the dialog takes `reply.progress` from the `tts` node, which reports the end of each sentence. With one, as in `qwen3-asr-qwen3-tts-customvoice`, the `aligner` node reads `["llm:text_delta", "tts:pcm_f32le", "tts:session", "dialog:session"]` and the dialog's `{kind: control, source: aligner, event: reply.progress}` input replaces the one from `tts`; a profile names exactly one source of `reply.progress`.

`method: sentence` keeps the sentence ends, so a truncated reply keeps the sentences whose audio was played to the end and drops the sentence the listener was hearing. `method: forced` also sends each finished sentence and its audio to the worker at `alignment_url` and reports the word ends it returns, so the cut can fall inside a sentence that finished; when the worker fails or exceeds `timeout_s`, that sentence keeps its sentence end. A sentence that an interruption cuts off before its audio is complete gets no word ends and counts as unheard in both methods.

The profiles declare an `aligner` node with `method: forced` and `alignment_url: "http://127.0.0.1:8010"`, so start the forced-alignment worker before `sgl-omni serve`: `python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:0 --language English --port 8010`. It needs the `qwen-asr` package, which pins a `transformers` release of its own, so run it from a separate virtual environment; the server checks at startup that the worker answers. The aligner sends each sentence with the language of its reply when the `llm:text_delta` chunks carry one, which they do once the recognizer has named the user's language; the worker aligns that sentence and reads out its digits in that language, such as `Chinese`, and aligns a sentence sent without a language in its `--language` (default `Chinese`). To run without it, set the aligner's `method: sentence` and remove `alignment_url`.

At the top of the profile, `timeout_s` (default 120) bounds the wait for the last replies after the input ends, and `max_pending_output_chunks` with `max_pending_output_bytes` bound the reply output held for a client that has stopped reading.

## Benchmarking

The turn latency benchmark streams one WAV for several turns and prints, per turn, the time from speech end to first text, first audio, final transcript and completion:

```bash
python benchmarks/realtime_turn_latency.py --url ws://127.0.0.1:8765/v1/realtime --audio question.wav --turns 5
```

## Known Limitations

- **Reasoning models must have thinking off.** The cascade speaks everything the language model returns as content, so a model that thinks by default needs `enable_thinking: false` in the node's `extra_body` (as in `qwen3-asr-qwen3.8-27b-customvoice`), or its reasoning is synthesized.
- **History is bounded.** A conversation whose history exceeds `max_history_chars` fails its next turn; open a new session.
- **Audio only.** The cascade takes audio input and produces text and audio; image input is not supported.
