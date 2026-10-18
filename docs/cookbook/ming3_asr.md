# Ming-3.0-asr-tiny

[Ming-3.0-asr-tiny](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny) transcribes Chinese and English speech. A Whisper audio encoder and a Ling V3 mixture-of-experts language model share the work. The encoder's embeddings fill the audio placeholder tokens of the transcription prompt.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

Startup uses one GPU. Preprocessing runs on the CPU. The audio encoder and the language model share that GPU.

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-asr-tiny
```

## Transcription

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@input.wav \
  -F language=en
```

The prompt is the checkpoint's transcription instruction followed by the audio. The chat template is not used. The model answers with a language label, a tab, and the transcript, and this endpoint returns only the transcript.

`language` accepts `en` or `zh` and prefills that language label. Without `language`, the model identifies the language itself. `prompt` adds one line of topic or hotword context after the instruction. Any other language, or a `prompt` that contains a newline, is rejected with HTTP 400.

Decoding is greedy unless `temperature` is set. Generation stops at the end-of-text token or after 256 new tokens. An upload longer than 30 seconds is split into 30-second clips, and the clip transcripts are joined.
