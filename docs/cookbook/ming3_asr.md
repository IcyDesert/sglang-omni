# Ming 3 ASR

Ming 3 ASR transcribes Chinese and English speech. The pipeline runs a Whisper
audio encoder and a Ling V3 mixture-of-experts language model. The encoder's
embeddings fill the audio placeholder tokens of the transcription prompt.

## Run locally

The default placement uses one GPU. Preprocessing runs on the CPU. The audio
encoder and the language model share GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-asr-tiny --host 127.0.0.1 --port 8011
```

Transcribe a recording:

```bash
curl http://127.0.0.1:8011/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny -F file=@input.wav -F language=en
```

The prompt is the checkpoint's transcription instruction followed by the audio.
The chat template is not used. The model answers with a language label, a tab,
and the transcript, and the endpoint returns only the transcript.

`language` accepts `en` or `zh` and prefills the language label, as the
checkpoint's language hint does. Without `language` the model identifies the
language itself. `prompt` adds topic or hotword context after the instruction
and must not contain newlines. Other languages and multi-line prompts are
rejected.

Decoding is greedy unless `temperature` is set, and stops at the end-of-text
token or after 256 new tokens. Uploads longer than 30 seconds are split into
30 second clips, and the clip transcripts are joined.

## Correctness baseline

Validation pins model revision `182ac2b94ee42ee9992286bca1f8df05aa803562` and
SeedTTS dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`. The client
sent each SeedTTS reference recording with its language at concurrency 32,
after a full unscored warmup pass, on one H200.

| Split | Entries | Error rate | Throughput | Mean latency |
|---|---|---|---|---|
| English | 1088 | **1.13% WER** | 46.9 req/s | 0.68 s |
| Chinese | 2020 | **0.57% CER** | 52.4 req/s | 0.61 s |

All entries succeeded. Chinese is scored per character.

The checkpoint's own code was the reference for parity on the first 200
entries of each split. It ran BF16 with SDPA, greedy decoding, and 256 new
tokens. Prompt token ids and audio embeddings were bitwise identical for all
400 recordings, with and without the language hint. Sequential transcripts
matched exactly as follows.

| Split | Without language | With language |
|---|---|---|
| English | 196/200 | 197/200 |
| Chinese | 200/200 | 200/200 |

The differing transcripts differ by one word that sounds alike, such as
"wasteland" and "wasteground". The language model's kernels differ numerically
from the reference implementation. An eager language model also differed on a
few English words (195/200 and 193/200). The reference decoder additionally
blocks repeated 10-grams, which this pipeline does not implement.

```bash
python -m pytest tests/unit_test/ming3_asr -q
python -m benchmarks.eval.benchmark_asr_seedtts --port 8011 \
  --model-path inclusionAI/Ming-3.0-asr-tiny --lang en \
  --concurrencies 32 --repeats 1 --warmup \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --output results/ming3_asr/asr_en.json
```

## Optimizations and parity

The language model decodes with SGLang CUDA graphs. The audio encoder runs
each 30 second chunk separately at batch size one, as the reference does.
Whisper attention has no padding mask, so padding chunks into one batch would
change their embeddings. Batch size one keeps embeddings bitwise identical to
the reference.

An encoder pass takes about 11 ms on an H200. The first pass at a new chunk
length takes about 56 ms, because PyTorch builds a cuDNN attention plan for
each sequence length. There are at most 1500 lengths, and plans are cached.
Another attention backend would avoid this cost, but its embeddings would no
longer match the reference bitwise.

The encoder's layer norms run in float32, as in the checkpoint's encoder. Ming 3
audio shares this encoder but keeps BF16 layer norms and batched chunks, which
match its own reference.
