# SPDX-License-Identifier: Apache-2.0
"""The forced-alignment worker's text helpers; they run where the worker's `qwen-asr` package is installed."""

import pytest

worker = pytest.importorskip("examples.streaming_cascade.alignment_worker")


@pytest.mark.parametrize(
    "language, words",
    [("English", "twenty one"), ("Chinese", "二十一"), ("Japanese", "21")],
)
def test_number_words_read_digits_in_the_request_language(language, words):
    assert worker.number_words("21", language) == words


@pytest.mark.parametrize(
    "language, spoken",
    [("English", "我有 three 个"), ("Chinese", "我有 三 个")],
)
def test_spoken_text_reads_digits_in_the_request_language(language, spoken):
    text, origin = worker.spoken_text("我有3个", language)
    assert text == spoken
    assert origin == [0, 1] + [2] * (len(spoken) - 3) + [3]


@pytest.mark.parametrize("language", ["Chinese", None])
def test_a_request_names_its_language_or_leaves_it_to_the_worker(language):
    body = {"text": "1", "audio": "", "sample_rate_hz": 24000, "audio_format": "pcm16"}
    if language is None:
        pass
    else:
        body["language"] = language
    assert worker.AlignmentRequest.model_validate(body).language == language
