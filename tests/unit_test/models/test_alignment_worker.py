# SPDX-License-Identifier: Apache-2.0
"""The forced-alignment worker's request contract; it runs where the worker's `qwen-asr` package is installed."""

import pytest

worker = pytest.importorskip("examples.streaming_cascade.alignment_worker")


@pytest.mark.parametrize("language", ["Chinese", None])
def test_a_request_names_its_language_or_leaves_it_to_the_worker(language):
    body = {"text": "1", "audio": "", "sample_rate_hz": 24000, "audio_format": "pcm16"}
    if language is None:
        pass
    else:
        body["language"] = language
    assert worker.AlignmentRequest.model_validate(body).language == language
