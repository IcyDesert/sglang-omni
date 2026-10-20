# SPDX-License-Identifier: Apache-2.0
"""Minimal SenseNova UMM routing and output-stage contracts."""

import threading

from sglang_omni.client.client import umm_segment_chunk
from sglang_omni.client.types import GenerateChunk, UMMSegment
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.pipeline.umm import (
    CONTINUATION_METADATA_KEY,
    Continuation,
    UMMController,
    UMMOutputRelay,
    route_umm,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

pytest_plugins = ["tests.unit_test.fixtures.sensenova_umm"]


def test_route_umm_uses_the_continuation_stage() -> None:
    continuation = Continuation("execution", 0, "generate", "generate", "nonce")
    payload = StagePayload(
        "request",
        OmniRequest("prompt", metadata={CONTINUATION_METADATA_KEY: continuation}),
        None,
    )
    assert route_umm("request", payload) == "generate"


def test_output_relay_forwards_stream_and_result() -> None:
    relay = UMMOutputRelay()
    relay.inbox.put(
        IncomingMessage(
            "request",
            "stream_chunk",
            StreamItem(0, {"kind": "image"}, "generate", {"modality": "image"}),
        )
    )
    relay.inbox.put(IncomingMessage("request", "new_request", {"type": "umm_result"}))
    thread = threading.Thread(target=relay.start)
    thread.start()
    try:
        stream = relay.outbox.get(timeout=2)
        result = relay.outbox.get(timeout=2)
    finally:
        relay.stop()
        thread.join(timeout=2)
    assert stream.type == "stream"
    assert stream.data == {"kind": "image"}
    assert stream.metadata == {"modality": "image"}
    assert result.type == "result"
    assert result.data == {"type": "umm_result"}


def test_streamed_usage_segment_uses_the_payload_stage() -> None:
    segment = UMMSegment("execution", 0, "usage", {"stage_name": "generate"})
    chunk = umm_segment_chunk(GenerateChunk(request_id="request"), segment.to_dict())
    assert chunk.stage_name == "generate"


def test_controller_route_and_output_relay_end_to_end(
    controller: UMMController, encoded_image: str
) -> None:
    dispatched = controller.transition(
        StagePayload("request", OmniRequest("prompt"), None)
    )
    assert dispatched is not None
    assert route_umm("request", dispatched) == "generate"
    dispatched.data = {
        "modality": "image",
        "image_b64": encoded_image,
        "image_usage": {"input_tokens": 3, "total_tokens": 3},
    }
    completed = controller.transition(dispatched)
    assert completed is not None
    assert route_umm("request", completed) == "output"
    assert isinstance(completed.data, dict)
    assert [segment["kind"] for segment in completed.data["segments"]] == [
        "image",
        "usage",
    ]

    relay = UMMOutputRelay()
    relay.inbox.put(IncomingMessage("request", "new_request", completed.data))
    thread = threading.Thread(target=relay.start)
    thread.start()
    try:
        message = relay.outbox.get(timeout=2)
    finally:
        relay.stop()
        thread.join(timeout=2)
    assert message.type == "result"
    assert isinstance(message.data, dict)
    assert message.data["segments"][1]["kind"] == "usage"
