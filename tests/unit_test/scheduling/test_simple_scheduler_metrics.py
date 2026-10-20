# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
import time

import pytest

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def test_concurrent_workers_report_actual_running_requests() -> None:
    release = threading.Event()

    def compute(payload: int) -> int:
        assert release.wait(3)
        return payload

    scheduler = SimpleScheduler(compute, max_concurrency=2)
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for i in range(3):
            scheduler.enqueue(IncomingMessage(str(i), "new_request", i))
        deadline = time.monotonic() + 2
        while (
            scheduler.metrics_snapshot()["num_running_reqs"] != 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 2,
            "num_queue_reqs": 1,
        }
        release.set()
        for _ in range(3):
            scheduler.outbox.get(timeout=3)
        deadline = time.monotonic() + 2
        while (
            scheduler.metrics_snapshot()["num_running_reqs"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 0,
            "num_queue_reqs": 0,
        }
    finally:
        release.set()
        scheduler.stop()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_waiting_running_and_abort() -> None:
    scheduler = None

    def compute(payload: int) -> int:
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 1,
            "num_queue_reqs": 1,
        }
        return payload

    scheduler = SimpleScheduler(compute)
    scheduler.enqueue(IncomingMessage("a", "new_request", 1))
    scheduler.enqueue(IncomingMessage("b", "new_request", 2))
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 2}
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_single(scheduler.next_message(), loop)
    finally:
        loop.close()
    scheduler.abort("b")
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 0}


def test_batch_failure_clears_running() -> None:
    def fail(payloads: list[int]) -> list[int]:
        assert scheduler.metrics_snapshot()["num_running_reqs"] == 2
        raise ValueError("failed")

    scheduler = SimpleScheduler(lambda x: x, batch_compute_fn=fail, max_batch_size=2)
    batch = [IncomingMessage(str(i), "new_request", i) for i in range(2)]
    for message in batch:
        scheduler.enqueue(message)
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError):
            scheduler.run_batch(batch, loop)
    finally:
        loop.close()
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 0}


def test_arrival_hook_and_metrics_both_receive_admitted_work() -> None:
    arrivals: list[str] = []

    def on_arrival(payload: StagePayload) -> None:
        arrivals.append(payload.request_id)

    scheduler = SimpleScheduler(
        lambda payload: payload, request_arrival_hook=on_arrival
    )
    payload = StagePayload(
        request_id="arrival",
        request=OmniRequest(inputs="prompt", metadata={"omni_metrics_enabled": True}),
        data="prompt",
    )
    scheduler.enqueue(IncomingMessage("arrival", "new_request", payload))
    assert arrivals == ["arrival"]
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 1}
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_single(scheduler.next_message(), loop)
    finally:
        loop.close()
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 0, "num_queue_reqs": 0}


def test_partial_batch_failure_reports_each_request_outcome() -> None:
    def compute_batch(payloads: list[StagePayload]) -> list[StagePayload | ValueError]:
        return [payloads[0], ValueError("invalid image request")]

    scheduler = SimpleScheduler(
        lambda payload: payload, batch_compute_fn=compute_batch, max_batch_size=2
    )
    messages = [
        IncomingMessage(
            request_id,
            "new_request",
            StagePayload(
                request_id=request_id,
                request=OmniRequest(
                    inputs="prompt", metadata={"omni_metrics_enabled": True}
                ),
                data="prompt",
            ),
        )
        for request_id in ("good", "bad")
    ]
    for message in messages:
        scheduler.enqueue(message)
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_batch(messages, loop)
    finally:
        loop.close()
    outcomes: dict[str, str] = {}
    results: dict[str, str] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        if message.type == "metrics" and message.data["event"] == "execution":
            outcomes[message.request_id] = message.data["outcome"]
        elif message.type in ("result", "error"):
            results[message.request_id] = message.type
        else:
            pass
    assert results == {"good": "result", "bad": "error"}
    assert outcomes == {"good": "completed", "bad": "error"}
