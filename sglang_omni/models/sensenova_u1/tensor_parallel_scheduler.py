# SPDX-License-Identifier: Apache-2.0
"""Serial execution of leader-dispatched work containing TP collectives."""

from __future__ import annotations

import asyncio
import threading
from typing import Protocol

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class RequestCompute(Protocol):
    def __call__(self, payload: StagePayload, /) -> StagePayload: ...


class RequestValidator(Protocol):
    def __call__(self, payload: StagePayload, /) -> None: ...


class ShutdownCallback(Protocol):
    def __call__(self) -> None: ...


class FailureCallback(Protocol):
    def __call__(self, exc: BaseException, /) -> None: ...


class TensorParallelScheduler(SimpleScheduler):
    """Drain dispatched TP forwards before cleanup, even after cancellation."""

    preserve_tp_work_on_abort: bool = True

    def __init__(
        self,
        request_compute: RequestCompute,
        *,
        validate_request: RequestValidator | None = None,
        shutdown_callback: ShutdownCallback | None = None,
    ) -> None:
        super().__init__(request_compute)
        self.validate_request: RequestValidator | None = validate_request
        self.wait_for_shutdown: bool = True
        self.shutdown_callback: ShutdownCallback | None = shutdown_callback
        self.stop_marker: IncomingMessage = IncomingMessage(
            request_id="", type="stream_done"
        )
        self.stop_requested: bool = False
        self.stop_lock: threading.Lock = threading.Lock()
        self.compute_error: Exception | None = None
        self.failure_callback: FailureCallback | None = None

    def abort(self, request_id: str) -> None:
        # note (AuFlow): Stage suppresses cancelled results; ranks must still run collectives.
        pass

    def next_message(self) -> IncomingMessage | None:
        message = super().next_message()
        if message is self.stop_marker:
            self.running = False
            return IncomingMessage(request_id="", type="wakeup")
        else:
            pass
        return message

    def start(self) -> None:
        try:
            super().start()
            if self.compute_error is not None:
                raise RuntimeError(
                    "TP forward failed; the collective sequence cannot be reused"
                ) from self.compute_error
            else:
                pass
        except BaseException as exc:
            # note (AuFlow): Cleanup can block in a broken collective. Notify the stage first so the supervisor can shut down the entire TP group.
            self.wait_for_shutdown = False
            if self.failure_callback is not None:
                self.failure_callback(exc)
            else:
                pass
            raise
        finally:
            self.running = False
            callback = self.shutdown_callback
            self.shutdown_callback = None
            if callback is not None:
                callback()
            else:
                pass

    def run_single(
        self, message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        try:
            super().run_single(message, loop)
        except Exception as exc:
            # note (AuFlow): A failed rank must not reuse collectives for the next request.
            self.compute_error = exc
            self.running = False
            raise

    def stop(self) -> None:
        with self.stop_lock:
            if not self.stop_requested:
                self.stop_requested = True
                self.inbox.put(self.stop_marker)
            else:
                pass
