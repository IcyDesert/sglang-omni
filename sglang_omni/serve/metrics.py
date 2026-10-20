# SPDX-License-Identifier: Apache-2.0
"""Instrument HTTP requests and preserve the metrics import entry points."""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang_omni.scheduling.metrics_snapshot import (
    BatchMetricEvent,
    ExecutionMetricEvent,
    QueueMetricEvent,
    SchedulerMetricEvent,
    StageMetricEvent,
)
from sglang_omni.serve.metrics_registry import (
    BackendMetricsCollector,
    validate_metrics_environment,
)
from sglang_omni.serve.request_metrics import RequestMetrics, TokenObservation

__all__ = [
    "BackendMetricsCollector",
    "BatchMetricEvent",
    "ExecutionMetricEvent",
    "MetricsMiddleware",
    "QueueMetricEvent",
    "RequestMetrics",
    "SchedulerMetricEvent",
    "StageMetricEvent",
    "TokenObservation",
    "validate_metrics_environment",
]


class MetricsMiddleware:
    """Pure ASGI instrumentation: do not buffer or parse streaming bodies."""

    def __init__(self, app: ASGIApp, metrics: RequestMetrics) -> None:
        self.app: ASGIApp = app
        self.metrics: RequestMetrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in {
            "/metrics",
            "/health",
            "/health_generate",
            "/favicon.ico",
        }:
            return await self.app(scope, receive, send)
        else:
            pass
        started_at_s = time.perf_counter()
        finished_at_s = None
        status = 500
        outcome: Literal[
            "incomplete", "completed", "disconnected", "cancelled", "error"
        ] = "incomplete"
        disconnected = False
        self.metrics.http_inflight.inc()

        async def wrapped_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            else:
                pass
            return message

        async def wrapped_send(message: Message) -> None:
            nonlocal status, finished_at_s, outcome
            if message["type"] == "http.response.start":
                status = message["status"]
            else:
                pass
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                finished_at_s = time.perf_counter()
                outcome = "completed"
            else:
                pass

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
            if disconnected and finished_at_s is None:
                outcome = "disconnected"
            else:
                pass
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            self.metrics.http_inflight.dec()
            method = scope["method"]
            if method not in {
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "HEAD",
                "OPTIONS",
            }:
                method = "OTHER"
            else:
                pass
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.http_requests.labels(method, route, str(status), outcome).inc()
            self.metrics.http_latency.labels(method, route, outcome).observe(
                (finished_at_s if finished_at_s is not None else time.perf_counter())
                - started_at_s
            )
