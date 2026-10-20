# SPDX-License-Identifier: Apache-2.0
"""Create isolated Prometheus collectors for request and backend metrics."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

METRICS_DEPENDENCY_ERROR: ModuleNotFoundError | None = None
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client.core import Metric
except ModuleNotFoundError as error:
    if error.name != "prometheus_client":
        raise
    else:
        METRICS_DEPENDENCY_ERROR = error
else:
    pass


from sglang_omni.scheduling.metrics_snapshot import ENGINE_GAUGES, BackendMetricFamily


def validate_metrics_environment() -> None:
    if METRICS_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Metrics require prometheus-client; install sglang-omni[metrics]."
        ) from METRICS_DEPENDENCY_ERROR
    else:
        pass
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        raise ValueError(
            "Omni metrics use a single HTTP-process registry. Unset "
            "PROMETHEUS_MULTIPROC_DIR before launching the service."
        )
    else:
        pass


class BackendMetricsCollector:
    """Export leader snapshots, excluding HTTP-owned metric families."""

    def __init__(self, model_name: str, owned_names: set[str]) -> None:
        self.model_name: str = model_name
        self.owned_names: set[str] = owned_names
        self.snapshots: list[tuple[str, list[BackendMetricFamily]]] = []

    def describe(self) -> list[Metric]:
        return []

    def collect(self) -> Iterator[Metric]:
        families: dict[str, Metric] = {}
        for stage, snapshot in self.snapshots:
            for backend_family in snapshot:
                name = backend_family["name"]
                if name in self.owned_names:
                    continue
                else:
                    pass
                family = families.setdefault(
                    name,
                    Metric(
                        name, backend_family["documentation"], backend_family["type"]
                    ),
                )
                if family.type != backend_family["type"]:
                    raise ValueError(f"Conflicting backend metric type: {name}")
                else:
                    pass
                for sample in backend_family["samples"]:
                    labels = {
                        **sample["labels"],
                        "model_name": self.model_name,
                        "omni_stage": stage,
                    }
                    family.add_sample(sample["name"], labels, sample["value"])
        yield from families.values()


@dataclass(kw_only=True)
class RequestMetricCollectors:
    """Prometheus objects owned by one HTTP service instance."""

    registry: CollectorRegistry
    e2e: Histogram
    ttft: Histogram
    itl: Histogram
    prompt_tokens: Counter
    generation_tokens: Counter
    token_lengths: dict[str, Histogram]
    cached_tokens: Counter
    requests: Counter
    aborted: Counter
    failed: Counter
    running: Gauge
    waiting: Gauge
    engine_gauges: dict[str, Gauge]
    queue_latency: Histogram
    batch_size: Histogram
    execution: Histogram
    stage_duration: Histogram
    audio_seconds: Counter
    images: Counter
    http_requests: Counter
    http_latency: Histogram
    http_inflight: Gauge
    backend: BackendMetricsCollector


def create_request_collectors(model_name: str) -> RequestMetricCollectors:
    validate_metrics_environment()
    registry: CollectorRegistry = CollectorRegistry()
    labels = ("model_name", "is_streaming")
    latency_buckets = (
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        2,
        4,
        6,
        8,
        10,
        20,
        40,
        60,
        80,
        100,
        200,
        400,
    )
    e2e: Histogram = Histogram(
        "sglang:e2e_request_latency_seconds",
        "Histogram of End-to-end request latency in seconds",
        labels,
        buckets=(*latency_buckets, 600, 1200, 1800, 2400),
        registry=registry,
    )
    ttft: Histogram = Histogram(
        "sglang:time_to_first_token_seconds",
        "Histogram of time to first token in seconds.",
        labels,
        buckets=latency_buckets,
        registry=registry,
    )
    itl: Histogram = Histogram(
        "sglang:inter_token_latency_seconds",
        "Histogram of inter-token latency in seconds.",
        ("model_name",),
        buckets=(
            0.002,
            0.004,
            0.006,
            0.008,
            0.010,
            0.015,
            0.020,
            0.025,
            0.030,
            0.035,
            0.040,
            0.060,
            0.080,
            0.100,
            0.200,
            0.400,
            0.600,
            0.800,
            1,
            2,
            4,
            6,
            8,
        ),
        registry=registry,
    )
    prompt_tokens: Counter = Counter(
        "sglang:prompt_tokens_total",
        "Number of prompt tokens processed.",
        labels,
        registry=registry,
    )
    generation_tokens: Counter = Counter(
        "sglang:generation_tokens_total",
        "Number of tokens generated.",
        labels,
        registry=registry,
    )
    token_buckets = (
        100,
        300,
        500,
        700,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        12500,
        15000,
        17500,
        20000,
        22500,
        25000,
        27500,
        30000,
        35000,
        40000,
        60000,
        80000,
        100000,
        200000,
        300000,
        400000,
        600000,
        800000,
        1000000,
        1100000,
    )
    token_lengths: dict[str, Histogram] = {
        name: Histogram(
            f"sglang:{name}_histogram",
            f"Histogram of {name.replace('_', ' ')} length.",
            ("model_name",),
            buckets=token_buckets,
            registry=registry,
        )
        for name in ("prompt_tokens", "uncached_prompt_tokens", "generation_tokens")
    }
    cached_tokens: Counter = Counter(
        "sglang:cached_tokens_total",
        "Number of cached prompt tokens by source.",
        ("model_name", "cache_source"),
        registry=registry,
    )
    requests: Counter = Counter(
        "sglang:num_requests_total",
        "Number of requests processed.",
        labels,
        registry=registry,
    )
    aborted: Counter = Counter(
        "sglang:num_aborted_requests_total",
        "Number of requests aborted.",
        ("model_name",),
        registry=registry,
    )
    failed: Counter = Counter(
        "sglang_omni:failed_requests_total",
        "Number of failed pipeline requests.",
        ("model_name",),
        registry=registry,
    )
    running: Gauge = Gauge(
        "sglang:num_running_reqs",
        "Number of requests running in a stage.",
        ("model_name", "stage"),
        registry=registry,
    )
    waiting: Gauge = Gauge(
        "sglang:num_queue_reqs",
        "Number of requests waiting in a stage.",
        ("model_name", "stage"),
        registry=registry,
    )
    engine_gauges: dict[str, Gauge] = {
        name: Gauge(
            f"sglang:{name}",
            f"Backend scheduler {name}.",
            ("model_name", "stage"),
            registry=registry,
        )
        for name in ENGINE_GAUGES
    }
    queue_latency: Histogram = Histogram(
        "sglang:queue_time_seconds",
        "Scheduler queue duration in seconds.",
        ("model_name", "stage"),
        registry=registry,
        buckets=(
            0,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            3,
            4,
            5,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2500,
            3000,
        ),
    )
    batch_size: Histogram = Histogram(
        "sglang_omni:scheduler_batch_size",
        "Requests dispatched per compute call.",
        ("model_name", "stage"),
        buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        registry=registry,
    )
    execution: Histogram = Histogram(
        "sglang_omni:stage_compute_duration_seconds",
        "Host compute duration, without device synchronization.",
        ("model_name", "stage", "outcome"),
        registry=registry,
    )
    stage_duration: Histogram = Histogram(
        "sglang_omni:stage_request_duration_seconds",
        "Stage dispatch through routing/cleanup, including scheduler waiting.",
        ("model_name", "stage", "outcome"),
        registry=registry,
    )
    audio_seconds: Counter = Counter(
        "sglang_omni:audio_output_seconds_total",
        "Completed output waveform duration before encoding/speed conversion.",
        ("model_name", "stage"),
        registry=registry,
    )
    images: Counter = Counter(
        "sglang_omni:generated_images_total",
        "Images in the completed output list, not diffusion steps.",
        ("model_name", "stage"),
        registry=registry,
    )
    http_requests: Counter = Counter(
        "sglang_omni:http_requests_total",
        "HTTP requests including errors.",
        ("method", "route", "status_code", "outcome"),
        registry=registry,
    )
    http_latency: Histogram = Histogram(
        "sglang_omni:http_request_duration_seconds",
        "HTTP duration through the final response body or disconnect.",
        ("method", "route", "outcome"),
        registry=registry,
        buckets=(*latency_buckets, 600, 1200, 1800, 2400),
    )
    http_inflight: Gauge = Gauge(
        "sglang_omni:http_requests_inflight",
        "Active HTTP requests.",
        registry=registry,
    )
    backend: BackendMetricsCollector = BackendMetricsCollector(
        model_name, {family.name for family in registry.collect()}
    )
    registry.register(backend)
    return RequestMetricCollectors(
        registry=registry,
        e2e=e2e,
        ttft=ttft,
        itl=itl,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        token_lengths=token_lengths,
        cached_tokens=cached_tokens,
        requests=requests,
        aborted=aborted,
        failed=failed,
        running=running,
        waiting=waiting,
        engine_gauges=engine_gauges,
        queue_latency=queue_latency,
        batch_size=batch_size,
        execution=execution,
        stage_duration=stage_duration,
        audio_seconds=audio_seconds,
        images=images,
        http_requests=http_requests,
        http_latency=http_latency,
        http_inflight=http_inflight,
        backend=backend,
    )
