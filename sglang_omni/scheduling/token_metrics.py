# SPDX-License-Identifier: Apache-2.0
"""Build host-only token telemetry after SGLang updates request output IDs."""

from typing import Protocol

from sglang_omni.scheduling.message import OutgoingMessage


class TokenMetricsRequest(Protocol):
    rid: str
    origin_input_ids: list[int]
    output_ids: list[int]


def batch_metrics_messages(
    scheduler_requests: list[TokenMetricsRequest], *, is_entry_rank: bool, now: float
) -> list[OutgoingMessage]:
    if not is_entry_rank:
        return []
    else:
        pass
    tracked: list[TokenMetricsRequest] = []
    messages: list[OutgoingMessage] = []
    for scheduler_request in scheduler_requests:
        payload = getattr(
            getattr(scheduler_request, "omni_data", None), "stage_payload", None
        )
        if payload is None or not payload.request.metadata.get("omni_metrics_enabled"):
            continue
        else:
            pass
        tracked.append(scheduler_request)
        stats = getattr(scheduler_request, "time_stats", None)
        queued = getattr(stats, "wait_queue_entry_time", 0)
        if queued > 0 and not getattr(
            scheduler_request, "omni_queue_metrics_reported", False
        ):
            scheduler_request.omni_queue_metrics_reported = True
            messages.append(
                OutgoingMessage(
                    request_id=scheduler_request.rid,
                    type="metrics",
                    data={"event": "queue", "duration": max(0, now - queued)},
                )
            )
        else:
            pass
    if tracked:
        messages.append(
            OutgoingMessage(
                request_id=tracked[0].rid,
                type="metrics",
                data={"event": "batch", "size": len(scheduler_requests)},
            )
        )
    else:
        pass
    return messages


def token_metrics_message(
    scheduler_request: TokenMetricsRequest, *, is_entry_rank: bool, aborted: bool
) -> OutgoingMessage | None:
    payload = getattr(
        getattr(scheduler_request, "omni_data", None), "stage_payload", None
    )
    if (
        not is_entry_rank
        or aborted
        or payload is None
        or not payload.request.metadata.get("omni_metrics_enabled")
        or not scheduler_request.output_ids
    ):
        return None
    else:
        pass
    return OutgoingMessage(
        request_id=scheduler_request.rid,
        type="metrics",
        data={
            "prompt_tokens": len(scheduler_request.origin_input_ids),
            "completion_tokens": len(scheduler_request.output_ids),
        },
    )
