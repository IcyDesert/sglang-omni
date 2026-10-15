# SPDX-License-Identifier: Apache-2.0
"""Session control events for scheduler tests."""

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import CONTROL, SESSION_CONTROL


def control_chunk(
    kind: str,
    t_start_ms: float = 0.0,
    *,
    seq: int = 0,
    turn: int | None = None,
    **fields: object,
) -> TimedChunk:
    """A session control event of type `kind` at `t_start_ms`; `fields` join its payload."""
    payload = {"type": kind, **fields}
    return TimedChunk(CONTROL, t_start_ms, 0, seq, payload, SESSION_CONTROL, turn=turn)
