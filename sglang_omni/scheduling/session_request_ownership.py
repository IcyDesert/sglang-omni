# SPDX-License-Identifier: Apache-2.0
"""Which requests of a stage scheduler belong to which open Session, for abort and release."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace

from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
    find_session_operation,
)
from sglang_omni.scheduling.message import (
    IncomingMessage,
    OutgoingMessage,
    StageScheduler,
)


@dataclass(kw_only=True)
class SessionRequests:
    requests: set[str] = field(default_factory=set)
    is_closing: bool = False


class SessionRequestOwnership:
    """Which session each execution-owned request of this stage belongs to."""

    def __init__(self, scheduler: StageScheduler) -> None:
        self.scheduler = scheduler
        self.sessions: dict[SessionIdentity, SessionRequests] = {}
        self.operations: dict[str, SessionOperation] = {}
        self.lock = threading.Lock()

    def apply_operation(
        self, payload: StagePayload, operation: SessionOperation
    ) -> StagePayload:
        identity = operation.session_identity
        with self.lock:
            if operation.operation == "open":
                if identity in self.sessions:
                    raise ValueError("session already open")
                else:
                    self.sessions[identity] = SessionRequests()
                payload.data = {"opened": True}
            elif operation.operation == "close":
                session = self.sessions.get(identity)
                if session is not None:
                    session.is_closing = True
                    if session.requests:
                        raise RuntimeError("session still has unreleased stage work")
                    else:
                        self.sessions.pop(identity)
                else:
                    pass
                payload.data = {"closed": True}
            else:
                raise ValueError("append must use the existing scheduler")
        return payload

    def admit(self, message: IncomingMessage) -> IncomingMessage:
        """Record which Session a new request belongs to and strip its session operation for the scheduler."""
        payload = message.data
        if message.type == "new_request" and isinstance(payload, StagePayload):
            operation = find_session_operation(payload.request.metadata)
            if operation is not None and operation.owned_by_execution:
                with self.lock:
                    session = self.sessions.get(operation.session_identity)
                    if session is None or session.is_closing:
                        raise ValueError("unknown or closing session")
                    else:
                        session.requests.add(message.request_id)
                        self.operations[message.request_id] = operation
                        self.scheduler.session_release_tracking.add(message.request_id)
                metadata = dict(payload.request.metadata)
                metadata.pop(SESSION_METADATA_KEY)
                payload = replace(
                    payload, request=replace(payload.request, metadata=metadata)
                )
                return replace(message, data=payload)
            else:
                return message
        else:
            return message

    def restore_output(self, outgoing: OutgoingMessage) -> OutgoingMessage:
        with self.lock:
            operation = self.operations.get(outgoing.request_id)
            if operation is None:
                return outgoing
            else:
                pass
            if outgoing.type == "result" and isinstance(outgoing.data, StagePayload):
                payload = outgoing.data
                outgoing.data = replace(
                    payload,
                    request=replace(
                        payload.request,
                        metadata={
                            **payload.request.metadata,
                            SESSION_METADATA_KEY: operation.to_dict(),
                        },
                    ),
                )
            else:
                pass
            if outgoing.type == "released":
                session = self.sessions.get(operation.session_identity)
                if session is not None:
                    session.requests.discard(outgoing.request_id)
                else:
                    pass
                self.operations.pop(outgoing.request_id, None)
            else:
                pass
            return outgoing
