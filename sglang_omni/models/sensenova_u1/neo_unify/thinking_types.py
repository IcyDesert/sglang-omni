"""Contracts for thinking sessions and prefix cache transfer."""

from typing import NotRequired, Protocol, TypedDict


class ThinkingSession(TypedDict):
    session_id: NotRequired[str]
    parent_rid: str
    input_ids: list[int]
    output_ids: list[int]


class RequestCacheMetadata(TypedDict, total=False):
    cached_tokens: int


class PrefixMetadata(TypedDict, total=False):
    dump_id: str
    data_file: str
    lock_file: str
    reuse_buffer: bool
    shape: list[int]
    layer_ids: list[int]
    dtype: str
    token_sha256: str
    data_bytes: int
    timings_ms: dict[str, float]
    session_reused_tokens: int
    request_meta_info: RequestCacheMetadata


class PrefixTimings(TypedDict, total=False):
    shared_to_pinned: float
    h2d: float
    import_wall: float
    export: dict[str, float]
    session_reused_tokens: int | None
    srt_cached_tokens: int | None
    request_wall: float
    total_wall: float


class ThinkingBackend(Protocol):
    @property
    def strict(self) -> bool: ...

    def generate_batch(
        self,
        input_ids: list[list[int]],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> list[list[int]]: ...

    def generate_batch_for_kv_transfer(
        self,
        input_ids: list[list[int]],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> tuple[list[list[int]], ThinkingSession | list[ThinkingSession]]: ...

    def dump_prefix_kv(self, input_ids: list[int], dump_id: str) -> None: ...

    def transfer_prefix_kv(
        self,
        input_ids: list[int],
        dump_id: str,
        session_context: ThinkingSession | None = None,
    ) -> PrefixMetadata: ...

    def close_transfer_context(
        self,
        session_context: ThinkingSession | list[ThinkingSession] | None,
    ) -> None: ...

    def fail(self, exc: BaseException) -> bool: ...
