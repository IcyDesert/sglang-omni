"""Srt thinking."""

import logging
import os
import time

import torch.utils.checkpoint
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    create_block_causal_mask,
)
from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    KV_DIAGNOSTIC_DIR,
    KV_TRANSFER_DIR,
    USE_SRT_KV_TRANSFER,
    dump_native_prefix_kv,
    load_srt_prefix_kv,
    merge_srt_prefix_caches,
    token_ids_sha256,
)
from sglang_omni.models.sensenova_u1.neo_unify.thinking_types import (
    PrefixTimings,
    ThinkingBackend,
    ThinkingSession,
)

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


def request_srt_think_tokens(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    thinking_backend: ThinkingBackend,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    max_think_tokens: int,
) -> tuple[list[list[int]], list[str], ThinkingSession | list[ThinkingSession] | None]:
    template = get_conv_template(self.template)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    request_ids = [
        input_ids[index, : int(length)].tolist()
        for index, length in enumerate(prefix_lengths.tolist())
    ]
    transfer_context = None
    if (
        os.environ.get(USE_SRT_KV_TRANSFER) == "1"
        and not os.environ.get(KV_DIAGNOSTIC_DIR)
        and os.environ.get(KV_TRANSFER_DIR)
        and hasattr(thinking_backend, "generate_batch_for_kv_transfer")
    ):
        output_ids, transfer_context = thinking_backend.generate_batch_for_kv_transfer(
            request_ids,
            max_think_tokens=max_think_tokens,
            eos_token_id=eos_token_id,
            think_end_token_id=think_end_token_id,
        )
    else:
        output_ids = thinking_backend.generate_batch(
            request_ids,
            max_think_tokens=max_think_tokens,
            eos_token_id=eos_token_id,
            think_end_token_id=think_end_token_id,
        )
    self.last_think_token_counts = [len(item) for item in output_ids]
    self.last_think_token_count = self.last_think_token_counts[0]
    try:
        think_texts = [
            tokenizer.decode(item, skip_special_tokens=False) for item in output_ids
        ]
    except Exception:  # noqa: BLE001 - retained sessions must close on decode errors.
        if hasattr(thinking_backend, "close_transfer_context"):
            thinking_backend.close_transfer_context(transfer_context)
        else:
            pass
        raise
    return (
        output_ids,
        think_texts,
        transfer_context,
    )


def import_srt_prefix_tokens(
    self: GenerationState,
    thinking_backend: ThinkingBackend,
    token_ids: list[int],
    session_context: ThinkingSession | None = None,
) -> tuple[DynamicCache, PrefixTimings]:
    started_at = time.perf_counter()
    transfer_id = token_ids_sha256(token_ids)[:12] + f"{time.time_ns():x}"
    metadata = thinking_backend.transfer_prefix_kv(
        token_ids, transfer_id, session_context
    )
    request_wall_ms = (time.perf_counter() - started_at) * 1000
    cache, timings = load_srt_prefix_kv(
        metadata,
        token_ids,
        self.language_model.config,
        self.device,
    )
    timings["request_wall"] = request_wall_ms
    timings["total_wall"] = (time.perf_counter() - started_at) * 1000
    return cache, timings


def try_import_srt_text_prefix(
    self: GenerationState,
    thinking_backend: ThinkingBackend,
    input_ids: torch.Tensor,
    prefix_length: int | torch.Tensor,
    name: str,
) -> tuple[DynamicCache | None, PrefixTimings]:
    token_ids = input_ids[0, : int(prefix_length)].tolist()
    try:
        return self._import_srt_prefix_tokens(thinking_backend, token_ids)
    except Exception as exc:
        logger.warning(
            f"SenseNova SRT {name} KV transfer failed; running the native prefix: {exc}"
        )
        return None, {}


def try_import_srt_text_prefix_batch(
    self: GenerationState,
    thinking_backend: ThinkingBackend,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    name: str,
) -> tuple[DynamicCache | None, list[PrefixTimings]]:
    caches = []
    timings = []
    lengths = [int(length) for length in prefix_lengths.tolist()]
    try:
        for index, length in enumerate(lengths):
            cache, row_timings = self._import_srt_prefix_tokens(
                thinking_backend,
                input_ids[index, :length].tolist(),
            )
            caches.append(cache)
            timings.append(row_timings)
        return (
            merge_srt_prefix_caches(caches, lengths, self.language_model.config),
            timings,
        )
    except Exception as exc:
        logger.warning(
            f"SenseNova SRT {name} KV transfer failed; running the native prefix: {exc}"
        )
        return None, []


def replay_srt_think_prefix(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    prefix_lengths: torch.Tensor,
    think_token_ids: list[list[int]],
    IMG_START_TOKEN: str,
    thinking_backend: ThinkingBackend | None = None,
    transfer_context: ThinkingSession | list[ThinkingSession] | None = None,
    require_transfer: bool = False,
) -> tuple[DynamicCache, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    append_ids = tokenizer(
        "\n\n" + IMG_START_TOKEN,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(self.device)
    replay_rows = [
        torch.cat(
            (
                input_ids[index, : int(prefix_length)],
                torch.tensor(
                    think_token_ids[index],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                append_ids[0],
            )
        )
        for index, prefix_length in enumerate(prefix_lengths.tolist())
    ]
    replay_lengths = torch.tensor(
        [row.shape[0] for row in replay_rows],
        dtype=torch.long,
        device=input_ids.device,
    )
    diagnostic_dir = os.environ.get(KV_DIAGNOSTIC_DIR)
    diagnostic_id = None
    diagnostic_token_ids = None
    if diagnostic_dir:
        if len(replay_rows) != 1:
            raise RuntimeError(
                "SenseNova KV diagnostic currently supports batch size 1"
            )
        else:
            pass
        if thinking_backend is None or not hasattr(thinking_backend, "dump_prefix_kv"):
            raise RuntimeError(
                "SenseNova KV diagnostic requires the managed SRT backend"
            )
        else:
            pass
        diagnostic_token_ids = replay_rows[0].tolist()
        diagnostic_id = token_ids_sha256(diagnostic_token_ids)[:16]
        thinking_backend.dump_prefix_kv(diagnostic_token_ids, diagnostic_id)
    else:
        pass
    max_length = int(replay_lengths.max().item())
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    else:
        pass
    replay_ids = torch.full(
        (len(replay_rows), max_length),
        int(pad_token_id or 0),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    key_valid_mask = torch.zeros(
        (len(replay_rows), max_length),
        dtype=torch.bool,
        device=input_ids.device,
    )
    for index, row in enumerate(replay_rows):
        replay_ids[index, : row.shape[0]] = row
        key_valid_mask[index, : row.shape[0]] = True
    t_idx = torch.arange(max_length, dtype=torch.long, device=replay_ids.device).expand(
        len(replay_rows), -1
    )
    batched_indexes = torch.stack(
        (t_idx, torch.zeros_like(t_idx), torch.zeros_like(t_idx)), dim=1
    )
    indexes = batched_indexes[0] if len(replay_rows) == 1 else batched_indexes
    attention_mask = {"full_attention": create_block_causal_mask(t_idx, key_valid_mask)}
    cache = hidden_states = None
    transfer_contexts = (
        transfer_context if isinstance(transfer_context, list) else [transfer_context]
    )
    use_transfer = (
        os.environ.get(USE_SRT_KV_TRANSFER) == "1"
        and diagnostic_dir is None
        and thinking_backend is not None
        and hasattr(thinking_backend, "transfer_prefix_kv")
        and all(context is not None for context in transfer_contexts)
        and len(transfer_contexts) == len(replay_rows)
    )
    if use_transfer:
        try:
            caches = []
            transfer_timings = []
            for row, context in zip(replay_rows, transfer_contexts):
                row_cache, row_timings = self._import_srt_prefix_tokens(
                    thinking_backend,
                    row.tolist(),
                    context,
                )
                caches.append(row_cache)
                transfer_timings.append(row_timings)
            cache = merge_srt_prefix_caches(
                caches,
                replay_lengths.tolist(),
                self.language_model.config,
            )
            self.last_srt_kv_transfer_timings_list = transfer_timings
            self.last_srt_kv_transfer_timings = transfer_timings[0]
            self.last_srt_kv_transfer_used = True
            self.last_srt_kv_transferred_prefixes.append("condition")
        except Exception as exc:
            if hasattr(thinking_backend, "close_transfer_context"):
                thinking_backend.close_transfer_context(transfer_contexts)
            else:
                pass
            self.last_srt_kv_transfer_used = False
            if require_transfer:
                raise RuntimeError(
                    "SenseNova compact mode requires SRT prefix KV transfer"
                ) from exc
            else:
                pass
            logger.warning(
                f"SenseNova SRT KV transfer failed; replaying the prefix: {exc}"
            )
    else:
        pass
    if cache is None:
        if require_transfer:
            raise RuntimeError(
                "SenseNova compact mode did not receive the condition prefix KV"
            )
        else:
            pass
        cache, hidden_states = self._t2i_prefix_forward(
            replay_ids, indexes, attention_mask
        )
    else:
        pass
    if diagnostic_id is not None:
        dump_native_prefix_kv(cache, diagnostic_token_ids, diagnostic_id)
    else:
        pass
    return cache, hidden_states, indexes, key_valid_mask, replay_lengths


# note (Codex): Preserve existing import and serialized class lookup names.
_import_srt_prefix_tokens = import_srt_prefix_tokens
_replay_srt_think_prefix = replay_srt_think_prefix
_request_srt_think_tokens = request_srt_think_tokens
_try_import_srt_text_prefix = try_import_srt_text_prefix
_try_import_srt_text_prefix_batch = try_import_srt_text_prefix_batch
