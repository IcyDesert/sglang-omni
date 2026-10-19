"""Text image prefix."""

import logging
from dataclasses import dataclass

import torch.utils.checkpoint
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    SenseNovaStageTimer,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    BatchedSRTThinkingFallbackRequired,
)
from sglang_omni.models.sensenova_u1.neo_unify.thinking_types import ThinkingBackend

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TextImagePrefix:
    condition_key_valid_mask: torch.Tensor
    condition_prefix_lengths: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    indexes_image_condition: torch.Tensor
    past_key_values_condition: DynamicCache
    think_text: str | list[str]
    use_direct_srt_prefix: bool


def text_image_prefix(
    IMG_START_TOKEN: str,
    attention_mask_condition_prefix: dict[str, torch.Tensor],
    compact_mode: bool,
    condition_key_valid_mask: torch.Tensor,
    condition_prefix_lengths: torch.Tensor,
    indexes_condition: torch.Tensor,
    indexes_image_condition: torch.Tensor,
    input_ids_condition: torch.Tensor,
    max_think_tokens: int,
    profile_stages: bool,
    profiler: SenseNovaStageTimer,
    prompts: list[str],
    self: GenerationState,
    think_mode: bool,
    think_text: str | list[str],
    thinking_backend: ThinkingBackend | None,
    token_h: int,
    token_w: int,
    tokenizer: PreTrainedTokenizerBase,
    use_direct_srt_prefix: bool,
) -> TextImagePrefix:
    if think_mode:
        used_srt = False
        if thinking_backend is not None:
            transfer_context = None
            profiler.mark("condition_prefill")
            try:
                (
                    think_token_ids,
                    think_texts,
                    transfer_context,
                ) = self._request_srt_think_tokens(
                    tokenizer,
                    thinking_backend,
                    input_ids_condition,
                    condition_prefix_lengths,
                    max_think_tokens,
                )
                profiler.mark("think_decode")
                (
                    past_key_values_condition,
                    prefix_hidden_states,
                    indexes_condition,
                    condition_key_valid_mask,
                    condition_prefix_lengths,
                ) = self._replay_srt_think_prefix(
                    tokenizer,
                    input_ids_condition,
                    condition_prefix_lengths,
                    think_token_ids,
                    IMG_START_TOKEN,
                    thinking_backend,
                    transfer_context,
                    require_transfer=compact_mode,
                )
                first_cache_layer = past_key_values_condition.layers[0]
                device = first_cache_layer.keys.device
                dtype = first_cache_layer.keys.dtype
                think_text = think_texts[0] if len(prompts) == 1 else think_texts
                del prefix_hidden_states, think_token_ids
                profiler.mark("think_replay_prefill")
                self.last_thinking_backend = "srt"
                self.last_thinking_backends = ["srt"] * len(prompts)
                used_srt = True
            except Exception as exc:
                use_direct_srt_prefix = False
                if hasattr(thinking_backend, "close_transfer_context"):
                    thinking_backend.close_transfer_context(transfer_context)
                else:
                    pass
                first_failure = thinking_backend.fail(exc)
                if compact_mode or thinking_backend.strict:
                    raise
                else:
                    pass
                if len(prompts) > 1:
                    raise BatchedSRTThinkingFallbackRequired(str(exc)) from exc
                else:
                    pass
                if first_failure:
                    logger.warning(
                        f"SenseNova SRT thinking is unavailable; this and later requests use the native decode: {exc}"
                    )
                else:
                    pass
        else:
            pass

        if not used_srt:
            outputs_condition = self._think_prefix_forward(
                input_ids=input_ids_condition,
                indexes=indexes_condition,
                attention_mask=attention_mask_condition_prefix,
            )
            past_key_values_condition = outputs_condition.past_key_values
            device = outputs_condition.logits.device
            dtype = outputs_condition.logits.dtype
            t_index_condition = indexes_condition[0].max().item()
            profiler.mark("condition_prefill")
            past_key_values_condition, t_index_condition, think_text = (
                self._generate_think(
                    tokenizer,
                    outputs_condition,
                    past_key_values_condition,
                    t_index_condition,
                    IMG_START_TOKEN,
                    max_think_tokens=max_think_tokens,
                )
            )
            profiler.mark("think_decode")
            self.last_thinking_backend = "native"
            self.last_thinking_backends = ["native"]
            self.last_think_token_counts = [self.last_think_token_count]
            if profile_stages:
                profiler.timings_ms["think_replay_prefill"] = 0.0
            else:
                pass
        else:
            pass
        if used_srt:
            use_direct_srt_prefix = self.last_srt_kv_transfer_used
            image_text_len = (
                int(condition_prefix_lengths[0].item())
                if len(prompts) == 1
                else condition_prefix_lengths
            )
        else:
            image_text_len = t_index_condition + 1
        indexes_image_condition = self._build_t2i_image_indexes(
            token_h,
            token_w,
            image_text_len,
            device=input_ids_condition.device,
        )
        if not used_srt:
            del outputs_condition
        else:
            pass
    else:
        past_key_values_condition = None
        if use_direct_srt_prefix:
            if len(prompts) == 1:
                (
                    past_key_values_condition,
                    self.last_srt_kv_transfer_timings,
                ) = self._try_import_srt_text_prefix(
                    thinking_backend,
                    input_ids_condition,
                    condition_prefix_lengths[0],
                    "condition",
                )
                self.last_srt_kv_transfer_timings_list = [
                    self.last_srt_kv_transfer_timings
                ]
            else:
                (
                    past_key_values_condition,
                    self.last_srt_kv_transfer_timings_list,
                ) = self._try_import_srt_text_prefix_batch(
                    thinking_backend,
                    input_ids_condition,
                    condition_prefix_lengths,
                    "condition",
                )
                if self.last_srt_kv_transfer_timings_list:
                    self.last_srt_kv_transfer_timings = (
                        self.last_srt_kv_transfer_timings_list[0]
                    )
                else:
                    pass
            self.last_srt_kv_transfer_used = past_key_values_condition is not None
            if self.last_srt_kv_transfer_used:
                self.last_srt_kv_transferred_prefixes.append("condition")
            else:
                use_direct_srt_prefix = False
                if compact_mode:
                    raise RuntimeError(
                        "SenseNova compact mode did not receive the condition "
                        "prefix KV"
                    )
                else:
                    pass
        else:
            pass
        if past_key_values_condition is None:
            (
                past_key_values_condition,
                prefix_hidden_states,
            ) = self._t2i_prefix_forward(
                input_ids_condition,
                indexes_condition,
                attention_mask_condition_prefix,
            )
            device = prefix_hidden_states.device
            dtype = prefix_hidden_states.dtype
            del prefix_hidden_states
        else:
            first_cache_layer = past_key_values_condition.layers[0]
            device = first_cache_layer.keys.device
            dtype = first_cache_layer.keys.dtype
        profiler.mark("condition_prefill")
        if profile_stages:
            profiler.timings_ms["think_decode"] = 0.0
            profiler.timings_ms["think_replay_prefill"] = 0.0
        else:
            pass

    return TextImagePrefix(
        condition_key_valid_mask=condition_key_valid_mask,
        condition_prefix_lengths=condition_prefix_lengths,
        device=device,
        dtype=dtype,
        indexes_image_condition=indexes_image_condition,
        past_key_values_condition=past_key_values_condition,
        think_text=think_text,
        use_direct_srt_prefix=use_direct_srt_prefix,
    )
