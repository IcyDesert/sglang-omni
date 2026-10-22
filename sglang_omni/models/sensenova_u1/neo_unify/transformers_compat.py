"""Small compatibility seams for the supported Transformers 4/5 window."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Protocol

import torch
import transformers
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

# Modified for SGLang; see this directory's README.md for upstream source.


class ModelForward(Protocol):
    def __call__(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        image_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Return hidden states and the optional updated prefix cache."""


class LegacyCausalMaskFactory(Protocol):
    def __call__(
        self,
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build a causal mask using explicit cache positions."""


class CausalMaskFactory(Protocol):
    def __call__(
        self,
        config: PretrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build a causal mask from the inputs and prefix cache."""


try:
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    def model_input_compat(func: ModelForward) -> ModelForward:
        return merge_with_config_defaults(capture_outputs(func))

except ImportError:
    from transformers.utils.generic import check_model_inputs

    def model_input_compat(func: ModelForward) -> ModelForward:
        return check_model_inputs(func)


@lru_cache(maxsize=None)
def parameter_names(
    mask_factory: CausalMaskFactory | LegacyCausalMaskFactory,
) -> frozenset[str]:
    return frozenset(inspect.signature(mask_factory).parameters)


def causal_mask_kwargs(
    mask_factory: CausalMaskFactory | LegacyCausalMaskFactory,
    *,
    config: PretrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.Tensor,
) -> dict[str, PretrainedConfig | torch.Tensor | Cache | None]:
    """Build arguments accepted by the installed create_causal_mask."""
    parameters = parameter_names(mask_factory)
    embedding_parameter = (
        "inputs_embeds" if "inputs_embeds" in parameters else "input_embeds"
    )
    candidates = {
        "config": config,
        embedding_parameter: inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    return {name: value for name, value in candidates.items() if name in parameters}


def pretrained_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Use the public dtype keyword supported throughout Transformers 4.57+."""
    return {"dtype": dtype}


def tied_weights_keys(output_key: str, input_key: str) -> list[str] | dict[str, str]:
    """Return the _tied_weights_keys shape expected by Transformers 4 or 5."""

    if Version(transformers.__version__).major >= 5:
        return {output_key: input_key}
    else:
        pass
    return [output_key]
