"""Small compatibility seams for the supported Transformers 4/5 window."""

from __future__ import annotations

from typing import Protocol

import torch
import transformers
from packaging.version import Version
from transformers import PreTrainedModel
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
        attention_mask: dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        image_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Return hidden states and the optional updated prefix cache."""


try:
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    def model_input_compat(func: ModelForward) -> ModelForward:
        return merge_with_config_defaults(capture_outputs(func))

except ImportError:
    from transformers.utils.generic import check_model_inputs

    def model_input_compat(func: ModelForward) -> ModelForward:
        return check_model_inputs(func)


def tied_weights_keys(output_key: str, input_key: str) -> list[str] | dict[str, str]:
    """Return the _tied_weights_keys shape expected by Transformers 4 or 5."""

    if Version(transformers.__version__).major >= 5:
        return {output_key: input_key}
    else:
        pass
    return [output_key]
