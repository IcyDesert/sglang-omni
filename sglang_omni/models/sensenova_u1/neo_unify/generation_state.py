"""Typed state shared by the generation routines."""

from typing import Protocol

import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)


class GenerationState(Protocol):
    config: NEOChatConfig
    language_model: Qwen3ForCausalLM | Qwen3MoeForCausalLM
    vision_model: NEOVisionModel
    device: torch.device
    fm_modules: nn.ModuleDict
    patch_size: int
    template: str
    downsample_ratio: float
    system_message: str
    concat_time_token_num: int
    noise_scale: float
    # note (Wes): Unrecognized noise modes retain the configured constant scale.
    noise_scale_mode: str
    noise_scale_base_image_seq_len: int
    noise_scale_max_value: float
    add_noise_scale_embedding: bool
    use_pixel_head: bool
    use_deep_fm_head: bool
    img_context_token_id: int | None
    img_start_token_id: int

    def extract_feature(
        self,
        pixel_values: torch.Tensor | None,
        gen_model: bool = False,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode image pixels into features for the language model."""

    def patchify(
        self, images: torch.Tensor, patch_size: int, channel_first: bool = False
    ) -> torch.Tensor:
        """Convert images into flattened patches."""

    def unpatchify(
        self,
        x: torch.Tensor,
        patch_size: int,
        h: int | None = None,
        w: int | None = None,
    ) -> torch.Tensor:
        """Reconstruct images from flattened patches."""

    def build_t2i_query(
        self,
        prompt_text: str,
        system_message: str | None = None,
        append_text: str | None = None,
    ) -> str:
        """Format a prompt for text-to-image generation."""

    def build_t2i_text_inputs(
        self, tokenizer: PreTrainedTokenizerBase, query: str | list[str]
    ) -> tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """Build token IDs, positions, attention masks, and valid prefix lengths."""

    def build_t2i_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int | torch.Tensor,
        device: torch.device | None,
    ) -> torch.Tensor:
        """Build temporal and spatial positions for image tokens."""

    def t2i_prefix_forward(
        self,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
    ) -> tuple[DynamicCache, torch.Tensor]:
        """Return the text prefix cache and hidden states."""

    def it2i_prefix_forward(
        self,
        input_imbeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
    ) -> tuple[DynamicCache, torch.Tensor]:
        """Return the image-edit prefix cache and hidden states."""

    def t2i_predict_v(
        self,
        input_embeds: torch.Tensor,
        indexes_image: torch.Tensor,
        attn_mask: dict[str, torch.Tensor | None] | None,
        past_key_values: DynamicCache,
        t: torch.Tensor,
        z: torch.Tensor,
        image_token_num: int,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Predict the image latent velocity for a denoising step."""

    def build_it2i_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build image-edit embeddings, positions, and attention masks."""

    def get_thw_indexes(
        self, input_ids: torch.Tensor, grid_hw: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return temporal and spatial positions for text and image tokens."""
