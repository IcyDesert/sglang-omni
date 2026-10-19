"""SenseNova mixture-of-experts language model."""

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from sglang_omni.models.sensenova_u1.cache_dit import (
    decoder_attention_type,
    decoder_layers,
)
from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    create_block_causal_mask,
    position_ids_from_indexes,
)
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOMoELLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.moe_layers import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeMLP,
    Qwen3MoeSparseMoeBlock,
)
from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    SenseNovaRMSNorm as _SenseNovaRMSNorm,
)
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    make_qwen3_rms_norm,
)
from sglang_omni.models.sensenova_u1.neo_unify.transformers_compat import (
    causal_mask_kwargs,
    model_input_compat,
    tied_weights_keys,
)

# Modified for SGLang; see this directory's README.md for upstream source.


__all__ = [
    "Qwen3MoeMLP",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoePreTrainedModel",
    "Qwen3MoeModel",
    "Qwen3MoeForCausalLM",
]


class Qwen3MoePreTrainedModel(PreTrainedModel):
    config: NEOMoELLMConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: list[str] = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement: list[str] = ["past_key_values"]
    _supports_flash_attn: bool = True
    _supports_sdpa: bool = True
    _supports_flex_attn: bool = True

    _can_compile_fullgraph: bool = False
    _supports_attention_backend: bool = True
    _can_record_outputs: dict[str, type[nn.Module]] = {
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: NEOMoELLMConfig) -> None:
        super().__init__(config)
        self.padding_idx: int | None = config.pad_token_id
        self.vocab_size: int = config.vocab_size

        self.embed_tokens: nn.Embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm: _SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_mot_gen: _SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing: bool = False
        self.has_sliding_layers: bool = "sliding_attention" in self.config.layer_types
        self.current_index: int = -1

        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        image_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if image_only:
            exist_non_image_gen_tokens = False
            exist_image_gen_tokens = True
        elif image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        else:
            pass

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            pass

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        else:
            pass

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        else:
            pass

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        else:
            pass

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(
                    input_ids.device
                )
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(
                        position_ids_from_indexes(indexes, 0)
                    ),
                }
                self.current_index = position_ids_from_indexes(indexes, 0).max()
        else:
            self.current_index = position_ids_from_indexes(indexes, 0).max()

        hidden_states = inputs_embeds

        layers = decoder_layers(
            self,
            update_cache=kwargs.get("update_cache", True),
            has_non_image_tokens=exist_non_image_gen_tokens,
            has_image_tokens=exist_image_gen_tokens,
        )
        for decoder_layer in layers[: self.config.num_hidden_layers]:
            attention_type = decoder_attention_type(self, decoder_layer)
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        elif not exist_non_image_gen_tokens:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(
                hidden_states[~image_gen_indicators]
            )
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(
                hidden_states[image_gen_indicators]
            )
            hidden_states = _hidden_states

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys(
        "lm_head.weight", "model.embed_tokens.weight"
    )
    _tp_plan: dict[str, str] = {"lm_head": "colwise_rep"}
    _pp_plan: dict[str, tuple[list[str], list[str]]] = {
        "lm_head": (["hidden_states"], ["logits"])
    }

    def __init__(self, config: NEOMoELLMConfig) -> None:
        super().__init__(config)
        self.model: Qwen3MoeModel = Qwen3MoeModel(config)
        self.vocab_size: int = config.vocab_size
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        else:
            pass

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeMLP",
]
