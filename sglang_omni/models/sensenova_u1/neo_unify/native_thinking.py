"""Native thinking."""

import os
from typing import Unpack

import torch.utils.checkpoint
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import (
    GenerationState,
    ThinkPrefixInputs,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    preallocate_think_cache,
)

# Modified for SGLang; see this directory's README.md for upstream source.


def t2i_prefix_forward(
    self: GenerationState,
    input_ids: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: dict[str, torch.Tensor],
) -> tuple[DynamicCache, torch.Tensor]:
    out = self.language_model.model(
        input_ids=input_ids,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=True,
    )
    return out.past_key_values, out.last_hidden_state


def it2i_prefix_forward(
    self: GenerationState,
    input_imbeds: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: dict[str, torch.Tensor],
    gen_indicators: torch.Tensor | None = None,
) -> tuple[DynamicCache, torch.Tensor]:
    out = self.language_model.model(
        inputs_embeds=input_imbeds,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=True,
        image_gen_indicators=(
            gen_indicators.view(1, -1) if gen_indicators is not None else None
        ),
    )
    return out.past_key_values, out.last_hidden_state


def think_prefix_forward(
    self: GenerationState, **kwargs: Unpack[ThinkPrefixInputs]
) -> CausalLMOutputWithPast:
    """Build the Think prefix cache without materialising per-token logits."""
    return self.language_model(use_cache=True, logits_to_keep=1, **kwargs)


def append_text_tokens_to_cache(
    self: GenerationState, cache: DynamicCache, t_idx: int, input_ids: torch.Tensor
) -> int:
    if input_ids.shape[1] == 0:
        return t_idx
    else:
        pass

    device = input_ids.device
    seq_len = input_ids.shape[1]
    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    t_indexes = torch.arange(
        t_idx + 1, t_idx + 1 + seq_len, dtype=torch.long, device=device
    )
    h_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
    w_indexes = torch.zeros(seq_len, dtype=torch.long, device=device)
    indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

    past_len = cache.get_seq_length()
    mask = torch.zeros(1, 1, seq_len, past_len + seq_len, device=device)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    causal_mask = torch.where(causal_mask == 1, 0.0, float("-inf"))
    mask[:, :, :, past_len:] = causal_mask
    attention_mask_dict = {"full_attention": mask}

    self.language_model.model(
        inputs_embeds=inputs_embeds,
        indexes=indexes,
        attention_mask=attention_mask_dict,
        past_key_values=cache,
        use_cache=True,
    )
    return t_idx + seq_len


def generate_think(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prefix_outputs: CausalLMOutputWithPast,
    past_key_values: DynamicCache,
    t_idx: int,
    IMG_START_TOKEN: str,
    max_think_tokens: int = 1024,
) -> tuple[DynamicCache, int, str]:
    template = get_conv_template(self.template)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    think_token_ids = []
    think_closed = False
    next_token = torch.argmax(prefix_outputs.logits[:, -1, :], dim=-1)
    append_ids = tokenizer(
        "\n\n" + IMG_START_TOKEN,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(self.device)
    if (
        self.device.type == "cuda"
        and os.environ.get("SENSENOVA_THINK_KV_CACHE", "preallocated") == "preallocated"
    ):
        preallocate_think_cache(
            past_key_values, max_think_tokens + append_ids.shape[1] + 1
        )
    else:
        pass

    for token_index in range(max_think_tokens):
        token_item = next_token.item()
        if token_item == eos_token_id:
            break
        else:
            pass
        if token_item == think_end_token_id:
            self.language_model.model.current_index = t_idx
            outputs = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            t_idx += 1
            think_token_ids.append(token_item)
            think_closed = True
            break
        else:
            pass

        if token_index == max_think_tokens - 1:
            break
        else:
            pass

        think_token_ids.append(token_item)

        self.language_model.model.current_index = t_idx
        outputs = self.language_model(
            input_ids=next_token.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        t_idx += 1

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

    if not think_closed:
        close_ids = torch.tensor(
            [[think_end_token_id]],
            dtype=append_ids.dtype,
            device=append_ids.device,
        )
        append_ids = torch.cat((close_ids, append_ids), dim=1)
        think_token_ids.append(think_end_token_id)
    else:
        pass
    t_idx = self._append_text_tokens_to_cache(past_key_values, t_idx, append_ids)

    think_text = tokenizer.decode(think_token_ids, skip_special_tokens=False)
    self.last_think_token_count = len(think_token_ids)

    return past_key_values, t_idx, think_text


def t2i_predict_v(
    self: GenerationState,
    input_embeds: torch.Tensor,
    indexes_image: torch.Tensor,
    attn_mask: dict[str, torch.Tensor | None] | None,
    past_key_values: DynamicCache,
    t: torch.Tensor,
    z: torch.Tensor,
    image_token_num: int,
    timestep_embeddings: torch.Tensor | None = None,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    B, L = z.shape[0], z.shape[1]

    outputs = self.language_model.model(
        inputs_embeds=input_embeds,
        image_only=True,
        indexes=indexes_image,
        attention_mask=attn_mask,
        past_key_values=past_key_values,
        update_cache=False,
        use_cache=True,
    )

    if self.use_pixel_head:
        merge_size = int(1 / self.downsample_ratio)
        token_h = image_size[1] // (self.patch_size * merge_size)
        token_w = image_size[0] // (self.patch_size * merge_size)

        img_reshaped = outputs.last_hidden_state[:, -image_token_num:].view(
            B, token_h, token_w, -1
        )
        img_2d = torch.einsum("b h w c -> b c h w", img_reshaped)
        img_2d = img_2d.contiguous().view(B, -1, token_h, token_w)

        smoothed_img_2d = self.fm_modules["fm_head"](img_2d)

        smoothed_reshaped = smoothed_img_2d.view(
            B,
            3,
            token_h,
            self.patch_size * merge_size,
            token_w,
            self.patch_size * merge_size,
        )
        smoothed_reshaped = torch.einsum(
            "b c h p w q -> b h w p q c", smoothed_reshaped
        )
        out_1d = smoothed_reshaped.contiguous().view(
            B, L, self.patch_size * merge_size * self.patch_size * merge_size * 3
        )
        x_pred = out_1d
    else:
        if self.use_deep_fm_head:
            x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B * L, -1),
                t.repeat(B * L),
            ).view(B, L, -1)
        else:
            x_pred = self.fm_modules["fm_head"](
                outputs.last_hidden_state[:, -image_token_num:].view(B, L, -1)
            ).view(B, L, -1)

    v_pred = (x_pred - z) / (1 - t).clamp_min(self.config.t_eps)
    return v_pred


# note (Codex): Preserve existing import and serialized class lookup names.
_append_text_tokens_to_cache = append_text_tokens_to_cache
_generate_think = generate_think
_it2i_prefix_forward = it2i_prefix_forward
_t2i_predict_v = t2i_predict_v
_t2i_prefix_forward = t2i_prefix_forward
_think_prefix_forward = think_prefix_forward
