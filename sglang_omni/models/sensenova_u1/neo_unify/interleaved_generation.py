"""Interleaved generation."""

import logging
from pathlib import Path
from typing import Literal

import torch.utils.checkpoint
from PIL import Image
from transformers import GenerationConfig, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    build_abs_positions_from_grid_hw,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.interleaved_text_image_denoise import (
    interleaved_text_image_denoise,
)
from sglang_omni.models.sensenova_u1.neo_unify.utils import load_image_native

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


@torch.no_grad()
def interleave_gen(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    images: list[Image.Image | str | Path] | None = None,
    generation_config: GenerationConfig | None = None,
    cfg_scale: float = 1.0,
    img_cfg_scale: float = 1.0,
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"] = "none",
    max_images: int = 10,
    enable_timestep_shift: bool = True,
    timestep_shift: float = 1.0,
    image_size: tuple[int, int] | list[tuple[int, int]] = (256, 256),
    num_steps: int = 30,
    IMG_START_TOKEN: str = "<img>",
    IMG_END_TOKEN: str = "</img>",
    IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
    method: Literal["euler"] = "euler",
    cfg_interval: tuple[float, float] = (0, 1),
    t_eps: float = 0.02,
    verbose: bool = False,
    system_message: str = "",
    think_mode: bool = False,
    seed: int = 0,
) -> tuple[str, list[torch.Tensor]]:
    self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    self.config.t_eps = t_eps

    if isinstance(image_size, tuple):
        image_size_list = [image_size] * max_images
    elif isinstance(image_size, list) and isinstance(image_size[0], tuple):
        image_size_list = image_size
        if len(image_size) < max_images:
            image_size_list += [image_size_list[-1]] * (
                max_images - len(image_size_list)
            )
        else:
            pass
    else:
        assert False, "image size should be a tuple or a list of tuple"

    if (
        generation_config
        and hasattr(generation_config, "max_new_tokens")
        and generation_config.max_new_tokens is not None
    ):
        max_new_tokens = generation_config.max_new_tokens
    else:
        max_new_tokens = 8192

    current_generated_tokens = 0

    if images is None:
        images = []
    else:
        pass

    template = get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    image_token_count = prompt.count("<image>")
    assert len(images) >= image_token_count
    if len(images) > image_token_count:
        prompt = "<image>\n" * (len(images) - image_token_count) + prompt
    else:
        pass

    pixel_values = []
    grid_hw = []
    for image in images:
        cur_pixel_values, cur_grid_hw = load_image_native(
            image,
            self.patch_size,
            self.downsample_ratio,
            min_pixels=512 * 512,
            max_pixels=min(2048 * 2048, (4096 * 4096) // max(1, len(images))),
            upscale=False,
        )
        grid_hw.append(cur_grid_hw.to(self.device))
        pixel_values.append(cur_pixel_values.to(self.device).to(torch.bfloat16))

    merge_size = int(1 / self.downsample_ratio)
    pv_tensor = torch.cat(pixel_values) if pixel_values else None
    ghw_tensor = torch.cat(grid_hw) if grid_hw else None

    template_cond = get_conv_template(self.template)
    template_cond.system_message = system_message
    template_cond.append_message(template_cond.roles[0], prompt)
    template_cond.append_message(template_cond.roles[1], None)
    query_cond = template_cond.get_prompt()

    if not think_mode:
        query_cond = query_cond + "<think>\n\n</think>\n\n"
    else:
        pass

    def replace_image_tokens(query: str, grid_hw_list: list[torch.Tensor]) -> str:
        for i in range(len(grid_hw_list)):
            num_patch_token = int(
                grid_hw_list[i][0, 0] * grid_hw_list[i][0, 1] * self.downsample_ratio**2
            )
            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        return query

    query_cond = replace_image_tokens(query_cond, grid_hw)
    input_embeds_cond, indexes_cond, attention_mask_cond = self._build_it2i_inputs(
        tokenizer, query_cond, pv_tensor, ghw_tensor
    )

    outputs_cond = self.language_model(
        inputs_embeds=input_embeds_cond,
        indexes=indexes_cond,
        attention_mask=attention_mask_cond,
        use_cache=True,
    )
    past_key_values_cond = outputs_cond.past_key_values
    t_index_cond = indexes_cond[0].max().item()

    question_text_uncondition = "<image>" * len(images)
    template_tu = get_conv_template(self.template)
    template_tu.system_message = self.system_message
    template_tu.append_message(template_tu.roles[0], question_text_uncondition)
    template_tu.append_message(template_tu.roles[1], None)
    query_text_uncond = template_tu.get_prompt()
    query_text_uncond = replace_image_tokens(query_text_uncond, grid_hw)

    input_embeds_tu, indexes_tu, attention_mask_tu = self._build_it2i_inputs(
        tokenizer, query_text_uncond, pv_tensor, ghw_tensor
    )
    outputs_tu = self.language_model(
        inputs_embeds=input_embeds_tu,
        indexes=indexes_tu,
        attention_mask=attention_mask_tu,
        use_cache=True,
    )
    past_key_values_tu = outputs_tu.past_key_values
    t_index_tu = indexes_tu[0].max().item()

    query_img_uncond = self._build_t2i_query("", append_text=IMG_START_TOKEN)
    input_embeds_iu, indexes_iu, attention_mask_iu = self._build_it2i_inputs(
        tokenizer, query_img_uncond
    )
    outputs_iu = self.language_model(
        inputs_embeds=input_embeds_iu,
        indexes=indexes_iu,
        attention_mask=attention_mask_iu,
        use_cache=True,
    )
    past_key_values_iu = outputs_iu.past_key_values

    generated_text = ""
    generated_images = []
    max_images = 10
    img_count = 0

    next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

    generator = torch.Generator(self.device).manual_seed(seed)
    while True:
        gen_tokens = []
        hit_max_tokens = False
        last_decoded = 0
        while True:
            token_item = next_token.item()
            if token_item == eos_token_id or token_item == self.img_start_token_id:
                break
            else:
                pass
            gen_tokens.append(token_item)
            current_generated_tokens += 1

            self.language_model.model.current_index = t_index_cond
            outputs_cond = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values_cond,
                use_cache=True,
            )
            past_key_values_cond = outputs_cond.past_key_values
            t_index_cond += 1
            next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)

            if verbose and len(gen_tokens) - last_decoded >= 16:
                partial = tokenizer.decode(
                    gen_tokens[last_decoded:], skip_special_tokens=True
                )
                print(partial, end="", flush=True)
                last_decoded = len(gen_tokens)
            else:
                pass

            if current_generated_tokens >= max_new_tokens:
                hit_max_tokens = True
                break
            else:
                pass

        if len(gen_tokens) > 0:
            chunk_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generated_text += chunk_text
            if verbose:
                remaining = tokenizer.decode(
                    gen_tokens[last_decoded:], skip_special_tokens=True
                )
                if remaining:
                    print(remaining, end="", flush=True)
                else:
                    pass
            else:
                pass
        else:
            pass

        if next_token.item() == eos_token_id or hit_max_tokens:
            break
        else:
            pass

        if next_token.item() == self.img_start_token_id:
            if img_count >= max_images:
                break
            else:
                pass

            generated_text += "<image>"
            if verbose:
                logger.info(f"Preparing diffusion for image {img_count + 1}")
            else:
                pass

            self.language_model.model.current_index = t_index_cond
            outputs_cond = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values_cond,
                use_cache=True,
            )
            past_key_values_cond = outputs_cond.past_key_values
            t_index_cond += 1

            self.language_model.model.current_index = t_index_tu
            outputs_tu = self.language_model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values_tu,
                use_cache=True,
            )
            past_key_values_tu = outputs_tu.past_key_values
            t_index_tu += 1

            device, gen_grid_hw, image_prediction = interleaved_text_image_denoise(
                cfg_interval,
                cfg_norm,
                cfg_scale,
                enable_timestep_shift,
                generated_images,
                generator,
                image_size_list,
                img_cfg_scale,
                img_count,
                indexes_iu,
                merge_size,
                num_steps,
                outputs_cond,
                past_key_values_cond,
                past_key_values_iu,
                past_key_values_tu,
                self,
                t_index_cond,
                t_index_tu,
                timestep_shift,
                verbose,
            )

            pred_img = image_prediction[0].unsqueeze(0).to(torch.bfloat16)
            raw_img = pred_img * 0.5 + 0.5
            img_mean = torch.tensor(
                [0.485, 0.456, 0.406], dtype=raw_img.dtype, device=device
            ).view(1, 3, 1, 1)
            img_std = torch.tensor(
                [0.229, 0.224, 0.225], dtype=raw_img.dtype, device=device
            ).view(1, 3, 1, 1)
            und_img = (raw_img - img_mean) / img_std
            c, h, w = und_img[0].shape
            ps = self.patch_size
            p_grid_h = h // ps
            p_grid_w = w // ps
            flatten_pixel_values = (
                und_img[0]
                .view(c, p_grid_h, ps, p_grid_w, ps)
                .permute(1, 3, 0, 2, 4)
                .reshape(p_grid_h * p_grid_w, c * ps**2)
            )
            vit_embeds = self.extract_feature(
                flatten_pixel_values, grid_hw=gen_grid_hw[:1]
            ).unsqueeze(0)

            img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
            img_end_embed = self.language_model.get_input_embeddings()(
                torch.tensor([[img_end_id]], device=device)
            )
            inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1)

            N_img_tokens = vit_embeds.shape[1]
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                gen_grid_hw[:1] // int(1 / self.downsample_ratio), device=device
            )

            def append_image_to_cache(
                cache: DynamicCache, t_idx: int
            ) -> tuple[CausalLMOutputWithPast, int]:
                past_len = cache.get_seq_length()
                tgt_len = N_img_tokens + 1

                t_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                t_indexes[:N_img_tokens] = t_idx + 1
                t_indexes[N_img_tokens] = t_idx + 2

                h_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                w_indexes = torch.zeros(tgt_len, dtype=torch.long, device=device)
                h_indexes[:N_img_tokens] = abs_pos_h
                w_indexes[:N_img_tokens] = abs_pos_w

                indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

                mask = torch.zeros(1, 1, tgt_len, past_len + tgt_len, device=device)
                mask[0, 0, :N_img_tokens, past_len + N_img_tokens] = float("-inf")
                attention_mask_dict = {"full_attention": mask}

                outputs = self.language_model(
                    inputs_embeds=inputs_embeds_img,
                    indexes=indexes,
                    attention_mask=attention_mask_dict,
                    past_key_values=cache,
                    use_cache=True,
                )
                return outputs, t_idx + 2

            outputs_cond, t_index_cond = append_image_to_cache(
                past_key_values_cond, t_index_cond
            )
            outputs_tu, t_index_tu = append_image_to_cache(
                past_key_values_tu, t_index_tu
            )

            next_token = torch.argmax(outputs_cond.logits[:, -1, :], dim=-1)
        else:
            pass

    return generated_text, generated_images
