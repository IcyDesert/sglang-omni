"""Interleaved images."""

from pathlib import Path
from typing import Literal

import torch.utils.checkpoint
from PIL import Image
from transformers import PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    build_abs_positions_from_grid_hw,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.interleaved_image_denoise import (
    interleaved_image_denoise,
)
from sglang_omni.models.sensenova_u1.neo_unify.utils import load_image_native

# Modified for SGLang; see this directory's README.md for upstream source.


@torch.no_grad()
def interleave_gen_image_only(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    gt_text: str,
    images: list[Image.Image | str | Path] | None = None,
    gt_images: list[Image.Image] | None = None,
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
) -> list[torch.Tensor]:
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

    if images is None:
        images = []
    else:
        pass

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
    query_cond = template_cond.get_prompt() + "<think>\n\n</think>\n\n"

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

    generated_images = []
    img_count = 0
    device = self.device

    def append_ids_to_cache(
        cache: DynamicCache, t_idx: int, input_ids: torch.Tensor
    ) -> int:
        if input_ids.shape[1] == 0:
            return t_idx
        else:
            pass
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

        self.language_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            attention_mask=attention_mask_dict,
            past_key_values=cache,
            use_cache=True,
        )
        return t_idx + seq_len

    def append_image_to_cache(
        cache: DynamicCache,
        t_idx: int,
        inputs_embeds_img: torch.Tensor,
        N_img_tokens: int,
        abs_pos_w: torch.Tensor,
        abs_pos_h: torch.Tensor,
    ) -> int:
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

        self.language_model(
            inputs_embeds=inputs_embeds_img,
            indexes=indexes,
            attention_mask=attention_mask_dict,
            past_key_values=cache,
            use_cache=True,
        )
        return t_idx + 2

    parts = gt_text.split("<image>")
    img_start_id_tensor = torch.tensor([[self.img_start_token_id]], device=device)

    for i, part in enumerate(parts):
        if len(part) > 0:
            if verbose:
                print(part, end="", flush=True)
            else:
                pass
            part_ids = tokenizer(part, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            t_index_cond = append_ids_to_cache(
                past_key_values_cond, t_index_cond, part_ids
            )
        else:
            pass

        if i < len(parts) - 1:
            if img_count >= max_images:
                break
            else:
                pass

            if verbose:
                print("<image>", end="", flush=True)
            else:
                pass

            t_index_cond = append_ids_to_cache(
                past_key_values_cond, t_index_cond, img_start_id_tensor
            )
            t_index_tu = append_ids_to_cache(
                past_key_values_tu, t_index_tu, img_start_id_tensor
            )

            image_prediction = interleaved_image_denoise(
                cfg_interval,
                cfg_norm,
                cfg_scale,
                device,
                enable_timestep_shift,
                generated_images,
                image_size,
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

            if gt_images is not None and img_count < len(gt_images):
                gt_img_pil = gt_images[img_count]
                gt_pixel_values, gt_grid_hw = load_image_native(
                    gt_img_pil,
                    self.patch_size,
                    self.downsample_ratio,
                    min_pixels=512 * 512,
                    max_pixels=(2048 * 2048),
                    upscale=False,
                )
                gt_pixel_values = gt_pixel_values.to(device).to(torch.bfloat16)

                flatten_pixel_values = gt_pixel_values
                gen_grid_hw_und = gt_grid_hw
            else:
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
                gen_grid_hw_und = torch.tensor([[p_grid_h, p_grid_w]], device=device)

            vit_embeds = self.extract_feature(
                flatten_pixel_values, grid_hw=gen_grid_hw_und[:1]
            ).unsqueeze(0)

            img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
            img_end_embed = self.language_model.get_input_embeddings()(
                torch.tensor([[img_end_id]], device=device)
            )
            inputs_embeds_img = torch.cat([vit_embeds, img_end_embed], dim=1)

            N_img_tokens = vit_embeds.shape[1]
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(
                gen_grid_hw_und[:1] // int(1 / self.downsample_ratio), device=device
            )

            t_index_cond = append_image_to_cache(
                past_key_values_cond,
                t_index_cond,
                inputs_embeds_img,
                N_img_tokens,
                abs_pos_w,
                abs_pos_h,
            )
            t_index_tu = append_image_to_cache(
                past_key_values_tu,
                t_index_tu,
                inputs_embeds_img,
                N_img_tokens,
                abs_pos_w,
                abs_pos_h,
            )

            img_count += 1
        else:
            pass

    return generated_images
