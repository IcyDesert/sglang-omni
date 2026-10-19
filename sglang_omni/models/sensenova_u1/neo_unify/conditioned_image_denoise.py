"""Conditioned image denoise."""

import logging
import math
from typing import Literal

import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    clear_flash_kv_cache,
    prepare_i2i_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import randn_with_seed
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.tensor_parallel import synchronize_tp_noise

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


def conditioned_image_denoise(
    batch_size: int,
    cfg_interval: tuple[float, float],
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"],
    cfg_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    enable_timestep_shift: bool,
    image_size: tuple[int, int],
    img_cfg_scale: float,
    indexes_image_condition: torch.Tensor,
    indexes_image_img_condition: torch.Tensor | None,
    indexes_image_uncondition: torch.Tensor | None,
    merge_size: int,
    num_steps: int,
    past_key_values_condition: DynamicCache,
    past_key_values_img_condition: DynamicCache | None,
    past_key_values_uncondition: DynamicCache | None,
    seed: int | list[int],
    self: GenerationState,
    timestep_shift: float,
    token_h: int,
    token_w: int,
) -> torch.Tensor:
    for layer_idx in range(len(past_key_values_condition.layers)):
        past_key_values_condition.layers[
            layer_idx
        ].keys = past_key_values_condition.layers[layer_idx].keys.expand(
            batch_size, *past_key_values_condition.layers[layer_idx].keys.shape[1:]
        )
        past_key_values_condition.layers[
            layer_idx
        ].values = past_key_values_condition.layers[layer_idx].values.expand(
            batch_size,
            *past_key_values_condition.layers[layer_idx].values.shape[1:],
        )
        if past_key_values_img_condition is not None:
            past_key_values_img_condition.layers[
                layer_idx
            ].keys = past_key_values_img_condition.layers[layer_idx].keys.expand(
                batch_size,
                *past_key_values_img_condition.layers[layer_idx].keys.shape[1:],
            )
            past_key_values_img_condition.layers[
                layer_idx
            ].values = past_key_values_img_condition.layers[layer_idx].values.expand(
                batch_size,
                *past_key_values_img_condition.layers[layer_idx].values.shape[1:],
            )
        else:
            pass
        if past_key_values_uncondition is not None:
            past_key_values_uncondition.layers[
                layer_idx
            ].keys = past_key_values_uncondition.layers[layer_idx].keys.expand(
                batch_size,
                *past_key_values_uncondition.layers[layer_idx].keys.shape[1:],
            )
            past_key_values_uncondition.layers[
                layer_idx
            ].values = past_key_values_uncondition.layers[layer_idx].values.expand(
                batch_size,
                *past_key_values_uncondition.layers[layer_idx].values.shape[1:],
            )
        else:
            pass

    prepare_i2i_flash_kv_cache(
        past_key_values_condition,
        current_len=token_h * token_w,
        batch_size=batch_size,
    )
    if past_key_values_img_condition is not None:
        prepare_i2i_flash_kv_cache(
            past_key_values_img_condition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
    else:
        pass
    if past_key_values_uncondition is not None:
        prepare_i2i_flash_kv_cache(
            past_key_values_uncondition,
            current_len=token_h * token_w,
            batch_size=batch_size,
        )
    else:
        pass
    logger.info(
        f"SenseNova-U1 I2I attention caches: condition={past_key_values_condition.layers[0].flash_cache_layout}, image_condition={past_key_values_img_condition.layers[0].flash_cache_layout if past_key_values_img_condition is not None else 'disabled'}, uncondition={past_key_values_uncondition.layers[0].flash_cache_layout if past_key_values_uncondition is not None else 'disabled'}"
    )

    grid_h = image_size[1] // self.patch_size
    grid_w = image_size[0] // self.patch_size
    grid_hw = torch.tensor([[grid_h, grid_w]] * batch_size, device=device)

    noise_scale = self.noise_scale
    if self.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(self.noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size**2) / base)
        noise_scale = scale * float(self.noise_scale)
        if self.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
        else:
            pass
    else:
        pass
    noise_scale = min(noise_scale, self.noise_scale_max_value)
    image_prediction = noise_scale * randn_with_seed(
        (batch_size, 3, image_size[1], image_size[0]),
        device=device,
        dtype=dtype,
        seed=seed,
    )

    synchronize_tp_noise(self, image_prediction)
    attention_mask_condition = {"full_attention": None}
    attention_mask_img_condition = {"full_attention": None}
    attention_mask_uncondition = {"full_attention": None}

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if enable_timestep_shift:
        timesteps = self._apply_time_schedule(
            timesteps, token_h * token_w, timestep_shift
        )
    else:
        pass
    cfg_active = self._build_i2i_cfg_schedule(timesteps, cfg_interval)
    denoise_embeddings = None
    if device.type == "npu":
        denoise_embeddings = self.fm_modules["timestep_embedder"](timesteps[:-1])
        if self.add_noise_scale_embedding:
            noise_level = timesteps.new_tensor(
                [noise_scale / self.noise_scale_max_value]
            )
            denoise_embeddings = denoise_embeddings + self.fm_modules[
                "noise_scale_embedder"
            ](noise_level)
        else:
            pass
    else:
        pass

    for step_i in range(num_steps):
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]
        use_cfg = cfg_active[step_i]

        z = self.patchify(image_prediction, self.patch_size * merge_size)
        image_input = self.patchify(
            image_prediction, self.patch_size, channel_first=True
        )
        image_embeds = self.extract_feature(
            image_input.view(batch_size * grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).view(batch_size, token_h * token_w, -1)
        if denoise_embeddings is not None:
            timestep_embeddings = denoise_embeddings[step_i].view(1, 1, -1)
        else:
            t_expanded = t.expand(batch_size * token_h * token_w)
            timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(
                batch_size, token_h * token_w, -1
            )
            if self.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(
                    t_expanded, noise_scale / self.noise_scale_max_value
                )
                timestep_embeddings += self.fm_modules["noise_scale_embedder"](
                    noise_scale_tensor
                ).view(batch_size, token_h * token_w, -1)
            else:
                pass
        image_embeds = image_embeds + timestep_embeddings

        out_cond = self._t2i_predict_v(
            image_embeds,
            indexes_image_condition,
            attention_mask_condition,
            past_key_values_condition,
            t,
            z,
            image_token_num=token_h * token_w,
            timestep_embeddings=timestep_embeddings,
            image_size=image_size,
        )

        if not use_cfg:
            v_pred = out_cond
        elif cfg_scale == 1 and img_cfg_scale == 1:
            v_pred = out_cond
        elif img_cfg_scale == 1:
            out_img_cond = self._t2i_predict_v(
                image_embeds,
                indexes_image_img_condition,
                attention_mask_img_condition,
                past_key_values_img_condition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            v_pred = out_img_cond + cfg_scale * (out_cond - out_img_cond)
        elif cfg_scale == img_cfg_scale:
            out_uncond = self._t2i_predict_v(
                image_embeds,
                indexes_image_uncondition,
                attention_mask_uncondition,
                past_key_values_uncondition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            v_pred = out_uncond + cfg_scale * (out_cond - out_uncond)
        else:
            out_img_cond = self._t2i_predict_v(
                image_embeds,
                indexes_image_img_condition,
                attention_mask_img_condition,
                past_key_values_img_condition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            out_uncond = self._t2i_predict_v(
                image_embeds,
                indexes_image_uncondition,
                attention_mask_uncondition,
                past_key_values_uncondition,
                t,
                z,
                image_token_num=token_h * token_w,
                timestep_embeddings=timestep_embeddings,
                image_size=image_size,
            )
            v_pred = (
                out_uncond
                + cfg_scale * (out_cond - out_img_cond)
                + img_cfg_scale * (out_img_cond - out_uncond)
            )
        if (cfg_scale > 1 or img_cfg_scale > 1) and use_cfg:
            if cfg_norm == "global":
                norm_v_condition = torch.norm(out_cond, dim=(1, 2), keepdim=True)
                norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                v_pred = v_pred * scale
            elif cfg_norm == "channel":
                norm_v_condition = torch.norm(out_cond, dim=-1, keepdim=True)
                norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(min=0, max=1.0)
                v_pred = v_pred * scale
            else:
                pass
        else:
            pass

        z = z + (t_next - t) * v_pred
        image_prediction = self.unpatchify(
            z, self.patch_size * merge_size, image_size[1], image_size[0]
        )

    clear_flash_kv_cache(past_key_values_condition)
    if past_key_values_img_condition is not None:
        clear_flash_kv_cache(past_key_values_img_condition)
    else:
        pass
    if past_key_values_uncondition is not None:
        clear_flash_kv_cache(past_key_values_uncondition)
    else:
        pass

    return image_prediction
