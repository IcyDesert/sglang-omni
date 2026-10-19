"""Text image denoise."""

import math
from typing import Literal

import torch.utils.checkpoint
from transformers.cache_utils import DynamicCache

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    npu_fia_available,
)
from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    clear_flash_kv_cache,
    prepare_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    SenseNovaStageTimer,
    optimized_scale,
    randn_with_seed,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.tensor_parallel import synchronize_tp_noise

# Modified for SGLang; see this directory's README.md for upstream source.


def text_image_denoise(
    batch_size: int,
    cfg_interval: tuple[float, float],
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"],
    cfg_scale: float,
    condition_key_valid_mask: torch.Tensor,
    condition_prefix_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    enable_timestep_shift: bool,
    image_size: tuple[int, int],
    indexes_image_condition: torch.Tensor,
    indexes_image_uncondition: torch.Tensor | None,
    merge_size: int,
    needs_cfg: bool,
    num_steps: int,
    past_key_values_condition: DynamicCache,
    past_key_values_uncondition: DynamicCache | None,
    profiler: SenseNovaStageTimer,
    seed: int | list[int],
    self: GenerationState,
    timestep_shift: float,
    token_h: int,
    token_w: int,
    uncondition_prefix_lengths: torch.Tensor | None,
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

    prepare_flash_kv_cache(
        past_key_values_condition,
        current_len=token_h * token_w,
        batch_size=batch_size,
        prefix_lengths=condition_prefix_lengths,
    )
    if past_key_values_uncondition is not None:
        prepare_flash_kv_cache(
            past_key_values_uncondition,
            current_len=token_h * token_w,
            batch_size=batch_size,
            prefix_lengths=uncondition_prefix_lengths,
        )
    else:
        pass

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
    if device.type == "npu" and batch_size > 1 and not npu_fia_available():
        condition_key_valid_mask = condition_key_valid_mask.expand(batch_size, -1)
        image_key_valid_mask = torch.ones(
            (batch_size, token_h * token_w),
            dtype=torch.bool,
            device=device,
        )
        denoise_key_valid_mask = torch.cat(
            [condition_key_valid_mask, image_key_valid_mask], dim=1
        )
        attention_mask_condition["full_attention"] = (
            denoise_key_valid_mask[:, None, None, :]
            .expand(-1, -1, token_h * token_w, -1)
            .contiguous()
        )
    else:
        pass
    attention_mask_uncondition = {"full_attention": None}

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    if enable_timestep_shift:
        timesteps = self._apply_time_schedule(
            timesteps, token_h * token_w, timestep_shift
        )
    else:
        pass
    cfg_active = self._build_cfg_schedule(timesteps, cfg_interval, needs_cfg)
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
    profiler.mark("denoise_prepare")

    for step_i in range(num_steps):
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]

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

        v_pred_condition = self._t2i_predict_v(
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

        if cfg_active[step_i]:
            v_pred_uncondition = self._t2i_predict_v(
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
            if cfg_norm == "cfg_zero_star":
                positive_flat = v_pred_condition.view(batch_size, -1)
                negative_flat = v_pred_uncondition.view(batch_size, -1)

                alpha = optimized_scale(positive_flat, negative_flat)
                alpha = alpha.view(
                    batch_size, *([1] * (len(v_pred_condition.shape) - 1))
                )
                alpha = alpha.to(positive_flat.dtype)

                if step_i <= 0:
                    v_pred = v_pred_condition * 0.0
                else:
                    v_pred = v_pred_uncondition * alpha + cfg_scale * (
                        v_pred_condition - v_pred_uncondition * alpha
                    )
            else:
                v_pred = v_pred_uncondition + cfg_scale * (
                    v_pred_condition - v_pred_uncondition
                )
                if cfg_norm == "global":
                    norm_v_condition = torch.norm(
                        v_pred_condition, dim=(1, 2), keepdim=True
                    )
                    norm_v_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(
                        min=0, max=1.0
                    )
                    v_pred = v_pred * scale
                elif cfg_norm == "channel":
                    norm_v_condition = torch.norm(
                        v_pred_condition, dim=-1, keepdim=True
                    )
                    norm_v_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
                    scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(
                        min=0, max=1.0
                    )
                    v_pred = v_pred * scale
                else:
                    pass
        else:
            v_pred = v_pred_condition

        z = z + (t_next - t) * v_pred

        image_prediction = self.unpatchify(
            z, self.patch_size * merge_size, image_size[1], image_size[0]
        )

    clear_flash_kv_cache(past_key_values_condition)
    if past_key_values_uncondition is not None:
        clear_flash_kv_cache(past_key_values_uncondition)
    else:
        pass

    return image_prediction
