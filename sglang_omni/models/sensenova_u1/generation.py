# SPDX-License-Identifier: Apache-2.0
"""Generate SenseNova images individually or in compatible batches."""

from __future__ import annotations

import logging

import torch
from pydantic import JsonValue
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.admission import (
    REFERENCE_MEMORY_ERROR_PREFIX,
    validate_output_dimensions,
)
from sglang_omni.models.sensenova_u1.cache_dit import SenseNovaCacheDit
from sglang_omni.models.sensenova_u1.image_processing import (
    encode_image,
    prepare_reference_images,
    resolve_edit_output_size,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_conditioned_generation import (
    it2i_generate,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.text_to_image_generation import (
    t2i_generate,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    image_guidance_branch_count,
)
from sglang_omni.proto.request import StagePayload

logger = logging.getLogger(__name__)


def generate_image(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    """Run the source model's native T2I or I2I path and return a PNG payload."""
    inputs = payload.request.inputs
    try:
        if isinstance(inputs, dict) and inputs.get("task") == "image_edit":
            return generate_image_edit(payload, model, tokenizer)
        else:
            pass
        return generate_text_to_image(payload, model, tokenizer)
    except Exception:
        unmount_cache_dit_after_failure(model)
        raise


def generate_images(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[StagePayload]:
    """Generate one scheduler batch, which batch keys limit to matching single-output T2I."""
    try:
        return generate_text_to_image_batch(payloads, model, tokenizer)
    except Exception:
        unmount_cache_dit_after_failure(model)
        raise


def text_to_image_options(payload: StagePayload) -> SenseNovaU1Sampling:
    prompt = payload.request.inputs
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty text prompt")
    else:
        pass
    return SenseNovaU1Sampling.from_params(payload.request.params)


def generate_text_to_image(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    prompt = payload.request.inputs
    options = text_to_image_options(payload)
    prepare_cache_dit(
        model,
        enable_cache_dit=options.enable_cache_dit,
        params=options.cache_dit_params,
        steps=options.num_inference_steps,
        cfg_scale=options.guidance_scale,
    )
    with torch.inference_mode():
        images = t2i_generate(
            model,
            tokenizer,
            prompt,
            sampling=options,
        )
    return encode_generated_images(
        payload, images, options.n, options.width, options.height
    )


def encode_generated_images(
    payload: StagePayload,
    images: torch.Tensor,
    image_count: int,
    width: int,
    height: int,
) -> StagePayload:
    """Encode one image as image_b64, or several as images_b64 in seed order."""
    if image_count == 1:
        return encode_image(payload, images, width, height)
    else:
        pass
    if not isinstance(images, torch.Tensor) or images.shape != (
        image_count,
        3,
        height,
        width,
    ):
        raise ValueError("SenseNova-U1 returned an invalid multi-output image tensor")
    else:
        pass
    encoded_images = []
    for index in range(image_count):
        encode_image(payload, images[index : index + 1], width, height)
        encoded_images.append(payload.data["image_b64"])
    payload.data = {
        "images_b64": encoded_images,
        "modality": "image",
        "finish_reason": "stop",
    }
    return payload


def generate_text_to_image_batch(
    payloads: list[StagePayload],
    model: NEOChatModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[StagePayload]:
    prompts = [payload.request.inputs for payload in payloads]
    options = [text_to_image_options(payload) for payload in payloads]
    first = options[0]
    assert all(
        sampling.batch_key() == first.batch_key() for sampling in options
    ), "image_generation_batch_key admits only matching T2I requests to a batch"

    prepare_cache_dit(
        model,
        enable_cache_dit=first.enable_cache_dit,
        params=first.cache_dit_params,
        steps=first.num_inference_steps,
        cfg_scale=first.guidance_scale,
    )

    logger.info(
        f"SenseNova-U1 T2I batch: size={len(payloads)}, "
        f"image_size={first.width}x{first.height}, "
        f"steps={first.num_inference_steps}, cfg={first.guidance_scale}"
    )
    with torch.inference_mode():
        images = t2i_generate(
            model,
            tokenizer,
            prompts,
            sampling=first,
            seeds=[sampling.seed for sampling in options],
        )
    expected_shape = (len(payloads), 3, first.height, first.width)
    if not isinstance(images, torch.Tensor) or images.shape != expected_shape:
        raise ValueError("SenseNova-U1 returned an invalid batched image tensor")
    else:
        pass
    return [
        encode_image(payload, images[index : index + 1], first.width, first.height)
        for index, payload in enumerate(payloads)
    ]


def generate_image_edit(
    payload: StagePayload, model: NEOChatModel, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    inputs = payload.request.inputs
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
    else:
        pass
    options = SenseNovaU1ImageEditSampling.from_params(payload.request.params)
    references = prepare_reference_images(inputs.get("image_b64"), options)
    width, height = resolve_edit_output_size(references[0].size, options)
    validate_output_dimensions(payload.request, width, height)
    output_images = []
    try:
        # note (Wes): Each output gets a fresh Cache-DiT context, so cached residuals never cross seeds.
        for output_index in range(options.n):
            prepare_cache_dit(
                model,
                enable_cache_dit=options.enable_cache_dit,
                params=options.cache_dit_params,
                steps=options.num_inference_steps,
                cfg_scale=options.guidance_scale,
                img_cfg_scale=options.img_cfg_scale,
                is_edit=True,
            )
            with torch.inference_mode():
                output_images.append(
                    it2i_generate(
                        model,
                        tokenizer,
                        prompt,
                        references,
                        sampling=options.model_copy(
                            update={
                                "width": width,
                                "height": height,
                                "n": 1,
                                "seed": options.seed + output_index,
                            }
                        ),
                    )
                )
    except torch.OutOfMemoryError as error:
        device = model.device
        if device.type not in ("cuda", "npu"):
            raise
        else:
            torch.get_device_module(device).empty_cache()
        memory_advice = "reduce references or output size"
        if options.do_resize:
            memory_advice += ", or lower input_max_pixels"
        else:
            pass
        if device.type == "cuda":
            memory_advice += (
                ", or lower generate.factory.dit_layerwise_resident_layers "
                "with layerwise offload enabled"
            )
        else:
            pass
        raise ValueError(
            f"{REFERENCE_MEMORY_ERROR_PREFIX}{len(references)} reference images exceeds "
            f"{device.type.upper()} memory; {memory_advice}"
        ) from error
    else:
        pass
    images = (
        torch.cat(output_images)
        if all(isinstance(image, torch.Tensor) for image in output_images)
        else None
    )
    return encode_generated_images(payload, images, options.n, width, height)


def unmount_cache_dit_after_failure(model: NEOChatModel) -> None:
    controller = model.__dict__.get("sensenova_cache_dit")
    if isinstance(controller, SenseNovaCacheDit):
        try:
            controller.unmount()
        except Exception:
            logger.exception(
                "Cache-DiT cleanup failed; the controller retained recovery state"
            )
    else:
        pass


def prepare_cache_dit(
    model: NEOChatModel,
    *,
    enable_cache_dit: bool | None,
    params: JsonValue,
    steps: int,
    cfg_scale: float,
    img_cfg_scale: float = 1.0,
    is_edit: bool = False,
) -> None:
    # note (Wes): Direct generation calls may bypass the factory controller.
    controller = model.__dict__.get("sensenova_cache_dit")
    if not isinstance(controller, SenseNovaCacheDit):
        return
    else:
        pass
    if not is_edit:
        branch_count = 2 if cfg_scale > 1 else 1
    else:
        branch_count = image_guidance_branch_count(cfg_scale, img_cfg_scale)
    controller.prepare(
        model.language_model.model,
        enabled=enable_cache_dit,
        params=params,
        steps=steps,
        branch_count=branch_count,
        cfg_interval=(0.0, 1.0),
    )
