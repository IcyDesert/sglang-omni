# SPDX-License-Identifier: Apache-2.0
"""SenseNova Images API routes, registered by the model package."""

from __future__ import annotations

import base64
import json
import logging
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)
from sglang_omni.serve.image_response import (
    ImageGenerationResponse,
    build_image_response,
)
from sglang_omni.serve.openai_api import (
    image_result_with_disconnect_watch,
    parse_image_size,
    read_image_edit_upload,
)
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.protocol import ImageGenerationRequest

logger = logging.getLogger(__name__)


def register_image_generations(app: FastAPI) -> None:
    """Expose SenseNova text-to-image generation as a minimal Images API."""

    @app.post("/v1/images/generations", response_model=ImageGenerationResponse)
    async def image_generations(
        image_request: ImageGenerationRequest, request: Request
    ) -> ImageGenerationResponse:
        if (
            image_request.model is not None
            and image_request.model != app.state.model_name
        ):
            raise HTTPException(status_code=400, detail="Unknown image model")
        else:
            pass
        try:
            width, height = parse_image_size(image_request.size)
            params = {
                "width": width,
                "height": height,
                "num_inference_steps": image_request.num_inference_steps,
                "guidance_scale": image_request.guidance_scale,
                "seed": image_request.seed,
                "n": image_request.n,
            }
            if image_request.enable_cache_dit is not None:
                params["enable_cache_dit"] = image_request.enable_cache_dit
            else:
                pass
            if image_request.cache_dit_params is not None:
                params["cache_dit_params"] = image_request.cache_dit_params
            else:
                pass
            SenseNovaU1Sampling.from_params(params)
            if not image_request.prompt.strip():
                raise ValueError("SenseNova-U1 requires a non-empty text prompt")
            else:
                pass
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        generate_request = GenerateRequest(
            model=app.state.model_name,
            prompt=image_request.prompt,
            extra_params=params,
            output_modalities=["image"],
            stream=False,
        )
        request_id = str(uuid.uuid4())
        try:
            chunk = await image_result_with_disconnect_watch(
                request, app.state.client, generate_request, request_id
            )
            encoded_outputs = chunk.images_b64 or (
                [chunk.image_b64] if chunk.image_b64 is not None else []
            )
            if len(encoded_outputs) != image_request.n:
                raise RuntimeError(
                    f"SenseNova-U1 expected {image_request.n} images, "
                    f"got {len(encoded_outputs)}"
                )
            else:
                pass
            return build_image_response(encoded_outputs, chunk.image_usage)
        except Exception as exc:
            logger.exception(f"Image generation failed for request {request_id}")
            raise HTTPException(
                status_code=generation_error_status_code(exc), detail=str(exc)
            ) from exc


def register_image_edits(app: FastAPI) -> None:
    """Expose SenseNova image editing as a minimal Images API."""

    @app.post("/v1/images/edits", response_model=ImageGenerationResponse)
    async def image_edits(
        request: Request,
        image: list[UploadFile] | None = File(default=None),  # noqa: B008
        image_array: list[UploadFile] | None = File(  # noqa: B008
            default=None, alias="image[]"
        ),
        prompt: str = Form(...),
        mask: UploadFile | None = File(default=None),  # noqa: B008
        model: str | None = Form(default=None),
        n: int = Form(default=1),
        response_format: str | None = Form(default=None),
        size: str | None = Form(default=None),
        seed: int | None = Form(default=None),
        num_inference_steps: int | None = Form(default=None),
        guidance_scale: float | None = Form(default=None),
        img_cfg_scale: float | None = Form(default=None),
        enable_cache_dit: str | None = Form(default=None),
        cache_dit_params: str | None = Form(default=None),
    ) -> ImageGenerationResponse:
        if model is not None and model != app.state.model_name:
            raise HTTPException(status_code=400, detail="Unknown image model")
        else:
            pass
        uploads = image or image_array
        if not uploads:
            raise HTTPException(
                status_code=422,
                detail="Field 'image' is required",
            )
        else:
            pass
        if mask is not None:
            raise HTTPException(
                status_code=400,
                detail="SenseNova-U1 image edits do not support masks yet",
            )
        else:
            pass
        resolved_response_format = (response_format or "b64_json").lower()
        if resolved_response_format != "b64_json":
            raise HTTPException(
                status_code=400,
                detail="SenseNova-U1 image edits only support response_format=b64_json",
            )
        else:
            pass
        try:
            if not 1 <= n <= 10:
                raise ValueError("SenseNova-U1 n must be an integer between 1 and 10")
            else:
                pass
            params = {"n": 1, "size_explicit": size is not None}
            if enable_cache_dit:
                normalized_cache_dit = enable_cache_dit.strip().lower()
                if normalized_cache_dit in ("true", "1", "yes", "on"):
                    params["enable_cache_dit"] = True
                elif normalized_cache_dit in ("false", "0", "no", "off"):
                    params["enable_cache_dit"] = False
                else:
                    raise ValueError("enable_cache_dit must be a boolean")
            else:
                pass
            if cache_dit_params is not None:
                parsed_cache_params = json.loads(cache_dit_params)
                if not isinstance(parsed_cache_params, dict):
                    raise ValueError("cache_dit_params must be a JSON object")
                else:
                    pass
                params["cache_dit_params"] = parsed_cache_params
            else:
                pass
            if size is not None:
                width, height = parse_image_size(size)
                params.update(width=width, height=height)
            else:
                pass
            for name, value in (
                ("num_inference_steps", num_inference_steps),
                ("guidance_scale", guidance_scale),
                ("img_cfg_scale", img_cfg_scale),
                ("seed", seed),
            ):
                if value is not None:
                    params[name] = value
                else:
                    pass
            edit_sampling = SenseNovaU1ImageEditSampling.from_params(params)
            params.update(
                width=edit_sampling.width,
                height=edit_sampling.height,
                num_inference_steps=edit_sampling.num_inference_steps,
                guidance_scale=edit_sampling.guidance_scale,
                img_cfg_scale=edit_sampling.img_cfg_scale,
                seed=edit_sampling.seed,
            )
            if not prompt.strip():
                raise ValueError("SenseNova-U1 requires a non-empty image edit prompt")
            else:
                pass
            reference_images = [
                await read_image_edit_upload(upload) for upload in uploads
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        encoded_references = [
            base64.b64encode(reference_image).decode("ascii")
            for reference_image in reference_images
        ]
        encoded_outputs = []
        image_usage = None
        parent_request_id = str(uuid.uuid4())
        try:
            for output_index in range(n):
                output_params = dict(params)
                output_params["seed"] = edit_sampling.seed + output_index
                generate_request = GenerateRequest(
                    model=app.state.model_name,
                    prompt={
                        "task": "image_edit",
                        "prompt": prompt,
                        "image_b64": encoded_references,
                    },
                    extra_params=output_params,
                    output_modalities=["image"],
                    stream=False,
                )
                request_id = f"{parent_request_id}:{output_index}"
                chunk = await image_result_with_disconnect_watch(
                    request, app.state.client, generate_request, request_id
                )
                if chunk.image_b64 is None:
                    raise RuntimeError("SenseNova-U1 generated no image")
                else:
                    pass
                if output_index == 0:
                    image_usage = chunk.image_usage
                else:
                    pass
                encoded_outputs.append(chunk.image_b64)
        except Exception as exc:
            logger.exception(f"Image edit failed for request {parent_request_id}")
            raise HTTPException(
                status_code=generation_error_status_code(exc), detail=str(exc)
            ) from exc
        return build_image_response(encoded_outputs, image_usage)


def register_images_api(app: FastAPI) -> None:
    """Register SenseNova's Images API routes on the initialized app."""
    register_image_generations(app)
    register_image_edits(app)
