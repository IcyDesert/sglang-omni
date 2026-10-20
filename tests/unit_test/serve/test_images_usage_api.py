# SPDX-License-Identifier: Apache-2.0
import base64
import io
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import JsonValue

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import GenerateChunk, GenerateRequest, UMMSegment
from sglang_omni.models.sensenova_u1.images_api import register_images_api
from sglang_omni.pipeline.umm import UMMController, UMMRequestError
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import create_app

pytest_plugins = ["tests.unit_test.fixtures.sensenova_umm"]


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize("fail_after_image", [False, True])
def test_images_usage_and_complete_result_consumption(
    editing: bool, outputs: int, fail_after_image: bool
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")

    class ImageClient(Client):
        def __init__(self) -> None:
            super().__init__(None)
            self.completed: int = 0

        async def generate(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncIterator[GenerateChunk]:
            image_count = request.extra_params["n"]
            image_usage = {
                "input_tokens": 3,
                "total_tokens": 3,
                "input_tokens_details": {
                    "text_tokens": 3,
                    "image_tokens": 16 if editing else 0,
                },
            }
            segments = [
                UMMSegment(
                    "execution",
                    index,
                    "image",
                    InlineImage.from_png(encoded_image).model_dump(),
                )
                for index in range(image_count)
            ]
            segments.append(
                UMMSegment(
                    "execution",
                    image_count,
                    "usage",
                    {"image_usage": image_usage, "stage_name": "generate"},
                )
            )
            yield self.default_result_builder(
                request_id,
                {
                    "type": "umm_result",
                    "execution_id": "execution",
                    "segments": [segment.to_dict() for segment in segments],
                    "finish_reason": "stop",
                },
            )
            if fail_after_image:
                raise RuntimeError("Backend completion failed")
            else:
                self.completed += 1

    backend = ImageClient()
    with TestClient(
        create_app(backend, model_name="test", image_api_registrar=register_images_api)
    ) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data={"prompt": "a red mug", "size": "64x32", "n": outputs},
                files={"image": ("reference.png", buffer.getvalue(), "image/png")},
            )
        else:
            response = client.post(
                "/v1/images/generations",
                json={"prompt": "a red mug", "size": "64x32", "n": outputs},
            )
    assert response.status_code == (500 if fail_after_image else 200)
    if fail_after_image:
        assert backend.completed == 0
    else:
        result = response.json()
        assert backend.completed == (outputs if editing else 1)
        assert len(result["data"]) == outputs
        assert all(image["output_format"] == "png" for image in result["data"])
        assert result["usage"]["input_tokens"] == result["usage"]["total_tokens"] == 3
        assert result["usage"]["input_tokens_details"] == {
            "text_tokens": 3,
            "image_tokens": 16 if editing else 0,
        }
        assert result["usage"]["image_count"] == outputs
        assert (result["usage"]["width"], result["usage"]["height"]) == (64, 32)


@pytest.mark.parametrize("editing", [False, True])
def test_images_endpoints_consume_real_umm_results(
    controller: UMMController, encoded_images: list[str], editing: bool
) -> None:
    output_images = iter(encoded_images)

    class GenerationCoordinator:
        async def submit(self, request_id: str, request: OmniRequest) -> JsonValue:
            phase = controller.transition(StagePayload(request_id, request, None))
            assert phase is not None
            phase.data = {
                "modality": "image",
                "images_b64": [
                    next(output_images) for _ in range(request.params.get("n", 1))
                ],
                "image_usage": {"input_tokens": 3, "total_tokens": 3},
            }
            completion = controller.transition(phase)
            assert completion is not None
            return completion.data

    application = create_app(
        Client(GenerationCoordinator()),
        image_api_registrar=register_images_api,
        interleaved_output=True,
    )
    with TestClient(application) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data={"prompt": "a mug", "size": "64x32", "n": 2},
                files={
                    "image": (
                        "input.png",
                        base64.b64decode(encoded_images[0]),
                        "image/png",
                    )
                },
            )
        else:
            response = client.post(
                "/v1/images/generations",
                json={"prompt": "a mug", "size": "64x32", "n": 2},
            )
    assert response.status_code == 200, response.text
    assert [image["b64_json"] for image in response.json()["data"]] == encoded_images[
        :2
    ]
    assert response.json()["usage"]["input_tokens"] == 3


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("capacity_rejection", [False, True])
def test_images_preserve_umm_rejection_status(
    encoded_image: str, editing: bool, capacity_rejection: bool
) -> None:
    class RejectedCoordinator:
        async def submit(self, request_id: str, request: OmniRequest) -> JsonValue:
            if capacity_rejection:
                raise QueueFullError()
            else:
                raise UMMRequestError("request exceeds the configured image budget")

    application = create_app(
        Client(RejectedCoordinator()),
        image_api_registrar=register_images_api,
        interleaved_output=True,
    )
    with TestClient(application) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data={"prompt": "edit", "size": "64x32"},
                files={
                    "image": ("input.png", base64.b64decode(encoded_image), "image/png")
                },
            )
        else:
            response = client.post(
                "/v1/images/generations", json={"prompt": "draw", "size": "64x32"}
            )
    assert response.status_code == (503 if capacity_rejection else 400)
