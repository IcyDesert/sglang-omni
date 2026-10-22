"""Native image-edit prefill using prepared reference images."""

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.image_denoise import (
    ImagePrefix,
    denoise_images,
)
from sglang_omni.models.sensenova_u1.neo_unify.utils import (
    SYSTEM_MESSAGE_FOR_GEN,
    load_image_native,
)
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1ImageEditSampling


@torch.no_grad()
def it2i_generate(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    images: list[Image.Image],
    *,
    sampling: SenseNovaU1ImageEditSampling,
) -> torch.Tensor:
    self.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    self.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    image_placeholders = prompt.count("<image>")
    assert len(images) >= image_placeholders
    if len(images) > image_placeholders:
        if image_placeholders == 0 and len(images) > 1:
            prompt = (
                "".join(f"Image-{index + 1}:<image>\n" for index in range(len(images)))
                + prompt
            )
        else:
            prompt = "<image>\n" * (len(images) - image_placeholders) + prompt
    else:
        pass
    reference_patches: list[torch.Tensor] = []
    reference_grids: list[torch.Tensor] = []
    for image in images:
        image_patches, image_grid = load_image_native(
            image, self.patch_size, self.downsample_ratio
        )
        reference_patches.append(
            image_patches.to(
                device=self.device, dtype=next(self.vision_model.parameters()).dtype
            )
        )
        reference_grids.append(image_grid.to(self.device))
    pixel_values = torch.cat(reference_patches)
    grid_shape = torch.cat(reference_grids)
    queries = [
        self.build_t2i_query(
            prompt,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text="<think>\n\n</think>\n\n<img>",
        )
    ]
    if sampling.img_cfg_scale == 1 and sampling.guidance_scale == 1:
        pass
    else:
        if (
            sampling.img_cfg_scale == 1
            or sampling.guidance_scale != sampling.img_cfg_scale
        ):
            queries.append(
                self.build_t2i_query("<image>" * len(images), append_text="<img>")
            )
        else:
            pass
        if sampling.img_cfg_scale != 1:
            queries.append(self.build_t2i_query("", append_text="<img>"))
        else:
            pass
    for image_grid in grid_shape:
        image_token_count = int(
            image_grid[0] * image_grid[1] * self.downsample_ratio**2
        )
        image_tokens = "<img>" + "<IMG_CONTEXT>" * image_token_count + "</img>"
        queries = [query.replace("<image>", image_tokens, 1) for query in queries]
    image_patch_size = self.patch_size * int(1 / self.downsample_ratio)
    prefixes: list[ImagePrefix] = []
    for query in queries:
        has_references = "<IMG_CONTEXT>" in query
        input_embeddings, indexes, attention_mask = self.build_it2i_inputs(
            tokenizer,
            query,
            pixel_values if has_references else None,
            grid_shape if has_references else None,
        )
        cache, _ = self.it2i_prefix_forward(input_embeddings, indexes, attention_mask)
        prefixes.append(
            ImagePrefix(
                cache=cache,
                image_indexes=self.build_t2i_image_indexes(
                    sampling.height // image_patch_size,
                    sampling.width // image_patch_size,
                    indexes[0].max() + 1,
                    device=input_embeddings.device,
                ),
                token_lengths=torch.tensor(
                    [cache.get_seq_length()], dtype=torch.long, device=self.device
                ),
            )
        )
    return denoise_images(
        self, sampling=sampling, prefixes=prefixes, seeds=[sampling.seed]
    )
