"""Native text-to-image prefill and multi-output generation."""

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.image_denoise import (
    ImagePrefix,
    denoise_images,
)
from sglang_omni.models.sensenova_u1.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling


@torch.no_grad()
def t2i_generate(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[str],
    *,
    sampling: SenseNovaU1Sampling,
    seeds: list[int] | None = None,
) -> torch.Tensor:
    assert self.concat_time_token_num == 0
    prompts = prompt if isinstance(prompt, list) else [prompt]
    if not prompts:
        raise ValueError("prompt batch must not be empty")
    elif seeds is None:
        seeds = [sampling.seed + index for index in range(sampling.n)]
    else:
        pass
    if isinstance(prompt, list) and len(prompts) != len(seeds):
        raise ValueError("Each batched prompt requires one generation seed")
    else:
        pass
    queries = [
        [
            self.build_t2i_query(
                text,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n<img>",
            )
            for text in prompts
        ]
    ]
    if sampling.guidance_scale > 1:
        queries.append([self.build_t2i_query("", append_text="<img>")])
    else:
        pass
    image_patch_size = self.patch_size * int(1 / self.downsample_ratio)
    prefixes: list[ImagePrefix] = []
    for branch_queries in queries:
        token_ids, indexes, attention_mask, _, token_lengths = (
            self.build_t2i_text_inputs(tokenizer, branch_queries)
        )
        cache, _ = self.t2i_prefix_forward(token_ids, indexes, attention_mask)
        prefixes.append(
            ImagePrefix(
                cache=cache,
                image_indexes=self.build_t2i_image_indexes(
                    sampling.height // image_patch_size,
                    sampling.width // image_patch_size,
                    token_lengths,
                    device=token_ids.device,
                ),
                token_lengths=token_lengths,
            )
        )
    return denoise_images(self, sampling=sampling, prefixes=prefixes, seeds=seeds)
