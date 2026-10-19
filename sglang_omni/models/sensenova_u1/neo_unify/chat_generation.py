"""Chat generation."""

import logging

import torch.utils.checkpoint
from transformers import GenerationConfig, PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.conversation import get_conv_template
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import (
    GenerationState,
    GenerationValue,
)

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


def chat(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    pixel_values: torch.Tensor | None,
    question: str,
    generation_config: dict[str, GenerationValue],
    history: list[tuple[str, str]] | None = None,
    return_history: bool = False,
    grid_hw: torch.Tensor | None = None,
    IMG_START_TOKEN: str = "<img>",
    IMG_END_TOKEN: str = "</img>",
    IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
    verbose: bool = False,
) -> str | tuple[str, list[tuple[str, str]]]:

    if history is None and pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question
    else:
        pass

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.img_context_token_id = img_context_token_id
    self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)

    template = get_conv_template(self.template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        logger.info(f"Dynamic image size: {grid_hw[0] * self.patch_size}")
    else:
        pass

    for i in range(grid_hw.shape[0]):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * self.downsample_ratio**2)
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(self.device)
    attention_mask = model_inputs["attention_mask"].to(self.device)
    generation_config["eos_token_id"] = eos_token_id
    generation_output = self.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        grid_hw=grid_hw,
        attention_mask=attention_mask,
        **generation_config,
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split(template.sep.strip())[0].strip()
    history.append((question, response))
    if return_history:
        return response, history
    else:
        query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
        query_to_print = query_to_print.replace(
            f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
        )
        if verbose:
            print(query_to_print, response)
        else:
            pass
        return response


@torch.no_grad()
def generate(
    self: GenerationState,
    pixel_values: torch.FloatTensor | None = None,
    input_ids: torch.FloatTensor | None = None,
    grid_hw: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    visual_features: torch.FloatTensor | None = None,
    generation_config: GenerationConfig | None = None,
    output_hidden_states: bool | None = None,
    **generate_kwargs: GenerationValue,
) -> torch.LongTensor:
    assert input_ids.shape[0] == 1
    assert self.img_context_token_id is not None
    indexes = self.get_thw_indexes(input_ids[0], grid_hw)
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values, grid_hw=grid_hw)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        indexes=indexes,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs
