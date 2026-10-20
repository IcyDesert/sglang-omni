# SPDX-License-Identifier: Apache-2.0
"""Check standalone native modules without loading accelerator dependencies."""

import importlib
import pickle
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Protocol
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig, Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


class NativeModuleLoader(Protocol):
    def __call__(self, module: str) -> ModuleType: ...


@pytest.fixture
def native_modules(monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeModuleLoader]:
    name = "sglang_omni.models.sensenova_u1.neo_unify"
    package = ModuleType(name)
    package.__path__ = [
        str(
            Path(__file__).resolve().parents[3]
            / "sglang_omni/models/sensenova_u1/neo_unify"
        )
    ]
    monkeypatch.setitem(sys.modules, name, package)
    try:
        yield lambda module: importlib.import_module(f"{name}.{module}")
    finally:
        for key in list(sys.modules):
            if key.startswith(f"{name}."):
                sys.modules.pop(key)
            else:
                pass


def test_moe_config_detects_expert_count_without_model_type(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("configuration_neo_chat")
    config = module.build_llm_config(
        {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32}
    )
    assert isinstance(config, module.NEOMoELLMConfig)
    assert config.num_experts == 4
    assert not module.is_moe_llm_config({"num_experts": 1})


def test_model_input_compat_is_available_on_installed_transformers(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("transformers_compat")
    assert callable(module.model_input_compat)


def test_deep_flow_head_constructs_and_accepts_timesteps(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("modeling_fm_modules")
    head = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    with torch.inference_mode():
        outputs = head(torch.randn(5, 8), torch.linspace(0, 1, 5))
    assert outputs.shape == (5, 12)
    assert torch.isfinite(outputs).all()
    assert "net.time_embed.mlp.0.weight" in head.state_dict()
    assert "net.final_layer.adaLN_modulation.1.weight" in head.state_dict()


@pytest.mark.parametrize("widths", [2048, [2048], [[2048]], [1024, 2048]])
def test_vision_config_normalizes_checkpoint_dimensions(
    native_modules: NativeModuleLoader, widths: int | list[int] | list[list[int]]
) -> None:
    module = native_modules("configuration_neo_vit")
    config = module.NEOVisionConfig(llm_hidden_size=widths, downsample_ratio=[0.5])
    expected = (1024, 2048) if widths == [1024, 2048] else (2048,)
    assert config.llm_hidden_size == expected
    assert config.downsample_ratio == (0.5,)
    restored = module.NEOVisionConfig.from_dict(config.to_dict())
    assert restored.llm_hidden_size == expected
    assert restored.downsample_ratio == (0.5,)


def test_flow_head_state_dict_round_trip(native_modules: NativeModuleLoader) -> None:
    module = native_modules("modeling_fm_modules")
    original = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    restored = module.FlowMatchingHead(8, 12, dim=16, layers=3, mlp_ratio=2).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    hidden_states = torch.randn(5, 8)
    timesteps = torch.linspace(0, 1, 5)
    with torch.inference_mode():
        torch.testing.assert_close(
            original(hidden_states, timesteps), restored(hidden_states, timesteps)
        )


def test_causal_mask_kwargs_preserve_transformers_version_contract(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("transformers_compat")

    def legacy_mask(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return input_embeds + cache_position.reshape(1, -1, 1)

    def current_mask(
        config: PretrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert position_ids is not None
        return inputs_embeds + position_ids.unsqueeze(-1)

    embeddings = torch.randn(1, 3, 4)
    positions = torch.arange(3)
    for factory in (legacy_mask, current_mask):
        keywords = module.causal_mask_kwargs(
            factory,
            config=PretrainedConfig(),
            inputs_embeds=embeddings,
            attention_mask=None,
            cache_position=positions,
            past_key_values=None,
            position_ids=positions.unsqueeze(0),
        )
        torch.testing.assert_close(
            factory(**keywords), embeddings + positions.reshape(1, -1, 1)
        )


def test_sdpa_grouped_query_matches_explicit_head_repetition(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("attention_backend")
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, 3, 4, 8, generator=generator)
    key = torch.randn(2, 5, 2, 8, generator=generator)
    value = torch.randn(2, 5, 2, 8, generator=generator)
    actual = module.sdpa_attn_func(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.repeat_interleave(2, dim=2).transpose(1, 2),
        value.repeat_interleave(2, dim=2).transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)


def test_seeded_generation_preserves_global_rng(
    native_modules: NativeModuleLoader,
) -> None:
    module = native_modules("generation_math")
    previous_state = torch.get_rng_state().clone()
    actual = module.randn_with_seed(
        (2, 3, 4), device="cpu", dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(seed))
            for seed in (7, 19)
        ]
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.get_rng_state(), previous_state)


@pytest.mark.parametrize("additional_tokens", [1, 4])
def test_preallocated_cache_preserves_appended_values(
    native_modules: NativeModuleLoader, additional_tokens: int
) -> None:

    prefix_cache = native_modules("prefix_cache")
    config = Qwen3Config(num_hidden_layers=1, num_key_value_heads=1, head_dim=2)
    cache = DynamicCache(config=config)
    prefix = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
    cache.update(prefix, prefix + 10, 0)
    prefix_cache.preallocate_think_cache(cache, additional_tokens)
    suffix = torch.ones(1, 1, 2, 2)
    keys, values = cache.update(suffix, suffix + 20, 0)
    torch.testing.assert_close(keys, torch.cat((prefix, suffix), dim=-2))
    torch.testing.assert_close(values, torch.cat((prefix + 10, suffix + 20), dim=-2))
    assert cache.get_seq_length() == 5


def test_prefix_transfer_rejects_mismatched_tokens_and_cleans_files(
    native_modules: NativeModuleLoader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:

    prefix_cache = native_modules("prefix_cache")
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_DIR", str(tmp_path))
    metadata_path = tmp_path / "prefix.json"
    buffer_path = tmp_path / "prefix.bin"
    metadata_path.write_text("{}")
    buffer_path.write_bytes(b"unchanged")
    metadata = {
        "dump_id": "prefix",
        "data_file": "prefix.bin",
        "shape": [1, 2, 1, 2, 2],
        "layer_ids": [0],
        "dtype": "torch.bfloat16",
        "token_sha256": "wrong",
    }
    config = Qwen3Config(num_hidden_layers=1, num_key_value_heads=1, head_dim=2)
    with pytest.raises(RuntimeError, match="token hash does not match"):
        prefix_cache.load_srt_prefix_kv(metadata, [1, 2], config, torch.device("cpu"))
    assert not metadata_path.exists()
    assert not buffer_path.exists()


@pytest.mark.parametrize(
    "template_name",
    ["Hermes-2", "internlm2-chat", "phi3-chat", "internvl2_5", "neo1_0"],
)
def test_conversation_templates_keep_request_histories_independent(
    native_modules: NativeModuleLoader, template_name: str
) -> None:
    conversations = native_modules("conversation")
    first = conversations.get_conv_template(template_name)
    second = conversations.get_conv_template(template_name)
    original_prompt = second.get_prompt()
    first.append_message(first.roles[0], "request-specific prompt")
    first.append_message(first.roles[1], None)
    assert "request-specific prompt" in first.get_prompt()
    assert second.get_prompt() == original_prompt
    assert (
        conversations.get_conv_template(template_name).get_prompt() == original_prompt
    )


def test_conversation_preserves_image_message_and_registration_contract(
    native_modules: NativeModuleLoader,
) -> None:
    conversations = native_modules("conversation")
    template = conversations.get_conv_template("neo1_0")
    template.name = "custom-review-template"
    template.append_message(template.roles[0], ("image question", [], "Default"))
    conversations.register_conv_template(template)
    assert (
        "image question" in conversations.get_conv_template(template.name).get_prompt()
    )
    with pytest.raises(AssertionError, match="has been registered"):
        conversations.register_conv_template(template)
    replacement = template.copy()
    replacement.set_system_message("replacement system prompt")
    conversations.register_conv_template(replacement, override=True)
    assert (
        "replacement system prompt"
        in conversations.get_conv_template(template.name).get_prompt()
    )


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [
        ("PatchDecoder_postps", "PatchDecoderPostPixelShuffle"),
        ("PatchDecoder_preps", "PatchDecoderPrePixelShuffle"),
        ("PatchDecoder_preps1", "PatchDecoderPrePixelShuffleSingleProjection"),
    ],
)
def test_legacy_decoder_import_and_pickle_contract(
    native_modules: NativeModuleLoader, legacy_name: str, canonical_name: str
) -> None:
    exports = native_modules("modeling_fm_modules")
    decoders = native_modules("image_decoders")
    legacy_class = getattr(exports, legacy_name)
    assert legacy_class is getattr(decoders, canonical_name)
    assert legacy_class is getattr(decoders, legacy_name)
    assert pickle.loads(pickle.dumps(legacy_class)) is legacy_class
    historical_pickle = f"c{exports.__name__}\n{legacy_name}\n.".encode("ascii")
    assert pickle.loads(historical_pickle) is legacy_class


@pytest.mark.parametrize("ratio", [0.5, [0.5], [[0.5]], [[[0.5]]], (0.5,)])
def test_vision_config_preserves_nested_ratio_round_trip(
    native_modules: NativeModuleLoader,
    ratio: (
        float | list[float] | list[list[float]] | list[list[list[float]]] | tuple[float]
    ),
) -> None:
    configuration = native_modules("configuration_neo_vit")
    config = configuration.NEOVisionConfig(downsample_ratio=ratio)
    assert config.downsample_ratio == (0.5,)
    restored = configuration.NEOVisionConfig.from_dict(config.to_dict())
    assert restored.downsample_ratio == (0.5,)


def test_interleaved_images_advance_sizes_and_stop_at_limit(
    native_modules: NativeModuleLoader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # note (Wes): Isolate orchestration from accelerator model type imports.
    state_types = ModuleType(
        "sglang_omni.models.sensenova_u1.neo_unify.generation_state"
    )
    state_types.GenerationState = Mock
    monkeypatch.setitem(sys.modules, state_types.__name__, state_types)
    generation = native_modules("interleaved_generation")
    device = torch.device("cpu")
    cache = Mock()
    cache.get_seq_length.return_value = 0
    model = Mock()
    model.device = device
    model.template = "neo1_0"
    model.system_message = ""
    model.patch_size = 1
    model.downsample_ratio = 1.0
    model.build_t2i_query.return_value = ""
    model.build_it2i_inputs.return_value = (
        torch.zeros(1, 1, 2),
        torch.zeros(3, 1, dtype=torch.long),
        None,
    )
    model.language_model.return_value = CausalLMOutputWithPast(
        logits=torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]), past_key_values=cache
    )
    model.language_model.get_input_embeddings.return_value = torch.nn.Embedding(
        4, 2, dtype=torch.bfloat16
    )
    model.extract_feature.return_value = torch.zeros(4, 2, dtype=torch.bfloat16)
    tokenizer = Mock()

    def token_id(token: str) -> int:
        return {"<img>": 1, "</img>": 2, "<IMG_CONTEXT>": 3}.get(token, 0)

    tokenizer.convert_tokens_to_ids.side_effect = token_id
    image_sizes = [(2 + index, 2) for index in range(10)]
    denoise = Mock(
        side_effect=[(device, torch.tensor([[2, 2]]), torch.zeros(1, 3, 2, 2))] * 10
        + [AssertionError("Generation exceeded the existing image limit")]
    )
    monkeypatch.setattr(generation, "interleaved_text_image_denoise", denoise)

    text, _ = generation.interleave_gen(
        model, tokenizer, "Draw a sequence", image_size=image_sizes
    )

    assert text == "<image>" * 10
    assert denoise.call_count == 10
    selected_sizes = [call.args[6][call.args[8]] for call in denoise.call_args_list]
    assert selected_sizes == image_sizes
