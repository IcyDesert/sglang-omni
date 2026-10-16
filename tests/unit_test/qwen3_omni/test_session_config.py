# SPDX-License-Identifier: Apache-2.0
"""The Qwen Session profile assembles without a cascade package."""

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import SessionNodeConfig
from sglang_omni.config.topology import compile_logical_processes
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSessionPipelineConfig
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.proto.session import ControlEdge
from sglang_omni.proto.streams import (
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    SESSION_CONTROL,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
)


def profile_config() -> Qwen3OmniSessionPipelineConfig:
    path = Path(__file__).resolve().parents[3] / "examples/qwen3_omni/session.yaml"
    return ConfigManager.from_file(str(path)).config


def test_qwen3_omni_profile_declares_the_native_node_beside_the_vad_on_one_gpu():
    config = profile_config()
    assert config.resolved_entry_stage == "preprocessing"
    with pytest.raises(ValueError, match="forks at 'preprocessing'"):
        config.session_route()
    graph = config.session_graph()
    assert graph.input_nodes() == ("vad", "qwen_omni")
    assert graph.reading_nodes("vad", SESSION_CONTROL) == ("qwen_omni",)
    assert graph.reading_nodes("input", PCM16) == ("vad", "qwen_omni")
    assert graph.reading_nodes("input", IMAGE_FRAME) == ("qwen_omni",)
    assert config.session_ports().input_modalities == ("audio", "image")
    assert graph.publish == {
        "vad": (SESSION_CONTROL,),
        "qwen_omni": (PCM16, SESSION_CONTROL, TEXT_DELTA, TRANSCRIPT_FORMAT),
    }
    assert "cascade" not in {s.name for s in config.stages}
    node = graph.nodes["qwen_omni"]
    assert node.execution == (
        "sglang_omni.models.qwen3_omni.session_execution.create_session_execution_factory"
    )
    assert node.controls == (ControlEdge("input", INTERRUPT, None, True),)
    assert set(node.stages) == {s.name for s in config.stages} - {"vad"}
    talker = next(s for s in config.stages if s.name == "talker_ar")
    assert talker.factory.enable_partial_start is True
    plan, stages = compile_logical_processes(config)
    expanded, topology = expand_replica_stages(stages, plan)
    placement = build_stage_placement_plan(
        config, stages_cfg=expanded, replica_instances=topology.replicas
    )
    qwen_stages = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")
    assert {placement.stages[name].gpu_ids for name in qwen_stages} == {(0,)}


def test_a_node_reading_every_input_format_must_take_camera_frames():
    config = profile_config()
    blind_vad = SessionNodeConfig(stages=["vad"], inputs=["input"])
    with pytest.raises(ValueError, match=r"nodes \['vad'\] read every format"):
        config.model_copy(
            update={"session_nodes": {**config.session_nodes, "vad": blind_vad}}
        ).session_ports()


def test_session_profile_round_trip_keeps_model_factory_arguments() -> None:
    config = profile_config()
    rebuilt = Qwen3OmniSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_ports() == config.session_ports()
    assert rebuilt.session_graph() == config.session_graph()
    assert rebuilt.session_limits().max_output_chunks == 1024
    for stage in config.stages:
        assert rebuilt.stage_factory_kwargs(stage.name) == config.stage_factory_kwargs(
            stage.name
        )
    assert config.realtime_deployment_factory == (
        "sglang_omni.serve.realtime.session_deployment.build_realtime_deployment"
    )
