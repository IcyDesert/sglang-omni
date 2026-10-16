# SPDX-License-Identifier: Apache-2.0
"""Cascade component protocols, configuration and launch profiles."""

import asyncio
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from sglang_omni.audio.segmentation import UtteranceSegmenter
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.streaming_cascade.base import (
    Replies,
    text_chunk,
    transcript_chunk,
)
from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.models.streaming_cascade.config import (
    ComponentGraphConfig,
    StreamingCascadePipelineConfig,
)
from sglang_omni.models.streaming_cascade.recognizers import probe_realtime_service
from sglang_omni.models.streaming_cascade.running_transcript import (
    RunningTranscript,
    RunningTranscriptConfig,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    INTERRUPT,
    MESSAGES,
    PCM16,
    PCM_F32,
    REPLY_PROGRESS,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    AudioPort,
    SessionPorts,
    control_type,
)
from tests.unit_test.fixtures.cascade_fakes import QUIET, build, drain
from tests.unit_test.fixtures.session_outputs import control_chunk

PROFILES = Path("examples/streaming_cascade/profiles")
PROFILE_NAMES = (
    "nemotron-qwen3-tts",
    "nemotron-stage-qwen3-tts",
    "qwen3-asr-qwen3-tts-customvoice",
    "qwen3-asr-qwen3.8-27b-customvoice",
)
HTTP_PROFILE = PROFILES / "nemotron-qwen3-tts" / "launch.yaml"
CASCADE_PORTS = SessionPorts(
    input=(AudioPort(PCM16, 16000),),
    output=AudioPort(PCM_F32, 24000),
    unit_ms=200,
    controls=frozenset({COMMIT, CONFIGURE, INTERRUPT, TRUNCATE}),
    hot_settings=frozenset({"instructions", "sampling", "turn_detection"}),
    sampling_parameters=("temperature", "max_new_tokens_per_unit"),
)


def profile_config(profile: str) -> StreamingCascadePipelineConfig:
    return ConfigManager.from_file(str(PROFILES / profile / "launch.yaml")).config


@pytest.mark.asyncio
async def test_speech_over_a_reply_makes_the_dialog_interrupt_before_turn_interrupted():
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world.", 1),
            control_chunk(SPEECH_STARTED),
        ],
    )
    assert [control_type(o) for o in outputs] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
        INTERRUPT,
        TURN_INTERRUPTED,
    ]


@pytest.mark.asyncio
async def test_a_truncate_after_turn_done_cuts_the_reply_that_was_still_playing():
    dialog = build({"kind": "dialog"})
    await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world. Second sentence.", 1),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, words=2, audio_ms=1000)
            ),
            lambda: dialog.control(
                control_chunk(REPLY_PROGRESS, turn=1, words=4, audio_ms=2000)
            ),
            control_chunk(TURN_DONE, turn=1),
            control_chunk(TRUNCATE, t_ms=1500),
        ],
    )
    assert dialog.messages[-1].content == "Hello world."
    await drain(dialog, [transcript_chunk("next", True)])
    await drain(dialog, [control_chunk(TRUNCATE, t_ms=0)])
    assert dialog.messages[-2].content == "Hello world."


@pytest.mark.asyncio
async def test_replies_forget_interrupted_turns_once_a_later_reply_starts():
    replies = Replies()
    for turn in (1, 2, 4):
        await replies.interrupt(turn)
    replies.start(3, asyncio.sleep(0))
    await replies.finish()
    assert replies.interrupted == {4}


def test_extra_body_cannot_set_the_fields_the_chat_kind_sets():
    with pytest.raises(ValueError, match="extra_body cannot set"):
        COMPONENTS.parse(
            {"kind": "openai_chat", **HTTP_SERVICE, "extra_body": {"stream": False}}
        )
    config = COMPONENTS.parse(
        {
            "kind": "openai_chat",
            **HTTP_SERVICE,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    )
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def declared(nodes=None, publish=None, **fields) -> StreamingCascadePipelineConfig:
    """The HTTP profile's declaration with `nodes` and `publish` replacing its own."""
    data = yaml.safe_load(HTTP_PROFILE.read_text())
    return StreamingCascadePipelineConfig(
        model_path="m",
        session_nodes={**data["session_nodes"], **(nodes or {})},
        session_publish={**data["session_publish"], **(publish or {})},
        **fields,
    )


HTTP_SERVICE = {"base_url": "http://127.0.0.1:8000/v1", "model": "test-model"}
DECLARATION_ERRORS = r"""
- [{nodes: {tts: {kind: openai_speech, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [input]}}}, produces]
- [{nodes: {dialog: {kind: dialog, inputs: ["asr:session", llm]}}}, cannot take]
- [{nodes: {asr: {kind: nope, inputs: [input]}}}, unknown component kind]
- [{nodes: {vad: {stages: [vad], inputs: [input], threshold: 0.5}}}, belong to a node of a kind]
- [{nodes: {asr: {kind: realtime_transcription, base_url: "ws://127.0.0.1:8000", model: test-asr, stages: [asr], inputs: [input]}}}, either stages or a kind]
- [{nodes: {asr: {stages: [ears], inputs: [input]}}}, no stage group]
- [{publish: {ghost: [session]}}, publish names unknown nodes]
- [{publish: {asr: [text_delta]}}, does not produce]
- [{publish: {tts: [pcm_f32le]}}, no published node ends turns]
- [{publish: {llm: [session]}}, turns end twice]
- [{publish: {dialog: []}}, open turns]
- [{nodes: {extra: {kind: tool, inputs: [llm], tools: {}}}}, unknown component kind]
"""


@pytest.mark.parametrize(
    "declaration, message",
    yaml.safe_load(DECLARATION_ERRORS),
)
def test_declaration_errors(declaration, message):
    with pytest.raises(ValueError, match=message):
        declared(**declaration)


def test_settings_merge_and_standard_names():
    settings = declared().component_graph()
    merged = settings.apply(
        {
            "instructions": "Be terse.",
            "sampling": {"temperature": 0.2, "max_new_tokens_per_unit": 64},
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 800},
            "nodes": {"tts": {"voice": "b"}, "asr": {"timeout_s": 30}},
        }
    )
    assert merged.node("dialog").component.system_prompt == "Be terse."
    llm = merged.node("llm").component
    local = "http://127.0.0.1:8000/v1"
    assert (llm.temperature, llm.max_tokens, llm.base_url) == (0.2, 64, local)
    assert merged.node("tts").component.voice == "b"
    assert merged.node("asr").component.timeout_s == 30
    assert settings.node("dialog").component.system_prompt != "Be terse."
    asr = settings.apply(
        {"nodes": {"asr": {"kind": "openai_transcription", **HTTP_SERVICE}}}
    ).node("asr")
    assert (asr.component.base_url, asr.inputs) == (local, ("input", "vad:session"))
    assert ComponentGraphConfig.model_validate_json(merged.model_dump_json()) == merged
    for update, message in [
        ({"nodes": {"ghost": {}}}, "unknown node"),
        ({"nodes": {"llm": {"inputs": ["input"]}}}, "rewire"),
        ({"nodes": {"tts": {"voice": 3}}}, "voice"),
        ({"nodes": {"llm": {"base_url": "file:///tmp"}}}, "base_url"),
        ({"publish": {}}, "unknown cascade settings"),
        ({"sampling": {"top_k": 3}}, "unsupported sampling"),
        ({"instructions": 3}, "string"),
    ]:
        with pytest.raises(ValueError, match=message):
            settings.apply(update)


TEXT_ONLY = """
nodes:
  - {name: asr, kind: openai_transcription, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [input]}
  - {name: dialog, kind: dialog, inputs: [asr, llm]}
  - {name: llm, kind: openai_chat, base_url: "http://127.0.0.1:8000/v1", model: test-model, inputs: [dialog]}
publish: {dialog: [session], llm: [text_delta, session]}
"""


def test_profile_graph_turns_formats_and_ports():
    settings = declared().component_graph()
    assert [n.name for n in settings.nodes if n.opens_turns] == ["dialog"]
    assert [n.name for n in settings.nodes if n.ends_turns] == ["llm", "tts"]
    assert settings.publish["llm"] == [TEXT_DELTA]
    hot_settings = frozenset({"instructions", "sampling"})
    assert settings.ports(200) == replace(CASCADE_PORTS, hot_settings=hot_settings)
    assert declared(unit_ms=100).session_ports().unit_ms == 100
    text_only = ComponentGraphConfig.model_validate(yaml.safe_load(TEXT_ONLY)).ports(
        200
    )
    assert (text_only.output, text_only.hot_settings) == (None, hot_settings)
    chat = settings.node("llm")
    assert (chat.consumes, chat.required) == ({MESSAGES, SESSION_CONTROL}, {MESSAGES})
    dialog_formats = {TRANSCRIPT_FORMAT, TEXT_DELTA, SESSION_CONTROL}
    assert settings.node("dialog").required == dialog_formats


def test_segmenter_gates_on_speech_controls():
    segmenter = UtteranceSegmenter()
    first = TimedChunk(AUDIO, 0, 100, 0, QUIET, format=PCM16)
    assert segmenter.step(first) == ["open", first]
    assert segmenter.step(control_chunk(SPEECH_STARTED, t_ms=0)) == []
    assert segmenter.step(control_chunk(SPEECH_STOPPED, t_ms=50)) == ["finish"]
    held = [
        TimedChunk(AUDIO, t, 100, 0, QUIET, format=PCM16) for t in range(100, 2600, 100)
    ]
    for chunk in held:
        assert segmenter.step(chunk) == []
    assert segmenter.pending_audio[0].t_start_ms == 600
    restart = control_chunk(SPEECH_STARTED, t_ms=2350)
    assert segmenter.step(restart) == ["open", *held[-3:]]
    assert segmenter.step(control_chunk(COMMIT)) == ["finish"]
    assert segmenter.step(control_chunk(COMMIT)) == ["empty"]
    assert segmenter.finish() == []


@pytest.mark.asyncio
async def test_realtime_transcription_streams_partials_then_final_per_utterance(
    realtime_transcription_server,
):
    base_url, appended, client_events = realtime_transcription_server
    frames = [bytes([index]) * 640 for index in range(4)]
    audio = [
        TimedChunk(AUDIO, index * 20, 20, index, frame, format=PCM16)
        for index, frame in enumerate(frames)
    ]
    spec = {
        "kind": "realtime_transcription",
        "base_url": base_url,
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    }
    outputs = await drain(build(spec), [*audio, control_chunk(COMMIT)])
    assert [chunk.payload for chunk in outputs] == [
        {"text": "hello", "is_final": False},
        {"text": "hello there", "is_final": False},
        {"text": "hello there", "is_final": True},
    ]
    assert appended == list(enumerate(frames))
    assert client_events == [
        "session.update",
        *["input_audio_buffer.append"] * 4,
        "sglang.input_audio.end",
        "session.close",
    ]


@pytest.mark.asyncio
async def test_openai_speech_flushes_each_reply_before_turn_done(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    inputs = [
        text_chunk("Hello wor", 1),
        text_chunk("ld. Second sentence here", 1),
        control_chunk(TURN_DONE, turn=1),
        text_chunk("Again", 2),
        control_chunk(TURN_DONE, turn=2),
    ]
    outputs = await drain(build({**tts, "max_sentence_words": 24}), inputs)
    outputs = [o for o in outputs if control_type(o) != REPLY_PROGRESS]
    assert spoken == ["Hello world.", "Second sentence here", "Again"]
    labels = [control_type(o) or o.format for o in outputs]
    runs = [label for i, label in enumerate(labels) if i == 0 or labels[i - 1] != label]
    assert runs == [PCM_F32, TURN_DONE] * 2
    assert [o.turn for o in outputs if o.format == SESSION_CONTROL] == [1, 2]
    samples = [
        np.frombuffer(o.payload, dtype="<f4") for o in outputs if o.modality == AUDIO
    ]
    assert np.concatenate(samples).tolist() == [0.0, 0.5, -0.5] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["english_tts"], indirect=True)
async def test_openai_speech_tags_each_sentence_with_its_language(backends):
    settings, _, spoken, _ = backends
    tts = settings.component_graph().node("tts").component.model_dump()
    inputs = [
        text_chunk("Hello world. 你好世界。", 1),
        control_chunk(TURN_DONE, turn=1),
    ]
    await drain(build({**tts, "language": "English"}), inputs)
    assert spoken[-1] == "你好世界。" and " ".join(spoken[:-1]) == "Hello world."


@pytest.mark.asyncio
@pytest.mark.parametrize("backends", ["unlabeled_rate"], indirect=True)
async def test_openai_speech_refuses_audio_of_an_unnamed_rate(backends):
    tts = backends[0].component_graph().node("tts").component.model_dump()
    with pytest.raises(ValueError, match="X-Sample-Rate"):
        await drain(
            build(tts), [text_chunk("Hello.", 1), control_chunk(TURN_DONE, turn=1)]
        )


@pytest.mark.asyncio
async def test_realtime_probe_requests_models_over_http(backends):
    """The loopback server answers only `/v1/models`, so a passing probe asked for it."""
    base_url = backends[0].component_graph().node("asr").component.base_url
    ws_url = base_url.replace("http", "ws").removesuffix("/v1")
    await probe_realtime_service(
        COMPONENTS.parse(
            {"kind": "realtime_transcription", "base_url": ws_url, "model": "test-asr"}
        )
    )


@pytest.mark.asyncio
async def test_running_transcript_cuts_utterances_at_speech_end_plus_tail():
    node = RunningTranscript(RunningTranscriptConfig(tail_ms=400))

    def text(t_ms: float, delta: str, full_text: str) -> TimedChunk:
        payload = {"text": delta, "full_text": full_text, "is_first_output": False}
        return TimedChunk("text", t_ms, 200, 0, payload, format="text")

    def unit(t_ms: float) -> TimedChunk:
        return TimedChunk(AUDIO, t_ms, 200, 0, b"", format=PCM16)

    async def inputs():
        yield text(0, "hello", "hello")
        yield control_chunk(SPEECH_STOPPED, t_ms=200)
        yield unit(200)
        yield text(400, " there", "hello there")
        yield unit(400)
        yield text(1000, "?", "hello there?")
        yield text(1200, " again", "hello there? again")
        for t_ms in (1200, 1400, 1600):
            yield unit(t_ms)
        yield control_chunk(SPEECH_STOPPED, t_ms=1300)

    outputs = [chunk async for chunk in node.run(inputs())]
    assert [(c.payload["text"], c.payload["is_final"]) for c in outputs] == [
        ("hello", False),
        ("hello there", False),
        ("hello there", True),
        ("again", False),
        ("again", True),
    ]
    assert all(c.format == TRANSCRIPT_FORMAT for c in outputs)
    replacement = RunningTranscript(RunningTranscriptConfig(tail_ms=0))
    replacement.adopt(node)
    assert replacement.running() == ""


@pytest.mark.asyncio
async def test_running_transcript_cuts_an_utterance_at_commit_plus_tail():
    node = RunningTranscript(RunningTranscriptConfig(tail_ms=400))

    async def inputs():
        payload = {"text": "hello", "full_text": "hello", "is_first_output": False}
        yield TimedChunk("text", 0, 200, 0, payload, format="text")
        yield TimedChunk(AUDIO, 0, 200, 0, b"", format=PCM16)
        yield control_chunk(COMMIT)
        for t_ms in (200, 400):
            yield TimedChunk(AUDIO, t_ms, 200, 0, b"", format=PCM16)

    outputs = [chunk async for chunk in node.run(inputs())]
    assert [(c.payload["text"], c.payload["is_final"]) for c in outputs] == [
        ("hello", False),
        ("hello", True),
    ]


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profiles_expand_components_into_independent_session_nodes(profile):
    config = profile_config(profile)
    graph = config.session_graph()
    assert set(graph.nodes) == set(config.session_nodes)
    assert graph.publish == {
        name: tuple(formats) for name, formats in config.session_publish.items()
    }
    assert config.session_ports() == CASCADE_PORTS
    assert (
        StreamingCascadePipelineConfig.model_validate(
            config.model_dump()
        ).session_graph()
        == graph
    )
    for name, node in config.session_nodes.items():
        if node.kind is not None:
            assert graph.nodes[name].stages == (name,)
            assert config.stage_named(name).process == "cascade"
            assert config.stage_factory_kwargs(name)["name"] == name
        else:
            assert graph.nodes[name].stages == tuple(node.stages)
    assert graph.nodes["dialog"].bound("llm").feedback
    assert graph.nodes["dialog"].bound("tts").feedback
