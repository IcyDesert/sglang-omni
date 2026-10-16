# SPDX-License-Identifier: Apache-2.0
"""QwenSessionExecution: utterances become turns answered from audio, transcribed and recorded."""

from __future__ import annotations

import asyncio
import base64

import pytest

from sglang_omni.models.qwen3_omni.session_execution import (
    RESPONSE_INSTRUCTION,
    SEEING_RESPONSE_INSTRUCTION,
    TRANSCRIPTION_INSTRUCTION,
    TRANSCRIPTION_PROMPT,
    ConversationItem,
    QwenSessionExecution,
    QwenSessionLimits,
)
from sglang_omni.pipeline.session_execution import SessionWork
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionGraph, SessionNode, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    IMAGE,
    IMAGE_FRAME,
    PCM16,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
)
from sglang_omni.serve.realtime.events import SessionObject
from sglang_omni.serve.realtime.session import (
    ConversationItem as LegacyConversationItem,
)
from sglang_omni.serve.realtime.session import RealtimeSession
from tests.unit_test.fixtures.qwen_session import (
    TEST_TIMEOUT_S,
    BlockingGenerationClient,
    CompletedGenerationClient,
    DetachedSessionWork,
    FailedGenerationClient,
    Harness,
)
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import pipeline, wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("has_audio", [False, True])
async def test_commit_answers_the_utterance_and_records_both_sides(
    has_audio: bool,
) -> None:
    client = CompletedGenerationClient(has_audio=has_audio)
    h = Harness(client, output_modalities=("text", "audio") if has_audio else ("text",))
    await h.open({"instructions": "Be brief."})
    await h.audio()
    await h.audio()
    await h.control(COMMIT)
    await h.until(TURN_DONE)
    await h.settle()
    labels = h.labels()
    order = [
        labels.index(TURN_QUEUED),
        labels.index(TURN_STARTED),
        labels.index(TEXT_DELTA),
        labels.index(TURN_DONE),
    ]
    assert order == sorted(order)
    assert labels.count(PCM16) == int(has_audio)
    assert h.turn_controls() == [
        (TURN_QUEUED, 1),
        (TURN_STARTED, 1),
        (TURN_DONE, 1),
    ]
    assert h.text() == "assistant words"
    transcripts = [o.payload for o in h.outputs if o.format == TRANSCRIPT_FORMAT]
    assert transcripts[-1] == {"text": "user words", "is_final": True}
    assert [(entry.role, entry.text) for entry in h.execution.conversation] == [
        ("user", "user words"),
        ("assistant", "assistant words"),
    ]
    assert client.audio_formats == ["wav", "pcm"]
    reply_request = client.reply_request()
    assert reply_request.messages[0].content == "Be brief."
    assert reply_request.metadata["audios"][0].startswith("data:audio/wav;base64,")
    if has_audio:
        (audio,) = [o for o in h.outputs if o.modality == AUDIO]
        assert audio.duration_ms == 48 * 1000 / 24000
        assert audio.t_start_ms == h.outputs[labels.index(TURN_STARTED)].t_start_ms
    else:
        pass
    await h.execution.close()
    assert h.work.released


@pytest.mark.asyncio
async def test_the_reply_sees_the_newest_camera_frames_of_its_utterance_and_the_transcription_does_not() -> (
    None
):
    client = CompletedGenerationClient(has_audio=False)
    limits = QwenSessionLimits(cleanup_timeout_s=1, max_utterance_frames=2)
    h = Harness(client, output_modalities=("text",), limits=limits)
    await h.open()

    async def frame(payload: bytes, t_ms: float) -> None:
        await h.append(TimedChunk(IMAGE, t_ms, 0, h.seq, payload, format=IMAGE_FRAME))

    jpeg = [b"\xff\xd8frame-" + bytes([index]) for index in range(3)]
    png = b"\x89PNGframe"
    await frame(jpeg[0], 0)
    await h.audio()
    await frame(jpeg[1], h.t_ms)
    await frame(png, h.t_ms)
    await h.audio()
    await h.control(COMMIT)
    await h.until(TURN_DONE)
    await h.settle()
    await frame(jpeg[2], 0)
    await h.audio()
    await h.control(COMMIT)
    await h.until(TURN_DONE, count=2)
    await h.settle()
    first, second = [
        r for r in client.requests if r.messages[0].content != TRANSCRIPTION_PROMPT
    ]
    assert first.metadata["images"] == [
        "data:image/jpeg;base64," + base64.b64encode(jpeg[1]).decode(),
        "data:image/png;base64," + base64.b64encode(png).decode(),
    ]
    assert first.metadata["audios"][0].startswith("data:audio/wav;base64,")
    assert first.messages[-1].content == SEEING_RESPONSE_INSTRUCTION
    assert "images" not in second.metadata
    assert second.messages[-1].content == RESPONSE_INSTRUCTION
    transcriptions = [
        r for r in client.requests if r.messages[0].content == TRANSCRIPTION_PROMPT
    ]
    assert len(transcriptions) == 2
    assert all(r.metadata == {} for r in transcriptions)
    await h.execution.close()


@pytest.mark.asyncio
async def test_turn_done_waits_for_the_final_transcript() -> None:
    client = CompletedGenerationClient(has_audio=False)
    client.release_transcription.clear()
    h = Harness(client, output_modalities=("text",))
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await asyncio.wait_for(client.transcription_started.wait(), TEST_TIMEOUT_S)
    await h.until(TEXT_DELTA)
    await asyncio.sleep(0.05)
    assert TURN_DONE not in h.labels()
    client.release_transcription.set()
    await h.until(TURN_DONE)
    labels = h.labels()
    final = next(
        index
        for index, o in enumerate(h.outputs)
        if o.format == TRANSCRIPT_FORMAT and o.payload["is_final"]
    )
    assert final < labels.index(TURN_DONE)
    await h.execution.close()


def test_replies_match_the_legacy_realtime_session_and_transcription_reads_the_instruction_first() -> (
    None
):
    client = BlockingGenerationClient()
    legacy = RealtimeSession(
        object(), client=client, model_name="qwen-test", supports_audio_output=True
    )
    legacy.session_object = SessionObject(
        id="session",
        model="qwen-test",
        instructions="custom instructions",
        modalities=["text", "audio"],
        temperature=0.3,
        max_response_output_tokens=32,
    )
    legacy.conversation = [
        LegacyConversationItem(role="user", text="earlier user", item_id="user"),
        LegacyConversationItem(
            role="assistant", text="earlier assistant", item_id="assistant"
        ),
    ]
    execution = QwenSessionExecution(
        client, "qwen-test", DetachedSessionWork(), output_modalities=("text", "audio")
    )
    execution.configure(
        {
            "instructions": "custom instructions",
            "sampling": {"temperature": 0.3, "max_new_tokens_per_unit": 32},
        }
    )
    execution.conversation = [
        ConversationItem("user", "earlier user", 1),
        ConversationItem("assistant", "earlier assistant", 1),
    ]
    audio_payload = "data:audio/wav;base64,fixture"
    assert (
        execution.build_response_request(audio_payload, (), ("text", "audio")).to_dict()
        == legacy.build_response_request(audio_payload).to_dict()
    )

    transcription = execution.build_transcription_request(audio_payload)
    assert [m.to_dict() for m in transcription.messages] == [
        {"role": "system", "content": TRANSCRIPTION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSCRIPTION_INSTRUCTION},
                {"type": "audio_url", "audio_url": {"url": audio_payload}},
            ],
        },
    ]
    assert transcription.metadata == {}
    assert transcription.output_modalities == ["text"]
    assert transcription.sampling.temperature == 0.0
    assert transcription.sampling.max_new_tokens == 32


@pytest.mark.asyncio
async def test_speech_over_the_reply_interrupts_it_and_truncate_forgets_it() -> None:
    client = BlockingGenerationClient()
    h = Harness(client)
    await h.open()
    await h.audio()
    await h.control(SPEECH_STARTED, t_ms=0.0)
    await h.audio()
    await h.control(SPEECH_STOPPED, t_ms=h.t_ms)
    await asyncio.wait_for(client.generation_started.wait(), TEST_TIMEOUT_S)
    await h.audio()
    await h.control(SPEECH_STARTED, t_ms=h.t_ms)
    await h.until(TURN_INTERRUPTED)
    await asyncio.wait_for(client.abort_started.wait(), TEST_TIMEOUT_S)
    assert len(client.aborted) == 1
    assert h.turn_controls()[-1] == (TURN_INTERRUPTED, 1)
    assert h.text() == "Hello"
    await h.settle()
    assert [(entry.role, entry.text) for entry in h.execution.conversation] == [
        ("user", "user words"),
        ("assistant", "Hello"),
    ]
    await h.control(TRUNCATE, t_ms=h.t_ms)
    assert [(entry.role, entry.text) for entry in h.execution.conversation] == [
        ("user", "user words"),
    ]
    await h.interrupt()
    await h.control(SPEECH_STOPPED, t_ms=h.t_ms)
    await h.until(TURN_STARTED, count=2)
    assert h.turn_controls()[-2:] == [(TURN_QUEUED, 2), (TURN_STARTED, 2)]
    await h.execution.close()
    assert not h.execution.turn_tasks and not h.execution.response_tasks
    assert not h.execution.cleanup_tasks


@pytest.mark.asyncio
async def test_truncate_before_the_interrupted_turn_is_recorded_still_forgets_it() -> (
    None
):
    client = BlockingGenerationClient()
    client.release_transcription.clear()
    h = Harness(client)
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await asyncio.wait_for(client.generation_started.wait(), TEST_TIMEOUT_S)
    await h.interrupt()
    await h.until(TURN_INTERRUPTED)
    await h.control(TRUNCATE, t_ms=h.t_ms)
    client.release_transcription.set()
    await h.settle()
    assert [(entry.role, entry.text) for entry in h.execution.conversation] == [
        ("user", "user words"),
    ]
    await h.execution.close()


@pytest.mark.asyncio
async def test_the_reply_stays_interruptible_until_the_session_returns_its_turn_done() -> (
    None
):
    client = CompletedGenerationClient()
    h = Harness(client, returns_turn_done=False)
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await h.until(TURN_DONE)
    await asyncio.sleep(0.05)
    assert h.execution.reply is not None and h.execution.turn_tasks
    await h.control(SPEECH_STARTED, t_ms=h.t_ms)
    await h.until(TURN_INTERRUPTED)
    await h.settle()
    assert h.turn_controls()[-2:] == [(TURN_DONE, 1), (TURN_INTERRUPTED, 1)]
    assert h.execution.interrupted_turn == 1
    await h.audio()
    await h.control(SPEECH_STOPPED, t_ms=h.t_ms)
    await h.until(TURN_DONE, count=2)
    await h.execution.control(h.outputs[-1])
    await h.settle()
    assert h.execution.reply is None
    assert h.turn_controls()[-1] == (TURN_DONE, 2)
    await h.execution.close()


@pytest.mark.asyncio
async def test_generation_failure_fails_the_session() -> None:
    h = Harness(FailedGenerationClient())
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await asyncio.wait_for(
        asyncio.gather(*h.execution.turn_tasks, return_exceptions=True), TEST_TIMEOUT_S
    )
    assert isinstance(h.error, RuntimeError)
    assert str(h.error) == "backend generation failed"
    assert [entry.role for entry in h.execution.conversation] == []
    await h.execution.close()


@pytest.mark.asyncio
async def test_a_failed_turn_stops_its_transcription() -> None:
    client = FailedGenerationClient()
    client.release_transcription.clear()
    h = Harness(client)
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await asyncio.wait_for(client.transcription_started.wait(), TEST_TIMEOUT_S)
    await asyncio.wait_for(
        asyncio.gather(*h.execution.turn_tasks, return_exceptions=True), TEST_TIMEOUT_S
    )
    assert not [
        task for task in asyncio.all_tasks() if task.get_coro().__name__ == "transcribe"
    ]
    assert str(h.error) == "backend generation failed"
    await h.execution.close()


@pytest.mark.asyncio
async def test_empty_commit_and_eos_bracket_turns() -> None:
    client = CompletedGenerationClient()
    h = Harness(client)
    await h.open()
    await h.control(COMMIT)
    assert h.turn_controls() == [(TURN_QUEUED, 1), (TURN_EMPTY, 1)]
    assert h.outputs[0].payload == {"text": "", "is_final": True}
    await h.audio()
    await h.audio(eos=True)
    assert h.turn_controls()[-3:] == [
        (TURN_QUEUED, 2),
        (TURN_STARTED, 2),
        (TURN_DONE, 2),
    ]
    assert len(h.execution.conversation) == 2
    await h.execution.close()


@pytest.mark.asyncio
async def test_configure_changes_instructions_and_sampling_for_later_turns() -> None:
    client = CompletedGenerationClient()
    h = Harness(client)
    await h.open({"instructions": "First.", "sampling": {"temperature": 0.1}})
    await h.control(CONFIGURE, instructions="Second.", sampling={"temperature": 0.9})
    await h.audio()
    await h.control(COMMIT)
    await h.until(TURN_DONE)
    reply_request = client.reply_request()
    assert reply_request.messages[0].content == "Second."
    assert reply_request.sampling.temperature == 0.9
    await h.execution.close()


@pytest.mark.asyncio
async def test_cleanup_failure_fails_the_session() -> None:
    class CleanupClient(BlockingGenerationClient):
        async def abort(self, request_id: str) -> None:
            self.abort_started.set()
            raise RuntimeError("control plane unavailable")

    client = CleanupClient()
    h = Harness(client)
    await h.open()
    await h.audio()
    await h.control(COMMIT)
    await asyncio.wait_for(client.generation_started.wait(), TEST_TIMEOUT_S)
    await h.interrupt()
    await asyncio.wait_for(
        asyncio.gather(*h.execution.cleanup_tasks.values()), TEST_TIMEOUT_S
    )
    with pytest.raises(RuntimeError, match="control plane unavailable"):
        await h.audio()
    await h.execution.close()
    assert not h.execution.abort_tasks


@pytest.mark.asyncio
async def test_a_failed_turn_fails_its_session_but_not_the_stages(tmp_path) -> None:
    async with pipeline(tmp_path) as (coordinator, _, _):
        clients = [FailedGenerationClient(), CompletedGenerationClient()]
        executions: list[QwenSessionExecution] = []

        def factory(work: SessionWork) -> QwenSessionExecution:
            executions.append(
                QwenSessionExecution(
                    clients[len(executions)],
                    "qwen-test",
                    work,
                    output_modalities=("text",),
                )
            )
            return executions[-1]

        coordinator.session_execution_factories["qwen"] = factory
        unit = b"\x00\x10" * 3200

        async def open_with_turn():
            identity = await coordinator.open_route(
                OmniRequest(None),
                stages=["source", "sink"],
                limits=None,
                session_id=None,
                execution="qwen",
            )
            await coordinator.append_session(
                identity, TimedChunk(AUDIO, 0, 200, 0, unit, format=PCM16)
            )
            await coordinator.append_session(
                identity, control_chunk(COMMIT, 200, seq=1)
            )
            return identity, coordinator.session_outputs(identity)

        _, outputs = await open_with_turn()

        async def read_all() -> None:
            async for _ in outputs:
                pass

        with pytest.raises(RuntimeError, match="backend generation failed"):
            await asyncio.wait_for(read_all(), TEST_TIMEOUT_S)
        await wait_until(lambda: not coordinator.sessions)
        assert not coordinator.session_unavailable_stages
        identity, outputs = await open_with_turn()
        async for output in outputs:
            if (
                output.kind == "data"
                and output.format == SESSION_CONTROL
                and output.payload["type"] == TURN_DONE
            ):
                break
            else:
                pass
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)


@pytest.mark.asyncio
async def test_a_node_publishing_no_audio_gets_its_turn_done_back(tmp_path) -> None:
    async with pipeline(tmp_path) as (coordinator, _, _):
        coordinator.session_execution_factories["qwen"] = lambda work: (
            QwenSessionExecution(
                CompletedGenerationClient(has_audio=False),
                "qwen-test",
                work,
                output_modalities=("text",),
            )
        )
        coordinator.session_route = None
        coordinator.session_graph = SessionGraph(
            nodes={
                "model": SessionNode(
                    stages=("source", "sink"), inputs=("input",), execution="qwen"
                )
            },
            publish={"model": (TEXT_DELTA, SESSION_CONTROL, TRANSCRIPT_FORMAT)},
        )
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        unit = b"\x00\x10" * 3200

        async def turn_done(turn: int) -> None:
            async for output in outputs:
                if (
                    output.kind == "data"
                    and output.format == SESSION_CONTROL
                    and output.payload["type"] == TURN_DONE
                    and output.turn == turn
                ):
                    return
                else:
                    pass

        for turn in (1, 2):
            seq = 2 * (turn - 1)
            await coordinator.append_session(
                identity,
                TimedChunk(AUDIO, 200 * (turn - 1), 200, seq, unit, format=PCM16),
            )
            await coordinator.append_session(
                identity, control_chunk(COMMIT, 200 * turn, seq=seq + 1)
            )
            await asyncio.wait_for(turn_done(turn), TEST_TIMEOUT_S)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
