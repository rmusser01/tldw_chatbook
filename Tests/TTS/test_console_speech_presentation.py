"""Exact disposable presentation ownership through the existing speech entries."""

from unittest.mock import AsyncMock

import pytest

from Tests.TTS.test_console_speak_autoplay import _SpeechRequestControllerStub
from tldw_chatbook.Audio.speech_presentation import (
    SpeechPresentation,
    SpeechPresentationOwner,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSEventHandler,
    TTSPlaybackLifecycle,
)


def presentation(*, message_id=None, utterance_token=None):
    return SpeechPresentation(
        SpeechPresentationOwner(
            "issued-request", object(), "session", 4, 7, message_id, utterance_token
        )
    )


@pytest.mark.asyncio
async def test_public_handsfree_entry_forwards_observation_and_finishes_once(
    monkeypatch,
):
    handler = TTSEventHandler()
    speech = presentation(message_id="reply", utterance_token=8)
    kwargs_seen, finished = {}, []

    async def generate(_text, _message_id, **kwargs):
        kwargs_seen.update(kwargs)
        kwargs["presentation"].set_transport("playing")
        kwargs["on_finished"](True)
        kwargs["on_finished"](False)

    monkeypatch.setattr(handler, "_prepare_tts_text", AsyncMock(return_value="Hello."))
    monkeypatch.setattr(handler, "_generate_tts", generate)
    await handler.speak_utterance(
        "Hello.", on_finished=finished.append, presentation=speech
    )
    observed = speech.read()
    assert finished == [True]
    assert observed.owner.utterance_token == 8
    assert observed.owner.character_id == 7
    assert kwargs_seen["voice"] is None
    assert kwargs_seen["resolution"] is None
    assert kwargs_seen["presentation"] is speech
    assert observed.state == "stopped"


def test_stale_lifecycle_terminal_settles_only_its_own_channel_despite_listener_failure():
    old, new = presentation(), presentation()
    current = [True]

    def raising_callback(_state):
        raise RuntimeError("listener failure")

    lifecycle = TTSPlaybackLifecycle(
        message_id="old",
        request_id=1,
        validator=lambda: current[0],
        callback=raising_callback,
        presentation=old,
    )
    assert lifecycle.report("playing")
    current[0] = False
    assert lifecycle.report_terminal("stopped")
    assert old.read().state == "stopped"
    assert new.read() is None
    assert not lifecycle.report_terminal("failed")


@pytest.mark.asyncio
async def test_guarded_preview_forwards_observation_without_a_conversation_message(
    monkeypatch,
):
    handler = TTSEventHandler()
    speech = presentation()
    captured = {}
    monkeypatch.setattr(
        handler, "_resolve_speech_request_identity", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(handler, "_prepare_tts_text", AsyncMock(return_value="sample"))

    async def admit(**kwargs):
        captured.update(kwargs)
        owner = kwargs["playback_lifecycle"]
        owner.report("playing")
        owner.report_terminal("stopped")

    monkeypatch.setattr(handler, "_admit_tts_generation", admit)
    monkeypatch.setattr(handler, "handle_tts_playback", AsyncMock())
    assert await handler.speak_guarded_utterance(
        "sample",
        assistant_kind="character",
        character_ref=None,
        expected_destination_fingerprint="sha256:" + "a" * 64,
        validator=lambda: True,
        presentation=speech,
    )
    assert captured["presentation"] is speech
    assert captured["playback_lifecycle"].presentation is speech
    assert speech.owner.message_id is None
    assert speech.read().state == "stopped"
    stop = handler.handle_tts_playback.await_args.args[0]
    assert stop.playback_lifecycle is captured["playback_lifecycle"]


@pytest.mark.asyncio
async def test_rate_limit_wrapper_preserves_optional_observation(monkeypatch):
    handler = TTSEventHandler()
    speech = presentation()
    generate = AsyncMock()
    monkeypatch.setattr(handler, "_generate_tts", generate)
    await handler._generate_tts_with_rate_limit(
        "hello", "message", None, presentation=speech
    )
    assert generate.await_args.kwargs["presentation"] is speech


@pytest.mark.asyncio
@pytest.mark.parametrize("character_id", [None, 7])
async def test_console_issues_exact_channel_and_hides_it_after_profile_or_session_change(
    character_id,
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character" if character_id else "generic",
        assistant_id=str(character_id) if character_id else None,
        assistant_authority_id="authority" if character_id else None,
        character_id=character_id,
    )
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Old reply."
    )
    controller = _SpeechRequestControllerStub(store, True)
    assert await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    event = controller.app_instance.post_message.call_args.args[0]
    speech = ConsoleMessageController.current_speech_presentation(controller)
    assert speech is event.presentation
    assert speech.owner.session_id == session.id
    assert speech.owner.message_id == message.id
    assert speech.owner.character_id == character_id
    assert speech.owner.scope_token is controller.app_instance.chachanotes_db
    event.playback_lifecycle.report("playing")
    controller.app_instance.chachanotes_db = object()
    assert ConsoleMessageController.current_speech_presentation(controller) is None
    assert event.playback_lifecycle.report_terminal("stopped")
    assert speech.read().state == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_tail", [False, True])
async def test_public_utterance_retains_completion_until_native_output_ack(
    monkeypatch, cancel_tail
):
    import asyncio
    import struct

    from Tests.Audio.test_streaming_sink_presentation import DelayedStream
    from tldw_chatbook.Audio.streaming_sink import StreamingPcmSink
    from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as module
    from tldw_chatbook.TTS.pcm_stream import SinkPlan

    handler = TTSEventHandler()
    speech = presentation(utterance_token=3)
    opened, fed = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    holder, finished = {}, []

    def backend(**kwargs):
        stream = DelayedStream(**kwargs)
        holder["stream"] = stream
        loop.call_soon_threadsafe(opened.set)
        return stream

    def make_sink(**kwargs):
        sink = StreamingPcmSink(stream_factory=backend, **kwargs)
        holder["sink"] = sink
        return sink

    async def source():
        yield struct.pack("<480h", *([16384] * 480))
        fed.set()

    async def generate(_text, message_id, **kwargs):
        await handler._stream_response_via_sink(
            SinkPlan(24000, 1, 0, None),
            source(),
            message_id=message_id,
            on_finished=kwargs["on_finished"],
            presentation=kwargs["presentation"],
            playback_lifecycle=kwargs["playback_lifecycle"],
        )

    monkeypatch.setattr(module, "StreamingPcmSink", make_sink)
    monkeypatch.setattr(handler, "_generate_tts", generate)
    monkeypatch.setattr(
        handler, "_prepare_tts_text", AsyncMock(return_value="sentence A")
    )
    monkeypatch.setattr(handler, "_post_tts_message", AsyncMock(return_value=True))
    task = asyncio.create_task(
        handler.speak_utterance(
            "sentence A",
            on_finished=finished.append,
            presentation=speech,
            validator=lambda: True,
        )
    )
    try:
        await asyncio.wait_for(opened.wait(), 2)
        await asyncio.wait_for(fed.wait(), 2)
        sink, stream = holder["sink"], holder["stream"]
        stream.tick()
        await asyncio.to_thread(sink.settle)
        await asyncio.sleep(0)
        assert sink.state == "draining" and not task.done()
        assert finished == []
        assert handler._active_stream_playback_owner is not None
        if cancel_tail:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished == [False]
            stream.finish_output()
            await asyncio.sleep(0)
            assert finished == [False]
        else:
            stream.finish_output()
            await asyncio.wait_for(task, 2)
            assert finished == [True]
        assert speech.read().state == "stopped"
        assert handler._active_stream_playback_owner is None
    finally:
        if "sink" in holder:
            holder["sink"].stop()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_legacy_transport_projects_real_pause_without_changing_admission_states(
    tmp_path,
):
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        _play_legacy_clip_and_await_completion,
    )
    from tldw_chatbook.TTS.audio_player import PlaybackState

    speech = presentation()
    path = tmp_path / "sample.wav"
    states = iter(
        [
            PlaybackState.PLAYING,
            PlaybackState.PAUSED,
            PlaybackState.PLAYING,
            PlaybackState.FINISHED,
        ]
    )
    observed = []

    class Player:
        def play(self, _path):
            return True

        def get_current_file(self):
            return path

        def get_state(self):
            return next(states)

    def project(state):
        speech.set_transport(state)
        observed.append(speech.read().state)

    assert _play_legacy_clip_and_await_completion(
        Player(),
        path,
        timeout_seconds=2,
        poll_interval_seconds=0,
        on_transport=project,
    )
    assert observed == ["playing", "paused", "playing"]
    assert not speech.read().timing_valid


def test_event_cannot_pair_a_lifecycle_with_another_requests_observation():
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSPlaybackEvent

    first, second = presentation(), presentation()
    lifecycle = TTSPlaybackLifecycle(
        message_id="request",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
        presentation=first,
    )
    with pytest.raises(ValueError, match="presentation"):
        TTSPlaybackEvent(
            "stop", "request", playback_lifecycle=lifecycle, presentation=second
        )


@pytest.mark.asyncio
async def test_cancelled_guarded_preview_settles_its_channel_without_stopping_new_voice(
    monkeypatch,
):
    import asyncio

    from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as module

    handler = TTSEventHandler()
    old, new = presentation(), presentation()
    admitted = asyncio.Event()
    monkeypatch.setattr(
        handler, "_resolve_speech_request_identity", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(handler, "_prepare_tts_text", AsyncMock(return_value="sample"))

    async def admit(**_kwargs):
        admitted.set()

    monkeypatch.setattr(handler, "_admit_tts_generation", admit)
    stops = []
    monkeypatch.setattr(module, "stop_live_sink", lambda: stops.append(True))
    task = asyncio.create_task(
        handler.speak_guarded_utterance(
            "sample",
            assistant_kind="character",
            character_ref=None,
            expected_destination_fingerprint="sha256:" + "a" * 64,
            validator=lambda: True,
            presentation=old,
        )
    )
    await asyncio.wait_for(admitted.wait(), 2)
    replacement = TTSPlaybackLifecycle(
        message_id="newer",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
        presentation=new,
    )
    replacement.report("playing")
    handler._active_stream_playback_owner = replacement
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stops == []
    assert handler._active_stream_playback_owner is replacement
    assert new.read().state == "playing"
    # No old handler owner remains: existing exact Stop reports failure rather
    # than claiming it stopped the replacement. Its own channel still settles.
    assert old.read().state == "failed"
