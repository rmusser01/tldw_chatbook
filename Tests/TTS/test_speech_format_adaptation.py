"""Adaptive response-format plumbing for Console reply speech.

TASK-32013: when the resolved TTS response format cannot be played on
this machine (no sink and no format-capable player binary), Console
speech requests are re-issued as WAV rather than synthesized into
guaranteed silence. These tests pin the plumbing: the service-level
override parameter and the `TTSEventHandler` wiring that computes it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.TTS import playback_capability as pc
from tldw_chatbook.TTS.effective_settings import TTSSelectionOverrides
from tldw_chatbook.TTS.request_admission import TTSRequestAdmissionCoordinator


def _bare_admission() -> TTSRequestAdmissionCoordinator:
    """A `TTSRequestAdmissionCoordinator` with no service wiring.

    `synthesize_default` only touches `self.synthesize_effective`, so a
    bare instance plus a captured fake is enough to pin how the override
    is folded into the explicit selection it builds.
    """
    return object.__new__(TTSRequestAdmissionCoordinator)


class TestSynthesizeDefaultFormatOverride:
    def test_format_override_becomes_explicit_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        admission = _bare_admission()
        captured: dict[str, object] = {}

        async def fake_effective(**kwargs):
            captured.update(kwargs)
            return ("response", "selection")

        monkeypatch.setattr(admission, "synthesize_effective", fake_effective)
        import asyncio

        asyncio.run(
            admission.synthesize_default(
                text="hello", response_format_override="wav"
            )
        )
        explicit = captured["explicit"]
        assert isinstance(explicit, TTSSelectionOverrides)
        assert explicit.response_format == "wav"

    def test_no_override_builds_no_explicit_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        admission = _bare_admission()
        captured: dict[str, object] = {}

        async def fake_effective(**kwargs):
            captured.update(kwargs)
            return ("response", "selection")

        monkeypatch.setattr(admission, "synthesize_effective", fake_effective)
        import asyncio

        asyncio.run(admission.synthesize_default(text="hello"))
        assert captured["explicit"] is None

    def test_voice_and_format_overrides_combine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        admission = _bare_admission()
        captured: dict[str, object] = {}

        async def fake_effective(**kwargs):
            captured.update(kwargs)
            return ("response", "selection")

        monkeypatch.setattr(admission, "synthesize_effective", fake_effective)
        import asyncio

        asyncio.run(
            admission.synthesize_default(
                text="hello",
                voice_override="shimmer",
                response_format_override="wav",
            )
        )
        explicit = captured["explicit"]
        assert isinstance(explicit, TTSSelectionOverrides)
        assert explicit.voice_mode == "exact"
        assert explicit.voice_id == "shimmer"
        assert explicit.response_format == "wav"


class _FakeByteStream:
    def __init__(self) -> None:
        self._chunks = [b"fake-audio-body"]

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeResponse:
    provider_id = "openai"
    model_id = "tts-1"
    byte_stream: _FakeByteStream

    def __init__(self, audio_format: str) -> None:
        self.audio_format = audio_format
        self.byte_stream = _FakeByteStream()

    async def aclose(self) -> None:
        pass


class _FakeTTSService:
    """Minimum `TTSEventHandler._generate_tts` surface, capturing overrides."""

    def __init__(self, preferences_format: str) -> None:
        self._preferences_format = preferences_format
        self.captured_format_override: str | None = "never-called"

    def preferences_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider_id="openai",
            speed=1.0,
            response_format=self._preferences_format,
        )

    def provider_descriptors(self):
        return []

    async def synthesize_default(
        self,
        *,
        text,
        voice_override=None,
        response_format_override=None,
        progress_sink=None,
        admission_authorizer=None,
    ):
        self.captured_format_override = response_format_override
        return _FakeResponse(response_format_override or self._preferences_format)


def _machine(monkeypatch: pytest.MonkeyPatch, *, sink: bool, players: dict):
    monkeypatch.setattr(pc, "sink_available", lambda: sink)
    monkeypatch.setattr(
        pc,
        "find_player_for_format",
        lambda fmt: next(
            (name for name, formats in players.items() if fmt in formats), None
        ),
    )


class TestGenerateTtsConsoleSpeechAdaptation:
    @pytest.mark.asyncio
    async def test_handsfree_utterance_adapts_unplayable_mp3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

        # Machine like stock Fedora with sounddevice installed: the sink
        # can play WAV but no player binary can decode the configured MP3.
        _machine(monkeypatch, sink=True, players={})
        service = _FakeTTSService(preferences_format="mp3")
        handler = TTSEventHandler()
        handler._tts_service = service
        monkeypatch.setattr(
            handler,
            "_play_utterance_legacy_artifact",
            lambda *args, **kwargs: _noop_async(),
        )
        outcomes: list[bool] = []
        await handler.speak_utterance("hello", on_finished=outcomes.append)
        assert service.captured_format_override == "wav"
        # The play path is no-op'd above, so the written artifact never
        # enters the play/cleanup cycle -- delete it through the handler's
        # own teardown instead of leaking it to interpreter-shutdown
        # deletion (where secure_delete_file loses builtins and logs).
        await handler.cleanup_tts_resources()

    @pytest.mark.asyncio
    async def test_handsfree_utterance_keeps_playable_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

        # Machine like macOS (afplay) or a Linux box with mpv: the
        # configured MP3 plays, so the request must not be rewritten.
        _machine(monkeypatch, sink=False, players={"mpv": {"mp3", "wav"}})
        service = _FakeTTSService(preferences_format="mp3")
        handler = TTSEventHandler()
        handler._tts_service = service
        monkeypatch.setattr(
            handler,
            "_play_utterance_legacy_artifact",
            lambda *args, **kwargs: _noop_async(),
        )
        outcomes: list[bool] = []
        await handler.speak_utterance("hello", on_finished=outcomes.append)
        assert service.captured_format_override is None
        await handler.cleanup_tts_resources()

    @pytest.mark.asyncio
    async def test_non_console_request_is_not_adapted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSEventHandler,
            TTSRequestEvent,
        )

        # An ad-hoc/explicit request (no hands-free on_finished, no Console
        # playback_lifecycle) must keep today's behavior bit-for-bit.
        _machine(monkeypatch, sink=True, players={})
        service = _FakeTTSService(preferences_format="mp3")
        handler = TTSEventHandler()
        handler._tts_service = service
        await handler.handle_tts_request(TTSRequestEvent(text="hello"))
        # The ad-hoc branch is fire-and-forget (`_admit_tts_generation`'s
        # bare else): yield the loop so the background generation task
        # reaches its synthesis call before the assertion reads the fake.
        for _ in range(20):
            await asyncio.sleep(0)
        assert service.captured_format_override is None
        await handler.cleanup_tts_resources()


def _noop_async():
    import asyncio

    fut = asyncio.get_event_loop().create_future()
    fut.set_result(None)
    return fut


class TestPlaybackFailureRemedy:
    @pytest.mark.asyncio
    async def test_unplayable_artifact_skips_play_and_toasts_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSEventHandler,
        )
        from tldw_chatbook.TTS import audio_player as ap_module

        # A machine where NOTHING can play audio: no sink, no player binary.
        # The mp3 request is not even adaptable (wav is unplayable too), so
        # synthesis produces an mp3 artifact nobody can ever play.
        _machine(monkeypatch, sink=False, players={})
        service = _FakeTTSService(preferences_format="mp3")
        handler = TTSEventHandler()
        handler._tts_service = service

        events: list[object] = []

        async def post(message):
            events.append(message)
            return True

        handler._post_tts_message = post

        def _no_player_spawn(*args, **kwargs):  # pragma: no cover
            raise AssertionError("play() must not run for an unplayable format")

        monkeypatch.setattr(ap_module, "get_audio_player", _no_player_spawn)

        outcomes: list[bool] = []
        await handler.speak_utterance("first sentence.", on_finished=outcomes.append)
        await handler.speak_utterance("second sentence.", on_finished=outcomes.append)
        await handler.cleanup_tts_resources()

        # Both utterances report failure (the loop must keep moving), and
        # exactly ONE user-visible remedy toast fires for the whole app run.
        assert outcomes == [False, False]
        remedy_errors = [
            e
            for e in events
            if type(e).__name__ == "TTSCompleteEvent" and getattr(e, "error", None)
        ]
        assert len(remedy_errors) == 1, [getattr(e, "error", None) for e in events]
        assert "player" in remedy_errors[0].error
