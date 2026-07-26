"""
Test cases for TTS improvements
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import ProgressSink, TTSProgress
from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.legacy_bridge import (
    LEGACY_ROUTES,
    LegacyBackendHost,
    LegacyTTSAdapter,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService

try:
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSEventHandler,
        TTSRequestEvent,
        TTSCompleteEvent,
        TTSPlaybackEvent,
        TTSProgressEvent,
        TTSExportEvent,
        play_audio_file,
        CostTracker,
    )
except ImportError:
    # Fallback imports if classes don't exist
    TTSEventHandler = None
    TTSRequestEvent = None
    TTSCompleteEvent = None
    TTSPlaybackEvent = None
    TTSProgressEvent = None
    TTSExportEvent = None
    play_audio_file = None
    CostTracker = None
try:
    from tldw_chatbook.TTS.audio_player import SimpleAudioPlayer, PlaybackState
except ImportError:
    SimpleAudioPlayer = None
    PlaybackState = None

if CostTracker is None:

    class CostTracker:
        pass

# cost_tracker module doesn't exist, create mock classes


class TTSProvider:
    pass


class TestTTSEventHandler:
    """Test TTS event handler improvements"""

    @pytest.fixture
    def handler(self):
        """Create a test handler"""

        class TestHandler(TTSEventHandler):
            def __init__(self):
                super().__init__()
                self.messages = []

            async def post_message(self, message):
                self.messages.append(message)

            def notify(self, message, severity="info"):
                pass

        return TestHandler()

    @pytest.mark.asyncio
    async def test_cooldown_cleanup(self, handler):
        """Test that cooldown dictionary is cleaned up"""
        # Add some old entries
        old_time = asyncio.get_event_loop().time() - 400  # More than 5 minutes ago
        handler._request_cooldown["old_message"] = old_time
        handler._request_cooldown["recent_message"] = asyncio.get_event_loop().time()

        # Trigger cleanup
        handler._cleanup_cooldown_dict(asyncio.get_event_loop().time())

        # Old entry should be removed
        assert "old_message" not in handler._request_cooldown
        assert "recent_message" in handler._request_cooldown

    @pytest.mark.asyncio
    async def test_cooldown_max_entries(self, handler):
        """Test that cooldown dictionary respects max entries"""
        # Fill up with max entries
        base_time = asyncio.get_event_loop().time()
        for i in range(handler.MAX_COOLDOWN_ENTRIES + 100):
            handler._request_cooldown[f"msg_{i}"] = base_time + i

        # Create a request that triggers cleanup
        event = TTSRequestEvent("Test text", "msg_new")
        with patch.object(handler, "_tts_service", None):
            await handler.handle_tts_request(event)

        # Should have removed oldest entries
        assert len(handler._request_cooldown) <= handler.MAX_COOLDOWN_ENTRIES
        # Oldest entries should be gone
        assert "msg_0" not in handler._request_cooldown

    @pytest.mark.asyncio
    async def test_initialize_tts_retrieves_only_the_bound_service(
        self,
        handler,
        monkeypatch,
    ):
        service = object()
        get_service = AsyncMock(return_value=service)
        get_setting = MagicMock(
            side_effect=AssertionError(
                "Console initialization must not read a second preference snapshot"
            )
        )
        monkeypatch.setattr(tts_events_module, "get_tts_service", get_service)
        monkeypatch.setattr(
            tts_events_module,
            "get_cli_setting",
            get_setting,
            raising=False,
        )

        await handler.initialize_tts()

        get_service.assert_awaited_once_with()
        get_setting.assert_not_called()
        assert handler._tts_service is service
        assert not hasattr(handler, "_tts_config")

    @pytest.mark.asyncio
    async def test_progress_events(self, handler):
        """Test that progress events are sent during generation"""
        chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4", b"chunk5"]

        async def mock_stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        class Response:
            provider_id = "openai"
            model_id = "tts-1"
            audio_format = "mp3"
            content_type = "audio/mpeg"
            metadata = {}

            def __init__(self):
                self.byte_stream = mock_stream()
                self.close_calls = 0

            async def aclose(self):
                self.close_calls += 1
                await self.byte_stream.aclose()

        response = Response()

        class Service:
            def __init__(self):
                self.calls = []

            def preferences_snapshot(self):
                return SimpleNamespace(provider_id="openai")

            async def synthesize_default(
                self,
                *,
                text,
                voice_override=None,
                progress_sink=None,
            ):
                self.calls.append((text, voice_override, progress_sink))
                if progress_sink is not None:
                    await progress_sink(
                        TTSProgress(status="Generating audio", fraction=0.5)
                    )
                return response

            async def generate_audio_stream(self, *_args, **_kwargs):
                raise AssertionError("Console must use synthesize_default")
                yield b""  # pragma: no cover

        service = Service()
        handler._tts_service = service

        generated_path = None
        try:
            # Generate TTS
            await handler._generate_tts("Test text", "test_msg", "alloy")
            generated_path = handler._audio_files["test_msg"]

            # Check for progress events
            progress_events = [
                m for m in handler.messages if isinstance(m, TTSProgressEvent)
            ]
            assert len(progress_events) >= 2  # At least initial and final
            assert progress_events[0].progress == 0.0
            assert progress_events[-1].progress == 1.0
            assert progress_events[-1].status == "Audio generation complete"
            assert service.calls == [("Test text", "alloy", service.calls[0][2])]
            assert response.close_calls == 1
        finally:
            await handler.cleanup_tts_resources()

        assert handler._audio_files == {}
        assert generated_path is not None
        assert not generated_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "provider_id",
            "configured_model",
            "configured_format",
            "expected_model",
            "expected_format",
            "expected_internal_id",
        ),
        (
            (
                "openai",
                "tts-1-hd",
                "opus",
                "tts-1-hd",
                "opus",
                "openai_official_tts-1-hd",
            ),
            (
                "elevenlabs",
                "eleven_multilingual_v2",
                "wav",
                "elevenlabs",
                "mp3",
                "elevenlabs_elevenlabs",
            ),
            (
                "kokoro",
                "kokoro",
                "mp3",
                "kokoro",
                "wav",
                "local_kokoro_default_onnx",
            ),
            (
                "chatterbox",
                "chatterbox",
                "mp3",
                "chatterbox",
                "wav",
                "local_chatterbox_default",
            ),
            (
                "higgs",
                "higgs-audio-v2",
                "wav",
                "higgs-audio-v2",
                "wav",
                "local_higgs_v2",
            ),
            (
                "alltalk",
                "alltalk",
                "mp3",
                "alltalk",
                "wav",
                "alltalk_default",
            ),
        ),
    )
    async def test_console_retained_provider_defaults_use_legacy_adapter(
        self,
        handler,
        monkeypatch,
        provider_id,
        configured_model,
        configured_format,
        expected_model,
        expected_format,
        expected_internal_id,
    ):
        captured: list[tuple[str, OpenAISpeechRequest]] = []

        def capture_generate(
            _host: LegacyBackendHost,
            internal_model_id: str,
            request: OpenAISpeechRequest,
            progress_sink: ProgressSink | None,
        ) -> AsyncIterator[bytes]:
            async def audio() -> AsyncIterator[bytes]:
                if progress_sink is not None:
                    await progress_sink(
                        TTSProgress(status="Generating audio", fraction=0.5)
                    )
                yield b"legacy-"
                yield provider_id.encode()

            captured.append((internal_model_id, request))
            return audio()

        monkeypatch.setattr(LegacyBackendHost, "generate", capture_generate)
        registry = TTSAdapterRegistry(
            specs=legacy_provider_specs(
                {},
                manager_factory=lambda _provider, _config: pytest.fail(
                    "Console request must stop at LegacyTTSAdapter"
                ),
            ),
            aliases={},
        )
        service = TTSService(
            registry,
            preferences_snapshot=TTSPreferencesSnapshot(
                provider_id=provider_id,
                model_mode="exact",
                model_id=configured_model,
                voice_mode="exact",
                voice_id="Voice/Case",
                response_format=configured_format,
                speed=1.25,
            ),
        )
        handler._tts_service = service
        artifact = None
        try:
            await handler._generate_tts(
                "Character response",
                f"legacy-{provider_id}",
                None,
            )
            completion = next(
                message
                for message in handler.messages
                if isinstance(message, TTSCompleteEvent)
            )
            artifact = completion.audio_file
            active = registry._slots[provider_id].active

            assert active is not None
            assert isinstance(active.adapter, LegacyTTSAdapter)
            assert LEGACY_ROUTES[expected_internal_id] == provider_id
            assert captured == [
                (
                    expected_internal_id,
                    OpenAISpeechRequest(
                        model=expected_model,
                        input="Character response",
                        voice="voice/case",
                        response_format=expected_format,
                        speed=1.25,
                    ),
                )
            ]
            assert artifact is not None
            assert artifact.suffix == f".{expected_format}"
            assert artifact.read_bytes() == b"legacy-" + provider_id.encode()
            progress_events = [
                message
                for message in handler.messages
                if isinstance(message, TTSProgressEvent)
            ]
            assert progress_events[0].progress == 0.0
            assert progress_events[-1].progress == 1.0
        finally:
            await handler.cleanup_tts_resources()
            await service.close()
            await service.wait_closed()

        assert artifact is not None
        assert not artifact.exists()

    @pytest.mark.asyncio
    async def test_export_functionality(self, handler, tmp_path):
        """Test audio export with custom naming"""
        # Create a mock audio file
        test_audio = tmp_path / "test_audio.mp3"
        test_audio.write_bytes(b"fake audio data")

        # Add to handler's audio files
        handler._audio_files["test_msg"] = test_audio

        # Export to custom location
        export_path = tmp_path / "exports" / "my_audio.mp3"
        event = TTSExportEvent("test_msg", export_path, include_metadata=True)

        await handler.handle_tts_export(event)

        # Check file was exported
        assert export_path.exists()
        assert export_path.read_bytes() == b"fake audio data"

        # Check metadata was created
        metadata_path = export_path.with_suffix(".mp3.json")
        assert metadata_path.exists()

    @pytest.mark.asyncio
    async def test_audio_cleanup_keeps_ownership_until_secure_delete_succeeds(
        self,
        handler,
        tmp_path,
        monkeypatch,
    ):
        test_audio = tmp_path / "retry-cleanup.wav"
        test_audio.write_bytes(b"audio")
        handler._audio_files["msg-retry"] = test_audio
        delete_results = iter((False, True))

        def delete(path):
            assert Path(path) == test_audio
            result = next(delete_results)
            if result:
                test_audio.unlink()
            return result

        monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)

        await handler._cleanup_audio_file("msg-retry")
        assert handler._audio_files == {"msg-retry": test_audio}
        assert test_audio.exists()

        await handler._cleanup_audio_file("msg-retry")
        assert handler._audio_files == {}
        assert not test_audio.exists()

    # --- task-559 unit 2: stop must actually interrupt playback ----------
    #
    # Previously `handle_tts_playback`'s "stop" branch only deleted the
    # cached audio file (`_cleanup_audio_file`) -- it never touched the
    # actual system audio player, so a "Stop" click did not silence audio
    # already playing (afplay/mpv/etc. keep streaming a deleted-but-open
    # file on Unix). Stop now also asks the shared `SimpleAudioPlayer`
    # singleton to stop, but ONLY when the message being stopped is the one
    # currently loaded -- the singleton holds a single global "now playing"
    # slot, and an unrelated message's cached-but-never-played file must not
    # be able to silence a different, actively-playing message (a real
    # scenario for legacy chat, where audio is not auto-played and multiple
    # messages can sit in "ready" state simultaneously).
    #
    # fix round 1: the FIRST fix (comparing against `self._audio_files`,
    # the same dict `_cleanup_audio_file` deletes from) had its own bug --
    # the "play" branch unconditionally schedules `_cleanup_audio_file(...,
    # delay=5.0)` the moment playback STARTS, not when it finishes. Any
    # clip that takes longer than 5s of wall-clock time between play and a
    # user's stop click has its `_audio_files` entry (and file) already
    # deleted, so the stop-guard found nothing and silently skipped calling
    # `player.stop()` -- for the COMMON case (Console auto-plays every
    # spoken message; anything over ~15 words exceeds 5s). These tests
    # exercise the REAL play -> (cleanup) -> stop lifecycle through
    # `handle_tts_playback` itself (not hand-seeded dicts) so this class of
    # bug can't hide behind an unrealistic setup again. The fix tracks
    # "what's currently loaded" in a separate `_last_played_audio_files`
    # map that is NOT subject to the 5s disk cleanup.

    @pytest.mark.asyncio
    async def test_stop_action_stops_playback_when_message_is_current(
        self, handler, tmp_path, monkeypatch
    ):
        test_audio = tmp_path / "clip.mp3"
        test_audio.write_bytes(b"fake audio data")
        handler._audio_files["msg-1"] = test_audio

        fake_player = MagicMock()
        fake_player.play.return_value = True
        fake_player.get_current_file.return_value = test_audio
        monkeypatch.setattr(
            "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
        )

        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="play", message_id="msg-1")
        )
        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="stop", message_id="msg-1")
        )

        fake_player.stop.assert_called_once()
        assert "msg-1" not in handler._audio_files

    @pytest.mark.asyncio
    async def test_stop_action_stops_playback_after_5s_cache_cleanup_already_ran(
        self, handler, tmp_path, monkeypatch
    ):
        """Reviewer repro (fix round 1). The play branch schedules the 5s
        cache cleanup as soon as playback STARTS -- simulate that cleanup
        having already run (bypassing only the `asyncio.sleep`, by calling
        the real `_cleanup_audio_file` with `delay=0`) before the user
        clicks stop. The player (per the mock) is still loaded with the
        same clip -- afplay/mpv keep streaming a deleted-but-open file
        descriptor on Unix -- so stop must still reach `player.stop()`."""
        test_audio = tmp_path / "clip.mp3"
        test_audio.write_bytes(b"fake audio data")
        handler._audio_files["msg-1"] = test_audio

        fake_player = MagicMock()
        fake_player.play.return_value = True
        fake_player.get_current_file.return_value = test_audio
        monkeypatch.setattr(
            "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
        )

        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="play", message_id="msg-1")
        )

        # The real cleanup code, run directly instead of waiting out the
        # scheduled asyncio.create_task(..., delay=5.0) -- delay=0 bypasses
        # only the sleep, not the logic.
        await handler._cleanup_audio_file("msg-1", delay=0)
        assert "msg-1" not in handler._audio_files  # cache entry really gone

        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="stop", message_id="msg-1")
        )

        fake_player.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_action_does_not_stop_unrelated_playing_message(
        self, handler, tmp_path, monkeypatch
    ):
        file_a = tmp_path / "a.mp3"
        file_a.write_bytes(b"a")
        file_b = tmp_path / "b.mp3"
        file_b.write_bytes(b"b")
        handler._audio_files["msg-a"] = file_a
        handler._audio_files["msg-b"] = file_b

        fake_player = MagicMock()
        fake_player.play.return_value = True
        fake_player.get_current_file.return_value = file_b  # b is playing
        monkeypatch.setattr(
            "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
        )

        # Only B was ever actually played -- A's file is cached but never
        # loaded into the player.
        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="play", message_id="msg-b")
        )

        await handler.handle_tts_playback(
            TTSPlaybackEvent(action="stop", message_id="msg-a")
        )

        fake_player.stop.assert_not_called()
        assert "msg-a" not in handler._audio_files  # cached file still cleared
        assert "msg-b" in handler._audio_files  # untouched, still playing

    @pytest.mark.asyncio
    async def test_stop_action_safe_when_nothing_cached(self, handler, monkeypatch):
        """Genuinely idle: no play ever happened for this id -- stop must
        be a silent no-op, not an error."""
        fake_player = MagicMock()
        monkeypatch.setattr(
            "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
        )

        event = TTSPlaybackEvent(action="stop", message_id="nonexistent")
        await handler.handle_tts_playback(event)  # must not raise

        fake_player.stop.assert_not_called()

    # --- fix round 2 (Qodo PR #867): single-slot tracker, no growth -----
    #
    # Round 1's `_last_played_audio_files` was a dict keyed by message id,
    # written on every "play" and only ever popped by a matching "stop" or
    # cleared at shutdown -- so an auto-played message the user never
    # explicitly stops (the common Console case: speak, listen, move on)
    # left a permanent entry. `SimpleAudioPlayer` itself is a single-slot
    # global singleton (one clip "current" system-wide at a time; every
    # `play()` stops whatever was previously loaded first), so tracking
    # more than one pending entry was never meaningful. Replaced with a
    # single `(message_id, path)` slot, overwritten on every play.

    @pytest.mark.asyncio
    async def test_play_path_tracks_only_a_single_slot_no_growth(
        self, handler, tmp_path, monkeypatch
    ):
        """Playing N different messages in a row (none of them ever
        explicitly stopped, mirroring Console's fire-and-auto-play flow)
        must never accumulate more than one tracked "last played" entry."""
        fake_player = MagicMock()
        fake_player.play.return_value = True
        monkeypatch.setattr(
            "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
        )

        last_audio = None
        for index in range(5):
            audio = tmp_path / f"clip-{index}.mp3"
            audio.write_bytes(b"x")
            handler._audio_files[f"msg-{index}"] = audio
            fake_player.get_current_file.return_value = audio
            await handler.handle_tts_playback(
                TTSPlaybackEvent(action="play", message_id=f"msg-{index}")
            )
            last_audio = audio

        # No per-message dict at all -- a single slot holding at most one
        # (message_id, path) pair, always the most recently played one.
        assert not hasattr(handler, "_last_played_audio_files")
        assert handler._last_played == ("msg-4", last_audio)


class TestAudioPlayer:
    """Test audio player improvements"""

    @pytest.fixture
    def player(self):
        """Create test player"""
        return SimpleAudioPlayer()

    def test_play_stop(self, player, tmp_path):
        """Test play and stop functionality"""
        # Create test audio file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal WAV header

        # Test play
        if player._player_cmd:  # Only test if player is available
            assert player.play(test_file)
            assert player.get_state() == PlaybackState.PLAYING

            # Test stop
            assert player.stop()
            assert player.get_state() == PlaybackState.IDLE
        else:
            # No player available on this system
            assert not player.play(test_file)

    def test_state_tracking(self, player):
        """Test state tracking"""
        assert player.get_state() == PlaybackState.IDLE
        assert not player.is_playing()

    def test_get_current_file_tracks_loaded_clip(self, player, tmp_path):
        """task-559 unit 2: exposes which file is currently loaded so a
        caller can decide whether a stop request actually applies to it."""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal WAV header

        assert player.get_current_file() is None

        if player._player_cmd:  # Only test if a player is available
            assert player.play(test_file)
            assert player.get_current_file() == test_file

            assert player.stop()
            assert player.get_current_file() is None


class TestCostTracker:
    """Test cost tracking functionality"""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create test tracker with temporary database"""
        db_path = tmp_path / "test_usage.db"
        return CostTracker(db_path)

    def test_cost_estimation(self, tracker):
        """Test cost estimation for different providers"""
        # OpenAI standard model
        cost = tracker.estimate_cost("openai", "tts-1", 1000)
        assert cost == 0.015  # $0.015 per 1K chars

        # OpenAI HD model
        cost = tracker.estimate_cost("openai", "tts-1-hd", 1000)
        assert cost == 0.030  # $0.030 per 1K chars

        # Local model (free)
        cost = tracker.estimate_cost("local", "kokoro", 10000)
        assert cost == 0.0

    def test_usage_tracking(self, tracker):
        """Test usage tracking and statistics"""
        # Track some usage
        record1 = tracker.track_usage(
            provider="openai",
            model="tts-1",
            text="Hello world",
            voice="alloy",
            format="mp3",
        )

        record2 = tracker.track_usage(
            provider="local",
            model="kokoro",
            text="This is a longer text for testing purposes",
            voice="af",
            format="wav",
        )

        # Check records
        assert record1.characters == 11
        assert record1.estimated_cost > 0
        assert record2.characters == 42
        assert record2.estimated_cost == 0.0  # Local is free

        # Check monthly usage
        monthly_chars = tracker.get_monthly_usage()
        assert monthly_chars == 53  # 11 + 42

        # Check monthly cost
        monthly_cost = tracker.get_monthly_cost()
        assert monthly_cost == record1.estimated_cost

    def test_free_tier_calculation(self, tracker):
        """Test free tier calculation"""
        # Update Google costs with free tier
        tracker.update_cost_info(
            provider="google",
            cost_per_1k_chars=0.016,
            free_tier_chars=1000000,  # 1M free chars
        )

        # First request should be free (under free tier)
        cost = tracker.estimate_cost("google", "wavenet", 50000)
        assert cost == 0.0

        # Track the usage
        tracker.track_usage(
            provider="google",
            model="wavenet",
            text="x" * 50000,
            voice="en-US-Wavenet-A",
            format="mp3",
        )

        # Next request partially in free tier
        cost = tracker.estimate_cost("google", "wavenet", 1000000)
        expected = (50000 / 1000.0) * 0.016  # Only 50K billable
        assert abs(cost - expected) < 0.001


def test_play_audio_file_security():
    """Test that play_audio_file is secure"""
    # Test path validation
    test_path = Path("/tmp/test.mp3")

    # Should handle non-existent files gracefully
    play_audio_file(test_path)  # Should not raise

    # Should reject non-audio extensions
    bad_path = Path("/tmp/test.exe")
    play_audio_file(bad_path)  # Should not raise, just log error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
