"""Regression coverage for Console speak (TTS) auto-play routing.

Context: the TTS completion handler (`TldwCli.handle_tts_complete_event` in
`tldw_chatbook/app.py`) deliberately does not auto-play for legacy
`ChatMessage`/`ChatMessageEnhanced` widgets -- it sets a "click play to
listen" state on the widget instead, because those widgets have their own
play control. Console has no such widget (no `ChatMessage`/
`ChatMessageEnhanced` is ever mounted for its messages), so a Console
`speak` action would previously synthesize audio and then go silent: there
is no play control for the user to click.

The fix: when a successful completion's message id is not claimed by any
mounted legacy widget, the handler now posts `TTSPlaybackEvent(action="play",
...)` itself so the existing `@on(TTSPlaybackEvent)` -> `handler.
handle_tts_playback` pipeline (already used by the legacy play button) plays
the audio immediately. The legacy widget-found branch was removed together
with the widget family (task-34890); this path is the only one.

Test level: these tests call `TldwCli.handle_tts_complete_event` directly
against a minimal duck-typed stand-in for `self` (exposing only `query`,
`post_message`, `notify`, `loguru_logger`), rather than booting the full
`TldwCli` app. `textual.on()` returns the decorated method "unaltered" (see
its own docstring), so calling the unbound method this way runs the exact
same production code that Textual's message dispatch would call. What this
does *not* exercise is Textual's real message-queue dispatch of the posted
`TTSPlaybackEvent` to the `@on(TTSPlaybackEvent)` handler, or the handler's
hand-off to the worker. Tests below exercise the real handler with a private
output coordinator and replacement native device boundaries.
"""

from __future__ import annotations

import asyncio
import io
import re
import threading
import wave
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App

from Tests.TTS.adapter_fakes import FakeAdapter
from Tests.TTS.owned_output_fakes import (  # noqa: F401
    ConsoleFilePlayer,
    drain_handler_playback,
    forbid_native_audio,
    wait_for_condition,
)
from Tests.TTS_Events.test_spoken_feedback_streaming import _RecordingSink
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSPlaybackEvent,
    TTSPlaybackLifecycle,
    TTSProgressEvent,
)
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.TTS import audio_output_runs
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.adapter_bootstrap import build_default_tts_service
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import (
    TTSAudioResponse,
    TTSConfigurationRevisionError,
    TTSProviderDescriptor,
    TTSProviderSpec,
)
from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    ConsoleAutoSpeakCoordinator,
)


class _FakeApp:
    """Minimal stand-in exposing only what handle_tts_complete_event touches."""

    def __init__(self, widgets=()):
        self._widgets = list(widgets)
        self.audio_output = AudioOutputOwnership()
        self.loguru_logger = MagicMock()
        self.notify = MagicMock()
        self.posted: list = []
        self.push_screen_wait = AsyncMock(return_value=False)
        self.worker_tasks: list[asyncio.Task] = []
        # Review round 2: `_offer_tts_global_override` reads this (when set)
        # to pick domain-accurate confirmation-dialog copy. `None` by
        # default -- exercises the same "no handler bound" fallback a real
        # app would hit if a token somehow outlived its issuing handler.
        self._tts_handler = None

    def query(self, widget_type):
        return [w for w in self._widgets if isinstance(w, widget_type)]

    def post_message(self, message) -> bool:
        self.posted.append(message)
        return True

    def run_worker(self, awaitable, **_kwargs):
        task = asyncio.create_task(awaitable)
        self.worker_tasks.append(task)
        return task

    async def _offer_tts_global_override(self, token: str) -> None:
        await TldwCli._offer_tts_global_override(self, token)


def _new_handler(app=None):
    handler = TTSEventHandler()
    handler.app = _FakeApp() if app is None else app
    return handler


class ProgressHost(App[None]):
    """Real Textual app exposing the DOMQuery used by the progress handler."""

    def __init__(self) -> None:
        super().__init__()
        self.loguru_logger = MagicMock()


@pytest.mark.asyncio
async def test_tts_progress_handler_supports_real_textual_dom_query() -> None:
    """Verify progress handling with Textual's concrete DOMQuery type."""
    host = ProgressHost()

    async with host.run_test():
        await TldwCli.handle_tts_progress_event(
            host,
            TTSProgressEvent(
                message_id="console-msg-1",
                progress=0.5,
                status="Generating audio",
            ),
        )

    host.loguru_logger.error.assert_not_called()


@pytest.mark.asyncio
async def test_app_routes_snapshot_event_without_logging_private_snapshot_data():
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="PRIVATE_AUTHORITY",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="PRIVATE_RESPONSE_TEXT",
    )
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
    )
    handler = MagicMock()
    handler.handle_tts_request = AsyncMock()
    fake_app = _FakeApp()
    fake_app._ensure_tts_handler = AsyncMock(return_value=handler)

    await TldwCli.handle_tts_message_speech_request_event(fake_app, event)

    handler.handle_tts_request.assert_awaited_once_with(event)
    rendered_logs = repr(fake_app.loguru_logger.method_calls)
    assert "PRIVATE_RESPONSE_TEXT" not in rendered_logs
    assert "PRIVATE_AUTHORITY" not in rendered_logs


@pytest.mark.asyncio
async def test_real_handler_supersedes_spawned_generation_and_discards_a(
    tmp_path, monkeypatch
) -> None:
    current_request = 1
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    never_release = asyncio.Event()
    posted: list[object] = []
    paths = [tmp_path / "a.mp3", tmp_path / "b.mp3"]

    def response(started: asyncio.Event) -> TTSAudioResponse:
        async def chunks():
            yield b"a" * (64 * 1024)
            started.set()
            await never_release.wait()

        return TTSAudioResponse(
            provider_id="openai",
            model_id="tts-model",
            audio_format="mp3",
            content_type="audio/mpeg",
            byte_stream=chunks(),
        )

    service = MagicMock()
    service.preferences_snapshot.return_value = SimpleNamespace(
        provider_id="openai",
        speed=1.0,
    )
    service.synthesize_default = AsyncMock(
        side_effect=[response(first_started), response(second_started)]
    )
    handler = _new_handler()
    handler._tts_service = service
    handler._create_tts_artifact = MagicMock(side_effect=paths)

    async def capture(message: object) -> bool:
        posted.append(message)
        return True

    handler._post_tts_message = capture
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.sink_available",
        lambda: False,
    )
    first_outcomes: list[bool] = []
    first_lifecycle = TTSPlaybackLifecycle(
        message_id="message-a",
        request_id=1,
        validator=lambda: current_request == 1,
        callback=lambda _state: None,
    )
    second_lifecycle = TTSPlaybackLifecycle(
        message_id="message-b",
        request_id=2,
        validator=lambda: current_request == 2,
        callback=lambda _state: None,
    )

    try:
        await handler._admit_tts_generation(
            text="First.",
            message_id="message-a",
            voice=None,
            resolution=None,
            outcome_callback=first_outcomes.append,
            playback_lifecycle=first_lifecycle,
        )
        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        assert paths[0].exists()

        current_request = 2
        await handler._admit_tts_generation(
            text="Second.",
            message_id="message-b",
            voice=None,
            resolution=None,
            playback_lifecycle=second_lifecycle,
        )
        await asyncio.wait_for(second_started.wait(), timeout=1.0)

        assert first_outcomes == [True]
        assert not paths[0].exists()
        assert "message-a" not in handler._audio_files
        assert not any(
            isinstance(message, (TTSCompleteEvent, TTSPlaybackEvent))
            and message.message_id == "message-a"
            for message in posted
        )
        assert handler._console_generation_owner.lifecycle is second_lifecycle
        assert handler._console_generation_owner.task.done() is False
    finally:
        current_request = 3
        for task in tuple(handler._active_tasks):
            task.cancel()
        await asyncio.gather(*tuple(handler._active_tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_app_snapshot_handler_unavailable_logs_only_safe_context():
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="PRIVATE_AUTHORITY",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="PRIVATE_RESPONSE_TEXT",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    fake_app = _FakeApp()
    fake_app._ensure_tts_handler = AsyncMock(return_value=None)
    fake_app.post_message = MagicMock(return_value=True)

    await TldwCli.handle_tts_message_speech_request_event(fake_app, event)

    fake_app.loguru_logger.error.assert_called_once_with(
        "TTS handler not initialized "
        "(operation=trusted_console_speech, outcome_code=handler_unavailable)"
    )
    rendered_logs = repr(fake_app.loguru_logger.method_calls)
    assert "PRIVATE_RESPONSE_TEXT" not in rendered_logs
    assert "PRIVATE_AUTHORITY" not in rendered_logs
    assert message.id not in rendered_logs
    fake_app.post_message.assert_called_once()
    completion = fake_app.post_message.call_args.args[0]
    assert isinstance(completion, TTSCompleteEvent)
    assert completion.message_id == message.id
    assert completion.error == "TTS service not available"
    event.report_outcome(True)
    assert outcomes == [False]


@pytest.mark.asyncio
async def test_trusted_snapshot_rejection_reports_auto_speak_failure_once() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="PRIVATE_RESPONSE_TEXT",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)
    store.update_message_content(message.id, "changed")
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        snapshot,
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    handler = _new_handler()
    handler.app = _FakeApp()

    await handler.handle_tts_request(event)

    assert outcomes == [False]
    assert "PRIVATE_RESPONSE_TEXT" not in repr(handler.app.loguru_logger.method_calls)


@pytest.mark.asyncio
async def test_auto_speak_outcome_callback_reaches_admitted_generation() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    handler = _new_handler()
    handler._validate_message_speech_snapshot = AsyncMock(return_value="Ready.")
    handler._prepare_tts_text = AsyncMock(return_value="Ready.")
    handler._resolve_message_speech_request = AsyncMock(return_value=object())
    handler._admit_tts_generation = AsyncMock()

    await handler.handle_tts_request(event)

    callback = handler._admit_tts_generation.await_args.kwargs["outcome_callback"]
    callback(True)
    callback(False)
    assert outcomes == [True]


@pytest.mark.asyncio
async def test_automatic_request_rejects_destination_change_before_admission() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    expected = "sha256:" + "a" * 64
    changed = "sha256:" + "b" * 64
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
        expected_destination_fingerprint=expected,
    )
    handler = _new_handler()
    handler._validate_message_speech_snapshot = AsyncMock(return_value="Ready.")
    handler._prepare_tts_text = AsyncMock(return_value="Ready.")
    resolution = object()
    handler._resolve_message_speech_request = AsyncMock(return_value=resolution)
    handler._destination_for_resolution = AsyncMock(
        return_value=MagicMock(fingerprint=changed)
    )
    handler._admit_tts_generation = AsyncMock()
    handler._post_tts_message = AsyncMock()

    await handler.handle_tts_request(event)

    handler._destination_for_resolution.assert_awaited_once_with(resolution)
    handler._admit_tts_generation.assert_not_awaited()
    assert outcomes == [False]


@pytest.mark.asyncio
async def test_automatic_request_rechecks_destination_immediately_before_synthesis() -> (
    None
):
    expected = "sha256:" + "a" * 64
    changed = "sha256:" + "b" * 64
    outcomes: list[bool] = []
    service = MagicMock()
    service.synthesize_default = AsyncMock()
    handler = _new_handler()
    handler._tts_service = service
    handler._destination_for_resolution = AsyncMock(
        return_value=MagicMock(fingerprint=changed)
    )
    handler._discard_tts_artifact = AsyncMock()
    handler._post_tts_message = AsyncMock()
    resolution = MagicMock(source="global", request=None)

    await handler._generate_tts(
        "Ready.",
        "message-1",
        None,
        resolution,
        outcome_callback=outcomes.append,
        expected_destination_fingerprint=expected,
    )

    handler._destination_for_resolution.assert_awaited_once_with(resolution)
    service.synthesize_default.assert_not_awaited()
    assert outcomes == [False]
    completion = handler._post_tts_message.await_args.args[0]
    assert isinstance(completion, TTSCompleteEvent)
    assert "destination changed" in completion.error.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["resolver", "admission"])
async def test_unexpected_trusted_request_failure_settles_once(
    failure_stage: str,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="PRIVATE_RESPONSE_TEXT",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    handler = _new_handler()
    handler._validate_message_speech_snapshot = AsyncMock(return_value="Ready.")
    handler._prepare_tts_text = AsyncMock(return_value="Ready.")
    handler._resolve_message_speech_request = AsyncMock(return_value=object())
    handler._admit_tts_generation = AsyncMock()
    handler._post_tts_message = AsyncMock()
    if failure_stage == "resolver":
        handler._resolve_message_speech_request.side_effect = RuntimeError("private")
    else:
        handler._admit_tts_generation.side_effect = RuntimeError("private")

    await handler.handle_tts_request(event)
    event.report_outcome(True)

    assert outcomes == [False]


@pytest.mark.asyncio
async def test_trusted_prepare_rejection_settles_once() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    handler = _new_handler()
    handler._validate_message_speech_snapshot = AsyncMock(return_value="Ready.")
    handler._prepare_tts_text = AsyncMock(return_value=None)

    await handler.handle_tts_request(event)
    event.report_outcome(True)

    assert outcomes == [False]


@pytest.mark.asyncio
async def test_trusted_request_cancellation_settles_once() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    handler = _new_handler()
    handler._validate_message_speech_snapshot = AsyncMock(
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await handler.handle_tts_request(event)
    event.report_outcome(True)

    assert outcomes == [False]


@pytest.mark.asyncio
async def test_trusted_request_cooldown_settles_once() -> None:
    outcomes: list[bool] = []
    handler = _new_handler()
    now = asyncio.get_running_loop().time()
    handler._request_cooldown = {"message-1": now}
    handler._last_cooldown_cleanup = now
    handler._post_tts_message = AsyncMock()
    handler._generate_tts_with_rate_limit = AsyncMock()

    await handler._admit_tts_generation(
        text="Ready.",
        message_id="message-1",
        voice=None,
        resolution=None,
        outcome_callback=outcomes.append,
    )

    assert outcomes == [False]
    handler._generate_tts_with_rate_limit.assert_not_called()


@pytest.mark.asyncio
async def test_retry_failed_auto_bypasses_only_its_existing_message_cooldown() -> None:
    outcomes: list[bool] = []
    handler = _new_handler()
    now = asyncio.get_running_loop().time()
    handler._request_cooldown = {"message-1": now, "other-message": now}
    handler._last_cooldown_cleanup = now
    handler._post_tts_message = AsyncMock(return_value=True)
    handler._generate_tts_with_rate_limit = AsyncMock()

    await handler._admit_tts_generation(
        text="Ready.",
        message_id="message-1",
        voice=None,
        resolution=None,
        outcome_callback=outcomes.append,
        retry_failed_auto=True,
    )

    handler._generate_tts_with_rate_limit.assert_called_once()
    assert handler._request_cooldown["other-message"] == now
    assert outcomes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("post_result", [False, RuntimeError("queue closed")])
async def test_post_tts_message_reports_queue_acceptance(post_result) -> None:
    handler = _new_handler()
    app = MagicMock()
    if isinstance(post_result, Exception):
        app.post_message.side_effect = post_result
    else:
        app.post_message.return_value = post_result
    handler.app = app

    accepted = await handler._post_tts_message(TTSCompleteEvent(message_id="m"))

    assert accepted is False


@pytest.mark.asyncio
async def test_final_completion_post_rejection_reports_generation_failure(
    tmp_path,
) -> None:
    async def chunks():
        yield b"audio"

    response = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-model",
        audio_format="mp3",
        content_type="audio/mpeg",
        byte_stream=chunks(),
    )
    service = MagicMock()
    service.preferences_snapshot.return_value = SimpleNamespace(
        provider_id="openai",
        speed=1.0,
    )
    service.synthesize_default = AsyncMock(return_value=response)
    handler = _new_handler()
    handler._tts_service = service
    handler._create_tts_artifact = MagicMock(return_value=tmp_path / "speech.mp3")
    posted: list[object] = []

    async def reject_completion(message: object) -> bool:
        posted.append(message)
        return not isinstance(message, TTSCompleteEvent)

    handler._post_tts_message = reject_completion
    outcomes: list[bool] = []

    await handler._generate_tts(
        "Private reply.",
        "message-1",
        None,
        outcome_callback=outcomes.append,
    )

    assert any(isinstance(message, TTSCompleteEvent) for message in posted)
    assert outcomes == [False]
    assert not (tmp_path / "speech.mp3").exists()
    assert "message-1" not in handler._audio_files


@pytest.mark.asyncio
async def test_manual_generation_preserves_legacy_service_call_shape(tmp_path) -> None:
    async def chunks():
        yield b"audio"

    class LegacyServiceDouble:
        def preferences_snapshot(self):
            return SimpleNamespace(provider_id="openai", speed=1.0)

        async def synthesize_default(
            self,
            *,
            text: str,
            voice_override: str | None,
            response_format_override: str | None = None,
            progress_sink,
        ):
            del text, voice_override, progress_sink
            return TTSAudioResponse(
                provider_id="openai",
                model_id="tts-1",
                audio_format="mp3",
                content_type="audio/mpeg",
                byte_stream=chunks(),
            )

    handler = _new_handler()
    handler._tts_service = LegacyServiceDouble()
    handler._create_tts_artifact = MagicMock(return_value=tmp_path / "manual.mp3")
    handler._post_tts_message = AsyncMock(return_value=True)

    await handler._generate_tts("Manual reply.", "message-1", None)

    assert any(
        isinstance(call.args[0], TTSCompleteEvent)
        for call in handler._post_tts_message.await_args_list
    )


class _SpeechRequestControllerStub:
    bind_speech_presentation = ConsoleMessageController.bind_speech_presentation
    current_speech_presentation = ConsoleMessageController.current_speech_presentation
    _begin_console_speech_presentation = (
        ConsoleMessageController._begin_console_speech_presentation
    )
    _settle_console_speech_presentation = (
        ConsoleMessageController._settle_console_speech_presentation
    )
    _schedule_console_speech_state_sync = (
        ConsoleMessageController._schedule_console_speech_state_sync
    )
    _dispatch_console_speech_stop_event = (
        ConsoleMessageController._dispatch_console_speech_stop_event
    )

    def __init__(self, store: ConsoleChatStore, post_result) -> None:
        self._store = store
        self._screen = MagicMock()
        self._screen._console_presentation_context.return_value = None
        self.app_instance = MagicMock()
        self.app_instance.audio_output = AudioOutputOwnership()
        if isinstance(post_result, Exception):
            self.app_instance.post_message.side_effect = post_result
        else:
            self.app_instance.post_message.return_value = post_result
        self._console_speaking_message_id = None
        self._console_speech_states: dict[str, str] = {}
        self._console_speech_request_generation = 0
        self._console_speech_lifetime_generation = 0
        self._console_speech_owner = None
        self._console_speech_pending_stop = None
        self.syncs = 0

    def _ensure_console_chat_store(self) -> ConsoleChatStore:
        return self._store

    async def _sync_native_console_chat_ui(self) -> None:
        self.syncs += 1


def _accept_owned_stop(message: object) -> bool:
    if isinstance(message, TTSPlaybackEvent):
        if message.playback_lifecycle is not None:
            message.playback_lifecycle.report_terminal("stopped")
        message.report_outcome(True)
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["manual", "automatic", "retry", "empty"])
async def test_console_markdown_reaches_adapter_through_speech_controls(
    mode, monkeypatch, tmp_path
) -> None:
    """Exercise controller, consent/destination resolution and real TTS admission.

    Args:
        mode: Manual, automatic, retry or empty-output speech path to exercise.
        monkeypatch: Fixture isolating provider and audio-output boundaries.
        tmp_path: Fixture providing storage for the test conversation.
    """
    captured = []
    audio = io.BytesIO()
    with wave.open(audio, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 24)

    class SpeechAdapter(FakeAdapter):
        def admitted_outbound_endpoint(self):
            return "https://speech.example.test/v1"

        async def synthesize(self, request, progress_sink=None):
            captured.append(request.text)
            if mode == "retry" and len(captured) == 1:
                raise RuntimeError("deliberate provider failure")
            return await super().synthesize(request, progress_sink)

    adapter = SpeechAdapter("openai", chunks=(audio.getvalue(),))
    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("openai", "OpenAI", False),
                factory=lambda _config: adapter,
                initial_config={
                    "app_config": {
                        "app_tts": {
                            "OPENAI_BASE_URL": "https://speech.example.test/v1",
                        }
                    }
                },
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="model",
            voice_mode="exact",
            voice_id="default",
            response_format="wav",
            speed=1.0,
        ),
    )
    handler = _new_handler()
    handler._tts_service = service
    store = ConsoleChatStore()
    session = store.create_session()
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.console_runtime = None
    handler.app = controller.app_instance
    queued = []
    errors = []
    drained = []

    def schedule(coroutine):
        queued.append(asyncio.create_task(coroutine))

    def post(event):
        if isinstance(event, TTSMessageSpeechRequestEvent):
            schedule(handler.handle_tts_request(event))
        elif isinstance(event, TTSCompleteEvent) and event.error:
            errors.append(event.error)
            if event.playback_lifecycle is not None:
                event.playback_lifecycle.report("failed")
        return True

    async def drain_audio(_plan, byte_source, *, playback_lifecycle, **_kwargs):
        playback_lifecycle.report("playing")
        drained.append(b"".join([chunk async for chunk in byte_source]))
        playback_lifecycle.report_terminal("stopped")
        return "success"

    async def settle():
        while queued or handler._active_tasks:
            tasks, queued[:] = [*queued, *handler._active_tasks], []
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            await asyncio.sleep(0)

    monkeypatch.setattr(handler, "_post_tts_message", AsyncMock(side_effect=post))
    monkeypatch.setattr(handler, "_stream_response_via_sink", drain_audio)
    monkeypatch.setattr(
        handler, "_create_tts_artifact", lambda _format: tmp_path / "reply.wav"
    )
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)
    controller.app_instance.post_message.side_effect = post
    coordinator = ConsoleAutoSpeakCoordinator(
        reserve_audio_intent=lambda: controller.app_instance.audio_output.begin_intent(
            tentative=True
        ),
        store_accessor=lambda: store,
        resolve_destination=handler.resolve_console_speech_destination,
        issue_message_speech=lambda *args: (
            ConsoleMessageController.request_console_message_speech(controller, *args)
        ),
        open_consent=lambda _modal, callback: callback(True),
        hands_free_active=lambda: False,
        sync_controls=lambda *_args: None,
        notify=lambda *_args: None,
        schedule=schedule,
    )
    coordinator.mount()
    content = "## Summary\n\nRead **this** [guide](https://example.test/private).\n\n```python\nsecret_code()\n```\n\n<div>Visible<br>details</div>\n\n- [x] Reviewed"
    if mode == "empty":
        content = "---\n\n<!-- formatting only -->"
    try:
        if mode != "manual":
            coordinator.request_enabled(True)
            await settle()
            assert session.speech_preferences.auto_speak
        message = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        store.append_stream_chunk(message.id, content)
        store.mark_message_complete(message.id)
        if mode == "manual":
            await ConsoleMessageController.request_console_message_speech(
                controller, message.id
            )
        await settle()
        if mode == "retry":
            assert session.speech_preferences.paused
            assert len(errors) == 1
            coordinator.request_retry()
            await settle()
        expected = "Summary. Read this guide. Code block omitted. Visible. details. Checked: Reviewed"
        assert captured == (
            [] if mode == "empty" else [expected] * (2 if mode == "retry" else 1)
        )
        assert len(drained) == (0 if mode == "empty" else 1)
        assert all(drained)
        assert controller._console_speaking_message_id is None
        assert store.get_message(message.id).content == content
        if mode != "retry":
            assert errors == []
            assert not session.speech_preferences.paused
    finally:
        coordinator.unmount()
        await settle()
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("post_result", [False, RuntimeError("queue closed")])
async def test_console_speech_post_rejection_settles_once(post_result) -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    controller = _SpeechRequestControllerStub(store, post_result)
    outcomes: list[bool] = []

    issued = await ConsoleMessageController.request_console_message_speech(
        controller,
        message.id,
        outcomes.append,
        expected_destination_fingerprint="sha256:" + "a" * 64,
    )

    assert issued is False
    assert outcomes == [False]
    assert controller._console_speaking_message_id is None


@pytest.mark.asyncio
async def test_console_speech_unexpected_snapshot_issue_failure_settles_once() -> None:
    store = MagicMock()
    store.issue_tts_message_speech_snapshot.side_effect = RuntimeError("private")
    controller = _SpeechRequestControllerStub(store, True)
    outcomes: list[bool] = []

    issued = await ConsoleMessageController.request_console_message_speech(
        controller,
        "message-1",
        outcomes.append,
    )

    assert issued is False
    assert outcomes == [False]
    controller.app_instance.post_message.assert_not_called()


@pytest.mark.asyncio
async def test_console_speech_request_waits_for_playback_start() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    controller = _SpeechRequestControllerStub(store, True)

    issued = await ConsoleMessageController.request_console_message_speech(
        controller,
        message.id,
    )

    assert issued is True
    assert controller._console_speech_states == {message.id: "generating"}
    event = controller.app_instance.post_message.call_args.args[0]
    event.report_outcome(True)
    await asyncio.sleep(0)

    assert controller._console_speech_states == {message.id: "generating"}
    event.playback_lifecycle.report("playing")
    await asyncio.sleep(0)
    assert controller._console_speech_states == {message.id: "playing"}
    assert controller._console_speaking_message_id == message.id


@pytest.mark.asyncio
async def test_stale_speech_outcome_cannot_replace_newer_request_state() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.post_message.side_effect = _accept_owned_stop

    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    first = controller.app_instance.post_message.call_args.args[0]
    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    second = controller.app_instance.post_message.call_args.args[0]

    first.report_outcome(False)
    await asyncio.sleep(0)
    assert controller._console_speech_states[message.id] == "generating"

    second.report_outcome(True)
    second.playback_lifecycle.report("playing")
    await asyncio.sleep(0)
    assert controller._console_speech_states[message.id] == "playing"


@pytest.mark.asyncio
async def test_generation_success_waits_for_authoritative_playback_lifecycle() -> None:
    """Synthesis delivery is not proof that the audio device started."""
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    controller = _SpeechRequestControllerStub(store, True)

    assert await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = controller.app_instance.post_message.call_args.args[0]

    request.report_outcome(True)
    await asyncio.sleep(0)
    assert controller._console_speech_states == {message.id: "generating"}

    assert request.playback_lifecycle.report("playing") is True
    await asyncio.sleep(0)
    assert controller._console_speech_states == {message.id: "playing"}

    assert request.playback_lifecycle.report("stopped") is True
    await asyncio.sleep(0)
    assert controller._console_speech_states == {message.id: "stopped"}
    assert controller._console_speaking_message_id is None


@pytest.mark.asyncio
async def test_completion_rejected_before_playback_reports_failed(tmp_path) -> None:
    """A full artifact cannot claim Playing when the Play post is rejected."""
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSPlaybackLifecycle,
    )

    states: list[str] = []
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    artifact = tmp_path / "reply.wav"
    artifact.write_bytes(b"RIFF")
    app = _FakeApp()
    app.post_message = MagicMock(return_value=False)

    await TldwCli.handle_tts_complete_event(
        app,
        TTSCompleteEvent(
            message_id="message-1",
            audio_file=artifact,
            playback_lifecycle=lifecycle,
        ),
    )

    assert states == ["failed"]


@pytest.mark.asyncio
async def test_missing_completion_artifact_reports_failed(tmp_path) -> None:
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSPlaybackLifecycle,
    )

    states: list[str] = []
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    app = _FakeApp()

    await TldwCli.handle_tts_complete_event(
        app,
        TTSCompleteEvent(
            message_id="message-1",
            audio_file=tmp_path / "missing.wav",
            playback_lifecycle=lifecycle,
        ),
    )

    assert states == ["failed"]
    assert not any(isinstance(item, TTSPlaybackEvent) for item in app.posted)


@pytest.mark.asyncio
async def test_stale_completion_discards_real_cached_artifact_and_settles(
    tmp_path,
) -> None:
    artifact = tmp_path / "stale.wav"
    artifact.write_bytes(b"RIFF")
    states: list[str] = []
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: False,
        callback=states.append,
    )
    handler = _new_handler()
    await handler._cache_audio_file("message-1", artifact, lifecycle)
    app = _FakeApp()
    app._tts_handler = handler

    await TldwCli.handle_tts_complete_event(
        app,
        TTSCompleteEvent(
            message_id="message-1",
            audio_file=artifact,
            playback_lifecycle=lifecycle,
        ),
    )

    assert not artifact.exists()
    assert "message-1" not in handler._audio_files
    assert "message-1" not in handler._audio_file_owners
    assert states == ["stopped"]
    assert not any(isinstance(item, TTSPlaybackEvent) for item in app.posted)


def test_terminal_lifecycle_is_not_current_and_cannot_restart() -> None:
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    assert lifecycle.report("stopped") is True

    assert lifecycle.is_current() is False
    assert lifecycle.report("playing") is False


@pytest.mark.asyncio
async def test_terminal_failed_completion_still_reports_actionable_error() -> None:
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    lifecycle.report_terminal("failed")
    app = _FakeApp()

    await TldwCli.handle_tts_complete_event(
        app,
        TTSCompleteEvent(
            message_id="message-1",
            error="playback failed; retry",
            playback_lifecycle=lifecycle,
        ),
    )

    app.notify.assert_called_once_with(
        "TTS failed: playback failed; retry",
        severity="error",
    )


@pytest.mark.asyncio
async def test_superseding_request_fences_unstarted_completion_without_stopping_output() -> (
    None
):
    store = ConsoleChatStore()
    session = store.create_session()
    first = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="First."
    )
    second = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Second."
    )
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.post_message.side_effect = _accept_owned_stop

    await ConsoleMessageController.request_console_message_speech(controller, first.id)
    first_request = controller.app_instance.post_message.call_args.args[0]
    await asyncio.sleep(0)

    await ConsoleMessageController.request_console_message_speech(controller, second.id)
    posted = [
        call.args[0] for call in controller.app_instance.post_message.call_args_list
    ]

    assert not any(isinstance(event, TTSPlaybackEvent) for event in posted)
    assert isinstance(posted[-1], TTSMessageSpeechRequestEvent)
    assert posted[-1].message_id == second.id
    assert first_request.playback_lifecycle.is_current() is False

    app = _FakeApp()
    artifact = MagicMock()
    artifact.exists.return_value = True
    await TldwCli.handle_tts_complete_event(
        app,
        TTSCompleteEvent(
            message_id=first.id,
            audio_file=artifact,
            playback_lifecycle=first_request.playback_lifecycle,
        ),
    )
    assert not any(isinstance(item, TTSPlaybackEvent) for item in app.posted)


@pytest.mark.asyncio
async def test_failed_manual_replacement_keeps_healthy_native_owner_and_its_stop_target(
    tmp_path, monkeypatch
):
    store = ConsoleChatStore()
    session = store.create_session()
    first, second = [
        store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=text
        )
        for text in ("first", "second")
    ]
    controller = _SpeechRequestControllerStub(store, True)
    post = controller.app_instance.post_message
    speak = ConsoleMessageController.request_console_message_speech
    assert await speak(controller, first.id)
    a = post.call_args.args[0]
    handler = _new_handler(controller.app_instance)
    controller.app_instance._tts_handler = handler
    sink, stream, finish = await _start_test_stream(
        handler, monkeypatch, a.playback_lifecycle, ticket=a.audio_intent
    )
    lease = handler._speech_outputs[0].run.lease
    try:
        assert await speak(controller, second.id)
        b = post.call_args.args[0]
        assert b.message_id == second.id
        assert not any(
            isinstance(call.args[0], TTSPlaybackEvent) for call in post.call_args_list
        )
        path = tmp_path / "replacement.wav"
        path.write_bytes(b"clip")
        no_plan = AsyncMock(return_value=None)
        monkeypatch.setattr(audio_output_runs.FileOutputRun, "prepare", no_plan)
        await handler._cache_audio_file(second.id, path, b.playback_lifecycle)
        await handler.handle_tts_playback(
            TTSPlaybackEvent(
                "play",
                second.id,
                expected_path=path,
                playback_lifecycle=b.playback_lifecycle,
                audio_intent=b.audio_intent,
            )
        )
        await wait_for_condition(lambda: b.playback_lifecycle.state == "failed")
        no_plan.assert_awaited_once_with(path)
        assert lease.is_current() and sink.state == "open"
        assert controller._console_speech_owner is a.playback_lifecycle
        assert controller._console_speaking_message_id == first.id
        assert a.playback_lifecycle.is_current()
        assert await _stop_file(handler, controller._console_speech_owner) == [True]
        assert sink.state == "stopped" and handler.app.audio_output.owner is None
    finally:
        finish.set()
        await stream
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_before_start_receipt", [True, False])
async def test_admitted_file_start_remains_addressable_across_failed_replacement(
    tmp_path, monkeypatch, failure_before_start_receipt
):
    from textual.widgets import Button
    from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService

    store = ConsoleChatStore()
    session = store.create_session()
    first, second = [
        store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=text
        )
        for text in ("first", "second")
    ]
    controller = _SpeechRequestControllerStub(store, True)
    handler = controller.app_instance._tts_handler = _new_handler(
        controller.app_instance
    )
    controller.app_instance.control_tts_playback = handler.handle_tts_playback
    controller._console_message_action_service = ConsoleMessageActionService(
        canvas_enabled_reader=lambda: False
    )
    controller._console_message_presentation = lambda message: message

    committed, return_start = threading.Event(), threading.Event()

    class HeldResult(ConsoleFilePlayer):
        def play_prepared(self, plan, **kwargs):
            accepted = super().play_prepared(plan, **kwargs)
            committed.set()
            assert return_start.wait(5)
            return accepted

    player = HeldResult(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    speak = ConsoleMessageController.request_console_message_speech
    post = controller.app_instance.post_message
    assert await speak(controller, first.id)
    first_request = post.call_args.args[0]
    path = tmp_path / "first.wav"
    path.write_bytes(b"first")
    await handler._cache_audio_file(first.id, path, first_request.playback_lifecycle)
    await handler.handle_tts_playback(
        TTSPlaybackEvent(
            "play",
            first.id,
            playback_lifecycle=first_request.playback_lifecycle,
            audio_intent=first_request.audio_intent,
            expected_path=path,
        )
    )
    try:
        await wait_for_condition(committed.is_set)
        assert player.state is PlaybackState.PLAYING
        assert first_request.playback_lifecycle.state == "generating"
        assert await speak(controller, second.id)
        replacement = post.call_args.args[0]
        if not failure_before_start_receipt:
            return_start.set()
            await wait_for_condition(
                lambda: first_request.playback_lifecycle.state == "playing"
            )
            assert controller._console_speech_states[first.id] == "playing"
            assert controller._console_speech_owner is replacement.playback_lifecycle
        replacement.report_outcome(False)
        assert replacement.playback_lifecycle.state == "failed"
        assert controller._console_speech_owner is first_request.playback_lifecycle
        assert (
            controller._console_speech_states[first.id]
            == first_request.playback_lifecycle.state
        )
        return_start.set()
        await wait_for_condition(
            lambda: first_request.playback_lifecycle.state == "playing"
        )
        assert controller._console_speech_states[first.id] == "playing"
        button = Button("Stop")
        button.console_action_id, button.console_message_id = "speak-stop", first.id
        assert await ConsoleMessageController.handle_console_message_action(
            controller, Button.Pressed(button)
        )
        assert player.retirement_confirmed(expected_token=player.token)
        assert controller.app_instance.audio_output.owner is None
    finally:
        return_start.set()
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_superseded_generation_settles_external_outcome_once() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    first = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="First."
    )
    second = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Second."
    )
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.post_message.side_effect = _accept_owned_stop
    outcomes: list[bool] = []

    await ConsoleMessageController.request_console_message_speech(
        controller,
        first.id,
        outcome_callback=outcomes.append,
    )
    first_request = controller.app_instance.post_message.call_args.args[0]

    await ConsoleMessageController.request_console_message_speech(controller, second.id)
    first_request.report_outcome(True)
    await asyncio.sleep(0)

    assert outcomes == [True]
    assert controller._console_speaking_message_id == second.id
    assert controller._console_speech_states[second.id] == "generating"


@pytest.mark.asyncio
async def test_session_switch_and_restore_same_id_invalidate_playback_owner() -> None:
    store = ConsoleChatStore()
    first_session = store.create_session()
    message = store.append_message(
        first_session.id, role=ConsoleMessageRole.ASSISTANT, content="Old."
    )
    second_session = store.create_session()
    store.switch_session(first_session.id)
    controller = _SpeechRequestControllerStub(store, True)

    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = controller.app_instance.post_message.call_args.args[0]
    store.switch_session(second_session.id)
    assert request.playback_lifecycle.is_current() is False

    replacement_session = type(first_session)(id=first_session.id, title="Restored")
    replacement_message = type(message)(
        id=message.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Replacement.",
        status="complete",
    )
    store.restore_state(
        sessions=[replacement_session],
        messages_by_session={replacement_session.id: [replacement_message]},
        active_session_id=replacement_session.id,
    )
    assert request.playback_lifecycle.is_current() is False


@pytest.mark.asyncio
async def test_screen_lifetime_invalidation_rejects_late_playback() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Ready."
    )
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.post_message.side_effect = _accept_owned_stop

    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = controller.app_instance.post_message.call_args.args[0]
    task = ConsoleMessageController.invalidate_console_speech_context(controller)
    assert task is not None
    await task

    assert request.playback_lifecycle.is_current() is False
    assert controller._console_speaking_message_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("post_failure", [False, RuntimeError("queue closed")])
async def test_context_invalidation_falls_back_to_real_handler_stop(
    post_failure,
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Ready."
    )
    controller = _SpeechRequestControllerStub(store, True)
    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = controller.app_instance.post_message.call_args.args[0]
    handler = _new_handler(controller.app_instance)
    controller.app_instance._tts_handler = handler
    sink, stream, finish = await _start_test_stream(
        handler, monkeypatch, request.playback_lifecycle, ticket=request.audio_intent
    )
    if isinstance(post_failure, Exception):
        controller.app_instance.post_message.side_effect = post_failure
    else:
        controller.app_instance.post_message.return_value = post_failure
    try:
        task = ConsoleMessageController.invalidate_console_speech_context(controller)
        assert task is not None
        await task
        assert sink.state == "stopped"
        assert handler.app.audio_output.owner is None
    finally:
        finish.set()
        await stream

    assert request.playback_lifecycle.state == "stopped"
    assert controller._console_speaking_message_id is None
    assert controller._console_speech_owner is None


@pytest.mark.asyncio
async def test_invalidation_fallback_exception_can_retry_retained_owner() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Ready."
    )
    controller = _SpeechRequestControllerStub(store, True)
    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = controller.app_instance.post_message.call_args.args[0]
    request.playback_lifecycle.report("playing")
    controller.app_instance.post_message.return_value = False
    attempts = 0

    async def handle_stop(event: TTSPlaybackEvent) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("handler unavailable")
        event.playback_lifecycle.report_terminal("stopped")
        event.report_outcome(True)

    controller.app_instance._tts_handler = SimpleNamespace(
        handle_tts_playback=handle_stop
    )

    first = ConsoleMessageController.invalidate_console_speech_context(controller)
    assert first is not None
    await first

    assert controller._console_speech_pending_stop is None
    assert controller._console_speech_owner is request.playback_lifecycle
    assert controller._console_speaking_message_id == message.id
    assert controller._console_speech_states[message.id] == "failed"

    second = ConsoleMessageController.invalidate_console_speech_context(controller)
    assert second is not None
    await second

    assert attempts == 2
    assert controller._console_speech_pending_stop is None
    assert controller._console_speech_owner is None
    assert controller._console_speaking_message_id is None
    assert controller._console_speech_states[message.id] == "stopped"


@pytest.mark.asyncio
async def test_restore_same_ids_rejects_delayed_invalidation_ui_settlement() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Original."
    )
    controller = _SpeechRequestControllerStub(store, True)
    posted: list[object] = []
    controller.app_instance.post_message.side_effect = lambda event: (
        posted.append(event) or True
    )
    await ConsoleMessageController.request_console_message_speech(
        controller, message.id
    )
    request = posted[-1]
    request.playback_lifecycle.report("playing")

    invalidation = ConsoleMessageController.invalidate_console_speech_context(
        controller
    )
    assert invalidation is not None
    await invalidation
    stop_event = posted[-1]
    replacement_session = type(session)(id=session.id, title="Restored")
    replacement_message = type(message)(
        id=message.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Replacement.",
        status="complete",
    )
    store.restore_state(
        sessions=[replacement_session],
        messages_by_session={replacement_session.id: [replacement_message]},
        active_session_id=replacement_session.id,
    )

    request.playback_lifecycle.report_terminal("stopped")
    stop_event.report_outcome(True)
    await asyncio.sleep(0)

    assert controller._console_speech_pending_stop is None
    assert controller._console_speech_owner is None
    assert controller._console_speaking_message_id is None
    assert message.id not in controller._console_speech_states


@pytest.mark.asyncio
async def test_ownership_state_remains_bounded_across_many_requests() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    controller = _SpeechRequestControllerStub(store, True)
    controller.app_instance.post_message.side_effect = _accept_owned_stop

    for index in range(10_000):
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"Ready {index}.",
        )
        await ConsoleMessageController.request_console_message_speech(
            controller, message.id
        )

    assert not hasattr(controller, "_console_speech_epochs")
    assert len(controller._console_speech_states) <= 2
    assert controller._console_speech_request_generation == 10_000


async def _dispatch_file(handler, artifact, lifecycle, *, outcome_callback=None):
    await handler._cache_audio_file(lifecycle.message_id, artifact, lifecycle)
    await handler.handle_tts_playback(
        TTSPlaybackEvent(
            action="play",
            message_id=lifecycle.message_id,
            playback_lifecycle=lifecycle,
            outcome_callback=outcome_callback,
        )
    )


async def _stop_file(handler, lifecycle, *, message_id=None):
    outcomes = []
    await handler.handle_tts_playback(
        TTSPlaybackEvent(
            action="stop",
            message_id=lifecycle.message_id if lifecycle else message_id,
            playback_lifecycle=lifecycle,
            outcome_callback=outcomes.append,
        )
    )
    return outcomes


def _lifecycle(message_id, states, *, request_id=1):
    return TTSPlaybackLifecycle(
        message_id=message_id,
        request_id=request_id,
        validator=lambda: True,
        callback=states.append,
    )


async def _start_test_stream(handler, monkeypatch, lifecycle, *, ticket=None):
    finish = asyncio.Event()
    opened = []
    from tldw_chatbook.Audio.streaming_sink import PumpResult, SinkStarted

    async def held_pump(sink, *_args, **_kwargs):
        sink.on_event(SinkStarted())
        opened.append(sink)
        await finish.wait()
        return PumpResult(outcome="stopped", bytes_fed=4)

    async def chunks():
        yield b"1234"

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _RecordingSink)
    monkeypatch.setattr(tts_events_module, "pump", held_pump)
    handler._post_tts_message = AsyncMock(return_value=True)
    token = tts_events_module._speech_intent.set(ticket)
    try:
        task = asyncio.create_task(
            handler._stream_response_via_sink(
                SimpleNamespace(
                    sample_rate=24000, channels=1, skip_bytes=0, data_bytes=None
                ),
                chunks(),
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
    finally:
        tts_events_module._speech_intent.reset(token)
    try:
        await wait_for_condition(lambda: opened and lifecycle.state == "playing")
    except BaseException:
        finish.set()
        await task
        raise
    return opened[0], task, finish


@pytest.mark.asyncio
async def test_playback_handler_reports_start_then_natural_stop(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "reply.wav"
    artifact.write_bytes(b"RIFF")
    states, outcomes = [], []
    lifecycle = _lifecycle("message-1", states)
    handler = _new_handler()
    player = ConsoleFilePlayer()
    player.release.clear()
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    try:
        await _dispatch_file(
            handler, artifact, lifecycle, outcome_callback=outcomes.append
        )
        assert await asyncio.to_thread(player.entered.wait, 3)
        assert states == []
        assert outcomes == []
        assert artifact.exists()
        player.release.set()
        await drain_handler_playback(handler)

        assert player.paths == [artifact]
        assert outcomes == [True]
        assert states == ["playing", "stopped"]
        assert handler.app.audio_output.owner is None
        assert handler._speech_outputs == []
    finally:
        player.release.set()
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_playback_handler_reports_player_start_failure(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "reply.wav"
    artifact.write_bytes(b"RIFF")
    states, outcomes = [], []
    handler = _new_handler()
    player = ConsoleFilePlayer(None)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    try:
        await _dispatch_file(
            handler,
            artifact,
            _lifecycle("message-1", states),
            outcome_callback=outcomes.append,
        )
        await drain_handler_playback(handler)
        assert outcomes == [False]
        assert states == ["failed"]
        assert handler.app.audio_output.owner is None
        assert player.stops == [player.retired_token]
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_message", [False, True])
async def test_stale_or_lifecycleless_stop_preserves_active_stream(
    monkeypatch, same_message
) -> None:
    owner = _lifecycle("message-b", [])
    stale = None if same_message else _lifecycle("message-a", [])
    handler = _new_handler()
    sink, stream, finish = await _start_test_stream(handler, monkeypatch, owner)
    try:
        outcomes = await _stop_file(handler, stale, message_id=owner.message_id)
        assert outcomes == [False]
        assert owner.state == "playing"
        assert sink.state == "open"
        assert handler.app.audio_output.owner is not None
    finally:
        finish.set()
        await stream
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_lifecycleless_stop_preserves_owned_artifact_before_play_reserves_owner(
    tmp_path,
    monkeypatch,
) -> None:
    async def chunks():
        yield b"audio"

    artifact = tmp_path / "pre-play.mp3"
    response = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-model",
        audio_format="mp3",
        content_type="audio/mpeg",
        byte_stream=chunks(),
    )
    service = MagicMock()
    service.preferences_snapshot.return_value = SimpleNamespace(
        provider_id="openai",
        speed=1.0,
    )
    service.synthesize_default = AsyncMock(return_value=response)
    lifecycle = TTSPlaybackLifecycle(
        message_id="pre-play-window",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    handler = _new_handler()
    handler._tts_service = service
    handler._create_tts_artifact = MagicMock(return_value=artifact)
    observed: dict[str, object] = {}

    async def intercept(message: object) -> bool:
        if (
            isinstance(message, TTSProgressEvent)
            and message.status == "Audio generation complete"
        ):
            outcomes: list[bool] = []
            await handler.handle_tts_playback(
                TTSPlaybackEvent(
                    action="stop",
                    message_id=lifecycle.message_id,
                    outcome_callback=outcomes.append,
                )
            )
            observed.update(
                outcomes=outcomes,
                artifact_exists=artifact.exists(),
                cached=handler._audio_files.get(lifecycle.message_id),
                cached_owner=getattr(handler, "_audio_file_owners", {}).get(
                    lifecycle.message_id
                ),
                generation_owner=handler._console_generation_owner,
                file_outputs=tuple(handler._speech_outputs),
            )
        return True

    handler._post_tts_message = intercept
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.sink_available",
        lambda: False,
    )

    await handler._admit_tts_generation(
        text="Reply.",
        message_id=lifecycle.message_id,
        voice=None,
        resolution=None,
        playback_lifecycle=lifecycle,
    )
    generation_owner = handler._console_generation_owner
    assert generation_owner is not None
    assert generation_owner.task is not None
    await generation_owner.task

    assert observed["outcomes"] == [False]
    assert observed["artifact_exists"] is True
    assert observed["cached"] == artifact
    assert observed["cached_owner"] is lifecycle
    assert observed["generation_owner"] is generation_owner
    assert observed["file_outputs"] == ()


@pytest.mark.asyncio
async def test_artifact_owner_metadata_tracks_replacement_discard_and_cleanup(
    tmp_path,
) -> None:
    old_artifact = tmp_path / "old.wav"
    new_artifact = tmp_path / "new.wav"
    old_artifact.write_bytes(b"old")
    new_artifact.write_bytes(b"new")
    old_owner = TTSPlaybackLifecycle(
        message_id="same-message",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    new_owner = TTSPlaybackLifecycle(
        message_id="same-message",
        request_id=2,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    handler = _new_handler()

    await handler._cache_audio_file("same-message", old_artifact, old_owner)
    await handler._cache_audio_file("same-message", new_artifact, new_owner)

    assert handler._audio_files == {"same-message": new_artifact}
    assert handler._audio_file_owners == {"same-message": new_owner}

    await handler._discard_tts_artifact(
        "same-message",
        old_artifact,
        artifact_owner=old_owner,
    )

    assert old_artifact.exists() is False
    assert new_artifact.read_bytes() == b"new"
    assert handler._audio_files == {"same-message": new_artifact}
    assert handler._audio_file_owners == {"same-message": new_owner}

    await handler._cleanup_audio_file(
        "same-message",
        artifact_owner=new_owner,
    )

    assert new_artifact.exists() is False
    assert handler._audio_files == {}
    assert handler._audio_file_owners == {}


@pytest.mark.asyncio
async def test_shutdown_late_delete_releases_artifact_and_exact_owner(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "late-shutdown.wav"
    artifact.write_bytes(b"private audio")
    lifecycle = TTSPlaybackLifecycle(
        message_id="shutdown-message",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    handler = _new_handler()
    await handler._cache_audio_file(
        lifecycle.message_id,
        artifact,
        lifecycle,
    )
    entered = threading.Event()
    release = threading.Event()
    log_messages: list[str] = []

    def delayed_delete(candidate) -> bool:
        assert candidate == artifact
        entered.set()
        assert release.wait(timeout=2.0)
        artifact.unlink()
        return True

    async def release_after_timeout() -> None:
        assert await asyncio.to_thread(entered.wait, 3)
        await asyncio.sleep(0.03)
        release.set()

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delayed_delete)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_SECURE_DELETE_TIMEOUT_SECONDS",
        0.01,
    )
    sink_id = tts_events_module.logger.add(
        log_messages.append,
        level="DEBUG",
        format="{message}",
    )
    release_task = asyncio.create_task(release_after_timeout())
    try:
        await handler.cleanup_tts_resources()
        await release_task
    finally:
        release.set()
        tts_events_module.logger.remove(sink_id)

    assert artifact.exists() is False
    assert handler._audio_files == {}
    assert handler._audio_file_owners == {}
    rendered_logs = "\n".join(log_messages)
    assert "Late TTS artifact cleanup could not be scheduled" not in rendered_logs
    assert "TypeError" not in rendered_logs


@pytest.mark.asyncio
async def test_missing_owned_play_artifact_clears_path_and_owner_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    lifecycle = TTSPlaybackLifecycle(
        message_id="missing-owned",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    missing_artifact = tmp_path / "missing.wav"
    handler = _new_handler()

    class Player(ConsoleFilePlayer):
        def prepare(self, path):
            assert path == missing_artifact and not path.exists()
            return None

    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", Player)
    await handler._cache_audio_file(
        lifecycle.message_id,
        missing_artifact,
        lifecycle,
    )

    await handler.handle_tts_playback(
        TTSPlaybackEvent(
            action="play",
            message_id=lifecycle.message_id,
            playback_lifecycle=lifecycle,
        )
    )

    await drain_handler_playback(handler)
    assert lifecycle.state == "failed"
    assert lifecycle.message_id not in handler._audio_files
    assert lifecycle.message_id not in handler._audio_file_owners


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["play", "stop"])
async def test_stale_controls_preserve_the_exact_owned_file(
    tmp_path, monkeypatch, action
):
    artifact = tmp_path / "newer.wav"
    artifact.write_bytes(b"RIFF-newer")
    states, outcomes = [], []
    cached_owner = _lifecycle("same-message", states, request_id=2)
    stale_owner = _lifecycle("same-message", []) if action == "play" else None
    handler = _new_handler()
    player = ConsoleFilePlayer(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    try:
        await _dispatch_file(handler, artifact, cached_owner)
        await wait_for_condition(lambda: cached_owner.state == "playing")
        (attempt,) = handler._speech_outputs
        await handler.handle_tts_playback(
            TTSPlaybackEvent(
                action=action,
                message_id=cached_owner.message_id,
                playback_lifecycle=stale_owner,
                outcome_callback=outcomes.append,
            )
        )

        assert outcomes == [False]
        assert states == ["playing"]
        assert attempt.run.lease.is_current()
        assert not player.stops
        if stale_owner is not None:
            assert stale_owner.state == "failed"
        assert handler._audio_files[cached_owner.message_id] == artifact
        assert handler._audio_file_owners[cached_owner.message_id] is cached_owner
        assert artifact.read_bytes() == b"RIFF-newer"
    finally:
        await _stop_file(handler, cached_owner)
        await drain_handler_playback(handler)
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_raises", [False, True])
async def test_unconfirmed_exact_file_stop_retains_owner_for_successful_retry(
    tmp_path, monkeypatch, stop_raises
) -> None:
    artifact = tmp_path / "retry.wav"
    artifact.write_bytes(b"RIFF-retry")
    states = []
    lifecycle = _lifecycle("message-1", states)
    handler = _new_handler()

    class Player(ConsoleFilePlayer):
        def stop(self, *, expected_token):
            if not self.can_stop and stop_raises:
                self.stops.append(expected_token)
                raise RuntimeError("player stop failed")
            return super().stop(expected_token=expected_token)

    player = Player(PlaybackState.PLAYING)
    player.can_stop = False
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    try:
        await _dispatch_file(handler, artifact, lifecycle)
        await wait_for_condition(lambda: lifecycle.state == "playing")
        (attempt,) = handler._speech_outputs

        assert await _stop_file(handler, lifecycle) == [False]
        await drain_handler_playback(handler)
        assert states == ["playing"]
        assert handler.app.audio_output.blocked
        assert handler._speech_outputs == [attempt]
        assert handler._audio_files[lifecycle.message_id] == artifact
        assert artifact.exists()
        assert player.stops == [player.token]

        player.can_stop = True
        assert await _stop_file(handler, lifecycle) == [True]
        await wait_for_condition(
            lambda: lifecycle.message_id not in handler._audio_files
        )
        assert not artifact.exists()
        assert player.stops == [player.token, player.token]
        assert states == ["playing", "stopped"]
        assert handler.app.audio_output.owner is None
        assert handler._speech_outputs == []
        assert lifecycle.message_id not in handler._audio_files
        assert lifecycle.message_id not in handler._audio_file_owners
    finally:
        player.can_stop = True
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_file_takes_over_stream_and_bare_stop_retires_the_current_file(
    tmp_path, monkeypatch
) -> None:
    stream_states, file_states = [], []
    stream_owner = _lifecycle("stream", stream_states)
    file_owner = _lifecycle("file", file_states, request_id=2)
    artifact = tmp_path / "file.wav"
    artifact.write_bytes(b"RIFF")
    handler = _new_handler()
    sink, stream, finish = await _start_test_stream(handler, monkeypatch, stream_owner)

    class Player(ConsoleFilePlayer):
        def play_prepared(self, plan, **kwargs):
            assert sink.state == "stopped"
            return super().play_prepared(plan, **kwargs)

    player = Player(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    try:
        await _dispatch_file(handler, artifact, file_owner)
        await wait_for_condition(lambda: file_owner.state == "playing")
        finish.set()
        await stream
        assert stream_states == ["playing", "stopped"]
        assert player.stops == []

        assert await _stop_file(handler, None) == [True]
        await drain_handler_playback(handler)
        assert file_states == ["playing", "stopped"]
        assert player.stops == [player.token]
        assert handler.app.audio_output.owner is None
        assert handler._speech_outputs == []
    finally:
        finish.set()
        await stream
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_exact_stop_during_owned_play_lock_wait_is_accepted(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "pending.wav"
    artifact.write_bytes(b"RIFF")
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    handler = _new_handler()
    handler._audio_files[lifecycle.message_id] = artifact
    player = ConsoleFilePlayer()
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    play_outcomes: list[bool] = []
    stop_outcomes: list[bool] = []
    await handler._audio_files_lock.acquire()
    try:
        play_task = asyncio.create_task(
            handler.handle_tts_playback(
                TTSPlaybackEvent(
                    action="play",
                    message_id=lifecycle.message_id,
                    playback_lifecycle=lifecycle,
                    outcome_callback=play_outcomes.append,
                )
            )
        )
        await asyncio.sleep(0)
        reserved_before_audio_lookup = any(
            attempt.lifecycle is lifecycle for attempt in handler._speech_outputs
        )
        stop_task = asyncio.create_task(
            handler.handle_tts_playback(
                TTSPlaybackEvent(
                    action="stop",
                    message_id=lifecycle.message_id,
                    playback_lifecycle=lifecycle,
                    outcome_callback=stop_outcomes.append,
                )
            )
        )
        await asyncio.sleep(0)
    finally:
        handler._audio_files_lock.release()

    await asyncio.gather(play_task, stop_task)
    await drain_handler_playback(handler)

    assert reserved_before_audio_lookup is True
    assert play_outcomes == [False]
    assert stop_outcomes == [True]
    assert player.paths == []
    assert lifecycle.state == "stopped"


@pytest.mark.asyncio
async def test_streaming_playback_reports_device_start_then_natural_drain(
    monkeypatch,
) -> None:
    from tldw_chatbook.Audio.streaming_sink import (
        PumpResult,
        SinkDrained,
        SinkStarted,
    )
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
        TTSPlaybackLifecycle,
    )

    async def fake_pump(sink, *_args, **_kwargs):
        await asyncio.to_thread(sink.on_event, SinkStarted())
        sink.state = "stopped"
        sink.on_event(SinkDrained())
        return PumpResult(outcome="drained", bytes_fed=4)

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _RecordingSink)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.pump",
        fake_pump,
    )
    states: list[str] = []
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    handler = _new_handler()
    handler._post_tts_message = AsyncMock(return_value=True)

    async def chunks():
        yield b"1234"

    outcome = await handler._stream_response_via_sink(
        SimpleNamespace(
            sample_rate=24_000,
            channels=1,
            skip_bytes=0,
            data_bytes=None,
        ),
        chunks(),
        message_id="message-1",
        playback_lifecycle=lifecycle,
    )

    assert outcome == "success"
    assert states == ["playing", "stopped"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pump_outcome", "expected_state"),
    [("stopped", "stopped"), ("failed", "failed")],
)
async def test_streaming_terminal_outcome_settles_lifecycle(
    pump_outcome,
    expected_state,
    monkeypatch,
) -> None:
    from tldw_chatbook.Audio.streaming_sink import PumpResult, SinkStarted

    async def fake_pump(sink, *_args, **_kwargs):
        await asyncio.to_thread(sink.on_event, SinkStarted())
        return PumpResult(outcome=pump_outcome, bytes_fed=4, reason="test")

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _RecordingSink)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.pump",
        fake_pump,
    )
    states: list[str] = []
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    handler = _new_handler()
    handler._post_tts_message = AsyncMock(return_value=True)

    async def chunks():
        yield b"1234"

    await handler._stream_response_via_sink(
        SimpleNamespace(
            sample_rate=24_000,
            channels=1,
            skip_bytes=0,
            data_bytes=None,
        ),
        chunks(),
        message_id="message-1",
        playback_lifecycle=lifecycle,
    )

    assert states == ["playing", expected_state]
    assert lifecycle.state == expected_state
    assert handler._speech_outputs == []
    assert handler.app.audio_output.owner is None


@pytest.mark.asyncio
async def test_streaming_playback_cancellation_releases_exact_owner(
    monkeypatch,
) -> None:
    from tldw_chatbook.Audio.streaming_sink import SinkStarted

    async def cancelled_pump(sink, *_args, **_kwargs):
        sink.on_event(SinkStarted())
        raise asyncio.CancelledError

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _RecordingSink)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.TTS_Events.tts_events.pump",
        cancelled_pump,
    )
    lifecycle = TTSPlaybackLifecycle(
        message_id="message-1",
        request_id=1,
        validator=lambda: True,
        callback=lambda _state: None,
    )
    handler = _new_handler()

    async def chunks():
        yield b"1234"

    with pytest.raises(asyncio.CancelledError):
        await handler._stream_response_via_sink(
            SimpleNamespace(
                sample_rate=24_000,
                channels=1,
                skip_bytes=0,
                data_bytes=None,
            ),
            chunks(),
            message_id="message-1",
            playback_lifecycle=lifecycle,
        )

    assert handler._speech_outputs == []
    assert handler.app.audio_output.owner is None


@pytest.mark.asyncio
async def test_app_unavailable_notice_rejection_still_settles_once() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    app = _FakeApp()
    app._ensure_tts_handler = AsyncMock(return_value=None)
    app.post_message = MagicMock(side_effect=RuntimeError("queue closed"))

    await TldwCli.handle_tts_message_speech_request_event(app, event)
    event.report_outcome(True)

    assert outcomes == [False]


@pytest.mark.asyncio
async def test_app_unavailable_notice_false_acceptance_settles_without_type_error() -> (
    None
):
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )
    outcomes: list[bool] = []
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
        outcome_callback=outcomes.append,
    )
    app = _FakeApp()
    app._ensure_tts_handler = AsyncMock(return_value=None)
    app.post_message = MagicMock(return_value=False)

    await TldwCli.handle_tts_message_speech_request_event(app, event)

    assert outcomes == [False]
    assert "TypeError" not in repr(app.loguru_logger.method_calls)


def test_auto_speak_outcome_callback_must_be_callable() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Ready.",
    )

    with pytest.raises(ValueError, match="outcome_callback"):
        TTSMessageSpeechRequestEvent(
            store.issue_tts_message_speech_snapshot(message.id),
            store.validate_tts_message_speech_snapshot,
            outcome_callback=False,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_effective_openai_destination_is_versioned_and_sanitized() -> None:
    app_tts = {
        "default_provider": "openai",
        "default_model_mode": "exact",
        "default_model": "pocket-tts",
        "default_voice_mode": "exact",
        "default_voice": "alba",
        "default_format": "wav",
        "default_speed": 1.0,
        "OPENAI_BASE_URL": "http://127.0.0.1:8765/v1/audio/speech",
        "OPENAI_AUTH_MODE": "none",
    }
    service = build_default_tts_service(
        {
            "COMPREHENSIVE_CONFIG_RAW": {"app_tts": app_tts},
            "APP_TTS_CONFIG": app_tts,
        }
    )
    handler = _new_handler()
    handler._tts_service = service

    try:
        destination = await handler.resolve_console_speech_destination(None, None)
    finally:
        await service.close()
        await service.wait_closed()

    assert destination is not None
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", destination.fingerprint)
    assert destination.provider_label == "OpenAI"
    assert destination.sanitized_destination == "http://127.0.0.1:8765"
    assert destination.charges_may_apply is False
    assert "speech" not in repr(destination)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "app_tts", "expected"),
    [
        (
            "openai",
            {"OPENAI_BASE_URL": "https://voice.example.test/v1/audio/speech"},
            "https://voice.example.test/v1/audio/speech",
        ),
        (
            "alltalk",
            {"ALLTALK_TTS_URL_DEFAULT": "https://alltalk.example.test/api/tts"},
            "https://alltalk.example.test/api/tts",
        ),
        (
            "alltalk",
            {
                "ALLTALK_TTS_URL": "https://runtime-alltalk.example.test",
                "ALLTALK_TTS_URL_DEFAULT": "https://saved-alltalk.example.test",
            },
            "https://runtime-alltalk.example.test",
        ),
        ("elevenlabs", {}, "https://api.elevenlabs.io"),
        ("kokoro", {}, "http://localhost"),
        ("chatterbox", {}, "http://localhost"),
        ("higgs", {}, "http://localhost"),
    ],
)
async def test_provider_destination_uses_applied_network_configuration(
    provider_id: str,
    app_tts: dict[str, str],
    expected: str,
) -> None:
    service = MagicMock()
    service.registry.provider_configuration_snapshot = AsyncMock(
        return_value=SimpleNamespace(
            applied_config={"app_config": {"app_tts": app_tts}}
        )
    )

    endpoint = await TTSEventHandler._effective_provider_endpoint(
        service,
        provider_id,
    )

    assert endpoint == expected


@pytest.mark.parametrize(
    ("applied_config", "expected"),
    [
        (
            {"mode": "external", "base_url": "https://audio.example.test:9443"},
            "https://audio.example.test:9443",
        ),
        (
            {"mode": "managed", "base_url": "https://dormant.invalid"},
            "http://localhost",
        ),
    ],
)
def test_audio_cpp_admitted_destination_uses_active_mode(
    applied_config: dict[str, str],
    expected: str,
) -> None:
    assert (
        TTSEventHandler._provider_endpoint_from_applied_config(
            "audio_cpp",
            applied_config,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_destination_authorization_uses_post_capacity_exact_lease_config() -> (
    None
):
    adapters: list[FakeAdapter] = []

    def factory(config: Mapping[str, Any]) -> FakeAdapter:
        adapter = FakeAdapter("openai")
        endpoint = config["app_config"]["app_tts"]["OPENAI_BASE_URL"]
        adapter.admitted_outbound_endpoint = (  # type: ignore[attr-defined]
            lambda: endpoint
        )
        adapters.append(adapter)
        return adapter

    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor("openai", "OpenAI", False),
                factory=factory,
                initial_config={
                    "generation": "one",
                    "app_config": {
                        "app_tts": {"OPENAI_BASE_URL": "https://a.example/v1"}
                    },
                },
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        max_concurrent_operations=1,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="model",
            voice_mode="exact",
            voice_id="default",
            response_format="wav",
            speed=1.0,
        ),
    )
    first = await service.synthesize_default(text="occupy capacity")
    authorization_started = asyncio.Event()
    observed: list[tuple[str, str]] = []

    def authorize(provider_id: str, endpoint: str) -> bool:
        observed.append((provider_id, endpoint))
        authorization_started.set()
        return endpoint == "https://a.example/v1"

    waiting = asyncio.create_task(
        service.synthesize_default(
            text="must not reach changed backend",
            admission_authorizer=authorize,
        )
    )
    await asyncio.sleep(0)
    assert not authorization_started.is_set()
    await registry.reconfigure_provider(
        "openai",
        {
            "generation": "two",
            "app_config": {"app_tts": {"OPENAI_BASE_URL": "https://b.example/v1"}},
        },
    )
    await first.aclose()

    with pytest.raises(TTSConfigurationRevisionError):
        await waiting

    assert observed[0][0] == "openai"
    assert observed[0][1] == "https://b.example/v1"
    assert adapters[-1].synthesize_calls == 0
    await service.close()
    await service.wait_closed()


def test_destination_fingerprint_includes_provider_and_normalized_endpoint() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "https://Voice.Example.test:443/v1/audio/speech"
    )

    openai = openai_destination_fingerprint("openai", endpoint)
    alltalk = openai_destination_fingerprint("alltalk", endpoint)

    assert openai != alltalk
    assert endpoint.origin == "https://voice.example.test"


@pytest.mark.asyncio
async def test_app_routes_global_override_decision_to_existing_handler() -> None:
    handler = MagicMock()
    handler.handle_tts_global_override_decision = AsyncMock()
    fake_app = _FakeApp()
    fake_app._ensure_tts_handler = AsyncMock(return_value=handler)
    event = TTSGlobalOverrideDecisionEvent("b" * 32, accepted=True)

    await TldwCli.handle_tts_global_override_decision_event(fake_app, event)

    handler.handle_tts_global_override_decision.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_console_speak_autoplay_when_no_legacy_widget_claims_message(tmp_path):
    """No legacy widget claims the message id (the Console case) -> the
    handler must post TTSPlaybackEvent(action="play") itself after the native
    complete-WAV artifact has been published."""
    audio_file = tmp_path / "clip.wav"
    complete_wav = b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 32
    audio_file.write_bytes(complete_wav)

    fake_app = _FakeApp(widgets=())
    event = TTSCompleteEvent(message_id="console-msg-1", audio_file=audio_file)

    await TldwCli.handle_tts_complete_event(fake_app, event)

    playback_events = [m for m in fake_app.posted if isinstance(m, TTSPlaybackEvent)]
    assert len(playback_events) == 1
    assert playback_events[0].action == "play"
    assert playback_events[0].message_id == "console-msg-1"
    assert audio_file.read_bytes() == complete_wav


class _FakeConsoleScreen:
    """Stand-in for the Console screen's speak-state surface (TASK-15422)."""

    def __init__(self, speaking_message_id: str | None) -> None:
        self._console_speaking_message_id = speaking_message_id
        self.sync_calls = 0

    async def _sync_native_console_chat_ui(self) -> None:
        """Count resyncs so the test can assert the row was repainted.

        Returns:
            None.
        """
        self.sync_calls += 1


@pytest.mark.asyncio
async def test_error_completion_clears_console_speaking_marker() -> None:
    """A failed generation must return the Console action row to 🔊.

    The error branch reset legacy `ChatMessage` widgets only; the Console
    transcript's action row renders from `_console_speaking_message_id`,
    which nothing cleared on failure — so after an instant failure the row
    kept "⏹ Stop speech" with no speech to stop (TASK-15422, observed live
    during the TASK-15420 UAT).
    """
    speaking = _FakeConsoleScreen("console-msg-1")
    other = _FakeConsoleScreen("console-msg-2")
    fake_app = _FakeApp(widgets=())
    fake_app.screen_stack = (other, speaking)
    event = TTSCompleteEvent(
        message_id="console-msg-1",
        error="The selected TTS model is not available",
    )

    await TldwCli.handle_tts_complete_event(fake_app, event)

    assert speaking._console_speaking_message_id is None
    assert speaking.sync_calls == 1
    # A screen speaking a DIFFERENT message keeps its own state untouched.
    assert other._console_speaking_message_id == "console-msg-2"
    assert other.sync_calls == 0


@pytest.mark.asyncio
async def test_adhoc_completion_autoplays_and_audio_is_cached_under_adhoc(tmp_path):
    """Regression pin for PR #850 review finding #2 (disproven): a
    `TTSRequestEvent(message_id=None)` does NOT orphan the auto-play path.

    `TTSEventHandler.handle_tts_request` normalizes ``message_id = event.
    message_id or "adhoc"`` *before* generation, so `_generate_tts` caches the
    audio under the truthy key ``"adhoc"`` and the completion event carries the
    same key -- `handle_tts_playback`'s ``_audio_files.get(event.message_id)``
    therefore resolves. This test pins both halves: (a) the app handler
    auto-plays an ``"adhoc"`` completion when no legacy widget claims it, and
    (b) the TTS handler's playback lookup finds audio cached under ``"adhoc"``.
    """
    audio_file = tmp_path / "adhoc.mp3"
    audio_file.write_bytes(b"fake-audio-bytes")

    # (a) app-level: adhoc completion with no legacy widget -> auto-play posted.
    fake_app = _FakeApp(widgets=())
    event = TTSCompleteEvent(message_id="adhoc", audio_file=audio_file)
    await TldwCli.handle_tts_complete_event(fake_app, event)
    playback_events = [m for m in fake_app.posted if isinstance(m, TTSPlaybackEvent)]
    assert len(playback_events) == 1
    assert playback_events[0].message_id == "adhoc"

    # (b) handler-level: the "adhoc" cache key resolves in handle_tts_playback's
    # lookup table (the normalization upstream guarantees it was cached there).
    handler = _new_handler()
    handler._audio_files = {"adhoc": audio_file}
    assert handler._audio_files.get(playback_events[0].message_id) == audio_file


@pytest.mark.asyncio
async def test_no_autoplay_and_no_click_notify_on_tts_error(tmp_path):
    """Regression pin: an error completion neither auto-plays nor claims
    success, regardless of legacy widget presence."""
    fake_app = _FakeApp(widgets=())
    event = TTSCompleteEvent(message_id="console-msg-2", error="synthesis failed")

    await TldwCli.handle_tts_complete_event(fake_app, event)

    playback_events = [m for m in fake_app.posted if isinstance(m, TTSPlaybackEvent)]
    assert playback_events == []
    fake_app.notify.assert_called_once_with(
        "TTS failed: synthesis failed", severity="error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", (True, False))
async def test_resolution_error_prompts_and_returns_exact_override_decision(
    accepted: bool,
) -> None:
    fake_app = _FakeApp(widgets=())
    fake_app.push_screen_wait.return_value = accepted
    token = "a" * 32
    event = TTSCompleteEvent(
        message_id="console-msg-override",
        error="Character voice profiles are unavailable.",
        global_override_token=token,
    )

    await TldwCli.handle_tts_complete_event(fake_app, event)
    await asyncio.gather(*fake_app.worker_tasks)

    decisions = [
        message
        for message in fake_app.posted
        if isinstance(message, TTSGlobalOverrideDecisionEvent)
    ]
    assert len(decisions) == 1
    assert decisions[0].token == token
    assert decisions[0].accepted is accepted
    fake_app.push_screen_wait.assert_awaited_once()
    assert not any(isinstance(message, TTSPlaybackEvent) for message in fake_app.posted)


@pytest.mark.asyncio
async def test_offer_global_override_names_character_domain_accurately() -> None:
    """Review round 2: a character-domain refusal keeps its existing copy."""
    fake_app = _FakeApp(widgets=())
    handler = MagicMock()
    handler.peek_global_override_voice_domain = MagicMock(return_value="character")
    fake_app._tts_handler = handler
    token = "d" * 32

    await fake_app._offer_tts_global_override(token)

    handler.peek_global_override_voice_domain.assert_called_once_with(token)
    dialog = fake_app.push_screen_wait.await_args.args[0]
    assert "character" in dialog.message.lower()
    fake_app.push_screen_wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_offer_global_override_names_default_profile_domain_not_character() -> (
    None
):
    """Review round 2 (the Critical this fixes): a default-profile refusal's
    dialog must never say "character" -- it fires for messages with no
    character context at all."""
    fake_app = _FakeApp(widgets=())
    handler = MagicMock()
    handler.peek_global_override_voice_domain = MagicMock(
        return_value="default_profile"
    )
    fake_app._tts_handler = handler
    token = "e" * 32

    await fake_app._offer_tts_global_override(token)

    dialog = fake_app.push_screen_wait.await_args.args[0]
    assert "character" not in dialog.message.lower()
    assert "default voice profile" in dialog.message.lower()


@pytest.mark.asyncio
async def test_offer_global_override_falls_back_to_neutral_copy_when_domain_unknown() -> (
    None
):
    """No bound handler (or an unknown/expired token) must not claim
    "character" either -- it genuinely does not know."""
    fake_app = _FakeApp(widgets=())
    assert fake_app._tts_handler is None
    token = "f" * 32

    await fake_app._offer_tts_global_override(token)

    dialog = fake_app.push_screen_wait.await_args.args[0]
    assert "character" not in dialog.message.lower()
    assert "global tts voice" in dialog.message.lower()


@pytest.mark.asyncio
async def test_resolution_prompt_failure_returns_decline_without_disclosure() -> None:
    fake_app = _FakeApp(widgets=())
    secret = "PRIVATE_DIALOG_FAILURE"
    fake_app.push_screen_wait.side_effect = RuntimeError(secret)
    token = "c" * 32
    event = TTSCompleteEvent(
        message_id="console-msg-override",
        error="Character voice profiles are unavailable.",
        global_override_token=token,
    )

    await TldwCli.handle_tts_complete_event(fake_app, event)
    await asyncio.gather(*fake_app.worker_tasks)

    decisions = [
        message
        for message in fake_app.posted
        if isinstance(message, TTSGlobalOverrideDecisionEvent)
    ]
    assert [(decision.token, decision.accepted) for decision in decisions] == [
        (token, False)
    ]
    assert secret not in repr(fake_app.loguru_logger.method_calls)
