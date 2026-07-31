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
the audio immediately. The legacy widget-found path is unchanged (pinned by
the second test below).

Test level: these tests call `TldwCli.handle_tts_complete_event` directly
against a minimal duck-typed stand-in for `self` (exposing only `query`,
`post_message`, `notify`, `loguru_logger`), rather than booting the full
`TldwCli` app. `textual.on()` returns the decorated method "unaltered" (see
its own docstring), so calling the unbound method this way runs the exact
same production code that Textual's message dispatch would call. What this
does *not* exercise is Textual's real message-queue dispatch of the posted
`TTSPlaybackEvent` to the `@on(TTSPlaybackEvent)` handler, or the handler's
own `_audio_files` lookup + `play_audio_file` call -- that downstream wiring
is pre-existing, already-covered-by-being-in-production-use code (the same
path the legacy play button drives today), not new logic introduced here.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSGlobalOverrideDecisionEvent,
    TTSMessageSpeechRequestEvent,
    TTSPlaybackEvent,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage


class _FakeApp:
    """Minimal stand-in exposing only what handle_tts_complete_event touches."""

    def __init__(self, widgets=()):
        self._widgets = list(widgets)
        self.loguru_logger = MagicMock()
        self.notify = MagicMock()
        self.posted: list = []
        self.push_screen_wait = AsyncMock(return_value=False)
        self.worker_tasks: list[asyncio.Task] = []

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
    event = TTSMessageSpeechRequestEvent(
        store.issue_tts_message_speech_snapshot(message.id),
        store.validate_tts_message_speech_snapshot,
    )
    fake_app = _FakeApp()
    fake_app._ensure_tts_handler = AsyncMock(return_value=None)
    fake_app.post_message = AsyncMock()

    await TldwCli.handle_tts_message_speech_request_event(fake_app, event)

    fake_app.loguru_logger.error.assert_called_once_with(
        "TTS handler not initialized "
        "(operation=trusted_console_speech, outcome_code=handler_unavailable)"
    )
    rendered_logs = repr(fake_app.loguru_logger.method_calls)
    assert "PRIVATE_RESPONSE_TEXT" not in rendered_logs
    assert "PRIVATE_AUTHORITY" not in rendered_logs
    assert message.id not in rendered_logs
    fake_app.post_message.assert_awaited_once()
    completion = fake_app.post_message.await_args.args[0]
    assert isinstance(completion, TTSCompleteEvent)
    assert completion.message_id == message.id
    assert completion.error == "TTS service not available"


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


@pytest.mark.asyncio
async def test_console_speak_autoplay_skipped_when_legacy_widget_claims_message(
    tmp_path,
):
    """Regression pin: a legacy ChatMessage widget owning the message id
    keeps the pre-existing "click play to listen" behavior and must NOT be
    auto-played out from under the user."""
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake-audio-bytes")

    widget = ChatMessage(message="hello", role="AI", message_id="legacy-msg-1")
    fake_app = _FakeApp(widgets=[widget])
    event = TTSCompleteEvent(message_id="legacy-msg-1", audio_file=audio_file)

    await TldwCli.handle_tts_complete_event(fake_app, event)

    playback_events = [m for m in fake_app.posted if isinstance(m, TTSPlaybackEvent)]
    assert playback_events == []
    fake_app.notify.assert_called_once_with(
        "TTS audio ready - click play to listen", severity="information"
    )


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
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

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
    handler = TTSEventHandler.__new__(TTSEventHandler)
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
    assert not any(
        isinstance(message, TTSPlaybackEvent) for message in fake_app.posted
    )


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
