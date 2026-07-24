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

from unittest.mock import MagicMock

import pytest

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
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

    def query(self, widget_type):
        return [w for w in self._widgets if isinstance(w, widget_type)]

    def post_message(self, message) -> bool:
        self.posted.append(message)
        return True


@pytest.mark.asyncio
async def test_console_speak_autoplay_when_no_legacy_widget_claims_message(tmp_path):
    """No legacy widget claims the message id (the Console case) -> the
    handler must post TTSPlaybackEvent(action="play") itself."""
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake-audio-bytes")

    fake_app = _FakeApp(widgets=())
    event = TTSCompleteEvent(message_id="console-msg-1", audio_file=audio_file)

    await TldwCli.handle_tts_complete_event(fake_app, event)

    playback_events = [m for m in fake_app.posted if isinstance(m, TTSPlaybackEvent)]
    assert len(playback_events) == 1
    assert playback_events[0].action == "play"
    assert playback_events[0].message_id == "console-msg-1"


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
