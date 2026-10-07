"""Task 14 (F18d): TTS message-widget index replaces full-DOM scans.

The TTS progress/completion handlers in ``tldw_chatbook/app_speech.py``
previously located the message widget for an event by running
``list(app.query(ChatMessage)) + list(app.query(ChatMessageEnhanced))`` --
two full-DOM walks per handler -- followed by a linear
``message_id_internal`` scan. Neither widget type is ever instantiated in
production today, so every walk was pure overhead per TTS event.

The replacement is a module-level index
(``tldw_chatbook.Widgets.Chat_Widgets.tts_widget_index``): both widget
classes register themselves in ``on_mount`` and unregister in
``on_unmount``; the handlers consult
``tts_widget_index.get_message_widgets(event.message_id)`` -- an O(1)
dict lookup that performs no DOM work at all when the registry is empty
(the production case).

Test level: the handler-facing tests call the unbound ``TldwCli`` handler
functions directly (the same seam Textual's ``@on`` dispatch drives) --
against a query-counting stand-in for the no-widget case, and against a
minimal real Textual ``App`` hosting the actual widgets for the
mounted-widget cases. ``recovered_root``/``recovered_profile`` are always
passed explicitly so widget construction never touches the real user data
dir or the recovery fence.
"""

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import MagicMock

import pytest
from textual.app import App

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSPlaybackEvent,
    TTSProgressEvent,
)
from tldw_chatbook.Widgets.Chat_Widgets import tts_widget_index
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import (
    ChatMessageEnhanced,
)


@pytest.fixture(autouse=True)
def _isolated_index() -> Any:
    """Snapshot and clear the module-level registry around every test."""
    saved = dict(tts_widget_index._MESSAGE_WIDGETS)
    tts_widget_index._MESSAGE_WIDGETS.clear()
    try:
        yield tts_widget_index
    finally:
        tts_widget_index._MESSAGE_WIDGETS.clear()
        tts_widget_index._MESSAGE_WIDGETS.update(saved)


class _QuerySpyApp:
    """Minimal stand-in exposing (and counting) only what the TTS handlers touch."""

    def __init__(self) -> None:
        self.query_calls = 0
        self.loguru_logger = MagicMock()
        self.notify = MagicMock()
        self.posted: list[Any] = []

    def query(self, selector: Any) -> list[Any]:
        self.query_calls += 1
        return []

    def post_message(self, message: Any) -> bool:
        self.posted.append(message)
        return True


class _WidgetHost(App[None]):
    """Real Textual app hosting pre-built chat message widgets."""

    def __init__(self, *widgets: Any) -> None:
        super().__init__()
        self.loguru_logger = MagicMock()
        self._hosted = list(widgets)
        self.notifications: list[str] = []

    def compose(self) -> Any:
        yield from self._hosted

    def notify(self, message: str = "", **_kwargs: Any) -> None:
        self.notifications.append(message)
        super().notify(message, **_kwargs)


def _enhanced(message_id: str, tmp_path: Any) -> ChatMessageEnhanced:
    """Build a fence-free ChatMessageEnhanced for the given message id."""
    return ChatMessageEnhanced(
        message="hello",
        role="AI",
        message_id=message_id,
        recovered_root=tmp_path,
        recovered_profile="tts-index-test",
    )


# (a) The production hot path: no registered widget -> zero DOM queries.


@pytest.mark.asyncio
async def test_tts_events_with_no_registered_widget_issue_zero_dom_queries(
    tmp_path: Any,
) -> None:
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVE" + b"\x00" * 32)
    app = _QuerySpyApp()

    for _ in range(5):
        await TldwCli.handle_tts_progress_event(
            app,
            TTSProgressEvent(
                message_id="console-msg-1",
                progress=0.42,
                status="Generating audio",
            ),
        )
    for _ in range(5):
        await TldwCli.handle_tts_complete_event(
            app,
            TTSCompleteEvent(message_id="console-msg-1", audio_file=audio_file),
        )

    assert app.query_calls == 0
    # Behavior preserved: unclaimed completions still auto-play (Console case).
    plays = [m for m in app.posted if isinstance(m, TTSPlaybackEvent)]
    assert len(plays) == 5
    assert all(play.action == "play" for play in plays)
    app.notify.assert_not_called()


# (b) A registered (mounted) widget receives its TTS state update.


@pytest.mark.asyncio
async def test_mounted_registered_widget_receives_its_tts_state_update(
    tmp_path: Any,
) -> None:
    audio_file = tmp_path / "ready.wav"
    audio_file.write_bytes(b"RIFF")
    widget = _enhanced("legacy-msg-1", tmp_path)
    host = _WidgetHost(widget)

    async with host.run_test():
        assert tts_widget_index.get_message_widgets("legacy-msg-1") == (widget,)

        await TldwCli.handle_tts_progress_event(
            host,
            TTSProgressEvent(
                message_id="legacy-msg-1", progress=0.5, status="mid"
            ),
        )
        assert widget.tts_progress == 0.5

        await TldwCli.handle_tts_complete_event(
            host,
            TTSCompleteEvent(message_id="legacy-msg-1", audio_file=audio_file),
        )
        assert widget.tts_state == "ready"
        assert widget.tts_audio_file == audio_file
        # The legacy "click play to listen" notice fired, not auto-play.
        assert host.notifications == ["TTS audio ready - click play to listen"]


# (c) Unmount removes the registration (lifecycle hook fires).


@pytest.mark.asyncio
async def test_unmount_removes_the_registration(tmp_path: Any) -> None:
    plain = ChatMessage(message="bye", role="User", message_id="legacy-msg-2")
    enhanced = _enhanced("legacy-msg-2", tmp_path)
    host = _WidgetHost(plain, enhanced)

    async with host.run_test():
        registered = tts_widget_index.get_message_widgets("legacy-msg-2")
        assert frozenset(id(w) for w in registered) == {id(plain), id(enhanced)}

        await plain.remove()
        # Identity-based removal: the enhanced sibling keeps its entry.
        assert tts_widget_index.get_message_widgets("legacy-msg-2") == (enhanced,)

        await enhanced.remove()
        assert tts_widget_index.get_message_widgets("legacy-msg-2") == ()

    # A progress event after the unmount must not find (or update) the widget.
    app = _QuerySpyApp()
    await TldwCli.handle_tts_progress_event(
        app,
        TTSProgressEvent(message_id="legacy-msg-2", progress=0.9, status="late"),
    )
    assert enhanced.tts_progress != 0.9


# (d) Two widgets sharing a message id both receive updates.


@pytest.mark.asyncio
async def test_two_widgets_sharing_a_message_id_both_receive_updates(
    tmp_path: Any,
) -> None:
    first = _enhanced("shared-msg-1", tmp_path)
    second = _enhanced("shared-msg-1", tmp_path)
    host = _WidgetHost(first, second)

    async with host.run_test():
        registered = tts_widget_index.get_message_widgets("shared-msg-1")
        assert frozenset(id(w) for w in registered) == {id(first), id(second)}

        await TldwCli.handle_tts_progress_event(
            host,
            TTSProgressEvent(
                message_id="shared-msg-1", progress=0.75, status="Generating"
            ),
        )
        assert first.tts_progress == 0.75
        assert second.tts_progress == 0.75


# (e) Weakref backstop: a missed unregister cannot resurrect a dead widget.


def test_weakref_backstop_never_resurrects_a_dead_widget() -> None:
    widget = ChatMessage(message="gone", role="User", message_id="dead-msg-1")
    tts_widget_index.register_message_widget("dead-msg-1", widget)
    assert tts_widget_index.get_message_widgets("dead-msg-1") == (widget,)

    # Simulate a missed on_unmount: drop every strong reference and collect.
    del widget
    gc.collect()

    assert tts_widget_index.get_message_widgets("dead-msg-1") == ()
    # The dead key is pruned from the registry, not just filtered on read.
    assert "dead-msg-1" not in tts_widget_index._MESSAGE_WIDGETS


# Registry contract unit tests.


def test_registration_is_idempotent_per_instance() -> None:
    widget = ChatMessage(message="dup", role="User", message_id="dup-msg-1")
    tts_widget_index.register_message_widget("dup-msg-1", widget)
    tts_widget_index.register_message_widget("dup-msg-1", widget)
    assert tts_widget_index.get_message_widgets("dup-msg-1") == (widget,)


def test_unregister_removes_only_the_matching_instance() -> None:
    kept = ChatMessage(message="keep", role="User", message_id="pair-msg-1")
    removed = ChatMessage(message="drop", role="User", message_id="pair-msg-1")
    tts_widget_index.register_message_widget("pair-msg-1", kept)
    tts_widget_index.register_message_widget("pair-msg-1", removed)

    tts_widget_index.unregister_message_widget("pair-msg-1", removed)

    assert tts_widget_index.get_message_widgets("pair-msg-1") == (kept,)


def test_missing_message_id_is_never_registered() -> None:
    widget = ChatMessage(message="anon", role="User")
    tts_widget_index.register_message_widget(None, widget)
    tts_widget_index.register_message_widget("", widget)
    assert tts_widget_index.get_message_widgets(None) == ()
    assert tts_widget_index.get_message_widgets("") == ()
    assert tts_widget_index._MESSAGE_WIDGETS == {}


def test_unregister_without_prior_registration_is_a_no_op() -> None:
    widget = ChatMessage(message="ghost", role="User", message_id="ghost-msg-1")
    tts_widget_index.unregister_message_widget("ghost-msg-1", widget)
    tts_widget_index.unregister_message_widget(None, widget)
    assert tts_widget_index._MESSAGE_WIDGETS == {}
