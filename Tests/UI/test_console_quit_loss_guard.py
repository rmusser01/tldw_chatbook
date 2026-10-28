"""Closing a tab names its parked unsent turns (TASK-33628.1).

Closing one tab already asked about its unsaved messages, unsent draft and
pending attachments, but not about a parked "Unsent turn needs attention"
recovery, which the close also discards.

These drive the real ``TldwCli`` on the real ``ChatScreen``, its real
controller, store and ``ConsoleRuntime``, and press the real keys.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_SETTLE_SECONDS = 20.0
_TEMPORARY_PROMPT = "temporary prompt beta"


async def _until(pilot, predicate, what: str, timeout: float = _SETTLE_SECONDS):
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


def _app_with_quit_recorder(monkeypatch, events: list[str]):
    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)

    async def _record_quit() -> None:
        # Stands in for the irreversible shutdown, as the other quit tests do.
        events.append("quit")

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_quit)
    return app


async def _mounted_console(app, pilot):
    await _until(
        pilot,
        lambda: (
            type(app.screen).__name__ == "ChatScreen"
            and bool(app.screen.query("#console-native-composer"))
        ),
        "the Console composer",
    )
    await pilot.pause(0.2)
    console = app.screen
    assert console._console_runtime() is app.console_runtime
    return console


async def _type_into_composer(pilot, console, text: str) -> ConsoleComposerBar:
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await pilot.pause()
    await pilot.press(*text)
    await _until(pilot, lambda: composer.draft_text() == text, "the typed draft")
    return composer


async def _temporary_chat_with_parked_turn(pilot, app, console) -> str:
    """Open a Temporary tab and send in it; the send parks as an unsent turn.

    A Temporary chat cannot store the send's durable capture, so the
    controller pauses the send before the provider and the runtime parks the
    turn on the "Unsent turn needs attention" shelf.
    """
    store = console._ensure_console_chat_store()
    before = {session.id for session in store.sessions()}
    await pilot.click("#console-new-temporary-tab")
    await _until(
        pilot,
        lambda: (
            store.active_session_id not in before
            and store.active_session_id is not None
        ),
        "the Temporary tab to open",
    )
    session_id = store.active_session_id
    await _type_into_composer(pilot, console, _TEMPORARY_PROMPT)
    await pilot.press("enter")
    runtime = app.console_runtime
    await _until(
        pilot,
        lambda: len(runtime.recoveries_for_session(session_id)) == 1,
        "the Temporary send to park as an unsent turn",
    )
    await pilot.pause(0.3)
    return session_id


async def test_tab_close_reports_a_parked_unsent_turn(monkeypatch):
    """AC#3: closing the Temporary tab names its parked unsent turn; Stay keeps it."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        temporary_id = await _temporary_chat_with_parked_turn(pilot, app, console)

        await pilot.click(f"#console-close-session-tab-{temporary_id}")
        await _until(
            pilot,
            lambda: (
                isinstance(app.screen, ConfirmationDialog)
                and str(app.screen.title).startswith("Close tab")
            ),
            "the close dialog",
        )
        close_dialog = app.screen
        assert "Unsent turns needing attention: 1" in close_dialog.message.splitlines()
        assert _TEMPORARY_PROMPT not in close_dialog.message
        await pilot.click(close_dialog.query_one("#cancel-button", Button))
        await _until(pilot, lambda: app.screen is console, "Stay on the close dialog")
        assert temporary_id in {session.id for session in store.sessions()}
        assert len(app.console_runtime.recoveries_for_session(temporary_id)) == 1
        assert events == []
