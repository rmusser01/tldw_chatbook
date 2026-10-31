"""Ctrl+Q asks before discarding unsent Console work (TASK-33628.1).

The app-level Console quit guard used to read only live runs, delegated
agents and queued prompts. An unsent composer draft, a Temporary chat's
messages, a staged attachment and a parked "Unsent turn needs attention"
recovery never raised its loss flag, so Ctrl+Q exited with no dialog and they
were gone on the next launch. Closing one tab already asked about drafts,
unsaved messages and attachments, but not parked turns either.

These drive the real ``TldwCli`` on the real ``ChatScreen``, its real
controller, store and ``ConsoleRuntime``, and press the real keys. Only the
irreversible shutdown at the end of an approved quit is replaced by a
recorder, as the other quit-flow tests do. Before the fix, each "asks" test
fails with "Ctrl+Q quit straight past ...": the quit ran without a dialog.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_SETTLE_SECONDS = 20.0
_DRAFT = "draft alpha unsent do not lose"
#: Longer than a tab label: the first send titles the tab with a cut-short
#: prefix of it, so the whole prompt must never reach a dialog.
_TEMPORARY_PROMPT = "temporary prompt beta, long enough that its tab label is cut short"
_NOTE = "Open tabs are not reopened next time. Saved chats stay in the Chats rail."


async def _until(pilot, predicate, what: str, timeout: float = _SETTLE_SECONDS):
    """Pump the app until ``predicate()`` holds, or fail naming ``what``."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError as exc:
        raise AssertionError(f"timed out waiting for {what}") from exc


def _quit_dialogs(app) -> list[ConfirmationDialog]:
    return [
        screen
        for screen in app.screen_stack
        if isinstance(screen, ConfirmationDialog) and screen.title == "Quit Chatbook?"
    ]


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


async def _ctrl_q_asks(pilot, app, events: list[str], what: str) -> ConfirmationDialog:
    """Press Ctrl+Q and wait for the quit dialog; fail if the app quit instead."""
    await pilot.press("ctrl+q")
    await _until(
        pilot,
        lambda: bool(_quit_dialogs(app)) or "quit" in events,
        what,
        timeout=5.0,
    )
    assert "quit" not in events, f"Ctrl+Q quit straight past {what}"
    (dialog,) = _quit_dialogs(app)
    assert app.screen is dialog
    return dialog


async def _stay(pilot, app, console) -> None:
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is console, "Stay to return to Console")
    await _until(
        pilot,
        lambda: app._quit_in_progress is False,
        "the quit guard to clear after Stay",
    )
    assert app.is_running
    assert app._shutting_down is False


def _message_rows(dialog: ConfirmationDialog) -> list[str]:
    return [row for row in dialog.message.splitlines() if row.strip()]


async def test_ctrl_q_with_only_a_composer_draft_asks_and_stay_keeps_it(monkeypatch):
    """AC#1/#2/#4/#5/#6: a lone typed draft is asked about; Stay keeps it."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        composer = await _type_into_composer(pilot, console, _DRAFT)

        dialog = await _ctrl_q_asks(pilot, app, events, "the unsent composer draft")

        rows = _message_rows(dialog)
        assert "Unsent drafts: 1" in rows
        title = console._session._session_close_display_title(session_id)
        assert f'Affected tabs: "{title}"' in rows
        assert _NOTE in rows
        # Only non-zero rows, and never the draft's content.
        assert not any(row.endswith(": 0") for row in rows), rows
        for absent in ("Live agent runs", "Delegated agents", "Unsent queued"):
            assert not any(row.startswith(absent) for row in rows), rows
        assert _DRAFT not in dialog.message
        assert "alpha" not in dialog.message

        await _stay(pilot, app, console)
        assert events == []
        assert composer.draft_text() == _DRAFT
        assert store.session_draft(session_id) == _DRAFT

        # The next Ctrl+Q asks again, and Quit then runs the shutdown.
        dialog = await _ctrl_q_asks(pilot, app, events, "the draft a second time")
        await pilot.click("#confirm-button")
        await _until(pilot, lambda: "quit" in events, "Quit to run the shutdown")


async def test_ctrl_q_from_another_screen_still_asks_about_a_console_draft(
    monkeypatch,
):
    """AC#1/#5: the guard is app-owned, so a draft left in Console still asks.

    The Console view is detached once Home is on screen; the quit inventory
    reads the draft the tab kept in the store, not a mounted composer.
    """
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        title = console._session._session_close_display_title(session_id)
        await _type_into_composer(pilot, console, _DRAFT)

        app.post_message(NavigateToScreen("home"))
        await _until(
            pilot,
            lambda: type(app.screen).__name__ == "HomeScreen",
            "the Home screen",
        )
        home = app.screen

        dialog = await _ctrl_q_asks(pilot, app, events, "the draft left in Console")

        rows = _message_rows(dialog)
        assert "Unsent drafts: 1" in rows
        assert f'Affected tabs: "{title}"' in rows
        assert _DRAFT not in dialog.message
        await pilot.press("escape")
        await _until(pilot, lambda: app.screen is home, "Stay to return to Home")
        await _until(
            pilot,
            lambda: app._quit_in_progress is False,
            "the quit guard to clear after Stay",
        )
        assert events == []
        assert store.session_draft(session_id) == _DRAFT


async def test_ctrl_q_with_nothing_to_lose_quits_without_asking(monkeypatch):
    """AC#1: an untouched Console tab quits at once, as it did before."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        await _mounted_console(app, pilot)
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: "quit" in events, "the quit to run", timeout=5.0)
        assert not _quit_dialogs(app)


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


async def test_ctrl_q_counts_a_temporary_chat_with_a_parked_unsent_turn(monkeypatch):
    """AC#2/#5/#6: Temporary messages and the parked turn are both counted."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = await _temporary_chat_with_parked_turn(pilot, app, console)
        unsaved = sum(
            message.persisted_message_id is None
            for message in store.all_messages_for_session(session_id)
        )

        dialog = await _ctrl_q_asks(
            pilot, app, events, "the Temporary chat's parked unsent turn"
        )

        rows = _message_rows(dialog)
        assert "Unsent turns needing attention: 1" in rows
        # The Temporary chat's own messages are never saved, so they count too.
        assert unsaved >= 1
        assert f"Temporary or unsaved messages: {unsaved}" in rows
        title = console._session._session_close_display_title(session_id)
        assert any(
            row.startswith("Affected tabs:") and f'"{title}"' in row for row in rows
        ), rows
        assert not any(row.endswith(": 0") for row in rows), rows
        # The tab's own label is the only text taken from the chat.
        assert _TEMPORARY_PROMPT not in dialog.message
        assert not any(
            "temporary prompt" in row
            for row in rows
            if not row.startswith("Affected tabs:")
        ), rows

        await _stay(pilot, app, console)
        assert events == []
        assert session_id in {session.id for session in store.sessions()}
        assert len(app.console_runtime.recoveries_for_session(session_id)) == 1


async def test_ctrl_q_with_only_a_pending_attachment_asks_and_stay_keeps_it(
    monkeypatch,
):
    """AC#1/#5: a staged attachment with an empty draft is asked about too."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        attachment = PendingAttachment(
            file_path="/sandbox/red-square.png",
            display_name="red-square.png",
            file_type="image",
            insert_mode="attachment",
            data=b"\x89PNG\r\n\x1a\n",
            mime_type="image/png",
            original_size=8,
            processed_size=8,
        )
        assert store.add_pending_attachment(session_id, attachment) is True
        await pilot.pause(0.2)

        dialog = await _ctrl_q_asks(pilot, app, events, "the staged attachment")

        rows = _message_rows(dialog)
        assert "Pending attachments: 1" in rows
        assert not any(row.startswith("Unsent draft") for row in rows), rows
        assert "red-square" not in dialog.message
        await _stay(pilot, app, console)
        assert events == []
        assert [
            item.attachment_id for item in store.pending_attachments(session_id)
        ] == [attachment.attachment_id]


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


async def test_quit_totals_equal_the_sum_of_every_tabs_close_impact(monkeypatch):
    """AC#3: three tabs with different losses; quit sums exactly what close counts."""
    events: list[str] = []
    app = _app_with_quit_recorder(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console = await _mounted_console(app, pilot)
        store = console._ensure_console_chat_store()
        first_id = store.active_session_id
        temporary_id = await _temporary_chat_with_parked_turn(pilot, app, console)
        # Two tabs that are not on screen: drafts and an attachment in the store.
        third = store.create_session(activate=False)
        store.set_session_draft(third.id, "third tab draft")
        store.set_session_draft(first_id, "first tab draft")
        store.add_pending_attachment(
            first_id,
            PendingAttachment(
                file_path="/sandbox/notes.png",
                display_name="notes.png",
                file_type="image",
                insert_mode="attachment",
                data=b"\x89PNG\r\n\x1a\n",
            ),
        )
        await pilot.pause(0.2)

        dialog = await _ctrl_q_asks(pilot, app, events, "three tabs of unsent work")

        close_impacts = [
            console._session._session_close_impact(session.id)
            for session in store.sessions()
        ]
        expected = {
            "Temporary or unsaved messages": sum(
                impact.transcript_message_count for impact in close_impacts
            ),
            "Unsent drafts": sum(impact.has_draft for impact in close_impacts),
            "Pending attachments": sum(
                impact.pending_attachment_count for impact in close_impacts
            ),
            "Unsent turns needing attention": sum(
                impact.unsent_turn_count for impact in close_impacts
            ),
            "Live agent runs": sum(
                impact.lifecycle.live_run_count for impact in close_impacts
            ),
        }
        assert expected["Unsent drafts"] >= 2
        assert expected["Pending attachments"] == 1
        assert expected["Unsent turns needing attention"] == 1
        rows = _message_rows(dialog)
        shown = {
            label: int(count)
            for label, _, count in (row.rpartition(": ") for row in rows)
            if label in expected
        }
        assert shown == {label: n for label, n in expected.items() if n}
        tabs_row = next(row for row in rows if row.startswith("Affected tabs:"))
        for session_id, impact in zip(
            (session.id for session in store.sessions()), close_impacts
        ):
            title = console._session._session_close_display_title(session_id)
            assert (f'"{title}"' in tabs_row) is impact.has_loss_risk, (
                title,
                tabs_row,
            )
        assert temporary_id in {session.id for session in store.sessions()}
        await _stay(pilot, app, console)
        assert events == []
