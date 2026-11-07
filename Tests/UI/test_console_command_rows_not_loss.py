"""Console command output and notices are not loss in close and quit (TASK-33628.16).

Owner decision, 2026-10-29: command output and notice rows (/help, /doctor,
error notices) do not count as loss in the tab-close and Ctrl+Q dialogs. They
never raise either question on their own, and beside real loss neither dialog
counts or lists them.

Console's own rows are SYSTEM transcript nodes the store never saves, and the
shared loss inventory counted every node with no saved id. So in a saved chat
with nothing else at stake, one /help made Ctrl+Q ask "Quit Chatbook?" with
"Temporary or unsaved messages: 1", and tab close asked the same.

These drive the real ``TldwCli`` on the real ``ChatScreen``, its real
controller, store, ``ConsoleRuntime`` and ``ChatPersistenceService`` (on an
in-memory ChaChaNotes DB, so a sent turn is really saved), and press the real
keys. Only the provider gateway (a fixed reply) and the irreversible shutdown
at the end of an approved quit are replaced, as the other quit-flow tests do.
Before the fix, each "alone" test fails because a dialog asks about the
command rows, and each mixed test fails on their extra
"Temporary or unsaved messages" count.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _build_console_send_test_app,
    _configure_native_ready_console,
    _select_llamacpp_console,
)
from Tests.UI.test_console_quit_loss_guard import (
    _ctrl_q_asks,
    _message_rows,
    _mounted_console,
    _quit_dialogs,
    _stay,
    _type_into_composer,
    _until,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_REPLY = "ok"
_SAVED_PROMPT = "Reply with exactly: ok"
_TEMPORARY_PROMPT = "Reply with exactly: ok, temporary"
_DRAFT = "draft beta unsent keep me"
_UNSAVED_LABEL = "Temporary or unsaved messages"

#: Each row kind: what is typed, and how its SYSTEM row starts.
_ROW_KINDS = {
    "help": ("/help", "Console commands:"),
    "doctor": ("/doctor", "Doctor"),
    # A command's error notice: /prefill pin with no text.
    "error notice": ("/prefill pin", "Usage: /prefill pin <text>."),
}


def _app(monkeypatch, events: list[str]):
    app = _build_console_send_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: CapturingGateway(chunks=(_REPLY,))

    async def _record_quit() -> None:
        # Stands in for the irreversible shutdown, as the other quit tests do.
        events.append("quit")

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", _record_quit)
    return app


def _system_rows(store, session_id: str) -> list:
    return [
        message
        for message in store.all_messages_for_session(session_id)
        if message.role is ConsoleMessageRole.SYSTEM
    ]


async def _send_and_wait_for_reply(pilot, console, store, session_id, prompt) -> None:
    await _type_into_composer(pilot, console, prompt)
    await pilot.press("enter")
    await _until(
        pilot,
        lambda: any(
            message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "complete"
            and message.content == _REPLY
            for message in store.all_messages_for_session(session_id)
        ),
        "the reply",
    )
    await _until(
        pilot,
        lambda: (
            console._session._session_close_impact(session_id) is not None
            and not console._session._session_close_impact(
                session_id
            ).lifecycle.has_loss_risk
        ),
        "the run to settle",
    )


async def _saved_chat(pilot, app):
    """Mount Console and save one real turn; nothing is left to lose."""
    console = await _mounted_console(app, pilot)
    await _select_llamacpp_console(console)
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    await _send_and_wait_for_reply(pilot, console, store, session_id, _SAVED_PROMPT)
    await _until(
        pilot,
        lambda: all(
            message.persisted_message_id is not None
            for message in store.all_messages_for_session(session_id)
        ),
        "the turn to be saved",
    )
    impact = console._session._session_close_impact(session_id)
    assert impact is not None and not impact.has_loss_risk, impact
    return console, store, session_id


async def _run_row_kind(pilot, console, store, session_id, kind: str) -> None:
    """Type a command, press Enter, and clear what it leaves in the composer."""
    typed, starts = _ROW_KINDS[kind]
    before = len(_system_rows(store, session_id))
    composer = await _type_into_composer(pilot, console, typed)
    await pilot.press("enter")
    # A bare command name opens the command popup, whose first Enter only
    # completes the name; the next Enter runs it.
    await _until(
        pilot,
        lambda: (
            len(_system_rows(store, session_id)) > before
            or composer.draft_text() == f"{typed} "
        ),
        f"{typed} to run or complete",
    )
    if len(_system_rows(store, session_id)) == before:
        await pilot.press("enter")
    await _until(
        pilot,
        lambda: len(_system_rows(store, session_id)) == before + 1,
        f"the {kind} row",
    )
    row = _system_rows(store, session_id)[-1]
    assert row.content.startswith(starts), row.content[:80]
    # The row is Console's own: never saved, even in a saved chat.
    assert row.persisted_message_id is None
    # The command's text stays in the composer; Ctrl+U clears it, so the
    # row is the only thing left that the old inventory counted.
    composer.focus()
    await pilot.press("ctrl+u")
    await _until(
        pilot,
        lambda: composer.draft_text() == "" and not store.session_draft(session_id),
        "the composer to clear",
    )


async def _click_close(pilot, app, session_id: str) -> None:
    await pilot.click(f"#console-close-session-tab-{session_id}")


def _close_dialog(app) -> ConfirmationDialog | None:
    screen = app.screen
    if isinstance(screen, ConfirmationDialog) and str(screen.title).startswith(
        "Close tab"
    ):
        return screen
    return None


def _counts(rows: list[str], labels) -> dict[str, int]:
    return {
        label: int(count)
        for label, _, count in (row.rpartition(": ") for row in rows)
        if label in labels and count.isdigit()
    }


@pytest.mark.parametrize("kind", list(_ROW_KINDS))
async def test_ctrl_q_does_not_ask_when_a_command_row_is_all_that_is_unsaved(
    monkeypatch, kind
):
    """AC#2: a saved chat holding only a command row quits without a question."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        await _run_row_kind(pilot, console, store, session_id, kind)

        await pilot.press("ctrl+q")
        await _until(
            pilot,
            lambda: "quit" in events or bool(_quit_dialogs(app)),
            "Ctrl+Q to quit or ask",
            timeout=5.0,
        )
        dialogs = _quit_dialogs(app)
        assert not dialogs, f"Ctrl+Q asked about the {kind} row: {dialogs[0].message!r}"
        assert events == ["quit"]


@pytest.mark.parametrize("kind", list(_ROW_KINDS))
async def test_tab_close_does_not_ask_when_a_command_row_is_all_that_is_unsaved(
    monkeypatch, kind
):
    """AC#2: closing that saved chat's tab closes it at once."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        await _run_row_kind(pilot, console, store, session_id, kind)

        await _click_close(pilot, app, session_id)
        await _until(
            pilot,
            lambda: (
                _close_dialog(app) is not None
                or session_id not in {session.id for session in store.sessions()}
            ),
            "the tab to close or ask",
        )
        dialog = _close_dialog(app)
        assert dialog is None, (
            f"tab close asked about the {kind} row: {dialog.message!r}"
        )
        assert session_id not in {session.id for session in store.sessions()}
        assert events == []


async def test_command_rows_beside_a_draft_ask_about_the_draft_alone(monkeypatch):
    """AC#3/#5: a real draft still asks; the command rows are not counted."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, session_id = await _saved_chat(pilot, app)
        for kind in _ROW_KINDS:
            await _run_row_kind(pilot, console, store, session_id, kind)
        assert len(_system_rows(store, session_id)) == len(_ROW_KINDS)
        composer = await _type_into_composer(pilot, console, _DRAFT)

        dialog = await _ctrl_q_asks(pilot, app, events, "the draft beside command rows")
        rows = _message_rows(dialog)
        assert "Unsent drafts: 1" in rows
        assert not any(row.startswith(_UNSAVED_LABEL) for row in rows), rows
        title = console._session._session_close_display_title(session_id)
        assert f'Affected tabs: "{title}"' in rows
        # Quit's totals are exactly the per-tab close impacts' sum.
        impacts = [
            console._session._session_close_impact(session.id)
            for session in store.sessions()
        ]
        assert sum(impact.transcript_message_count for impact in impacts) == 0
        assert sum(impact.has_draft for impact in impacts) == 1
        await _stay(pilot, app, console)
        assert events == []
        assert composer.draft_text() == _DRAFT

        await _click_close(pilot, app, session_id)
        await _until(pilot, lambda: _close_dialog(app) is not None, "the close dialog")
        close_dialog = _close_dialog(app)
        close_rows = close_dialog.message.splitlines()
        assert "Unsent draft: yes" in close_rows
        assert not any(row.startswith(_UNSAVED_LABEL) for row in close_rows), close_rows
        await pilot.click(close_dialog.query_one("#cancel-button", Button))
        await _until(pilot, lambda: app.screen is console, "Stay on the close dialog")
        assert session_id in {session.id for session in store.sessions()}
        assert len(_system_rows(store, session_id)) == len(_ROW_KINDS)


async def test_command_rows_beside_temporary_messages_are_left_out_of_the_count(
    monkeypatch,
):
    """AC#3/#5: a Temporary chat's real messages still ask, counted without
    the command rows; a saved tab holding only a command row is not named."""
    events: list[str] = []
    app = _app(monkeypatch, events)
    async with app.run_test(size=(160, 48)) as pilot:
        console, store, saved_id = await _saved_chat(pilot, app)
        await _run_row_kind(pilot, console, store, saved_id, "help")

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
        temporary_id = store.active_session_id
        await _until(
            pilot,
            lambda: console._console_visible_draft_session_id == temporary_id,
            "the Temporary tab's composer to become current",
        )
        await pilot.pause()
        await _send_and_wait_for_reply(
            pilot, console, store, temporary_id, _TEMPORARY_PROMPT
        )
        real_unsaved = [
            message
            for message in store.all_messages_for_session(temporary_id)
            if message.persisted_message_id is None
        ]
        assert [message.role for message in real_unsaved] == [
            ConsoleMessageRole.USER,
            ConsoleMessageRole.ASSISTANT,
        ]
        for kind in _ROW_KINDS:
            await _run_row_kind(pilot, console, store, temporary_id, kind)
        assert len(_system_rows(store, temporary_id)) == len(_ROW_KINDS)

        dialog = await _ctrl_q_asks(pilot, app, events, "the Temporary chat")
        rows = _message_rows(dialog)
        labels = (_UNSAVED_LABEL, "Unsent drafts")
        assert _counts(rows, labels) == {_UNSAVED_LABEL: len(real_unsaved)}, rows
        tabs_row = next(row for row in rows if row.startswith("Affected tabs:"))
        saved_title = console._session._session_close_display_title(saved_id)
        temporary_title = console._session._session_close_display_title(temporary_id)
        assert f'"{temporary_title}"' in tabs_row
        assert f'"{saved_title}"' not in tabs_row, tabs_row
        # Quit's totals are exactly the per-tab close impacts' sum.
        impacts = [
            console._session._session_close_impact(session.id)
            for session in store.sessions()
        ]
        assert sum(impact.transcript_message_count for impact in impacts) == len(
            real_unsaved
        )
        await _stay(pilot, app, console)
        assert events == []

        await _click_close(pilot, app, temporary_id)
        await _until(pilot, lambda: _close_dialog(app) is not None, "the close dialog")
        close_dialog = _close_dialog(app)
        close_rows = close_dialog.message.splitlines()
        assert f"{_UNSAVED_LABEL}: {len(real_unsaved)}" in close_rows, close_rows
        await pilot.click(close_dialog.query_one("#cancel-button", Button))
        await _until(pilot, lambda: app.screen is console, "Stay on the close dialog")
        assert temporary_id in {session.id for session in store.sessions()}
