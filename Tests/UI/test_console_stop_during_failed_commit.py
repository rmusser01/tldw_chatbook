"""Stop pressed while a send's durable commit runs, and the commit then fails.

TASK-33621.59. Owner decision 2026-10-29: treat it as stopped. Stop wins:
the send is abandoned, its text and attachments go back to the draft, and
nothing offers it again -- no "Send could not be saved" card with Retry send,
and no unsent-turn shelf entry.

Measured on dev 8ae979d81b before the fix, in both windows below: the send
paused as PERSISTENCE, the card offered Retry send, the shelf offered the
turn (Restore / Discard) and held its attachment, and the unsent echo stayed
in the transcript, although the user had asked to stop.

The two windows are the two ways a commit fails:

- before its reservation: the store's preflight raises, and the controller
  takes the pause (TASK-33621.53, ``pause_unreserved_failed_commit``);
- inside its reservation: the persistence write raises, and the store pauses
  the send itself.

While the commit runs the run state is VALIDATING, so the composer's Stop
and Ctrl+G are not offered; the command palette's "Console: Stop this tab's
run" (``action_stop_console_run``, also /stop) is the Stop a user can press
then, so it is the one pressed here.

This drives the stock received-intent path: the real ChatScreen, runtime,
controller and store, over a file-backed database, so the commit runs on its
worker thread as in the app. Only the provider call is a double; the commit
is held on that thread until Stop has been pressed, then fails.
"""

from __future__ import annotations

import asyncio
import functools
import sqlite3
import struct
import threading
import zlib

import pytest
from textual.widgets import Button

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
    _until,
)
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

FOLLOW_UP = "A follow-up after Stop"
REPLY = "received intent reply"
NOT_SAVED = "Send could not be saved"
NOT_SENT_SHELF = "Not sent:"
WINDOWS = ("before its reservation", "inside its reservation")
# No spaces: the UI PR gate census lists these node ids one per line.
WINDOW_IDS = ("before-reservation", "inside-reservation")


def _png() -> bytes:
    """A valid 1x1 PNG, so the attachment passes the commit's checks."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


def _hold_then_fail_commit(monkeypatch, store, window: str):
    """Hold the first durable commit on its thread, then make it fail once.

    Returns:
        ``(entered, release)``: set when the commit is held; set to let it
        go on and fail. Later commits run unchanged.
    """
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        entered.set()
        assert release.wait(20), "the held commit was never released"

    if window == "before its reservation":
        # The store's preflight runs before its reservation exists:
        # "Durable identity owner changed." and its siblings come from here.
        from tldw_chatbook.Chat import console_legacy_flat_roots as flat_roots

        original_acceptance = flat_roots.durable_acceptance
        original_commit = store.commit_durable_turn
        failed: list[bool] = []

        def fail_once(acceptance, nodes):
            if not failed:
                failed.append(True)
                raise RuntimeError("Durable identity owner changed.")
            return original_acceptance(acceptance, nodes)

        def held_commit(acceptance):
            if not entered.is_set():
                hold()  # outside the store's lock: Stop needs it
            return original_commit(acceptance)

        monkeypatch.setattr(flat_roots, "durable_acceptance", fail_once)
        monkeypatch.setattr(store, "commit_durable_turn", held_commit)
    else:
        # The persistence write runs inside the reservation; a write lock
        # held past the busy timeout fails it exactly like this, live.
        persistence = store.persistence
        original_write = persistence.commit_durable_turn

        @functools.wraps(original_write)
        def held_write(**kwargs):
            if not entered.is_set():
                hold()
                raise sqlite3.OperationalError("database is locked")
            return original_write(**kwargs)

        monkeypatch.setattr(persistence, "commit_durable_turn", held_write)
    return entered, release


async def _send_with_failing_commit(case, monkeypatch, window: str):
    """Stage an image, press Enter, and wait until the commit is held."""
    assert config_module.save_setting_to_cli_config(
        "console", "exchange_capture", False
    )
    _allow_second_turns(case.host)
    case.probe.release.set()  # do not hold the configuration read
    store, session_id = case.store, case.session.id
    entered, release = _hold_then_fail_commit(monkeypatch, store, window)
    attachment = PendingAttachment(
        "/f59-picture.png",
        "f59-picture.png",
        "image",
        "attachment",
        data=_png(),
        mime_type="image/png",
    )
    assert store.add_pending_attachment(session_id, attachment)
    _send(case, "enter")
    assert await _until(entered.is_set, 20)
    preparation = store.preparation_for_session(session_id)
    assert preparation is not None
    assert preparation.state is ConsoleTurnPreparationState.COMMITTING
    # The send took the image into its custody; it is not staged any more.
    assert store.pending_attachments(session_id) == []
    assert case.provider_calls == []
    return attachment, release, preparation


def _turn_ended(case) -> bool:
    return not any(
        row.session_id == case.session.id for row in case.runtime._turn_custody.values()
    )


@pytest.mark.parametrize("window", WINDOWS, ids=WINDOW_IDS)
async def test_stop_during_a_failing_commit_gives_the_send_back_and_frees_the_chat(
    monkeypatch, window
):
    """AC#1-#3: stopped, text and image back in the draft, next send answered."""
    async with _received_console_case(
        monkeypatch, f"stop-failed-commit-{window.split()[0]}", durable=True
    ) as case:
        store, controller = case.store, case.controller
        session_id = case.session.id
        attachment, release, preparation = await _send_with_failing_commit(
            case, monkeypatch, window
        )
        owner = controller._ordinary_native_commit_owner(preparation.preparation_id)
        try:
            # Stop, then Stop again before the commit returns, as a user might.
            await case.console.action_stop_console_run()
            assert owner.explicit_stop
            await case.console.action_stop_console_run()
        finally:
            release.set()
        assert await _until(lambda: _turn_ended(case), 15)
        await asyncio.sleep(0.3)  # let the screen paint the settled turn

        painted = _painted(case.host)
        # Stop won: no pause, no card offering Retry send, no shelf entry.
        assert store.preparation_for_session(session_id) is None
        assert store.preparation_by_id(preparation.preparation_id) is None
        assert NOT_SAVED not in painted
        assert not case.console.query_one("#console-trace-retry", Button).display
        assert case.runtime.recoveries_for_session(session_id) == ()
        assert NOT_SENT_SHELF not in painted
        state = controller.run_state_for(session_id)
        assert state.status is ConsoleRunStatus.STOPPED, state
        assert "Blocked" not in painted
        # The message and its image are back where the user left them.
        assert case.composer.draft_text() == case.draft
        assert store.session_draft(session_id) == case.draft
        assert [item.attachment_id for item in store.pending_attachments(session_id)] == [
            attachment.attachment_id
        ]
        # The unsent echo is gone, and nothing reached the provider.
        assert not [
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.USER
        ]
        assert case.provider_calls == []

        # The next send in this chat is answered.
        case.composer.load_draft(FOLLOW_UP)
        case.composer.focus()
        assert await _until(lambda: case.host.focused is case.composer, 5)
        assert await _until(lambda: store.session_draft(session_id) == FOLLOW_UP, 5)
        _send(case, "enter")
        assert await _until(lambda: len(case.provider_calls) == 1, 20)
        assert await _until(lambda: REPLY in _painted(case.host), 20)
        assert await _until(lambda: _turn_ended(case), 15)
        assert len(case.provider_calls) == 1
        assert NOT_SAVED not in _painted(case.host)
        assert case.runtime.recoveries_for_session(session_id) == ()


@pytest.mark.parametrize("window", WINDOWS, ids=WINDOW_IDS)
async def test_a_failing_commit_with_no_stop_still_pauses_for_retry(
    monkeypatch, window
):
    """AC#4: without Stop the failure still offers Retry send and keeps the turn."""
    async with _received_console_case(
        monkeypatch, f"failed-commit-no-stop-{window.split()[0]}", durable=True
    ) as case:
        store, session_id = case.store, case.session.id
        _attachment, release, preparation = await _send_with_failing_commit(
            case, monkeypatch, window
        )
        release.set()
        assert await _until(lambda: _turn_ended(case), 15)
        held = store.preparation_for_session(session_id)
        assert held is not None
        assert held.preparation_id == preparation.preparation_id
        assert held.state is ConsoleTurnPreparationState.PAUSED, held.state
        assert held.pause_kind is ConsolePreparationPauseKind.PERSISTENCE
        assert await _until(lambda: NOT_SAVED in _painted(case.host), 5)
        retry = case.console.query_one("#console-trace-retry", Button)
        assert retry.display and str(retry.label) == "Retry send"
        # The unsent turn, image included, waits for the card or the shelf.
        assert len(case.runtime.recoveries_for_session(session_id)) == 1
        assert store.pending_attachments(session_id) == []
        assert case.provider_calls == []
