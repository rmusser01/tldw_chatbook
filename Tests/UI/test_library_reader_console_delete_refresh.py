"""An open Library reader re-reads its transcript after a Console Delete or Undo.

TASK-33628.10, seen live during the TASK-33628.6 check: after a Console
Delete, the Library list's message count updated but the reader beside it
kept its earlier load ("4 of 4 messages") until the app restarted. Library
is a reused screen, so a return visit re-reads the Conversations list; the
reader skipped its own re-read because a message delete, undo or edit never
changes the conversation's version, the only thing the reader compared.

Each case drives the production app (``TldwCli``): the real Library and
Console screens, the real router (Library is suspended and resumed, never
rebuilt), the real Console store and delete flow, and one file-backed
ChaChaNotes database behind both screens.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

# The real ChatScreen/store goes through config-participant admission, which
# the per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile.
pytestmark = pytest.mark.bootstrap_profile

_TITLE = "Reader refresh chat"
_BODIES = (
    "First question: name a colour.",
    "First answer: blue.",
    "Second question: name a fruit.",
    "Second answer: apple.",
)


async def _wait_until(
    pilot: Any,
    predicate: Callable[[], bool],
    debug: Callable[[], Any] | None = None,
    *,
    timeout: float = 30.0,
) -> None:
    # Wall clock, generous: a loaded machine slows every pause, and each wait
    # returns the moment its condition holds.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.05)
    assert predicate(), debug() if debug is not None else "condition not reached"


def _painted(app: Any) -> str:
    """Return the text actually painted on the active screen."""
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _seed_console_conversation(app: Any, db: CharactersRAGDB) -> str:
    """Save four turns through the Console's own persistence path."""
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(
            db, workspace_registry=app.workspace_registry_service
        )
    )
    session = store.create_session(title=_TITLE)
    for index, body in enumerate(_BODIES):
        store.append_message(
            session.id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content=body,
            persist=True,
        )
    assert session.persisted_conversation_id is not None
    return session.persisted_conversation_id


def _build_app(tmp_path: Any) -> tuple[Any, CharactersRAGDB, ChatConversationService]:
    db = CharactersRAGDB(tmp_path / "reader-refresh.db", client_id="reader-refresh")
    app = _build_test_app(configured_default="library")
    app.chachanotes_db = db
    local = app.local_chat_conversation_service = ChatConversationService(db)
    app.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=local, server_service=None
    )
    return app, db, local


def _reader_debug(library: Any) -> Callable[[], Any]:
    def describe() -> str:
        state = library._conversations_state.reader_state
        counts = [
            record.get("message_count")
            for record in library._conversations_state.page_records
        ]
        return (
            f"reader total={state.message_total} rows={len(state.messages)} "
            f"loaded_generation={state.loaded_generation} loading={state.loading} "
            f"error={state.error!r} revisions={[m.revision for m in state.messages]} "
            f"list message_count={counts}"
        )

    return describe


async def _open_in_library_reader(app: Any, pilot: Any, cid: str) -> Any:
    """Open the seeded conversation in the Library reader; return the screen."""
    await _wait_until(
        pilot,
        lambda: (
            type(app.screen).__name__ == "LibraryScreen"
            and bool(app.screen.query("#library-row-browse-conversations"))
        ),
        lambda: type(app.screen).__name__,
    )
    library = app.screen
    library.query_one("#library-row-browse-conversations", Button).press()
    await _wait_until(
        pilot,
        lambda: (
            library._conversations_state.reader_state.loaded_id == cid
            and library._conversations_state.reader_state.loaded_actions_eligible
        ),
        _reader_debug(library),
    )
    assert library._conversations_state.reader_state.message_total == 4
    # The reader's status line here carries the factory app's source-snapshot
    # error (no media/notes services), so read the painted transcript rows.
    await _wait_until(pilot, lambda: _BODIES[3] in _painted(app), lambda: _painted(app))
    return library


async def _resume_in_console(app: Any, pilot: Any, library: Any, cid: str) -> Any:
    """Press the reader's Resume conversation; return the Console screen."""
    library.query_one("#library-conversation-open-console", Button).press()
    await _wait_until(
        pilot,
        lambda: (
            type(app.screen).__name__ == "ChatScreen"
            and any(
                session.persisted_conversation_id == cid
                and session.id
                == app.screen._ensure_console_chat_store().active_session_id
                for session in app.screen._ensure_console_chat_store().sessions()
            )
        ),
        lambda: type(app.screen).__name__,
    )
    console = app.screen
    store = console._ensure_console_chat_store()
    await _wait_until(
        pilot,
        lambda: len(store.messages_for_session(store.active_session_id)) == 4,
        lambda: store.messages_for_session(store.active_session_id),
    )
    return console


async def _delete_from_third_message(app: Any, pilot: Any, console: Any) -> None:
    """Delete message 3 (and the one after it) through More > Delete > confirm."""
    from tldw_chatbook.Widgets.Console import ConsoleTranscript

    store = console._ensure_console_chat_store()
    target = store.messages_for_session(store.active_session_id)[2].id
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    transcript.select_message(target)
    await console._sync_native_console_chat_ui()
    opener = f"#console-message-action-more-{target}"
    await _wait_until(pilot, lambda: bool(console.query(opener)))
    console.query_one(opener, Button).press()
    await _wait_until(
        pilot, lambda: bool(console.query("#console-message-more-delete"))
    )
    console.query_one("#console-message-more-delete", Button).press()
    confirm = f"#console-message-action-delete-confirm-{target}"
    await _wait_until(pilot, lambda: bool(console.query(confirm)))
    console.query_one(confirm, Button).press()
    await _wait_until(pilot, lambda: bool(app.screen.query("#console-delete-receipt")))


def _live_message_ids(db: CharactersRAGDB, cid: str) -> list[str]:
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT id FROM messages WHERE conversation_id = ? AND deleted = 0 "
            "ORDER BY timestamp ASC, rowid ASC",
            (cid,),
        ).fetchall()
    return [row["id"] for row in rows]


async def _return_to_library(app: Any, pilot: Any, library: Any) -> None:
    await app.handle_screen_navigation(NavigateToScreen("library"))
    await _wait_until(pilot, lambda: app.screen is library)


@pytest.mark.asyncio
async def test_console_delete_updates_the_open_library_reader(tmp_path) -> None:
    """AC#1/#2: the reader left open in Library shows the post-delete count."""
    app, db, _local = _build_app(tmp_path)
    cid = _seed_console_conversation(app, db)
    try:
        async with app.run_test(size=(160, 44)) as pilot:
            library = await _open_in_library_reader(app, pilot, cid)
            console = await _resume_in_console(app, pilot, library, cid)

            await _delete_from_third_message(app, pilot, console)
            app.screen.query_one("#console-delete-receipt-done", Button).press()
            await _wait_until(pilot, lambda: app.screen is console)
            live = _live_message_ids(db, cid)
            assert len(live) == 2

            await _return_to_library(app, pilot, library)
            await _wait_until(
                pilot,
                lambda: (
                    library._conversations_state.reader_state.message_total == 2
                    and library._conversations_state.reader_state.loaded_actions_eligible
                ),
                _reader_debug(library),
            )
            reader = library._conversations_state.reader_state
            assert [message.message_id for message in reader.messages] == live
            assert [message.text for message in reader.messages] == list(_BODIES[:2])
            await _wait_until(
                pilot,
                lambda: (
                    _BODIES[1] in _painted(app)
                    and _BODIES[2] not in _painted(app)
                    and _BODIES[3] not in _painted(app)
                ),
                lambda: _painted(app),
            )
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_console_delete_then_undo_rereads_the_open_library_reader(
    tmp_path,
) -> None:
    """AC#1/#2: after Undo the reader re-reads the restored rows."""
    app, db, local = _build_app(tmp_path)
    cid = _seed_console_conversation(app, db)
    try:
        async with app.run_test(size=(160, 44)) as pilot:
            library = await _open_in_library_reader(app, pilot, cid)
            loaded_before = library._conversations_state.reader_state
            console = await _resume_in_console(app, pilot, library, cid)

            await _delete_from_third_message(app, pilot, console)
            app.screen.query_one("#console-delete-receipt-undo", Button).press()
            await _wait_until(pilot, lambda: len(_live_message_ids(db, cid)) == 4)
            await _wait_until(pilot, lambda: app.screen is console)

            await _return_to_library(app, pilot, library)
            # Undo bumped every restored row's version, so their saved
            # revisions changed; a reader that re-read them carries the new
            # ones (an untouched reader still holds the pre-delete load).
            saved = local.get_library_conversation_messages(cid, message_limit=10)
            current = {m["id"]: m["revision"] for m in saved["messages"]}
            await _wait_until(
                pilot,
                lambda: (
                    library._conversations_state.reader_state.loaded_actions_eligible
                    and {
                        m.message_id: m.revision
                        for m in library._conversations_state.reader_state.messages
                    }
                    == current
                ),
                _reader_debug(library),
            )
            reader = library._conversations_state.reader_state
            assert reader.loaded_generation != loaded_before.loaded_generation
            assert reader.message_total == 4
            assert [message.text for message in reader.messages] == list(_BODIES)
            await _wait_until(pilot, lambda: _BODIES[3] in _painted(app))
    finally:
        db.close_connection()
