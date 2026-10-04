"""The off-loop runner behind Console Delete and Undo (TASK-33628.5).

``run_durable_off_loop`` moves the durable half of a Delete or an Undo off
the event loop and hands its outcome back to the loop. Two properties make
that safe for UI-thread state:

* ``settle`` -- which applies the write to the Console store and releases
  the store's fences -- always runs on the event-loop thread;
* it runs exactly once when the write finishes, even if the task awaiting
  the write was cancelled meanwhile. Cancelling a task does not stop the
  thread already writing, so without this a cancelled awaiter would leave
  the store showing rows the database had deleted, with its fences held.

A ``:memory:`` ChaChaNotes database is thread-local, so the write runs
inline for one. The store-level cases use a real FILE-backed database.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_message_delete import (
    delete_subtree_off_loop,
    restore_subtree_off_loop,
    run_durable_off_loop,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# The real Console store goes through config-participant admission, which the
# per-test sandbox refuses (RecoveryRequired); keep the collection-time profile.
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]

_FILE_DB = SimpleNamespace(is_memory_db=False)
_MEMORY_DB = SimpleNamespace(is_memory_db=True)


async def _until(predicate: Any, what: str, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


async def test_a_memory_database_writes_inline_on_the_loop():
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []

    def call() -> str:
        seen.append(("call", threading.get_ident()))
        return "written"

    def settle(result: Any, error: BaseException | None) -> str:
        seen.append(("settle", threading.get_ident()))
        assert error is None
        return f"{result} and applied"

    assert await run_durable_off_loop(_MEMORY_DB, call, settle) == "written and applied"
    assert seen == [("call", loop_thread), ("settle", loop_thread)]


async def test_a_file_database_writes_off_the_loop_and_settles_on_it():
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def call() -> int:
        seen["call"] = threading.get_ident()
        return 3

    def settle(result: Any, error: BaseException | None) -> int:
        seen["settle"] = threading.get_ident()
        assert error is None
        return result + 1

    assert await run_durable_off_loop(_FILE_DB, call, settle) == 4
    assert seen["call"] != loop_thread
    assert seen["settle"] == loop_thread


@pytest.mark.parametrize("database", [_FILE_DB, _MEMORY_DB], ids=["file", "memory"])
async def test_a_failed_write_reaches_settle_and_then_the_awaiter(database):
    def call() -> None:
        raise ValueError("Resolve pending dispatch before deleting this message.")

    errors: list[BaseException | None] = []

    def settle(_result: Any, error: BaseException | None) -> None:
        errors.append(error)
        assert error is not None
        raise error

    with pytest.raises(ValueError, match="pending dispatch"):
        await run_durable_off_loop(database, call, settle)
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


async def test_a_cancelled_awaiter_still_settles_once_on_the_loop():
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    entered, release = threading.Event(), threading.Event()
    settled: list[tuple[Any, int]] = []

    def call() -> str:
        entered.set()
        assert release.wait(10)
        return "committed"

    def settle(result: Any, error: BaseException | None) -> None:
        settled.append((result, threading.get_ident()))
        raise RuntimeError("nobody is listening any more")

    try:
        awaiter = loop.create_task(run_durable_off_loop(_FILE_DB, call, settle))
        await _until(entered.is_set, "the write to start")
        awaiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await awaiter
        assert settled == [], "settled before the write had finished"

        release.set()
        await _until(lambda: bool(settled), "the finished write to settle")
        await asyncio.sleep(0.05)
        import gc

        gc.collect()
        await asyncio.sleep(0)
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)
    assert settled == [("committed", loop_thread)]
    assert unhandled == [], unhandled


# --- The store: a cancelled awaiter cannot strand it ---------------------------


def _chain(db: CharactersRAGDB, count: int) -> tuple[str, list[str]]:
    conversation_id = db.add_conversation({"title": "Off loop"})
    ids: list[str] = []
    parent = None
    with db.transaction():
        for index in range(count):
            role = "user" if index % 2 == 0 else "assistant"
            parent = db.add_message(
                {
                    "id": f"m{index:03d}",
                    "conversation_id": conversation_id,
                    "sender": role,
                    "role": role,
                    "content": f"m{index:03d} text",
                    "parent_message_id": parent,
                    "timestamp": f"2026-09-30T00:00:{index:02d}.000000+00:00",
                }
            )
            ids.append(parent)
    db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=ids[-1], before_message_id=None
    )
    return conversation_id, ids


def _open(db: CharactersRAGDB, conversation_id: str) -> tuple[Any, str, dict[str, str]]:
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_conversation_hydration import (
        console_messages_from_conversation_tree,
    )

    tree = ChatConversationService(db).get_conversation_tree(
        conversation_id, depth_cap=10_000, root_limit=10_000
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.restore_persisted_session(
        title="Off loop",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=console_messages_from_conversation_tree(tree, db=db),
        active_leaf_persisted_id=db.get_conversation_active_cursor(conversation_id)[0],
    )
    native = {
        node.persisted_message_id: node.id
        for node in store._nodes_by_session[session.id].values()
    }
    return store, session.id, native


def _gate(db: CharactersRAGDB, name: str) -> tuple[threading.Event, threading.Event]:
    entered, release = threading.Event(), threading.Event()
    real = getattr(db, name)

    def held(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)

    setattr(db, name, held)
    return entered, release


def _fences_held(store: Any, session_id: str) -> bool:
    return bool(store._fork_source_transitions.get(session_id)) or bool(
        store._voice_promotion_mutation_admissions.get(session_id)
    )


def _live(db: CharactersRAGDB, ids: list[str]) -> list[bool]:
    return [db.get_message_by_id(message_id) is not None for message_id in ids]


async def test_a_delete_whose_awaiter_is_cancelled_still_lands_in_the_store(tmp_path):
    db = CharactersRAGDB(tmp_path / "off-loop.db", "off-loop")
    conversation_id, ids = _chain(db, 12)
    store, session_id, native = _open(db, conversation_id)
    entered, release = _gate(db, "soft_delete_message_subtree")

    loop = asyncio.get_running_loop()
    awaiter = loop.create_task(delete_subtree_off_loop(store, native[ids[4]]))
    try:
        await _until(entered.is_set, "the durable delete to start")
        assert _fences_held(store, session_id)
        awaiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await awaiter
        # Still writing: the store still shows the rows and keeps its fences.
        assert len(store._nodes_by_session[session_id]) == 12
        assert _fences_held(store, session_id)
    finally:
        release.set()

    await _until(
        lambda: not _fences_held(store, session_id), "the fences to be released"
    )
    assert _live(db, ids) == [True] * 4 + [False] * 8
    assert {
        node.persisted_message_id
        for node in store._nodes_by_session[session_id].values()
    } == set(ids[:4])


async def test_an_undo_whose_awaiter_is_cancelled_still_lands_in_the_store(tmp_path):
    db = CharactersRAGDB(tmp_path / "off-loop.db", "off-loop")
    conversation_id, ids = _chain(db, 12)
    store, session_id, native = _open(db, conversation_id)
    deleted, _held = await delete_subtree_off_loop(store, native[ids[4]])
    assert _live(db, ids) == [True] * 4 + [False] * 8
    entered, release = _gate(db, "restore_message_subtree")

    loop = asyncio.get_running_loop()
    awaiter = loop.create_task(restore_subtree_off_loop(store, deleted))
    try:
        await _until(entered.is_set, "the durable undo to start")
        assert _fences_held(store, session_id)
        awaiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await awaiter
        assert len(store._nodes_by_session[session_id]) == 4
    finally:
        release.set()

    await _until(
        lambda: not _fences_held(store, session_id), "the fences to be released"
    )
    assert _live(db, ids) == [True] * 12
    assert len(store._nodes_by_session[session_id]) == 12
    assert [
        m.persisted_message_id for m in store.messages_for_session(session_id)
    ] == ids


async def test_a_refused_delete_changes_nothing_and_releases_the_fences(tmp_path):
    """A pending dispatch refuses inside the write transaction: nothing lands."""
    db = CharactersRAGDB(tmp_path / "off-loop.db", "off-loop")
    conversation_id, ids = _chain(db, 6)
    store, session_id, native = _open(db, conversation_id)

    def refuse(*_args: Any, **_kwargs: Any) -> bool:
        from tldw_chatbook.DB.ChaChaNotes_DB import InputError

        raise InputError("cursor owned by a pending dispatch")

    db.set_conversation_active_leaf = refuse  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="pending dispatch"):
        await delete_subtree_off_loop(store, native[ids[2]])

    assert _live(db, ids) == [True] * 6
    assert len(store._nodes_by_session[session_id]) == 6
    assert not _fences_held(store, session_id)
