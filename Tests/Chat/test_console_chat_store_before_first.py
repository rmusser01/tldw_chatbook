"""Store contract for positioning immediately before a root prompt."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore


class _CursorDB:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None, str | None]] = []

    def set_conversation_active_cursor(
        self,
        conversation_id: str,
        *,
        active_leaf_message_id: str | None,
        before_message_id: str | None,
    ) -> bool:
        self.calls.append(
            (conversation_id, active_leaf_message_id, before_message_id)
        )
        return self.result


class _RaisingCursorDB(_CursorDB):
    def set_conversation_active_cursor(
        self,
        conversation_id: str,
        *,
        active_leaf_message_id: str | None,
        before_message_id: str | None,
    ) -> bool:
        self.calls.append(
            (conversation_id, active_leaf_message_id, before_message_id)
        )
        raise RuntimeError("cursor write failed")


def _restore_root(
    store: ConsoleChatStore,
    *,
    role: ConsoleMessageRole = ConsoleMessageRole.USER,
    persisted_message_id: str = "root",
) -> tuple[ConsoleChatSession, ConsoleChatMessage]:
    root = ConsoleChatMessage(
        role=role,
        content="prompt",
        persisted_message_id=persisted_message_id,
    )
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[root],
        active_leaf_persisted_id=persisted_message_id,
    )
    return session, root


def _state(
    store: ConsoleChatStore, session_id: str
) -> tuple[str | None, list[str], int, int]:
    return (
        store.active_leaf(session_id),
        store.active_path_message_ids(session_id),
        store.payload_revision(session_id),
        store.conversation_context_epoch(session_id),
    )


def test_temporary_before_first_is_success_without_durable_write() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session = store.create_session(title="Temporary")
    root = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="prompt"
    )
    payload_before = store.payload_revision(session.id)
    epoch_before = store.conversation_context_epoch(session.id)

    assert store.set_active_path_before(session.id, root.id) is True

    assert store.active_path_message_ids(session.id) == []
    assert store.payload_revision(session.id) == payload_before + 1
    assert store.conversation_context_epoch(session.id) == epoch_before + 1
    assert db.calls == []

    assert store.set_active_path_before(session.id, root.id) is True
    assert store.payload_revision(session.id) == payload_before + 2
    assert store.conversation_context_epoch(session.id) == epoch_before + 1


def test_persisted_before_first_writes_marker_and_keeps_empty_path() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session, root = _restore_root(store)
    payload_before = store.payload_revision(session.id)
    epoch_before = store.conversation_context_epoch(session.id)

    assert store.set_active_path_before(session.id, root.id) is True

    assert store.active_path_message_ids(session.id) == []
    assert store.payload_revision(session.id) == payload_before + 1
    assert store.conversation_context_epoch(session.id) == epoch_before + 1
    assert db.calls[-1] == ("conversation", None, "root")


def test_persisted_native_root_without_durable_id_keeps_rewind_but_fails() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[],
        active_leaf_persisted_id=None,
    )
    root = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="pending"
    )

    assert store.set_active_path_before(session.id, root.id) is False

    assert store.active_path_message_ids(session.id) == []
    assert db.calls == []


def test_writer_false_keeps_in_memory_rewind() -> None:
    from loguru import logger as loguru_logger

    db = _CursorDB(result=False)
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session, root = _restore_root(store)

    records: list[dict[str, object]] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(message.record), level="WARNING"
    )
    try:
        assert store.set_active_path_before(session.id, root.id) is False
    finally:
        loguru_logger.remove(sink_id)

    assert store.active_path_message_ids(session.id) == []
    assert db.calls == [("conversation", None, "root")]
    cursor_warnings = [
        record
        for record in records
        if "Failed to persist Console before-first cursor" in str(record["message"])
    ]
    assert len(cursor_warnings) == 1
    assert cursor_warnings[0]["extra"] == {
        "session_id": session.id,
        "conversation_id": "conversation",
    }
    assert root.persisted_message_id == "root"
    assert root.content not in str(cursor_warnings[0]["message"])
    assert root.persisted_message_id not in str(cursor_warnings[0]["message"])


def test_writer_exception_keeps_in_memory_rewind() -> None:
    db = _RaisingCursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session, root = _restore_root(store)

    assert store.set_active_path_before(session.id, root.id) is False

    assert store.active_path_message_ids(session.id) == []
    assert db.calls == [("conversation", None, "root")]


@pytest.mark.parametrize(
    "persistence",
    [
        SimpleNamespace(db=None),
        SimpleNamespace(db=SimpleNamespace()),
        SimpleNamespace(
            db=SimpleNamespace(set_conversation_active_cursor="not callable")
        ),
    ],
    ids=["db-unavailable", "writer-unavailable", "writer-not-callable"],
)
def test_unavailable_cursor_writer_keeps_in_memory_rewind(
    persistence: SimpleNamespace,
) -> None:
    store = ConsoleChatStore(persistence=persistence)
    session, root = _restore_root(store)

    assert store.set_active_path_before(session.id, root.id) is False

    assert store.active_path_message_ids(session.id) == []


def test_wrong_role_raises_without_mutation() -> None:
    store = ConsoleChatStore()
    session = store.create_session(title="Temporary")
    root = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    state_before = _state(store, session.id)

    with pytest.raises(
        ValueError, match="Before-first target must be a root user message"
    ):
        store.set_active_path_before(session.id, root.id)

    assert _state(store, session.id) == state_before


def test_durable_node_with_imported_parent_raises_without_mutation() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="first",
        persisted_message_id="root",
    )
    child = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="follow-up",
        persisted_message_id="child",
        parent_message_id="root",
    )
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[root, child],
        active_leaf_persisted_id="child",
    )
    state_before = _state(store, session.id)

    with pytest.raises(
        ValueError, match="Before-first target must be a root user message"
    ):
        store.set_active_path_before(session.id, child.id)

    assert _state(store, session.id) == state_before
    assert db.calls == []


def test_durable_imported_root_remains_valid_after_native_root_repair() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    first = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="first",
        persisted_message_id="first",
    )
    second = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="second",
        persisted_message_id="second",
    )
    session = store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=[first, second],
        active_leaf_persisted_id="second",
    )
    assert store._native_parent_by_message[second.id] == first.id
    assert second.parent_message_id is None

    assert store.set_active_path_before(session.id, second.id) is True

    assert store.active_path_message_ids(session.id) == []
    assert db.calls == [("conversation", None, "second")]


def test_temporary_descendant_uses_native_parent_and_raises_without_mutation() -> None:
    store = ConsoleChatStore()
    session = store.create_session(title="Temporary")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="U1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="A1")
    u2 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="U2"
    )
    assert u2.parent_message_id is None
    state_before = _state(store, session.id)

    with pytest.raises(
        ValueError, match="Before-first target must be a root user message"
    ):
        store.set_active_path_before(session.id, u2.id)

    assert _state(store, session.id) == state_before


def test_live_persisted_descendant_uses_native_parent_without_mutation() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    session, _root = _restore_root(store)
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="A1"
    )
    u2 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="U2"
    )
    # Acceptance hydration assigns the live owner its durable identity without
    # rewriting this persisted-parent field; native ancestry remains authoritative.
    live_u2 = store._nodes_by_session[session.id][u2.id]
    live_u2.persisted_message_id = "live-u2"
    assert live_u2.parent_message_id is None
    assert store._native_parent_by_message[u2.id] is not None
    state_before = _state(store, session.id)

    with pytest.raises(
        ValueError, match="Before-first target must be a root user message"
    ):
        store.set_active_path_before(session.id, u2.id)

    assert _state(store, session.id) == state_before
    assert db.calls == []


def test_unknown_message_raises_without_mutation() -> None:
    store = ConsoleChatStore()
    session = store.create_session(title="Temporary")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="U1")
    state_before = _state(store, session.id)

    with pytest.raises(KeyError, match="Unknown Console message: missing"):
        store.set_active_path_before(session.id, "missing")

    assert _state(store, session.id) == state_before
