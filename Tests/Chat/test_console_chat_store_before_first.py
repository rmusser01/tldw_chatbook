"""Store contract for positioning immediately before a root prompt."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession, ConsoleChatStore


class _CursorDB:
    def __init__(
        self,
        result: bool = True,
        *,
        summary_pair: tuple[str | None, str | None] = (None, None),
    ) -> None:
        self.result = result
        self.summary_pair = summary_pair
        self.calls: list[tuple[str, str | None, str | None]] = []

    def set_conversation_active_cursor(
        self,
        conversation_id: str,
        *,
        active_leaf_message_id: str | None,
        before_message_id: str | None,
    ) -> bool:
        self.calls.append((conversation_id, active_leaf_message_id, before_message_id))
        return self.result

    def set_conversation_active_leaf(
        self, conversation_id: str, message_id: str | None
    ) -> None:
        self.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=message_id,
            before_message_id=None,
        )

    def get_conversation_context_summary(
        self, _conversation_id: str
    ) -> tuple[str | None, str | None]:
        return self.summary_pair


class _RaisingCursorDB(_CursorDB):
    def set_conversation_active_cursor(
        self,
        conversation_id: str,
        *,
        active_leaf_message_id: str | None,
        before_message_id: str | None,
    ) -> bool:
        self.calls.append((conversation_id, active_leaf_message_id, before_message_id))
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


def _root_and_reply() -> tuple[ConsoleChatMessage, ConsoleChatMessage]:
    root = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="current durable prompt",
        persisted_message_id="root-user",
        parent_message_id=None,
    )
    reply = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        persisted_message_id="reply",
        parent_message_id="root-user",
    )
    return root, reply


def _restore_with_cursor(
    store: ConsoleChatStore,
    all_nodes: list[ConsoleChatMessage],
    *,
    active_leaf_persisted_id: str | None,
    active_leaf_before_persisted_id: str | None,
) -> ConsoleChatSession:
    return store.restore_persisted_session(
        title="Saved",
        workspace_id=None,
        persisted_conversation_id="conversation",
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_persisted_id,
        active_leaf_before_persisted_id=active_leaf_before_persisted_id,
    )


def test_unset_cursor_falls_back_to_newest_leaf_and_repairs_cursor() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root, reply = _root_and_reply()

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id=None,
    )

    assert store.active_path_message_ids(session.id) == [root.id, reply.id]
    assert store.active_leaf(session.id) == reply.id
    assert db.calls == [("conversation", "reply", None)]


def test_valid_leaf_restores_its_branch_without_fallback() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root, reply = _root_and_reply()

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id="root-user",
        active_leaf_before_persisted_id=None,
    )

    assert store.active_path_message_ids(session.id) == [root.id]
    assert store.active_leaf(session.id) == root.id
    assert db.calls == []


def test_valid_before_first_restores_empty_path_and_durable_text() -> None:
    root, reply = _root_and_reply()
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id="root-user",
    )

    assert store.active_path_message_ids(session.id) == []
    assert store.active_leaf(session.id) is None
    assert store.session_draft(session.id) == "current durable prompt"
    assert session.has_user_work is True
    assert db.calls == []


def test_valid_leaf_wins_over_marker_and_repairs_companion() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root, reply = _root_and_reply()

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id="reply",
        active_leaf_before_persisted_id="root-user",
    )

    assert store.active_path_message_ids(session.id) == [root.id, reply.id]
    assert store.active_leaf(session.id) == reply.id
    assert store.session_draft(session.id) == ""
    assert db.calls == [("conversation", "reply", None)]


def test_dangling_leaf_ignores_valid_marker_and_repairs_to_fallback() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    root, reply = _root_and_reply()

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id="missing-leaf",
        active_leaf_before_persisted_id="root-user",
    )

    assert store.active_path_message_ids(session.id) == [root.id, reply.id]
    assert store.active_leaf(session.id) == reply.id
    assert store.session_draft(session.id) == ""
    assert db.calls == [("conversation", "reply", None)]


@pytest.mark.parametrize(
    ("nodes_factory", "marker"),
    [
        (_root_and_reply, "missing-marker"),
        (
            lambda: (
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="root answer",
                    persisted_message_id="root-assistant",
                    parent_message_id=None,
                ),
            ),
            "root-assistant",
        ),
        (
            lambda: (
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="root",
                    persisted_message_id="root-user",
                    parent_message_id=None,
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="nested prompt",
                    persisted_message_id="nested-user",
                    parent_message_id="root-user",
                ),
            ),
            "nested-user",
        ),
    ],
    ids=["dangling", "non-user", "non-root"],
)
def test_invalid_marker_only_falls_back_and_repairs_cursor(
    nodes_factory,
    marker: str,
) -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    nodes = list(nodes_factory())

    session = _restore_with_cursor(
        store,
        nodes,
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id=marker,
    )

    fallback = nodes[-1]
    assert store.active_leaf(session.id) == fallback.id
    assert store.session_draft(session.id) == ""
    assert db.calls == [("conversation", fallback.persisted_message_id, None)]


@pytest.mark.parametrize(
    ("active_leaf", "before"),
    [
        (None, "missing-marker"),
        ("missing-leaf", None),
        ("missing-leaf", "missing-marker"),
    ],
)
def test_invalid_non_null_cursor_on_empty_tree_clears_both_components(
    active_leaf: str | None,
    before: str | None,
) -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))

    session = _restore_with_cursor(
        store,
        [],
        active_leaf_persisted_id=active_leaf,
        active_leaf_before_persisted_id=before,
    )

    assert store.active_path_message_ids(session.id) == []
    assert store.active_leaf(session.id) is None
    assert db.calls == [("conversation", None, None)]


def test_unset_cursor_on_empty_tree_does_not_write() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))

    session = _restore_with_cursor(
        store,
        [],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id=None,
    )

    assert store.active_path_message_ids(session.id) == []
    assert store.active_leaf(session.id) is None
    assert db.calls == []


def test_legacy_flat_root_marker_remains_valid_after_native_repair() -> None:
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))
    u1 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="u1",
        persisted_message_id="u1",
        parent_message_id=None,
    )
    a1 = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="a1",
        persisted_message_id="a1",
        parent_message_id=None,
    )
    u2 = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="u2",
        persisted_message_id="u2",
        parent_message_id=None,
    )

    session = _restore_with_cursor(
        store,
        [u1, a1, u2],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id="u2",
    )

    assert store._native_parent_by_message[u2.id] == a1.id
    assert u2.parent_message_id is None
    assert store.active_path_message_ids(session.id) == []
    assert store.active_leaf(session.id) is None
    assert store.session_draft(session.id) == "u2"
    assert session.has_user_work is True
    assert db.calls == []


def test_attachment_only_root_restores_no_staged_input_or_user_work() -> None:
    attachment = MessageAttachment(
        data=b"attachment",
        mime_type="image/png",
        display_name="prompt.png",
        position=0,
    )
    root = ConsoleChatMessage(
        role=ConsoleMessageRole.USER,
        content="",
        persisted_message_id="image-root",
        parent_message_id=None,
        image_data=b"attachment",
        image_mime_type="image/png",
        attachments=(attachment,),
    )
    db = _CursorDB()
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))

    session = _restore_with_cursor(
        store,
        [root],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id="image-root",
    )

    assert store.active_path_message_ids(session.id) == []
    assert store.session_draft(session.id) == ""
    assert session.has_user_work is False
    assert session.pending_attachments == []
    assert store.get_message(root.id).attachments == (attachment,)
    assert db.calls == []


def test_before_first_restore_still_resolves_context_summary_after_path() -> None:
    root, reply = _root_and_reply()
    db = _CursorDB(summary_pair=("earlier recap", "reply"))
    store = ConsoleChatStore(persistence=SimpleNamespace(db=db))

    session = _restore_with_cursor(
        store,
        [root, reply],
        active_leaf_persisted_id=None,
        active_leaf_before_persisted_id="root-user",
    )

    assert store.active_path_message_ids(session.id) == []
    assert store.session_context_summary(session.id) == ("earlier recap", reply.id)
    assert db.calls == []


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
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="U2")
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
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="A1")
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="U2")
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
