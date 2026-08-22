from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat import console_chat_store as console_store_module
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
    ConsoleLibraryPolicyDefaults,
)
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


class _Contribution:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.writer = None

    def write(self, *, writer, conversation_id, message_ids) -> None:
        self.calls += 1
        self.writer = writer
        if self.fail:
            raise RuntimeError("injected contribution failure")
        sequence = writer.next_trajectory_sequence()
        writer.execute(
            "INSERT INTO message_trajectory_metadata "
            "(message_id, conversation_id, turn_id, seq, event_kind, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                message_ids["assistant"],
                conversation_id,
                message_ids["user"],
                sequence,
                "accepted",
                '{"version":1}',
            ),
        )


def _store(tmp_path, name="promotion.db"):
    db = CharactersRAGDB(tmp_path / name, "promotion-test")
    service = ChatPersistenceService(db)
    store = ConsoleChatStore(
        persistence=service,
        library_policy_defaults=ConsoleLibraryPolicyDefaults(
            ConsoleAutoRetrieve.AUTOMATIC,
            ConsoleAssistantLibraryAccess.ALLOWED,
        ),
    )
    return db, service, store


def _memory_state(store, session_id):
    session = next(item for item in store.sessions() if item.id == session_id)
    messages = store._tree_nodes_parent_first(session_id)
    return (
        session.ephemeral,
        session.persisted_conversation_id,
        session.title,
        session.library_policy_holder.snapshot,
        session.library_policy_holder.explicitly_staged,
        session.library_policy_holder.save_pending,
        session.rag_scope_holder.scope,
        tuple(
            (
                message.id,
                message.persisted_message_id,
                message.parent_message_id,
                message.attachments,
            )
            for message in messages
        ),
    )


def _conversation_count(db):
    return db.get_connection().execute("SELECT COUNT(*) FROM conversations").fetchone()[0]


def test_staged_identity_is_immutable_and_staging_does_not_mutate_session():
    store = ConsoleChatStore()
    session = store.create_session(title="Temporary", ephemeral=True)
    before = _memory_state(store, session.id)

    identity = store.stage_first_persistence(session.id)

    assert isinstance(identity, console_store_module.ConsoleStagedConversationIdentity)
    assert identity.title == "Temporary"
    assert _memory_state(store, session.id) == before
    with pytest.raises(FrozenInstanceError):
        identity.title = "changed"


@pytest.mark.parametrize("failure_point", ["conversation", "policy"])
def test_conversation_and_policy_failures_leave_state_exact_and_retry_once(
    tmp_path, monkeypatch, failure_point
):
    db, service, store = _store(tmp_path, "retry.db")
    session = store.create_session(title="Retry me", ephemeral=True)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        attachments=(MessageAttachment(b"payload", "text/plain", "note.txt", 0),),
    )
    session.rag_scope_holder.set(
        RagScope(
            items=(ScopeItem("media", "m1"),),
            updated_at="2026-08-22T00:00:00Z",
        )
    )
    before = _memory_state(store, session.id)
    attempts = 0
    publish_transaction_states = []
    original_publish = store.publish_committed_identity

    def publish_after_commit(session_id, identity):
        publish_transaction_states.append(db.get_connection().in_transaction)
        original_publish(session_id, identity)

    monkeypatch.setattr(store, "publish_committed_identity", publish_after_commit)

    if failure_point == "conversation":
        original_create = service.create_conversation

        def fail_once(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected conversation failure")
            return original_create(**kwargs)

        monkeypatch.setattr(service, "create_conversation", fail_once)
    else:
        original_insert = service.console_library_policy_repository.insert

        def fail_once(conversation_id, candidate):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected policy failure")
            return original_insert(conversation_id, candidate)

        monkeypatch.setattr(
            service.console_library_policy_repository,
            "insert",
            fail_once,
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
        store.promote_ephemeral_session(session.id)

    assert _memory_state(store, session.id) == before
    assert _conversation_count(db) == 0
    assert publish_transaction_states == []

    conversation_id = store.promote_ephemeral_session(session.id)

    assert conversation_id is not None
    assert _conversation_count(db) == 1
    assert session.persisted_conversation_id == conversation_id
    assert session.ephemeral is False
    assert publish_transaction_states == [False]


def test_promotion_is_atomic_for_policy_lineage_attachments_and_contribution(
    tmp_path,
):
    db, service, store = _store(tmp_path, "bundle.db")
    session = store.create_session(title="Bundle", ephemeral=True)
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="look",
        attachments=(
            MessageAttachment(b"zero", "image/png", "zero.png", 0),
            MessageAttachment(b"one", "image/png", "one.png", 1),
        ),
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="done",
    )
    contribution = _Contribution()

    conversation_id = store.promote_ephemeral_session(
        session.id, contributions=(contribution,)
    )

    assert conversation_id is not None
    live_user = store._nodes_by_session[session.id][user.id]
    live_assistant = store._nodes_by_session[session.id][assistant.id]
    durable_policy = service.console_library_policy_repository.read(conversation_id)
    assert durable_policy.durable_policy is not None
    rows = db.get_messages_for_conversation(conversation_id, limit=100)
    assert [row["id"] for row in rows] == [
        live_user.persisted_message_id,
        live_assistant.persisted_message_id,
    ]
    assert rows[1]["parent_message_id"] == live_user.persisted_message_id
    attachments = db.get_attachments_for_messages([live_user.persisted_message_id])
    assert [row["position"] for row in attachments[live_user.persisted_message_id]] == [1]
    trajectory = db.get_connection().execute(
        "SELECT message_id, seq FROM message_trajectory_metadata WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    assert tuple(trajectory) == (live_assistant.persisted_message_id, 1)
    with pytest.raises(RuntimeError, match="active contribution"):
        contribution.writer.next_trajectory_sequence()


def test_contribution_failure_rolls_back_bundle_and_preserves_retryability(tmp_path):
    db, _service, store = _store(tmp_path, "contribution-failure.db")
    session = store.create_session(title="Still temporary", ephemeral=True)
    store.stage_session_library_policy(
        session.id,
        ConsoleLibraryPolicyCandidate(
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
        ),
    )
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="hi")
    failing = _Contribution(fail=True)
    before = _memory_state(store, session.id)

    with pytest.raises(RuntimeError, match="injected contribution failure"):
        store.promote_ephemeral_session(session.id, contributions=(failing,))

    assert _memory_state(store, session.id) == before
    assert _conversation_count(db) == 0
    failing.fail = False
    assert store.promote_ephemeral_session(session.id, contributions=(failing,))
    assert _conversation_count(db) == 1


def test_promotion_rejects_unresolved_operation_before_any_write(tmp_path):
    db, _service, store = _store(tmp_path, "unresolved.db")
    session = store.create_session(ephemeral=True)
    store.set_unresolved_promotion_operation(session.id, "future-preparation")

    with pytest.raises(
        RuntimeError, match="Finish or discard the pending turn before saving"
    ):
        store.promote_ephemeral_session(session.id)

    assert session.ephemeral is True
    assert _conversation_count(db) == 0
