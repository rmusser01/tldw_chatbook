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
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


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
        session.workspace_id,
        session.library_policy_holder.snapshot,
        session.library_policy_holder.explicitly_staged,
        session.library_policy_holder.save_pending,
        session.library_policy_hydrated,
        session.rag_scope_holder.scope,
        store._unresolved_promotion_operations.get(session_id),
        store._active_leaf_by_session.get(session_id),
        store._context_summary_by_session.get(session_id),
        tuple(
            (
                message.id,
                message.role,
                message.content,
                message.status,
                message.persisted_message_id,
                message.parent_message_id,
                message.attachments,
                message.image_data,
                message.image_mime_type,
            )
            for message in messages
        ),
    )


def _conversation_count(db):
    return db.get_connection().execute("SELECT COUNT(*) FROM conversations").fetchone()[0]


def _bundle_counts(db):
    connection = db.get_connection()
    return tuple(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "conversations",
            "messages",
            "console_conversation_library_policy",
            "message_attachments",
            "message_trajectory_metadata",
        )
    )


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


def _workspace_store(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.sqlite", "workspace-test")
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspace.sqlite", client_id="workspace-test")
    )
    registry.create_workspace(workspace_id="workspace-a", name="Workspace A")
    service = ChatPersistenceService(db, workspace_registry=registry)
    return db, registry, service, ConsoleChatStore(persistence=service)


@pytest.mark.parametrize("failure_point", ("policy", "message", "attachment", "contribution"))
def test_chat_failure_never_projects_cross_database_workspace_membership(
    tmp_path, monkeypatch, failure_point
):
    db, registry, service, store = _workspace_store(tmp_path)
    session = store.create_session(workspace_id="workspace-a", ephemeral=True)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        attachments=(
            MessageAttachment(b"zero", "image/png", "zero.png", 0),
            MessageAttachment(b"one", "image/png", "one.png", 1),
        ),
    )
    contribution = _Contribution(fail=failure_point == "contribution")
    if failure_point == "policy":
        monkeypatch.setattr(
            service.console_library_policy_repository,
            "insert",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("policy fail")),
        )
    elif failure_point == "message":
        monkeypatch.setattr(
            service,
            "create_message",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("message fail")),
        )
    elif failure_point == "attachment":
        monkeypatch.setattr(
            db,
            "set_message_attachments",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("attachment fail")),
        )

    with pytest.raises(RuntimeError):
        store.promote_ephemeral_session(session.id, contributions=(contribution,))

    assert _conversation_count(db) == 0
    assert registry.list_workspace_conversations("workspace-a") == ()


def test_workspace_projection_failure_keeps_commit_and_retries_idempotently(
    tmp_path, monkeypatch
):
    db, registry, service, store = _workspace_store(tmp_path)
    session = store.create_session(workspace_id="workspace-a", ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    original_link = registry.link_membership
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("workspace unavailable")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(registry, "link_membership", fail_once)

    conversation_id = store.promote_ephemeral_session(session.id)

    assert conversation_id is not None
    assert session.ephemeral is False
    assert _conversation_count(db) == 1
    assert store.has_pending_workspace_projection(session.id)
    assert store.promote_ephemeral_session(session.id) is None
    assert not store.has_pending_workspace_projection(session.id)
    assert [row.item_id for row in registry.list_workspace_conversations("workspace-a")] == [conversation_id]
    assert store.retry_pending_workspace_projection(session.id)
    assert store.promote_ephemeral_session(session.id) is None
    assert _conversation_count(db) == 1


def test_workspace_projection_reconciles_after_restart_without_duplicate_membership(
    tmp_path, monkeypatch
):
    db, registry, service, store = _workspace_store(tmp_path)
    session = store.create_session(workspace_id="workspace-a", ephemeral=True)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    original_link = registry.link_membership
    monkeypatch.setattr(
        registry,
        "link_membership",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("workspace unavailable")
        ),
    )
    conversation_id = store.promote_ephemeral_session(session.id)
    assert registry.list_workspace_conversations("workspace-a") == ()

    monkeypatch.setattr(registry, "link_membership", original_link)
    restarted = ConsoleChatStore(persistence=service)
    restored = restarted.restore_persisted_session(
        title="Restarted",
        workspace_id="workspace-a",
        persisted_conversation_id=conversation_id,
        all_nodes=(),
    )
    assert restarted.has_pending_workspace_projection(restored.id)
    assert restarted.retry_pending_workspace_projection(restored.id)
    assert restarted.retry_pending_workspace_projection(restored.id)
    memberships = registry.list_workspace_conversations("workspace-a")
    assert [row.item_id for row in memberships] == [conversation_id]
    assert _conversation_count(db) == 1


def test_shadowed_legacy_persist_cannot_escape_atomic_workspace_promotion(
    tmp_path, monkeypatch
):
    db, registry, service, store = _workspace_store(tmp_path)
    session = store.create_session(workspace_id="workspace-a", ephemeral=True)
    session.user_display_name_override = "Rowan"
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    before = _memory_state(store, session.id)
    original_persist = store.persist_session_if_needed
    shadow_calls = 0

    def unsafe_legacy_shadow(session_id, *, strict_roleplay_context=False):
        nonlocal shadow_calls
        shadow_calls += 1
        original_persist(
            session_id,
            strict_roleplay_context=strict_roleplay_context,
        )
        raise RuntimeError("strict roleplay/project-context failure")

    monkeypatch.setattr(store, "persist_session_if_needed", unsafe_legacy_shadow)
    monkeypatch.setattr(
        service.console_library_policy_repository,
        "insert",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("atomic policy failure")
        ),
    )

    with pytest.raises(RuntimeError, match="atomic policy failure"):
        store.promote_ephemeral_session(session.id)

    assert shadow_calls == 0
    assert _memory_state(store, session.id) == before
    assert _bundle_counts(db) == (0, 0, 0, 0, 0)
    assert registry.list_workspace_conversations("workspace-a") == ()
    binding = store.library_policy_coordinator._holders[session.id]
    assert binding.conversation_id is None


def test_promotion_without_atomic_adapter_refuses_before_any_write():
    class NonAtomicPersistence:
        db = None

        def __init__(self):
            self.create_calls = 0

        def create_conversation(self, **_kwargs):
            self.create_calls += 1
            return "legacy-conversation"

    persistence = NonAtomicPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(title="Temporary", ephemeral=True)
    before = _memory_state(store, session.id)

    with pytest.raises(RuntimeError, match="atomic promotion"):
        store.promote_ephemeral_session(session.id)

    assert persistence.create_calls == 0
    assert _memory_state(store, session.id) == before


@pytest.mark.parametrize(
    "failure_point",
    (
        "conversation",
        "policy",
        "message-0",
        "message-1",
        "attachment-0",
        "attachment-sidecar",
        "active-leaf",
        "context-summary",
        "contribution-0",
        "contribution-1",
    ),
)
def test_each_bundle_write_boundary_rolls_back_exactly_and_retries_once(
    tmp_path, monkeypatch, failure_point
):
    db, service, store = _store(tmp_path, f"boundary-{failure_point}.db")
    session = store.create_session(title="Boundary", ephemeral=True)
    user = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="hello",
        attachments=(
            MessageAttachment(b"zero", "image/png", "zero.png", 0),
            MessageAttachment(b"one", "image/png", "one.png", 1),
        ),
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="hi"
    )
    store._context_summary_by_session[session.id] = ("summary", assistant.id)
    contributions = [_Contribution(), _Contribution()]
    before = _memory_state(store, session.id)
    failure = RuntimeError(f"injected {failure_point}")

    def fail_once(original, predicate=lambda *_args, **_kwargs: True):
        failed = False

        def wrapped(*args, **kwargs):
            nonlocal failed
            if not failed and predicate(*args, **kwargs):
                failed = True
                raise failure
            return original(*args, **kwargs)

        return wrapped

    if failure_point == "conversation":
        monkeypatch.setattr(
            service, "create_conversation", fail_once(service.create_conversation)
        )
    elif failure_point == "policy":
        repository = service.console_library_policy_repository
        monkeypatch.setattr(repository, "insert", fail_once(repository.insert))
    elif failure_point.startswith("message-"):
        target = int(failure_point.rsplit("-", 1)[1])
        call_index = -1
        original = service.create_message

        def fail_message(*args, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == target:
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "create_message", fail_message)
    elif failure_point == "attachment-0":
        monkeypatch.setattr(
            db,
            "add_message",
            fail_once(
                db.add_message,
                lambda payload: payload.get("image_data") == b"zero",
            ),
        )
    elif failure_point == "attachment-sidecar":
        monkeypatch.setattr(
            db,
            "set_message_attachments",
            fail_once(db.set_message_attachments, lambda _message_id, rows: bool(rows)),
        )
    elif failure_point == "active-leaf":
        monkeypatch.setattr(
            db,
            "set_conversation_active_leaf",
            fail_once(db.set_conversation_active_leaf),
        )
    elif failure_point == "context-summary":
        monkeypatch.setattr(
            db,
            "set_conversation_context_summary",
            fail_once(db.set_conversation_context_summary),
        )
    else:
        contributions[int(failure_point.rsplit("-", 1)[1])].fail = True

    expected_error = (
        "injected contribution failure"
        if failure_point.startswith("contribution-")
        else f"injected {failure_point}"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        store.promote_ephemeral_session(session.id, contributions=contributions)

    assert _memory_state(store, session.id) == before
    assert _bundle_counts(db) == (0, 0, 0, 0, 0)
    for contribution in contributions:
        contribution.fail = False
    conversation_id = store.promote_ephemeral_session(
        session.id, contributions=contributions
    )
    assert conversation_id is not None
    assert _bundle_counts(db) == (1, 2, 1, 1, 2)
    row = db.get_message_by_id(
        store._nodes_by_session[session.id][user.id].persisted_message_id
    )
    assert row["image_data"] == b"zero"
    attachments = db.get_attachments_for_messages([row["id"]])
    assert attachments[row["id"]][0]["data"] == b"one"
