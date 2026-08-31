"""Shared-prefix trace lineage for Console conversation forks."""

from __future__ import annotations

from collections.abc import Iterable
import sqlite3

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> Iterable[CharactersRAGDB]:
    database = CharactersRAGDB(":memory:", "console-trace-fork-test")
    yield database
    database.close_connection()


def _conversation(db: CharactersRAGDB, title: str) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": title})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": title,
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _root_with_surface(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> tuple[str, str, str, str, str]:
    conversation_id, message_id = _conversation(db, "source")
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        policy = repository.ensure_policy(
            cursor,
            FrozenTracePolicy(
                policy_id=new_opaque_id(),
                credential_filter_version="cred-v1",
                pii_redaction_enabled=False,
                pii_ruleset_revision_id=None,
            ),
        )
        revision = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_create",
            live_message_id=message_id,
        )
        node = repository.append_surface_node(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="message",
            reference=SemanticRevisionRef(revision.revision_id),
        )
        repository.append_event(
            cursor,
            segment_id=segment.segment_id,
            sequence=0,
            event_type="surface_append",
            surface_node_id=node.node_id,
        )
    return (
        conversation_id,
        owner.owner_id,
        segment.segment_id,
        policy.policy_id,
        node.node_id,
    )


def _reserve_call(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    owner_id: str,
    segment_id: str,
    policy_id: str,
    turn_id: str,
    event_sequence: int,
    call_sequence: int,
) -> str:
    call = repository.reserve_call(
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        turn_id=turn_id,
        run_id=f"run-{turn_id}",
        call_sequence=call_sequence,
        idempotency_key=f"idem-{owner_id}-{turn_id}-{call_sequence}",
        policy_id=policy_id,
    )
    repository.append_event(
        cursor,
        segment_id=segment_id,
        sequence=event_sequence,
        event_type="call_boundary",
        call_id=call.call_id,
    )
    return call.call_id


def _payload_counts(cursor: sqlite3.Cursor) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for table, byte_expression in (
        ("console_trace_semantic_revisions", "0"),
        ("console_trace_artifacts", "COALESCE(SUM(byte_length), 0)"),
        ("console_trace_surface_nodes", "0"),
        ("console_trace_events", "0"),
        ("console_trace_calls", "0"),
    ):
        row = cursor.execute(
            f"SELECT COUNT(*), {byte_expression} FROM {table}"
        ).fetchone()
        assert row is not None
        result[table] = (int(row[0]), int(row[1]))
    return result


def test_fork_attaches_shared_boundary_without_copying_trace_payload_rows(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, source_owner_id, source_segment_id, policy_id, source_head_id = (
        _root_with_surface(db, repository)
    )
    child_id, _ = _conversation(db, "child")

    with db.transaction(immediate=True) as cursor:
        source_call = _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=1,
            call_sequence=0,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None
        assert boundary.parent_segment_id == source_segment_id
        assert boundary.inherited_through_sequence == 1
        assert boundary.inherited_surface_head_id == source_head_id
        before = _payload_counts(cursor)

        child_owner = repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )

        assert _payload_counts(cursor) == before
        assert child_owner.conversation_id == child_id
        child_segment = repository.get_segment(cursor, child_owner.root_segment_id)
        assert child_segment is not None
        assert child_segment.parent_segment_id == source_segment_id
        assert child_segment.inherited_through_sequence == 1
        assert child_segment.inherited_surface_head_id == source_head_id
        assert [
            call.call_id
            for call in repository.read_conversation_call_lineage(cursor, child_id)
        ] == [source_call]


def test_fork_lineage_is_stable_across_divergence_and_nested_forks(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, source_owner_id, source_segment_id, policy_id, _source_head_id = (
        _root_with_surface(db, repository)
    )
    child_id, _ = _conversation(db, "child")
    grandchild_id, _ = _conversation(db, "grandchild")

    with db.transaction(immediate=True) as cursor:
        inherited_call = _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=1,
            call_sequence=0,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None
        child_owner = repository.attach_fork_owner(
            cursor,
            conversation_id=child_id,
            boundary=boundary,
        )

        source_only_call = _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-source-suffix",
            event_sequence=2,
            call_sequence=0,
        )
        child_call = _reserve_call(
            cursor,
            repository,
            owner_id=child_owner.owner_id,
            segment_id=child_owner.root_segment_id,
            policy_id=policy_id,
            turn_id="turn-child-suffix",
            event_sequence=2,
            call_sequence=0,
        )

        child_calls = repository.read_conversation_call_lineage(cursor, child_id)
        assert [call.call_id for call in child_calls] == [inherited_call, child_call]
        assert source_only_call not in {call.call_id for call in child_calls}
        assert [
            call.call_id
            for call in repository.read_conversation_call_lineage(cursor, source_id)
        ] == [inherited_call, source_only_call]

        inherited_boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=child_id,
            included_turn_ids=("turn-1",),
        )
        assert inherited_boundary is not None
        assert inherited_boundary.source_owner_id == child_owner.owner_id
        assert inherited_boundary.parent_segment_id == source_segment_id
        assert inherited_boundary.inherited_through_sequence == 1

        nested_boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=child_id,
            included_turn_ids=("turn-child-suffix",),
        )
        assert nested_boundary is not None
        grandchild_owner = repository.attach_fork_owner(
            cursor,
            conversation_id=grandchild_id,
            boundary=nested_boundary,
        )
        assert grandchild_owner.owner_id != child_owner.owner_id
        assert [
            call.call_id
            for call in repository.read_conversation_call_lineage(
                cursor, grandchild_id
            )
        ] == [inherited_call, child_call]


def test_attach_fork_owner_rejects_stale_or_foreign_source_ownership(
    db: CharactersRAGDB,
) -> None:
    repository = ConsoleTraceRepository()
    source_id, source_owner_id, source_segment_id, policy_id, _source_head_id = (
        _root_with_surface(db, repository)
    )
    child_id, _ = _conversation(db, "child")
    with db.transaction(immediate=True) as cursor:
        _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=1,
            call_sequence=0,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None
        repository.detach_owner(
            cursor,
            owner_id=source_owner_id,
            detached_at="2026-08-30T12:00:00Z",
        )
        with pytest.raises(ValueError, match="fork_boundary_owner"):
            repository.attach_fork_owner(
                cursor,
                conversation_id=child_id,
                boundary=boundary,
            )

    assert repository.capture_fork_boundary(
        db.get_connection().cursor(),
        conversation_id=child_id,
        included_turn_ids=("turn-1",),
    ) is None


def test_temporary_fork_propagates_durable_prefix_and_attaches_it_once_on_save(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    repository = service.console_trace_repository
    source_id, source_owner_id, source_segment_id, policy_id, _source_head_id = (
        _root_with_surface(db, repository)
    )
    with db.transaction(immediate=True) as cursor:
        source_call = _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=1,
            call_sequence=0,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None

    store = ConsoleChatStore(persistence=service)
    temporary_parent = store.create_session(
        title="temporary parent",
        ephemeral=True,
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
    )
    temporary_parent.fork_projection = True
    temporary_parent.fork_trace_boundary = boundary
    parent_message = store.append_message(
        temporary_parent.id,
        role=ConsoleMessageRole.USER,
        content="in-memory prefix",
        persist=False,
    )
    fence = store.issue_fork_fence(parent_message.id)
    snapshot = store.stage_fork_snapshot(
        fence,
        title="temporary child",
        fork_session_id="temporary-child",
        fork_conversation_id=None,
    )
    child = store.register_fork_snapshot(snapshot)

    assert snapshot.trace_boundary == boundary
    assert child.ephemeral is True
    assert child.fork_trace_boundary == boundary
    source_counts = db.get_connection().execute(
        "SELECT COUNT(*) FROM console_trace_calls"
    ).fetchone()[0]

    child_conversation_id = store.promote_ephemeral_session(child.id)

    assert child_conversation_id is not None
    assert child.ephemeral is False
    with db.transaction() as cursor:
        assert cursor.execute(
            "SELECT COUNT(*) FROM console_trace_calls"
        ).fetchone()[0] == source_counts
        assert [
            call.call_id
            for call in repository.read_conversation_call_lineage(
                cursor,
                child_conversation_id,
            )
        ] == [source_call]


@pytest.mark.asyncio
async def test_durable_source_can_stage_temporary_fork_with_shared_trace_boundary(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    repository = service.console_trace_repository
    source_id, source_owner_id, source_segment_id, policy_id, _source_head_id = (
        _root_with_surface(db, repository)
    )
    source_message_id = db.get_connection().execute(
        "SELECT id FROM messages WHERE conversation_id = ?",
        (source_id,),
    ).fetchone()[0]
    db.set_conversation_active_leaf(source_id, source_message_id)
    with db.transaction(immediate=True) as cursor:
        _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id=source_message_id,
            event_sequence=1,
            call_sequence=0,
        )

    store = ConsoleChatStore(persistence=service)
    hydrated = console_messages_from_conversation_tree(
        ChatConversationService(db).get_conversation_tree(source_id),
        db=db,
    )
    assert hydrated[0].turn_id is None
    assert hydrated[0].trace_turn_id == source_message_id
    session = store.restore_persisted_session(
        title="source",
        workspace_id=None,
        persisted_conversation_id=source_id,
        all_nodes=hydrated,
        active_leaf_persisted_id=source_message_id,
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
    )
    await store.hydrate_session_library_policy(session.id)
    boundary_message = store.messages_for_session(session.id)[0]

    snapshot = store.stage_fork_snapshot(
        store.issue_fork_fence(boundary_message.id),
        title="temporary child",
        fork_session_id="temporary-child",
        fork_conversation_id=None,
        destination_durable=False,
    )
    child = store.register_fork_snapshot(snapshot)

    assert snapshot.durable is False
    assert snapshot.source_conversation_id is None
    assert snapshot.trace_boundary is not None
    assert snapshot.messages[0].source_persisted_message_id is None
    assert child.ephemeral is True
    assert child.fork_trace_boundary == snapshot.trace_boundary


def test_failed_temporary_fork_promotion_keeps_source_and_child_ownership(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ChatPersistenceService(db)
    repository = service.console_trace_repository
    source_id, source_owner_id, source_segment_id, policy_id, _source_head_id = (
        _root_with_surface(db, repository)
    )
    with db.transaction(immediate=True) as cursor:
        _reserve_call(
            cursor,
            repository,
            owner_id=source_owner_id,
            segment_id=source_segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=1,
            call_sequence=0,
        )
        boundary = repository.capture_fork_boundary(
            cursor,
            conversation_id=source_id,
            included_turn_ids=("turn-1",),
        )
        assert boundary is not None

    store = ConsoleChatStore(persistence=service)
    child = store.create_session(title="temporary child", ephemeral=True)
    child.fork_projection = True
    child.fork_trace_boundary = boundary
    store.append_message(
        child.id,
        role=ConsoleMessageRole.USER,
        content="not yet saved",
        persist=False,
    )

    def fail_attach(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced trace attach failure")

    monkeypatch.setattr(repository, "attach_fork_owner", fail_attach)
    with pytest.raises(RuntimeError, match="forced trace attach failure"):
        store.promote_ephemeral_session(child.id)

    assert child.ephemeral is True
    assert child.persisted_conversation_id is None
    with db.transaction() as cursor:
        source_owner = repository.get_owner(cursor, source_owner_id)
        assert source_owner is not None and source_owner.attached
        assert source_owner.conversation_id == source_id
