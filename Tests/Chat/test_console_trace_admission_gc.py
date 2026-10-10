"""Canonical revision identity survives the admission-to-reservation gap."""

from contextlib import closing

from Tests.Chat.test_console_trace_runtime import _saved_message, _semantic_request
from Tests.Chat.test_console_semantic_revision_coordinator import (
    _message,
    _reference_under_policies,
)
from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator
from tldw_chatbook.Chat.console_prepared_request import (
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_provenance import ConsoleRequestRoute
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _complete_migration(database):
    with database.transaction(immediate=True) as cursor:
        cursor.execute(
            "UPDATE console_trace_migration_state SET status = 'logical_complete'"
        )


def test_collection_between_admission_and_reservation_preserves_exact_revision():
    with closing(CharactersRAGDB(":memory:", "admission-gc")) as database:
        conversation = database.add_conversation({"title": "admitted send"})
        _, revision = _saved_message(database, conversation, "synthetic request")
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        prepared = prepare_provider_request(
            _semantic_request(
                [{"role": "user", "content": "synthetic request"}], [revision], policy
            ),
            wire_style="distinct_roles",
            provider="openai",
            model="gpt-test",
            capacity=resolve_request_capacity(context_window_tokens=None),
        )
        _complete_migration(database)
        assert (
            TraceGarbageCollector(database).collect(request_id=new_opaque_id()).status
            == "completed"
        )
        boundary = ConsoleTraceBoundaryFactory(database)(
            prepared, None, ConsoleRequestRoute.FRESH
        )
        assert boundary._current_revision_id == revision.revision_id


def test_unreferenced_revision_is_collected_after_canonical_message_deletion():
    with closing(CharactersRAGDB(":memory:", "admission-gc-delete")) as database:
        conversation = database.add_conversation({"title": "canonical owner"})
        message_id, revision = _saved_message(
            database, conversation, "synthetic request"
        )
        _complete_migration(database)
        collector = TraceGarbageCollector(database)
        collector.collect(request_id=new_opaque_id())
        with database.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT revision_id FROM console_trace_semantic_revisions WHERE revision_id = ?",
                    (revision.revision_id,),
                ).fetchone()
                is not None
            )
        with database.transaction(immediate=True) as cursor:
            SemanticRevisionCoordinator(database).mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_delete",
                hard_delete=True,
            )
        collector.collect(request_id=new_opaque_id())
        with database.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT revision_id FROM console_trace_semantic_revisions WHERE source_message_id = ?",
                    (message_id,),
                ).fetchone()
                is None
            )


def test_canonical_ancestry_cannot_retain_payload_through_another_owners_policy():
    with closing(CharactersRAGDB(":memory:", "admission-gc-shared-policy")) as database:
        first_conversation, first_message = _message(
            database, content="archived private body"
        )
        second_conversation, second_message = _message(
            database, content="independent body"
        )
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        old_revision, _node, _policies = _reference_under_policies(
            database,
            conversation_id=first_conversation,
            message_id=first_message,
            policies=(policy,),
        )
        _reference_under_policies(
            database,
            conversation_id=second_conversation,
            message_id=second_message,
            policies=(policy,),
        )
        repository = ConsoleTraceRepository()
        with database.transaction(immediate=True) as cursor:
            result = SemanticRevisionCoordinator(database).mutate_message(
                cursor,
                message_id=first_message,
                creation_reason="message_edit",
                mutate=lambda scoped: scoped.execute(
                    "UPDATE messages SET content = ? WHERE id = ?",
                    ("current body", first_message),
                ),
            )
            binding = repository.get_revision_policy_binding(
                cursor,
                revision_id=old_revision,
                policy_id=policy.policy_id,
            )
            assert binding is not None and binding.artifact_id is not None
            owner = cursor.execute(
                "SELECT owner_id FROM console_trace_owners WHERE conversation_id = ?",
                (first_conversation,),
            ).fetchone()[0]
            for call in cursor.execute(
                "SELECT call_id FROM console_trace_calls WHERE owner_id = ?",
                (owner,),
            ).fetchall():
                repository.advance_call_state(
                    cursor,
                    call_id=call[0],
                    target=TraceCallState.NOT_DISPATCHED,
                    occurred_at="2026-10-04T12:00:00Z",
                )
            repository.detach_owner(
                cursor, owner_id=owner, detached_at="2026-10-04T12:01:00Z"
            )
        _complete_migration(database)
        assert (
            TraceGarbageCollector(database).collect(request_id=new_opaque_id()).status
            == "completed"
        )
        with database.transaction() as cursor:
            assert (
                repository.get_revision_policy_binding(
                    cursor,
                    revision_id=old_revision,
                    policy_id=policy.policy_id,
                )
                is None
            )
            assert repository.get_artifact(cursor, binding.artifact_id) is None
            assert repository.get_semantic_revision(cursor, old_revision) is not None
            current = repository.get_semantic_revision(
                cursor, result.current_revision_id
            )
            assert (
                current is not None and current.predecessor_revision_id == old_revision
            )
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
