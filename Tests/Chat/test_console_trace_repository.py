"""Transaction and invariants for the Console semantic trace repository."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping, MutableMapping
import hashlib
import inspect
from pathlib import Path
import sqlite3
from typing import cast

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    InvalidTraceTransition,
    SemanticRevisionRef,
    SurfaceReplacement,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    HeaderComponentRef,
    IntegrityState,
    TraceIdentityConflict,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "console-trace-repository-test")
    yield database
    database.close_connection()


@pytest.fixture
def repository() -> ConsoleTraceRepository:
    return ConsoleTraceRepository()


def _conversation_with_message(db: CharactersRAGDB) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "semantic trace"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "must never enter semantic revision metadata",
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="cred-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def _owned_root(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> tuple[str, str, str]:
    conversation_id, message_id = _conversation_with_message(db)
    with db.transaction() as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
    return conversation_id, message_id, owner.owner_id


def _surface_and_header(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    message_id: str,
    owner_id: str,
) -> tuple[str, str, str, str, str]:
    owner = repository.get_owner(cursor, owner_id)
    assert owner is not None
    policy = repository.ensure_policy(cursor, _policy())
    revision = repository.ensure_semantic_revision(
        cursor,
        source_conversation_id=conversation_id,
        source_message_id=message_id,
        revision_sequence=0,
        normalized_role="user",
        content_kind="text",
        creation_reason="message_created",
        live_message_id=message_id,
    )
    node = repository.append_surface_node(
        cursor,
        segment_id=owner.root_segment_id,
        sequence=0,
        predecessor_node_id=None,
        component_kind="message",
        reference=SemanticRevisionRef(revision.revision_id),
    )
    repository.append_event(
        cursor,
        segment_id=owner.root_segment_id,
        sequence=0,
        event_type="surface_append",
        surface_node_id=node.node_id,
    )
    header = repository.create_or_reuse_request_header(
        cursor,
        provider_name="openai",
        model_name="gpt-test",
        route_identity="chat_completions",
        endpoint_identity="public_api",
        generation_parameters={"temperature": 0},
        adapter_defaults={},
        response_format={},
        reasoning_controls={},
        components=(),
    )
    return (
        owner.root_segment_id,
        policy.policy_id,
        revision.revision_id,
        node.node_id,
        header.header_id,
    )


def _bound_call(
    cursor: sqlite3.Cursor,
    repository: ConsoleTraceRepository,
    *,
    conversation_id: str,
    message_id: str,
    owner_id: str,
    idempotency_key: str,
) -> str:
    segment_id, policy_id, _revision_id, node_id, header_id = _surface_and_header(
        cursor,
        repository,
        conversation_id=conversation_id,
        message_id=message_id,
        owner_id=owner_id,
    )
    call = repository.reserve_call(
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        turn_id="turn-1",
        run_id="run-1",
        call_sequence=0,
        idempotency_key=idempotency_key,
        policy_id=policy_id,
    )
    return repository.bind_call(
        cursor,
        call_id=call.call_id,
        surface_node_id=node_id,
        request_header_id=header_id,
        provider_name="openai",
        model_name="gpt-test",
        route_identity="chat_completions",
    ).call_id


def test_artifact_exact_reuse_and_seeded_digest_collision_get_distinct_ids(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    sanitized = b'{"safe":"value"}'
    digest = hashlib.sha256(sanitized).hexdigest()
    with db.transaction() as cursor:
        first = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=sanitized,
            media_type="application/json",
            normalization_version="json-v1",
        )
        exact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=sanitized,
            media_type="application/json",
            normalization_version="json-v1",
        )
        collision_id = new_opaque_id()
        cursor.execute(
            """
            INSERT INTO console_trace_artifacts(
              artifact_id, identity_digest, media_type, normalization_version,
              sanitized_bytes, byte_length
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                collision_id,
                digest,
                "text/plain",
                "text-v1",
                sqlite3.Binary(b"different"),
                9,
            ),
        )
        distinct = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=sanitized,
            media_type="text/plain",
            normalization_version="text-v1",
        )
    assert exact.artifact_id == first.artifact_id
    assert distinct.artifact_id not in {first.artifact_id, collision_id}
    assert distinct.identity_digest == digest
    assert (
        repository.get_artifact(db.get_connection().cursor(), distinct.artifact_id)
        == distinct
    )


def test_revision_policy_binding_reuses_exact_and_rejects_incompatible(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, _owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        policy = repository.ensure_policy(cursor, _policy())
        revision = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_created",
            live_message_id=message_id,
        )
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b"safe",
            media_type="text/plain",
            normalization_version="text-v1",
        )
        before = repository.get_graph_epoch(cursor)
        first = repository.bind_revision_policy(
            cursor,
            revision_id=revision.revision_id,
            policy_id=policy.policy_id,
            artifact_id=artifact.artifact_id,
        )
        exact = repository.bind_revision_policy(
            cursor,
            revision_id=revision.revision_id,
            policy_id=policy.policy_id,
            artifact_id=artifact.artifact_id,
        )
        assert repository.get_graph_epoch(cursor) == before + 1
        with pytest.raises(TraceIdentityConflict, match="revision_policy_binding"):
            repository.bind_revision_policy(
                cursor,
                revision_id=revision.revision_id,
                policy_id=policy.policy_id,
                omission_reason_code="capture_disabled",
            )
    assert exact == first


def test_semantic_revision_is_opaque_metadata_without_body_or_digest(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id = _conversation_with_message(db)
    with db.transaction() as cursor:
        first = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_created",
            live_message_id=message_id,
        )
        exact = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_id,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_created",
            live_message_id=message_id,
        )
        with pytest.raises(TraceIdentityConflict, match="semantic_revision"):
            repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_id,
                revision_sequence=0,
                normalized_role="assistant",
                content_kind="text",
                creation_reason="message_created",
                live_message_id=message_id,
            )
        columns = {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(console_trace_semantic_revisions)"
            )
        }
    assert first == exact
    assert {"body", "digest", "history", "history_list"}.isdisjoint(columns)
    assert "must never enter" not in repr(first)


def test_semantic_revision_epoch_matrix_counts_live_and_predecessor_edges_once(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, no_edge_message = _conversation_with_message(db)
    message_ids = [no_edge_message]
    for label in ("live", "predecessor", "both"):
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": label,
            }
        )
        assert message_id is not None
        message_ids.append(message_id)

    with db.transaction() as cursor:
        no_edge = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[0],
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="locator_retired",
            live_locator_retired_at="2026-08-28T12:00:00Z",
        )
        assert repository.get_graph_epoch(cursor) == 0
        assert (
            repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_ids[0],
                revision_sequence=0,
                normalized_role="user",
                content_kind="text",
                creation_reason="locator_retired",
                live_locator_retired_at="2026-08-28T12:00:00Z",
            )
            == no_edge
        )
        assert repository.get_graph_epoch(cursor) == 0

        live = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[1],
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_created",
            live_message_id=message_ids[1],
        )
        assert repository.get_graph_epoch(cursor) == 1
        assert (
            repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_ids[1],
                revision_sequence=0,
                normalized_role="user",
                content_kind="text",
                creation_reason="message_created",
                live_message_id=message_ids[1],
            )
            == live
        )
        assert repository.get_graph_epoch(cursor) == 1

        predecessor_base = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[2],
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="locator_retired",
            live_locator_retired_at="2026-08-28T12:00:00Z",
        )
        predecessor = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[2],
            revision_sequence=1,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_edited",
            predecessor_revision_id=predecessor_base.revision_id,
            live_locator_retired_at="2026-08-28T12:00:01Z",
        )
        assert repository.get_graph_epoch(cursor) == 2
        assert (
            repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_ids[2],
                revision_sequence=1,
                normalized_role="user",
                content_kind="text",
                creation_reason="message_edited",
                predecessor_revision_id=predecessor_base.revision_id,
                live_locator_retired_at="2026-08-28T12:00:01Z",
            )
            == predecessor
        )
        assert repository.get_graph_epoch(cursor) == 2

        both_base = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[3],
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="locator_retired",
            live_locator_retired_at="2026-08-28T12:00:00Z",
        )
        both = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=conversation_id,
            source_message_id=message_ids[3],
            revision_sequence=1,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_edited",
            predecessor_revision_id=both_base.revision_id,
            live_message_id=message_ids[3],
        )
        assert repository.get_graph_epoch(cursor) == 3
        assert (
            repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_ids[3],
                revision_sequence=1,
                normalized_role="user",
                content_kind="text",
                creation_reason="message_edited",
                predecessor_revision_id=both_base.revision_id,
                live_message_id=message_ids[3],
            )
            == both
        )
        assert repository.get_graph_epoch(cursor) == 3


def test_request_header_reuses_only_complete_exact_header(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    with db.transaction() as cursor:
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b"schema",
            media_type="application/schema+json",
            normalization_version="json-v1",
        )
        component = HeaderComponentRef("tool_schema", 0, artifact.artifact_id)
        first = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="responses",
            endpoint_identity="public_api",
            generation_parameters={"temperature": 0},
            adapter_defaults={},
            response_format={"type": "text"},
            reasoning_controls={},
            components=(component,),
        )
        exact = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="responses",
            endpoint_identity="public_api",
            generation_parameters={"temperature": 0},
            adapter_defaults={},
            response_format={"type": "text"},
            reasoning_controls={},
            components=(component,),
        )
        changed = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="responses",
            endpoint_identity="public_api",
            generation_parameters={"temperature": 1},
            adapter_defaults={},
            response_format={"type": "text"},
            reasoning_controls={},
            components=(component,),
        )
    assert exact.header_id == first.header_id
    assert changed.header_id != first.header_id
    assert first.components == (component,)


def test_request_header_nested_json_is_exact_and_rejects_coercing_collisions(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    with db.transaction() as cursor:

        def create(parameters: Mapping[str, object]):
            return repository.create_or_reuse_request_header(
                cursor,
                provider_name="openai",
                model_name="gpt-test",
                route_identity="responses",
                endpoint_identity="public_api",
                generation_parameters=parameters,
                adapter_defaults={},
                response_format={},
                reasoning_controls={},
                components=(),
            )

        string_key = create({"nested": {"1": "x", "items": [1, True, None]}})
        reordered = create({"nested": {"items": [1, True, None], "1": "x"}})
        distinct = create({"nested": {"1": "x", "items": [True, 1, None]}})
        with pytest.raises(ValueError, match="generation_parameters"):
            create({"nested": {1: "x"}})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="generation_parameters"):
            create({"nested": ("coerced", "array")})
        with pytest.raises(ValueError, match="generation_parameters"):
            create({"nested": object()})
        count = cursor.execute(
            "SELECT COUNT(*) FROM console_trace_request_headers"
        ).fetchone()[0]

    assert reordered.header_id == string_key.header_id
    assert distinct.header_id != string_key.header_id
    assert count == 2


def test_request_header_accepts_mapping_implementations_recursively(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    parameters = UserDict(
        {
            "nested": UserDict({"items": [1, True, None]}),
        }
    )
    with db.transaction() as cursor:
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="responses",
            endpoint_identity="public_api",
            generation_parameters=parameters,
            adapter_defaults=UserDict(),
            response_format=UserDict(),
            reasoning_controls=UserDict(),
            components=(),
        )

    assert header.generation_parameters == {
        "nested": {"items": (1, True, None)},
    }


def test_request_header_json_state_is_deeply_immutable(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    with db.transaction() as cursor:
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="responses",
            endpoint_identity="public_api",
            generation_parameters={"nested": {"items": [1, True, None]}},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )

    root = cast(MutableMapping[str, object], header.generation_parameters)
    nested = cast(MutableMapping[str, object], root["nested"])
    items = cast(list[object], nested["items"])
    with pytest.raises(TypeError):
        root["mutated"] = True
    with pytest.raises(TypeError):
        nested["mutated"] = True
    with pytest.raises(AttributeError):
        items.append("mutated")


def test_call_reservation_reuses_exact_key_and_rejects_ambiguous_identity(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, policy_id, *_ = _surface_and_header(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
        )
        before = repository.get_graph_epoch(cursor)
        first = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="immutable-send-key",
            policy_id=policy_id,
        )
        exact = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="immutable-send-key",
            policy_id=policy_id,
        )
        assert repository.get_graph_epoch(cursor) == before + 1
        with pytest.raises(TraceIdentityConflict, match="call_reservation"):
            repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="different-turn",
                run_id="run-1",
                call_sequence=0,
                idempotency_key="immutable-send-key",
                policy_id=policy_id,
            )
    assert exact.call_id == first.call_id


def test_call_reservation_rejects_reverse_and_crossed_identity_conflicts(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, policy_id, *_ = _surface_and_header(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
        )
        repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="identity-a",
            policy_id=policy_id,
        )
        repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=1,
            idempotency_key="identity-b",
            policy_id=policy_id,
        )
        with pytest.raises(TraceIdentityConflict, match="call_reservation"):
            repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id="run-1",
                call_sequence=0,
                idempotency_key="different-key",
                policy_id=policy_id,
            )
        with pytest.raises(TraceIdentityConflict, match="call_reservation"):
            repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id="run-1",
                call_sequence=1,
                idempotency_key="identity-a",
                policy_id=policy_id,
            )


def test_call_reservation_reconciles_insert_constraint_race(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    class ReservationRaceCursor:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self._cursor = cursor
            self.connection = cursor.connection
            self.injected = False

        def execute(
            self,
            sql: str,
            parameters: tuple[object, ...] = (),
        ) -> sqlite3.Cursor:
            if not self.injected and "INSERT INTO console_trace_calls(" in sql:
                self.injected = True
                self._cursor.execute(sql, (new_opaque_id(), *parameters[1:]))
                self._cursor.execute(
                    """UPDATE console_trace_graph_epoch
                          SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
                        WHERE singleton_id = 1"""
                )
            return self._cursor.execute(sql, parameters)

    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, policy_id, *_ = _surface_and_header(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
        )
        racing_cursor = ReservationRaceCursor(cursor)
        before = repository.get_graph_epoch(cursor)
        call = repository.reserve_call(
            cast(sqlite3.Cursor, racing_cursor),
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="insert-race",
            policy_id=policy_id,
        )
        count = cursor.execute(
            "SELECT COUNT(*) FROM console_trace_calls WHERE idempotency_key = ?",
            ("insert-race",),
        ).fetchone()[0]

    assert racing_cursor.injected is True
    assert call.idempotency_key == "insert-race"
    assert count == 1
    assert repository.get_graph_epoch(db.get_connection().cursor()) == before + 1


def test_lookup_before_write_requires_caller_transaction(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    policy = _policy()

    with pytest.raises(RuntimeError, match="caller_transaction_required"):
        repository.ensure_policy(db.get_connection().cursor(), policy)

    assert repository.get_policy(db.get_connection().cursor(), policy.policy_id) is None


def test_lookup_before_write_claims_write_intent_and_committed_call_retries(
    tmp_path: Path,
    repository: ConsoleTraceRepository,
) -> None:
    database_path = tmp_path / "semantic-trace-concurrency.sqlite"
    primary = CharactersRAGDB(database_path, "console-trace-primary")
    secondary = CharactersRAGDB(database_path, "console-trace-secondary")
    try:
        conversation_id, message_id, owner_id = _owned_root(primary, repository)
        with primary.transaction() as cursor:
            segment_id, policy_id, *_ = _surface_and_header(
                cursor,
                repository,
                conversation_id=conversation_id,
                message_id=message_id,
                owner_id=owner_id,
            )
            first = repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id="run-1",
                call_sequence=0,
                idempotency_key="committed-retry",
                policy_id=policy_id,
            )

        with secondary.transaction() as cursor:
            exact = repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id="run-1",
                call_sequence=0,
                idempotency_key="committed-retry",
                policy_id=policy_id,
            )
        assert exact.call_id == first.call_id

        policy = repository.get_policy(primary.get_connection().cursor(), policy_id)
        assert policy is not None
        with primary.transaction() as cursor:
            repository.ensure_policy(cursor, policy)
            secondary_connection = secondary.get_connection()
            secondary_connection.execute("PRAGMA busy_timeout = 0")
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                secondary_connection.execute("BEGIN IMMEDIATE")
            if secondary_connection.in_transaction:
                secondary_connection.rollback()
    finally:
        secondary.close_connection()
        primary.close_connection()


def test_bind_lifecycle_ordered_events_and_db_monotonic_rejection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, policy_id, _revision_id, node_id, header_id = _surface_and_header(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
        )
        call = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="call-order",
            policy_id=policy_id,
        )
        before_bind = repository.get_graph_epoch(cursor)
        bound = repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node_id,
            request_header_id=header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
        )
        assert repository.get_graph_epoch(cursor) == before_bind + 1
        repository.bind_call(
            cursor,
            call_id=call.call_id,
            surface_node_id=node_id,
            request_header_id=header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
        )
        assert repository.get_graph_epoch(cursor) == before_bind + 1
        repository.append_event(
            cursor,
            segment_id=segment_id,
            sequence=1,
            event_type="call_boundary",
            call_id=call.call_id,
        )
        repository.append_event(
            cursor,
            segment_id=segment_id,
            sequence=2,
            event_type="request_header_selection",
            call_id=call.call_id,
            request_header_id=header_id,
        )
        with pytest.raises(sqlite3.IntegrityError, match="append order"):
            repository.append_event(
                cursor,
                segment_id=segment_id,
                sequence=1,
                event_type="call_boundary",
                call_id=call.call_id,
            )
        epoch_before_lifecycle = repository.get_graph_epoch(cursor)
        started = repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-28T12:00:00Z",
        )
        responding = repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.RESPONSE_STARTED,
            occurred_at="2026-08-28T12:00:01Z",
        )
        complete = repository.advance_call_state(
            cursor,
            call_id=call.call_id,
            target=TraceCallState.COMPLETE,
            occurred_at="2026-08-28T12:00:02Z",
            usage={"input_tokens": 4},
            integrity_state="complete",
        )
        assert repository.get_graph_epoch(cursor) == epoch_before_lifecycle
    assert bound.state is TraceCallState.RESERVED
    assert started.state is TraceCallState.DISPATCH_STARTED
    assert responding.state is TraceCallState.RESPONSE_STARTED
    assert complete.state is TraceCallState.COMPLETE
    assert [
        event.sequence
        for event in repository.read_events(db.get_connection().cursor(), segment_id)
    ] == [0, 1, 2]
    assert [
        item.call_id
        for item in repository.read_calls(db.get_connection().cursor(), owner_id)
    ] == [call.call_id]


def test_dispatch_started_retry_is_exactly_idempotent(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        call_id = _bound_call(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
            idempotency_key="dispatch-retry",
        )
        first = repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-28T12:00:00Z",
            provider_operation_inactive=True,
        )
        retry = repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-28T12:00:00Z",
            provider_operation_inactive=True,
        )
        with pytest.raises(TraceIdentityConflict, match="call_lifecycle_retry"):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.DISPATCH_STARTED,
                occurred_at="2026-08-28T12:00:09Z",
            )
        with pytest.raises(InvalidTraceTransition):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.RESERVED,
                occurred_at="2026-08-28T12:00:00Z",
            )

    assert retry == first


def test_abandoned_retry_requires_authorization_evidence(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        call_id = _bound_call(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
            idempotency_key="abandoned-evidence-retry",
        )
        repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-28T12:00:00Z",
        )
        abandoned = repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.ABANDONED,
            occurred_at="2026-08-28T12:00:01Z",
            provider_operation_inactive=True,
        )
        assert (
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.ABANDONED,
                occurred_at="2026-08-28T12:00:01Z",
                provider_operation_inactive=True,
            )
            == abandoned
        )
        with pytest.raises(TraceIdentityConflict, match="call_lifecycle_retry"):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.ABANDONED,
                occurred_at="2026-08-28T12:00:01Z",
            )


def test_response_and_terminal_retries_require_compatible_set_once_fields(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        call_id = _bound_call(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
            idempotency_key="response-retry",
        )
        repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.DISPATCH_STARTED,
            occurred_at="2026-08-28T12:00:00Z",
        )
        response = repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.RESPONSE_STARTED,
            occurred_at="2026-08-28T12:00:01Z",
        )
        assert (
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.RESPONSE_STARTED,
                occurred_at="2026-08-28T12:00:01Z",
            )
            == response
        )
        with pytest.raises(TraceIdentityConflict, match="call_lifecycle_retry"):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.RESPONSE_STARTED,
                occurred_at="2026-08-28T12:00:08Z",
            )

        complete = repository.advance_call_state(
            cursor,
            call_id=call_id,
            target=TraceCallState.COMPLETE,
            occurred_at="2026-08-28T12:00:02Z",
            usage={"input_tokens": 4, "cached": True},
            integrity_state="complete",
            omission_reason_code="response_sanitized",
        )
        assert (
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.COMPLETE,
                occurred_at="2026-08-28T12:00:02Z",
                usage={"input_tokens": 4, "cached": True},
                integrity_state="complete",
                omission_reason_code="response_sanitized",
            )
            == complete
        )
        conflicts: tuple[tuple[str, int, IntegrityState, str], ...] = (
            ("2026-08-28T12:00:03Z", 4, "complete", "response_sanitized"),
            ("2026-08-28T12:00:02Z", 5, "complete", "response_sanitized"),
            ("2026-08-28T12:00:02Z", 4, "incomplete", "response_sanitized"),
            ("2026-08-28T12:00:02Z", 4, "complete", "capture_disabled"),
        )
        with pytest.raises(TraceIdentityConflict, match="call_lifecycle_retry"):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.COMPLETE,
                occurred_at="2026-08-28T12:00:02Z",
                usage={"input_tokens": 4, "cached": 1},
                integrity_state="complete",
                omission_reason_code="response_sanitized",
            )
        for timestamp, token_count, integrity, omission in conflicts:
            with pytest.raises(TraceIdentityConflict, match="call_lifecycle_retry"):
                repository.advance_call_state(
                    cursor,
                    call_id=call_id,
                    target=TraceCallState.COMPLETE,
                    occurred_at=timestamp,
                    usage={"input_tokens": token_count, "cached": True},
                    integrity_state=integrity,
                    omission_reason_code=omission,
                )
        with pytest.raises(InvalidTraceTransition):
            repository.advance_call_state(
                cursor,
                call_id=call_id,
                target=TraceCallState.STOPPED,
                occurred_at="2026-08-28T12:00:02Z",
            )


def test_child_segment_inherits_surface_and_nearest_owner_detach_shadows_parent(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    parent_conversation, parent_message, parent_owner_id = _owned_root(db, repository)
    child_conversation, child_message = _conversation_with_message(db)
    with db.transaction() as cursor:
        parent_segment, _policy_id, _revision_id, head_id, _header_id = (
            _surface_and_header(
                cursor,
                repository,
                conversation_id=parent_conversation,
                message_id=parent_message,
                owner_id=parent_owner_id,
            )
        )
        epoch_before_child = repository.get_graph_epoch(cursor)
        child = repository.create_segment(
            cursor,
            parent_segment_id=parent_segment,
            inherited_through_sequence=0,
            inherited_surface_head_id=head_id,
        )
        assert repository.get_graph_epoch(cursor) == epoch_before_child + 1
        effective_parent = repository.get_effective_owner(cursor, child.segment_id)
        assert effective_parent is not None
        assert effective_parent.owner_id == parent_owner_id
        child_owner = repository.attach_owner(
            cursor,
            conversation_id=child_conversation,
            root_segment_id=child.segment_id,
        )
        assert repository.get_effective_owner(cursor, child.segment_id) == child_owner
        child_revision = repository.ensure_semantic_revision(
            cursor,
            source_conversation_id=child_conversation,
            source_message_id=child_message,
            revision_sequence=0,
            normalized_role="user",
            content_kind="text",
            creation_reason="message_created",
            live_message_id=child_message,
        )
        inherited_append = repository.append_surface_node(
            cursor,
            segment_id=child.segment_id,
            sequence=1,
            predecessor_node_id=head_id,
            component_kind="message",
            reference=SemanticRevisionRef(child_revision.revision_id),
        )
        assert inherited_append.predecessor_node_id == head_id
        detached = repository.detach_owner(
            cursor,
            owner_id=child_owner.owner_id,
            detached_at="2026-08-28T12:00:00Z",
        )
    assert detached.attached is False
    effective = repository.get_effective_owner(
        db.get_connection().cursor(), child.segment_id
    )
    assert effective is not None
    assert effective.owner_id == child_owner.owner_id
    assert effective.attached is False


def test_surface_replacement_and_reads_preserve_frozen_shape(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, _policy_id, revision_id, first_node_id, _header_id = (
            _surface_and_header(
                cursor,
                repository,
                conversation_id=conversation_id,
                message_id=message_id,
                owner_id=owner_id,
            )
        )
        second = repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=1,
            predecessor_node_id=first_node_id,
            component_kind="message",
            reference=TraceOmission("message", "capture_disabled"),
        )
        replacement_node = repository.append_surface_node(
            cursor,
            segment_id=segment_id,
            sequence=2,
            predecessor_node_id=second.node_id,
            component_kind="message",
            reference=SemanticRevisionRef(revision_id),
        )
        replacement = repository.append_surface_replacement(
            cursor,
            segment_id=segment_id,
            replacement=SurfaceReplacement(
                predecessor_head_id=second.node_id,
                start_node_id=second.node_id,
                end_node_id=second.node_id,
                start_sequence=1,
                end_sequence=1,
                replacement_node_id=replacement_node.node_id,
            ),
        )
        earlier_replacement = repository.append_surface_replacement(
            cursor,
            segment_id=segment_id,
            replacement=SurfaceReplacement(
                predecessor_head_id=second.node_id,
                start_node_id=first_node_id,
                end_node_id=first_node_id,
                start_sequence=0,
                end_sequence=0,
                replacement_node_id=replacement_node.node_id,
            ),
        )
    assert repository.read_surface_nodes(db.get_connection().cursor(), segment_id) == (
        repository.get_surface_node(db.get_connection().cursor(), first_node_id),
        second,
        replacement_node,
    )
    assert repository.read_surface_replacements(
        db.get_connection().cursor(), segment_id
    ) == (earlier_replacement, replacement)


def test_response_link_is_exact_revision_or_sanitized_artifact(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, message_id, owner_id = _owned_root(db, repository)
    with db.transaction() as cursor:
        segment_id, policy_id, revision_id, _node_id, _header_id = _surface_and_header(
            cursor,
            repository,
            conversation_id=conversation_id,
            message_id=message_id,
            owner_id=owner_id,
        )
        first_call = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=0,
            idempotency_key="response-revision",
            policy_id=policy_id,
        )
        revision_link = repository.store_response_link(
            cursor,
            call_id=first_call.call_id,
            response=SemanticRevisionRef(revision_id),
        )
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b"safe response",
            media_type="text/plain",
            normalization_version="text-v1",
        )
        second_call = repository.reserve_call(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            turn_id="turn-1",
            run_id="run-1",
            call_sequence=1,
            idempotency_key="response-artifact",
            policy_id=policy_id,
        )
        artifact_link = repository.store_response_link(
            cursor,
            call_id=second_call.call_id,
            response=TraceContentRef(artifact.artifact_id, "artifact"),
        )
    assert revision_link.verification_outcome == "verified_equal"
    assert revision_link.semantic_revision_id == revision_id
    assert artifact_link.verification_outcome == "sanitized_artifact"
    assert artifact_link.artifact_id == artifact.artifact_id


def test_caller_transaction_owns_rollback_and_repository_never_commits(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    connection = db.get_connection()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    connection.execute("BEGIN IMMEDIATE")
    statements.clear()
    cursor = connection.cursor()
    segment = repository.create_segment(cursor)
    repository_statements = tuple(statements)
    connection.rollback()
    connection.set_trace_callback(None)
    assert repository.get_segment(connection.cursor(), segment.segment_id) is None
    assert not any(
        statement.lstrip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
        for statement in repository_statements
    )


def test_epoch_changes_only_for_new_reachability_roots_and_edges(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    conversation_id, _message_id = _conversation_with_message(db)
    with db.transaction() as cursor:
        assert repository.get_graph_epoch(cursor) == 0
        segment = repository.create_segment(cursor)
        assert repository.get_graph_epoch(cursor) == 0
        policy = repository.ensure_policy(cursor, _policy())
        artifact = repository.store_sanitized_artifact(
            cursor,
            sanitized_bytes=b"safe",
            media_type="text/plain",
            normalization_version="text-v1",
        )
        assert repository.get_graph_epoch(cursor) == 0
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        assert repository.get_graph_epoch(cursor) == 1
        repository.get_owner(cursor, owner.owner_id)
        repository.get_artifact(cursor, artifact.artifact_id)
        repository.ensure_policy(cursor, policy)
        assert repository.get_graph_epoch(cursor) == 1
        repository.detach_owner(
            cursor,
            owner_id=owner.owner_id,
            detached_at="2026-08-28T12:00:00Z",
        )
        assert repository.get_graph_epoch(cursor) == 2


def test_chat_persistence_service_exposes_same_database_repository(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    assert isinstance(service.console_trace_repository, ConsoleTraceRepository)
    assert service.console_trace_repository is service.console_trace_repository


def test_repository_api_has_no_canonical_body_digest_or_history_list_parameters() -> (
    None
):
    forbidden = {"body", "digest", "history", "history_list", "messages"}
    for name, method in inspect.getmembers(
        ConsoleTraceRepository, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        assert forbidden.isdisjoint(inspect.signature(method).parameters), name
