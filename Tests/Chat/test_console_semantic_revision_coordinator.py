"""Transactional semantic revision coordination for model-visible messages."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from Tests.DB.fixtures.chachanotes_v54 import genuine_v54_database
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    SemanticRevisionRef,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_FILTER_VERSION,
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizationResult,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "semantic-revision-test")
    yield database
    database.close_connection()


def _message(
    db: CharactersRAGDB, *, content: str = "before", role: str = "user"
) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "semantic revisions"})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "role": role,
            "content": content,
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _untracked_message(
    db: CharactersRAGDB, *, content: str = "before", role: str = "user"
) -> tuple[str, str]:
    """Insert a genuine legacy-style row without going through a public writer."""

    conversation_id = db.add_conversation({"title": "untracked semantic revisions"})
    assert conversation_id is not None
    message_id = db._generate_uuid()
    now = db._get_current_utc_timestamp_iso()
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            """INSERT INTO messages(
                   id, conversation_id, sender, content, timestamp,
                   last_modified, client_id, version, deleted, role)
                 VALUES (?, ?, 'user', ?, ?, ?, ?, 1, 0, ?)""",
            (message_id, conversation_id, content, now, now, db.client_id, role),
        )
    return conversation_id, message_id


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )


def _reference_under_policies(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    message_id: str,
    policy_count: int = 2,
    policies: tuple[FrozenTracePolicy, ...] | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    repository = ConsoleTraceRepository()
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        current_row = cursor.execute(
            """SELECT revision_id
                 FROM console_trace_semantic_revisions
                WHERE source_message_id = ? AND live_message_id = ?""",
            (message_id, message_id),
        ).fetchone()
        if current_row is None:
            revision = repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=conversation_id,
                source_message_id=message_id,
                revision_sequence=0,
                normalized_role="user",
                content_kind="text",
                creation_reason="capture",
                live_message_id=message_id,
            )
        else:
            revision = repository.get_semantic_revision(cursor, str(current_row[0]))
            assert revision is not None
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
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        policy_ids: list[str] = []
        requested_policies = policies or tuple(_policy() for _ in range(policy_count))
        for sequence, requested_policy in enumerate(requested_policies):
            policy = repository.ensure_policy(cursor, requested_policy)
            policy_ids.append(policy.policy_id)
            call = repository.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id=f"turn-{sequence}",
                run_id=f"run-{sequence}",
                call_sequence=sequence,
                idempotency_key=f"semantic-revision-{new_opaque_id()}",
                policy_id=policy.policy_id,
            )
            repository.bind_call(
                cursor,
                call_id=call.call_id,
                surface_node_id=node.node_id,
                request_header_id=header.header_id,
                provider_name="openai",
                model_name="gpt-test",
                route_identity="chat_completions",
            )
    return revision.revision_id, node.node_id, tuple(policy_ids)


def test_initial_revision_is_digest_free_metadata_pointing_to_live_message(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db, content="not duplicated", role="system")
    coordinator = SemanticRevisionCoordinator(db)

    with db.transaction(immediate=True) as cursor:
        revision = coordinator.ensure_current_revision(cursor, message_id=message_id)
        same = coordinator.ensure_current_revision(cursor, message_id=message_id)

    assert same == revision
    assert revision.source_conversation_id == conversation_id
    assert revision.source_message_id == message_id
    assert revision.live_message_id == message_id
    assert revision.predecessor_revision_id is None
    assert revision.normalized_role == "system"
    columns = {
        str(row[1])
        for row in db.get_connection().execute(
            "PRAGMA table_info(console_trace_semantic_revisions)"
        )
    }
    assert not {"digest", "hash", "content", "body"} & columns
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_trace_artifacts")
        .fetchone()[0]
        == 0
    )


def test_genuine_pre_v55_message_acquires_lazy_revision_without_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "genuine-pre-v55.sqlite"
    with genuine_v54_database(path) as historical:
        message_id = str(
            historical.get_connection().execute("SELECT id FROM messages").fetchone()[0]
        )

    db = CharactersRAGDB(path, client_id="lazy-v55")
    try:
        coordinator = SemanticRevisionCoordinator(db)
        with db.transaction(immediate=True) as cursor:
            first = coordinator.ensure_current_revision(cursor, message_id=message_id)
        with db.transaction(immediate=True) as cursor:
            second = coordinator.ensure_current_revision(cursor, message_id=message_id)
        assert second.revision_id == first.revision_id
        assert second.creation_reason == "legacy_reference"
        assert second.normalized_role == "user"
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_trace_artifacts")
            .fetchone()[0]
            == 0
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_trace_revision_bindings")
            .fetchone()[0]
            == 0
        )
    finally:
        db.close_connection()


def test_two_policy_edit_materializes_once_binds_each_policy_and_advances_epoch_once(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(
        db, content="before sk-live-abcdefghijklmnop"
    )
    old_revision_id, old_node_id, policy_ids = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    coordinator = SemanticRevisionCoordinator(db)
    before_epoch = ConsoleTraceRepository().get_graph_epoch(
        db.get_connection().cursor()
    )

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("after", message_id),
            ),
        )

    assert result.previous_revision_id == old_revision_id
    assert result.current_revision_id is not None
    assert result.replacement_id is not None
    assert result.materialized_policy_ids == tuple(sorted(policy_ids))
    assert (
        ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
        == before_epoch + 1
    )
    old = (
        db.get_connection()
        .execute(
            "SELECT live_message_id, live_locator_retired_at FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        )
        .fetchone()
    )
    assert old[0] is None and old[1] is not None
    new = (
        db.get_connection()
        .execute(
            "SELECT predecessor_revision_id, live_message_id, revision_sequence FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (result.current_revision_id,),
        )
        .fetchone()
    )
    assert tuple(new) == (old_revision_id, message_id, 1)
    bindings = (
        db.get_connection()
        .execute(
            "SELECT policy_id, artifact_id, omission_reason_code FROM console_trace_revision_bindings WHERE revision_id = ? ORDER BY policy_id",
            (old_revision_id,),
        )
        .fetchall()
    )
    assert [row[0] for row in bindings] == sorted(policy_ids)
    assert len({row[1] for row in bindings}) == 1
    assert all(row[2] is None for row in bindings)
    artifact = bytes(
        db.get_connection()
        .execute(
            "SELECT sanitized_bytes FROM console_trace_artifacts WHERE artifact_id = ?",
            (bindings[0][1],),
        )
        .fetchone()[0]
    )
    assert b"sk-live-abcdefghijklmnop" not in artifact
    assert b"before" in artifact
    persisted_envelope = json.loads(artifact)
    assert persisted_envelope["role"] == "user"
    assert (
        not {
            "message_id",
            "conversation_id",
            "parent_message_id",
        }
        & persisted_envelope.keys()
    )
    assert (
        db.get_connection()
        .execute("SELECT content FROM messages WHERE id = ?", (message_id,))
        .fetchone()[0]
        == "after"
    )
    replacement = (
        db.get_connection()
        .execute(
            "SELECT start_node_id, end_node_id FROM console_trace_surface_replacements WHERE replacement_id = ?",
            (result.replacement_id,),
        )
        .fetchone()
    )
    assert tuple(replacement) == (old_node_id, old_node_id)


def test_hard_delete_preserves_reachable_projection_and_removes_canonical_row(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db, content="delete me")
    old_revision_id, _node_id, policy_ids = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policy_count=1,
    )
    coordinator = SemanticRevisionCoordinator(db)

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="hard_delete",
            hard_delete=True,
        )

    assert result.deleted is True
    assert result.current_revision_id is not None
    assert result.materialized_policy_ids == policy_ids
    assert (
        db.get_connection()
        .execute("SELECT 1 FROM messages WHERE id = ?", (message_id,))
        .fetchone()
        is None
    )
    preserved = (
        db.get_connection()
        .execute(
            "SELECT live_message_id, live_locator_retired_at FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        )
        .fetchone()
    )
    assert preserved[0] is None and preserved[1] is not None


def test_sanitizer_failure_persists_only_content_free_policy_omission(
    db: CharactersRAGDB,
) -> None:
    canary = "sk-live-never-persist-this-canary"
    conversation_id, message_id = _message(db, content=canary)
    old_revision_id, _node_id, policy_ids = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policy_count=2,
    )

    class UnavailableSanitizer:
        def sanitize(self, value: object) -> CredentialSanitizationResult:
            del value
            return CredentialSanitizationResult(
                available=False,
                value=None,
                omission_reason_code=CREDENTIAL_SANITIZER_UNAVAILABLE,
                detector_version=CREDENTIAL_FILTER_VERSION,
            )

    coordinator = SemanticRevisionCoordinator(  # type: ignore[arg-type]
        db, sanitizer=UnavailableSanitizer()
    )
    with db.transaction(immediate=True) as cursor:
        coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = 'after' WHERE id = ?", (message_id,)
            ),
        )

    rows = (
        db.get_connection()
        .execute(
            "SELECT policy_id, binding_outcome, artifact_id, omission_reason_code FROM console_trace_revision_bindings WHERE revision_id = ? ORDER BY policy_id",
            (old_revision_id,),
        )
        .fetchall()
    )
    assert [tuple(row) for row in rows] == [
        (policy_id, "omission", None, CREDENTIAL_SANITIZER_UNAVAILABLE)
        for policy_id in sorted(policy_ids)
    ]
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM console_trace_artifacts")
        .fetchone()[0]
        == 0
    )
    trace_dump = "\n".join(
        str(tuple(row))
        for table in (
            "console_trace_revision_bindings",
            "console_trace_semantic_revisions",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
        )
        for row in db.get_connection().execute(f"SELECT * FROM {table}")
    )
    assert canary not in trace_dump


@pytest.mark.parametrize(
    "policy",
    [
        FrozenTracePolicy(
            policy_id=new_opaque_id(),
            credential_filter_version="credentials-v999",
            pii_redaction_enabled=False,
            pii_ruleset_revision_id=None,
        ),
        FrozenTracePolicy(
            policy_id=new_opaque_id(),
            credential_filter_version=CREDENTIAL_FILTER_VERSION,
            pii_redaction_enabled=True,
            pii_ruleset_revision_id=new_opaque_id(),
        ),
    ],
)
def test_unsupported_frozen_policy_fails_closed_without_running_weaker_sanitizer(
    db: CharactersRAGDB, policy: FrozenTracePolicy
) -> None:
    canary = "Call +1-415-555-0199 with sk-live-policy-canary-secret"
    conversation_id, message_id = _message(db, content=canary)
    old_revision_id, _node_id, _policy_ids = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policies=(policy,),
    )

    class MustNotRunSanitizer:
        def sanitize(self, value: object) -> CredentialSanitizationResult:
            raise AssertionError(f"weaker sanitizer received {type(value).__name__}")

    coordinator = SemanticRevisionCoordinator(  # type: ignore[arg-type]
        db, sanitizer=MustNotRunSanitizer()
    )
    with db.transaction(immediate=True) as cursor:
        coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = 'after' WHERE id = ?", (message_id,)
            ),
        )

    connection = db.get_connection()
    assert tuple(
        connection.execute(
            """SELECT binding_outcome, artifact_id, omission_reason_code
                 FROM console_trace_revision_bindings
                WHERE revision_id = ? AND policy_id = ?""",
            (old_revision_id, policy.policy_id),
        ).fetchone()
    ) == ("omission", None, "trace_redaction_policy_unsupported")
    assert (
        connection.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )
    trace_dump = "\n".join(
        str(tuple(row))
        for table in (
            "console_trace_revision_bindings",
            "console_trace_semantic_revisions",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
        )
        for row in connection.execute(f"SELECT * FROM {table}")
    )
    assert "+1-415-555-0199" not in trace_dump
    assert "sk-live-policy-canary-secret" not in trace_dump


def test_sanitizer_detector_version_mismatch_cannot_bind_artifact(
    db: CharactersRAGDB,
) -> None:
    canary = "sk-live-detector-mismatch-canary"
    conversation_id, message_id = _message(db, content=canary)
    old_revision_id, _node_id, policy_ids = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policies=(_policy(),),
    )

    class WrongDetectorSanitizer:
        def sanitize(self, value: object) -> CredentialSanitizationResult:
            return CredentialSanitizationResult(
                available=True,
                value=value,
                omission_reason_code=None,
                detector_version="credentials-v999",
            )

    coordinator = SemanticRevisionCoordinator(  # type: ignore[arg-type]
        db, sanitizer=WrongDetectorSanitizer()
    )
    with db.transaction(immediate=True) as cursor:
        coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = 'after' WHERE id = ?", (message_id,)
            ),
        )

    connection = db.get_connection()
    assert tuple(
        connection.execute(
            """SELECT binding_outcome, artifact_id, omission_reason_code
                 FROM console_trace_revision_bindings
                WHERE revision_id = ? AND policy_id = ?""",
            (old_revision_id, policy_ids[0]),
        ).fetchone()
    ) == ("omission", None, "trace_redaction_policy_unsupported")
    assert (
        connection.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )


def test_exception_rolls_back_content_lineage_and_clears_authorization(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db)
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policy_count=1,
    )
    coordinator = SemanticRevisionCoordinator(db)
    before_epoch = ConsoleTraceRepository().get_graph_epoch(
        db.get_connection().cursor()
    )

    def explode(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "UPDATE messages SET content = 'transient', role = 'system' WHERE id = ?",
            (message_id,),
        )
        raise RuntimeError("injected mutation failure")

    with pytest.raises(RuntimeError, match="injected mutation failure"):
        with db.transaction(immediate=True) as cursor:
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=explode,
            )

    persisted_message = (
        db.get_connection()
        .execute("SELECT content, role FROM messages WHERE id = ?", (message_id,))
        .fetchone()
    )
    assert persisted_message is not None
    assert tuple(persisted_message) == ("before", "user")
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_trace_semantic_revisions WHERE source_message_id = ?",
            (message_id,),
        )
        .fetchone()[0]
        == 1
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT live_message_id FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        )
        .fetchone()[0]
        == message_id
    )
    assert (
        ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
        == before_epoch
    )
    with pytest.raises(sqlite3.DatabaseError, match="semantic mutation"):
        db.get_connection().execute(
            "UPDATE messages SET content = 'still blocked' WHERE id = ?",
            (message_id,),
        )


def test_authorization_is_cleared_after_success(db: CharactersRAGDB) -> None:
    conversation_id, message_id = _message(db)
    _reference_under_policies(
        db,
        conversation_id=conversation_id,
        message_id=message_id,
        policy_count=0,
    )
    coordinator = SemanticRevisionCoordinator(db)
    with db.transaction(immediate=True) as cursor:
        coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = 'after' WHERE id = ?", (message_id,)
            ),
        )
    with pytest.raises(sqlite3.DatabaseError, match="semantic mutation"):
        db.get_connection().execute(
            "UPDATE messages SET content = 'bypass' WHERE id = ?", (message_id,)
        )


def test_role_only_mutation_creates_role_accurate_successor(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db, role="user")
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db, conversation_id=conversation_id, message_id=message_id, policy_count=1
    )
    coordinator = SemanticRevisionCoordinator(db)
    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="role_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET role = 'system' WHERE id = ?", (message_id,)
            ),
        )
    artifact = (
        db.get_connection()
        .execute(
            """SELECT artifact.sanitized_bytes
             FROM console_trace_revision_bindings AS binding
             JOIN console_trace_artifacts AS artifact USING (artifact_id)
            WHERE binding.revision_id = ?""",
            (old_revision_id,),
        )
        .fetchone()[0]
    )
    assert json.loads(bytes(artifact))["role"] == "user"
    assert (
        db.get_connection()
        .execute(
            "SELECT normalized_role FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (result.current_revision_id,),
        )
        .fetchone()[0]
        == "system"
    )


@pytest.mark.parametrize(
    "mutation_kind",
    ["exact_message_noop", "attachment_noop", "message_cas_miss"],
)
def test_noop_mutation_preserves_current_revision_and_trace_graph(
    db: CharactersRAGDB, mutation_kind: str
) -> None:
    conversation_id, message_id = _message(db)
    db.set_message_attachments(
        message_id,
        [
            {
                "position": 1,
                "data": b"unchanged",
                "mime_type": "text/plain",
                "display_name": "same.txt",
            }
        ],
    )
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db, conversation_id=conversation_id, message_id=message_id, policy_count=1
    )
    coordinator = SemanticRevisionCoordinator(db)
    connection = db.get_connection()
    before_epoch = ConsoleTraceRepository().get_graph_epoch(connection.cursor())
    before_counts = {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "console_trace_semantic_revisions",
            "console_trace_artifacts",
            "console_trace_revision_bindings",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
            "console_trace_events",
        )
    }

    def noop(cursor: sqlite3.Cursor) -> None:
        if mutation_kind == "exact_message_noop":
            cursor.execute(
                "UPDATE messages SET content = content WHERE id = ?", (message_id,)
            )
        elif mutation_kind == "attachment_noop":
            cursor.execute(
                """UPDATE message_attachments
                      SET display_name = display_name
                    WHERE message_id = ? AND position = 1""",
                (message_id,),
            )
        else:
            update = cursor.execute(
                """UPDATE messages SET content = 'after'
                    WHERE id = ? AND content = 'stale-cas-value'""",
                (message_id,),
            )
            assert update.rowcount == 0

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=noop,
        )

    assert result.previous_revision_id == old_revision_id
    assert result.current_revision_id == old_revision_id
    assert result.replacement_id is None
    assert result.materialized_policy_ids == ()
    assert result.deleted is False
    assert ConsoleTraceRepository().get_graph_epoch(connection.cursor()) == before_epoch
    assert {
        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in before_counts
    } == before_counts
    assert (
        connection.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()[0]
        == "before"
    )
    assert tuple(
        connection.execute(
            """SELECT data, mime_type, display_name
                 FROM message_attachments
                WHERE message_id = ? AND position = 1""",
            (message_id,),
        ).fetchone()
    ) == (b"unchanged", "text/plain", "same.txt")


def _untracked_semantic_state(db: CharactersRAGDB) -> dict[str, int]:
    connection = db.get_connection()
    return {
        "revisions": int(
            connection.execute(
                "SELECT COUNT(*) FROM console_trace_semantic_revisions"
            ).fetchone()[0]
        ),
        "artifacts": int(
            connection.execute(
                "SELECT COUNT(*) FROM console_trace_artifacts"
            ).fetchone()[0]
        ),
        "bindings": int(
            connection.execute(
                "SELECT COUNT(*) FROM console_trace_revision_bindings"
            ).fetchone()[0]
        ),
        "locators": int(
            connection.execute(
                """SELECT COUNT(*) FROM console_trace_semantic_revisions
                    WHERE live_message_id IS NOT NULL
                       OR live_locator_retired_at IS NOT NULL"""
            ).fetchone()[0]
        ),
        "epoch": ConsoleTraceRepository().get_graph_epoch(connection.cursor()),
    }


def test_untracked_message_stale_cas_does_not_create_semantic_state(
    db: CharactersRAGDB,
) -> None:
    _conversation_id, message_id = _untracked_message(db)
    coordinator = SemanticRevisionCoordinator(db)
    before = _untracked_semantic_state(db)

    def stale_cas(cursor: sqlite3.Cursor) -> None:
        update = cursor.execute(
            """UPDATE messages SET content = 'after'
                WHERE id = ? AND content = 'stale-cas-value'""",
            (message_id,),
        )
        assert update.rowcount == 0

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=stale_cas,
        )

    assert result.previous_revision_id is None
    assert result.current_revision_id is None
    assert result.replacement_id is None
    assert result.materialized_policy_ids == ()
    assert result.deleted is False
    assert _untracked_semantic_state(db) == before


def test_untracked_message_semantic_noop_does_not_create_semantic_state(
    db: CharactersRAGDB,
) -> None:
    _conversation_id, message_id = _untracked_message(db)
    coordinator = SemanticRevisionCoordinator(db)
    before = _untracked_semantic_state(db)

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                "UPDATE messages SET content = content WHERE id = ?", (message_id,)
            ),
        )

    assert result.previous_revision_id is None
    assert result.current_revision_id is None
    assert result.replacement_id is None
    assert result.materialized_policy_ids == ()
    assert result.deleted is False
    assert _untracked_semantic_state(db) == before


def test_untracked_message_successful_first_mutation_creates_one_successor(
    db: CharactersRAGDB,
) -> None:
    _conversation_id, message_id = _untracked_message(db)
    coordinator = SemanticRevisionCoordinator(db)
    before_epoch = ConsoleTraceRepository().get_graph_epoch(
        db.get_connection().cursor()
    )

    with db.transaction(immediate=True) as cursor:
        result = coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: scoped.execute(
                """UPDATE messages SET content = 'after', role = 'system'
                    WHERE id = ?""",
                (message_id,),
            ),
        )

    revisions = (
        db.get_connection()
        .execute(
            """SELECT revision_id, predecessor_revision_id, revision_sequence,
                  live_message_id, live_locator_retired_at, normalized_role
             FROM console_trace_semantic_revisions
            ORDER BY revision_sequence"""
        )
        .fetchall()
    )
    assert len(revisions) == 2
    assert tuple(revisions[0][:3]) == (result.previous_revision_id, None, 0)
    assert revisions[0][3] is None and revisions[0][4] is not None
    assert revisions[0][5] == "user"
    assert tuple(revisions[1][:4]) == (
        result.current_revision_id,
        result.previous_revision_id,
        1,
        message_id,
    )
    assert revisions[1][4] is None
    assert revisions[1][5] == "system"
    assert result.replacement_id is None
    assert result.materialized_policy_ids == ()
    assert (
        ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
        == before_epoch + 1
    )


def test_reachable_policy_discovery_is_bounded_by_surface_graph_not_call_depth(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db)
    repository = ConsoleTraceRepository()
    with db.transaction(immediate=True) as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
        revision_id = str(
            cursor.execute(
                """SELECT revision_id
                     FROM console_trace_semantic_revisions
                    WHERE source_message_id = ? AND live_message_id = ?""",
                (message_id, message_id),
            ).fetchone()[0]
        )
        revision = repository.get_semantic_revision(cursor, revision_id)
        assert revision is not None
        predecessor_id = None
        head = None
        for sequence in range(40):
            head = repository.append_surface_node(
                cursor,
                segment_id=segment.segment_id,
                sequence=sequence,
                predecessor_node_id=predecessor_id,
                component_kind="message",
                reference=SemanticRevisionRef(revision.revision_id),
            )
            repository.append_event(
                cursor,
                segment_id=segment.segment_id,
                sequence=sequence,
                event_type="surface_append",
                surface_node_id=head.node_id,
            )
            predecessor_id = head.node_id
        assert head is not None
        policy = repository.ensure_policy(cursor, _policy())
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="chat_completions",
            endpoint_identity="public_api",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        for sequence in range(80):
            call = repository.reserve_call(
                cursor,
                owner_id=owner.owner_id,
                segment_id=segment.segment_id,
                turn_id="bounded-turn",
                run_id="bounded-run",
                call_sequence=sequence,
                idempotency_key=f"bounded-{sequence}-{new_opaque_id()}",
                policy_id=policy.policy_id,
            )
            repository.bind_call(
                cursor,
                call_id=call.call_id,
                surface_node_id=head.node_id,
                request_header_id=header.header_id,
                provider_name="openai",
                model_name="gpt-test",
                route_identity="chat_completions",
            )

    progress_callbacks = 0

    def count_progress() -> int:
        nonlocal progress_callbacks
        progress_callbacks += 1
        return 0

    connection = db.get_connection()
    connection.set_progress_handler(count_progress, 100)
    try:
        policy_ids = SemanticRevisionCoordinator._reachable_policy_ids(
            connection.cursor(), revision.revision_id
        )
    finally:
        connection.set_progress_handler(None, 0)

    assert policy_ids == (policy.policy_id,)
    assert progress_callbacks < 500


@pytest.mark.parametrize("transaction_action", ["commit", "rollback"])
def test_callback_cannot_escape_authorized_transaction(
    db: CharactersRAGDB, transaction_action: str
) -> None:
    conversation_id, message_id = _message(db)
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db, conversation_id=conversation_id, message_id=message_id, policy_count=1
    )
    coordinator = SemanticRevisionCoordinator(db)

    def escape(cursor: sqlite3.Cursor) -> None:
        getattr(cursor.connection, transaction_action)()
        cursor.connection.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "UPDATE messages SET content = 'escaped' WHERE id = ?", (message_id,)
        )

    with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
        with db.transaction(immediate=True) as cursor:
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=escape,
            )
    assert (
        db.get_connection()
        .execute("SELECT content FROM messages WHERE id = ?", (message_id,))
        .fetchone()[0]
        == "before"
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT live_message_id FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        )
        .fetchone()[0]
        == message_id
    )
    with pytest.raises(sqlite3.DatabaseError, match="semantic mutation"):
        db.get_connection().execute(
            "UPDATE messages SET content = 'still blocked' WHERE id = ?", (message_id,)
        )


@pytest.mark.parametrize("savepoint_action", ["release", "rollback"])
def test_callback_cannot_end_top_level_savepoint_and_escape_coordinator(
    db: CharactersRAGDB, savepoint_action: str
) -> None:
    conversation_id, message_id = _message(db)
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db, conversation_id=conversation_id, message_id=message_id, policy_count=1
    )
    coordinator = SemanticRevisionCoordinator(db)
    connection = db.get_connection()
    connection.execute("SAVEPOINT caller_owned")
    escaped_boundary = False

    def escape_outer(cursor: sqlite3.Cursor) -> None:
        nonlocal escaped_boundary
        statement = (
            "RELEASE SAVEPOINT caller_owned"
            if savepoint_action == "release"
            else "ROLLBACK TO SAVEPOINT caller_owned"
        )
        cursor.execute(statement)
        escaped_boundary = True
        raise RuntimeError("savepoint boundary escaped")

    try:
        with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
            coordinator.mutate_message(
                connection.cursor(),
                message_id=message_id,
                creation_reason="message_edit",
                mutate=escape_outer,
            )
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK TO SAVEPOINT caller_owned")
            connection.execute("RELEASE SAVEPOINT caller_owned")

    assert escaped_boundary is False
    assert (
        connection.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()[0]
        == "before"
    )
    assert (
        connection.execute(
            "SELECT live_message_id FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        ).fetchone()[0]
        == message_id
    )


def test_caught_callback_exception_rolls_back_only_coordinator_subtransaction(
    db: CharactersRAGDB,
) -> None:
    conversation_id, message_id = _message(db)
    old_revision_id, _node_id, _policies = _reference_under_policies(
        db, conversation_id=conversation_id, message_id=message_id, policy_count=1
    )
    coordinator = SemanticRevisionCoordinator(db)
    before_epoch = ConsoleTraceRepository().get_graph_epoch(
        db.get_connection().cursor()
    )

    def fail_after_mutation(cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "UPDATE messages SET content = 'transient' WHERE id = ?", (message_id,)
        )
        raise RuntimeError("caught callback failure")

    with db.transaction(immediate=True) as cursor:
        with pytest.raises(RuntimeError, match="caught callback failure"):
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=fail_after_mutation,
            )
        cursor.execute(
            "UPDATE conversations SET title = 'outer committed' WHERE id = ?",
            (conversation_id,),
        )

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()[0]
        == "before"
    )
    assert (
        connection.execute(
            "SELECT live_message_id FROM console_trace_semantic_revisions WHERE revision_id = ?",
            (old_revision_id,),
        ).fetchone()[0]
        == message_id
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_trace_revision_bindings WHERE revision_id = ?",
            (old_revision_id,),
        ).fetchone()[0]
        == 0
    )
    assert ConsoleTraceRepository().get_graph_epoch(connection.cursor()) == before_epoch
    assert (
        connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == "outer committed"
    )


def test_pre_v55_mutation_advances_epoch_once_and_retry_once(tmp_path: Path) -> None:
    path = tmp_path / "pre-v55-mutation.sqlite"
    with genuine_v54_database(path) as historical:
        message_id = str(
            historical.get_connection().execute("SELECT id FROM messages").fetchone()[0]
        )
    db = CharactersRAGDB(path, client_id="pre-v55-mutation")
    try:
        coordinator = SemanticRevisionCoordinator(db)
        before = ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
        with pytest.raises(RuntimeError, match="retry"):
            with db.transaction(immediate=True) as cursor:
                coordinator.mutate_message(
                    cursor,
                    message_id=message_id,
                    creation_reason="message_edit",
                    mutate=lambda scoped: (_ for _ in ()).throw(RuntimeError("retry")),
                )
        assert (
            ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
            == before
        )
        with db.transaction(immediate=True) as cursor:
            coordinator.mutate_message(
                cursor,
                message_id=message_id,
                creation_reason="message_edit",
                mutate=lambda scoped: scoped.execute(
                    "UPDATE messages SET role = 'system' WHERE id = ?", (message_id,)
                ),
            )
        assert (
            ConsoleTraceRepository().get_graph_epoch(db.get_connection().cursor())
            == before + 1
        )
    finally:
        db.close_connection()


def test_coordinator_requires_caller_owned_transaction(db: CharactersRAGDB) -> None:
    _conversation_id, message_id = _message(db)
    cursor = db.get_connection().cursor()
    coordinator = SemanticRevisionCoordinator(db)

    with pytest.raises(RuntimeError, match="caller_transaction_required"):
        coordinator.ensure_current_revision(cursor, message_id=message_id)
    with pytest.raises(RuntimeError, match="caller_transaction_required"):
        coordinator.mutate_message(
            cursor,
            message_id=message_id,
            creation_reason="message_edit",
            mutate=lambda scoped: None,
        )


def test_chat_persistence_service_exposes_shared_coordinator(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)

    assert service.semantic_revision_coordinator.db is db
    assert (
        service.semantic_revision_coordinator.repository
        is service.console_trace_repository
    )
