"""Behavioral contract for the immutable, conversation-owned Canvas repository."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

try:
    from tldw_chatbook.Canvas.limits import CanvasRepositoryLimits
    from tldw_chatbook.Canvas.repository import (
        CanvasConflictError,
        CanvasImportBatch,
        CanvasImportDocument,
        CanvasImportRevision,
        CanvasNotFoundError,
        CanvasQuotaError,
        CanvasRepository,
        CanvasRepositoryError,
        CanvasValidationError,
    )
except (ImportError, ModuleNotFoundError) as exc:  # Expected during the RED run.
    _REPOSITORY_IMPORT_ERROR: Exception | None = exc
else:
    _REPOSITORY_IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def repository_api_is_available() -> None:
    """Report a missing Task 2.1 API as a test failure, not collection noise."""

    assert _REPOSITORY_IMPORT_ERROR is None, (
        "Task 2.1 Canvas repository API is not implemented: "
        f"{type(_REPOSITORY_IMPORT_ERROR).__name__}"
    )


@pytest.fixture
def db(tmp_path: Path):
    database = CharactersRAGDB(tmp_path / "canvas.sqlite", client_id="canvas-tests")
    try:
        yield database
    finally:
        database.close_connection()


def _owner(db: CharactersRAGDB, title: str = "owner") -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": title})
    assert conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "Canvas origin",
        }
    )
    assert message_id is not None
    return conversation_id, message_id


def _create(
    repository: CanvasRepository,
    conversation_id: str,
    message_id: str,
    *,
    title: str = "Map",
    source: str = "<main>one</main>",
):
    return repository.create_canvas(
        conversation_id,
        title=title,
        source=source,
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-1",
    )


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_create_list_and_exact_read_return_frozen_typed_values(db) -> None:
    """A mutable/untyped return or a list operation leaking source fails here."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    created = _create(
        repository,
        conversation_id,
        message_id,
        title="🗺️ Map",
        source="<main>λ</main>",
    )

    assert created.identity.canvas_id
    assert created.identity.conversation_id == conversation_id
    assert created.identity.deleted_at is None
    assert created.revision.sequence == 1
    assert created.revision.parent_revision_id is None
    assert created.revision.title == "🗺️ Map"
    assert created.revision.source == "<main>λ</main>"
    assert created.revision.source_bytes == len("<main>λ</main>".encode())
    assert created.revision.content_sha256 == _sha256("<main>λ</main>")
    assert created.revision.runtime_profile == "canvas-v1"
    assert created.revision.origin_message_id == message_id

    identities = repository.list_identities(conversation_id)
    metadata = repository.list_revision_metadata(conversation_id)
    exact = repository.read_revision(conversation_id, created.revision.revision_id)
    assert identities == (created.identity,)
    assert len(metadata) == 1
    assert metadata[0].revision_id == created.revision.revision_id
    assert metadata[0].content_sha256 == created.revision.content_sha256
    assert not hasattr(metadata[0], "source")
    assert exact == created.revision
    assert not hasattr(exact, "__dict__")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        exact.title = "mutated"  # type: ignore[misc]


def test_branching_append_allocates_monotonic_sequence_and_preserves_titles(db) -> None:
    """Appending from history must branch without mutating either prior title/source."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    root = _create(repository, conversation_id, message_id).revision
    child = repository.append_revision(
        conversation_id,
        root.canvas_id,
        parent_revision_id=root.revision_id,
        title="Map v2",
        source="<main>two</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-2",
    )
    sibling = repository.append_revision(
        conversation_id,
        root.canvas_id,
        parent_revision_id=root.revision_id,
        title="Renamed branch",
        source=root.source,
        runtime_profile="canvas-v1",
        actor_kind="user_rename",
        origin_message_id=message_id,
        origin_turn_id="turn-3",
    )

    assert (root.sequence, child.sequence, sibling.sequence) == (1, 2, 3)
    assert child.parent_revision_id == sibling.parent_revision_id == root.revision_id
    assert repository.read_revision(conversation_id, root.revision_id).title == "Map"
    assert repository.read_revision(conversation_id, child.revision_id).source == (
        "<main>two</main>"
    )
    assert repository.read_revision(conversation_id, sibling.revision_id).title == (
        "Renamed branch"
    )


def test_owner_origin_parent_and_digest_checks_fail_without_mutation(db) -> None:
    """Foreign origins/parents and caller-forged identities must never create rows."""

    first_conversation, first_message = _owner(db, "first")
    second_conversation, second_message = _owner(db, "second")
    repository = CanvasRepository(db)
    first = _create(repository, first_conversation, first_message).revision
    second = _create(repository, second_conversation, second_message).revision

    with pytest.raises(CanvasValidationError) as foreign_origin:
        repository.append_revision(
            first_conversation,
            first.canvas_id,
            parent_revision_id=first.revision_id,
            title="wrong origin",
            source="<main>x</main>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=second_message,
            origin_turn_id="turn-x",
        )
    assert foreign_origin.value.code == "origin_owner_mismatch"

    with pytest.raises(CanvasValidationError) as foreign_parent:
        repository.append_revision(
            first_conversation,
            first.canvas_id,
            parent_revision_id=second.revision_id,
            title="wrong parent",
            source="<main>x</main>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=first_message,
            origin_turn_id="turn-x",
        )
    assert foreign_parent.value.code == "parent_owner_mismatch"
    assert len(repository.list_revision_metadata(first_conversation)) == 1

    connection = db.get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO canvas_revisions "
            "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
            "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
            "origin_turn_id, created_at, deleted_at) "
            "VALUES (?, ?, NULL, 1, ?, 'canvas-v1', ?, ?, ?, 'assistant', ?, ?, ?, NULL)",
            (
                str(uuid4()),
                first.canvas_id,
                "forged",
                "<main>forged</main>",
                "0" * 64,
                len(b"<main>forged</main>"),
                first_message,
                "turn-forged",
                "2026-09-03T00:00:00.000Z",
            ),
        )


def test_database_rejects_revision_update_delete_and_foreign_canvas_parent(db) -> None:
    """SQL outside the repository cannot mutate/delete history or cross-link graphs."""

    first_conversation, first_message = _owner(db, "first")
    second_conversation, second_message = _owner(db, "second")
    repository = CanvasRepository(db)
    first = _create(repository, first_conversation, first_message).revision
    second = _create(repository, second_conversation, second_message).revision
    connection = db.get_connection()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        connection.execute(
            "UPDATE canvas_revisions SET title = ? WHERE id = ?",
            ("tampered", first.revision_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="deletion authorization"):
        connection.execute(
            "DELETE FROM canvas_revisions WHERE id = ?", (first.revision_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="deletion authorization"):
        connection.execute(
            "DELETE FROM canvas_documents WHERE id = ?", (first.canvas_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="ownership is immutable"):
        connection.execute(
            "UPDATE canvas_documents SET conversation_id = ? WHERE id = ?",
            (second_conversation, first.canvas_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="origin owner"):
        connection.execute(
            "INSERT INTO canvas_revisions "
            "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
            "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
            "origin_turn_id, created_at, deleted_at) "
            "VALUES (?, ?, ?, 2, ?, 'canvas-v1', ?, ?, ?, 'assistant', ?, ?, ?, NULL)",
            (
                str(uuid4()),
                first.canvas_id,
                first.revision_id,
                "cross-owned origin",
                "<main>x</main>",
                _sha256("<main>x</main>"),
                len(b"<main>x</main>"),
                second_message,
                "turn-cross-origin",
                "2026-09-03T00:00:00.000Z",
            ),
        )
    with pytest.raises(sqlite3.IntegrityError, match="parent"):
        connection.execute(
            "INSERT INTO canvas_revisions "
            "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
            "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
            "origin_turn_id, created_at, deleted_at) "
            "VALUES (?, ?, ?, 2, ?, 'canvas-v1', ?, ?, ?, 'assistant', ?, ?, ?, NULL)",
            (
                str(uuid4()),
                first.canvas_id,
                second.revision_id,
                "cross-linked",
                "<main>x</main>",
                _sha256("<main>x</main>"),
                len(b"<main>x</main>"),
                first_message,
                "turn-cross",
                "2026-09-03T00:00:00.000Z",
            ),
        )


def test_injected_first_and_later_revision_failures_roll_back_and_retry(db) -> None:
    """A failure after an identity or sequence read leaves no partial write."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_canvas_first BEFORE INSERT ON canvas_revisions "
        "WHEN NEW.sequence = 1 BEGIN SELECT RAISE(ABORT, 'injected first'); END"
    )
    with pytest.raises(CanvasRepositoryError) as first_failure:
        _create(repository, conversation_id, message_id)
    assert first_failure.value.code == "storage_failure"
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 0
    )
    connection.execute("DROP TRIGGER fail_canvas_first")

    created = _create(repository, conversation_id, message_id)
    connection.execute(
        "CREATE TRIGGER fail_canvas_second BEFORE INSERT ON canvas_revisions "
        "WHEN NEW.sequence = 2 BEGIN SELECT RAISE(ABORT, 'injected second'); END"
    )
    with pytest.raises(CanvasRepositoryError):
        repository.append_revision(
            conversation_id,
            created.identity.canvas_id,
            parent_revision_id=created.revision.revision_id,
            title="retry",
            source="<main>retry</main>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=message_id,
            origin_turn_id="turn-retry",
        )
    assert len(repository.list_revision_metadata(conversation_id)) == 1
    connection.execute("DROP TRIGGER fail_canvas_second")
    retried = repository.append_revision(
        conversation_id,
        created.identity.canvas_id,
        parent_revision_id=created.revision.revision_id,
        title="retry",
        source="<main>retry</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-retry",
    )
    assert retried.sequence == 2


def test_two_real_connections_allocate_unique_sequences_without_lost_update(
    tmp_path: Path,
) -> None:
    """Replacing BEGIN IMMEDIATE with a deferred transaction makes this race unsafe."""

    path = tmp_path / "race.sqlite"
    first_db = CharactersRAGDB(path, client_id="canvas-race-a")
    second_db = CharactersRAGDB(path, client_id="canvas-race-b")
    try:
        conversation_id, message_id = _owner(first_db)
        root = _create(CanvasRepository(first_db), conversation_id, message_id).revision
        barrier = threading.Barrier(2)

        def append(database: CharactersRAGDB, title: str):
            try:
                barrier.wait(timeout=5)
                return CanvasRepository(database).append_revision(
                    conversation_id,
                    root.canvas_id,
                    parent_revision_id=root.revision_id,
                    title=title,
                    source=f"<main>{title}</main>",
                    runtime_profile="canvas-v1",
                    actor_kind="assistant",
                    origin_message_id=message_id,
                    origin_turn_id=f"turn-{title}",
                )
            finally:
                database.close_connection()

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(append, first_db, "left")
            right_future = executor.submit(append, second_db, "right")
            left = left_future.result(timeout=10)
            right = right_future.result(timeout=10)

        assert {left.sequence, right.sequence} == {2, 3}
        rows = (
            first_db.get_connection()
            .execute(
                "SELECT sequence, title FROM canvas_revisions "
                "WHERE canvas_id = ? ORDER BY sequence",
                (root.canvas_id,),
            )
            .fetchall()
        )
        assert [row[0] for row in rows] == [1, 2, 3]
        assert {row[1] for row in rows[1:]} == {"left", "right"}
    finally:
        first_db.close_connection()
        second_db.close_connection()


def test_repository_quotas_are_injectable_and_refusals_do_not_prune(db) -> None:
    """Each durable quota must reject atomically while retaining prior revisions."""

    conversation_id, message_id = _owner(db)
    limits = CanvasRepositoryLimits(
        max_canvases_per_conversation=2,
        max_revisions_per_canvas=2,
        max_source_bytes_per_conversation=12,
        max_source_bytes_per_revision=8,
        max_title_bytes=32,
        max_origin_turn_id_bytes=64,
    )
    repository = CanvasRepository(db, limits=limits)
    first = _create(repository, conversation_id, message_id, source="1234")
    second = _create(repository, conversation_id, message_id, source="5678")

    with pytest.raises(CanvasQuotaError) as canvas_limit:
        _create(repository, conversation_id, message_id, source="x")
    assert canvas_limit.value.code == "canvas_count"

    child = repository.append_revision(
        conversation_id,
        first.identity.canvas_id,
        parent_revision_id=first.revision.revision_id,
        title="child",
        source="abcd",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=message_id,
        origin_turn_id="turn-child",
    )
    with pytest.raises(CanvasQuotaError) as revision_limit:
        repository.append_revision(
            conversation_id,
            first.identity.canvas_id,
            parent_revision_id=child.revision_id,
            title="overflow",
            source="z",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=message_id,
            origin_turn_id="turn-overflow",
        )
    assert revision_limit.value.code == "revision_count"

    with pytest.raises(CanvasQuotaError) as source_limit:
        repository.append_revision(
            conversation_id,
            second.identity.canvas_id,
            parent_revision_id=second.revision.revision_id,
            title="too much total",
            source="z",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=message_id,
            origin_turn_id="turn-source",
        )
    assert source_limit.value.code == "conversation_source_bytes"
    assert len(repository.list_identities(conversation_id)) == 2
    assert len(repository.list_revision_metadata(conversation_id)) == 3

    with pytest.raises(CanvasQuotaError) as per_revision:
        CanvasRepository(
            db,
            limits=CanvasRepositoryLimits(
                max_source_bytes_per_revision=3,
                max_source_bytes_per_conversation=100,
            ),
        ).append_revision(
            conversation_id,
            second.identity.canvas_id,
            parent_revision_id=second.revision.revision_id,
            title="large",
            source="four",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=message_id,
            origin_turn_id="turn-large",
        )
    assert per_revision.value.code == "revision_source_bytes"


def test_soft_delete_restore_hints_owner_lifecycle_and_no_canvas_sync_rows(db) -> None:
    """Canvas lifecycle follows its owner locally without creating a sync entity."""

    conversation_id, message_id = _owner(db)
    child_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": message_id,
            "sender": "user",
            "role": "user",
            "content": "Child whose FK update is semantic-guarded",
        }
    )
    assert child_message_id is not None
    repository = CanvasRepository(db)
    created = _create(repository, conversation_id, message_id)
    repository.set_reopen_hint(conversation_id, created.identity.canvas_id)
    assert repository.get_reopen_hint(conversation_id) == created.identity.canvas_id

    deleted = repository.soft_delete_canvas(conversation_id, created.identity.canvas_id)
    assert deleted.deleted_at is not None
    assert repository.list_identities(conversation_id) == ()
    with pytest.raises(CanvasNotFoundError):
        repository.read_revision(conversation_id, created.revision.revision_id)
    assert repository.get_reopen_hint(conversation_id) is None

    restored = repository.restore_canvas(conversation_id, created.identity.canvas_id)
    assert restored.deleted_at is None
    assert repository.list_identities(conversation_id) == (restored,)

    assert db.soft_delete_conversation(conversation_id, expected_version=1) is True
    assert repository.list_identities(conversation_id) == ()
    assert db.restore_conversation(conversation_id, expected_version=2) is True
    assert repository.list_identities(conversation_id) == (restored,)
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM sync_log WHERE entity LIKE 'canvas%'")
        .fetchone()[0]
        == 0
    )

    with pytest.raises(sqlite3.IntegrityError), db.transaction() as cursor:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    purged = repository.hard_purge_conversation(conversation_id)
    assert (purged.canvases_deleted, purged.revisions_deleted) == (1, 1)
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,))
        .fetchone()[0]
        == 0
    )
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        .fetchone()[0]
        == 0
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM canvas_documents")
        .fetchone()[0]
        == 0
    )


def test_valid_import_preserves_ids_branch_graph_source_and_reopen_hint(db) -> None:
    """A validated import must preserve its exact immutable graph, not reallocate it."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    canvas_id = str(uuid4())
    root_id = str(uuid4())
    left_id = str(uuid4())
    right_id = str(uuid4())
    created_at = "2026-09-03T12:00:00.000Z"
    sources = ("<main>root</main>", "<main>left</main>", "<main>right</main>")
    batch = CanvasImportBatch(
        conversation_id=conversation_id,
        documents=(
            CanvasImportDocument(
                canvas_id=canvas_id,
                conversation_id=conversation_id,
                created_at=created_at,
                deleted_at=None,
            ),
        ),
        revisions=(
            CanvasImportRevision(
                revision_id=root_id,
                canvas_id=canvas_id,
                parent_revision_id=None,
                sequence=1,
                title="Imported",
                runtime_profile="canvas-v1",
                source=sources[0],
                content_sha256=_sha256(sources[0]),
                source_bytes=len(sources[0].encode("utf-8")),
                actor_kind="user_import",
                origin_message_id=message_id,
                origin_turn_id="import-root",
                created_at=created_at,
                deleted_at=None,
            ),
            CanvasImportRevision(
                revision_id=left_id,
                canvas_id=canvas_id,
                parent_revision_id=root_id,
                sequence=2,
                title="Left",
                runtime_profile="canvas-v1",
                source=sources[1],
                content_sha256=_sha256(sources[1]),
                source_bytes=len(sources[1].encode("utf-8")),
                actor_kind="user_import",
                origin_message_id=message_id,
                origin_turn_id="import-left",
                created_at=created_at,
                deleted_at=None,
            ),
            CanvasImportRevision(
                revision_id=right_id,
                canvas_id=canvas_id,
                parent_revision_id=root_id,
                sequence=3,
                title="Right",
                runtime_profile="canvas-v1",
                source=sources[2],
                content_sha256=_sha256(sources[2]),
                source_bytes=len(sources[2].encode("utf-8")),
                actor_kind="user_import",
                origin_message_id=message_id,
                origin_turn_id="import-right",
                created_at=created_at,
                deleted_at=None,
            ),
        ),
        reopen_canvas_id=canvas_id,
    )

    result = repository.import_batch(batch)
    assert (result.canvases_imported, result.revisions_imported) == (1, 3)
    assert repository.get_reopen_hint(conversation_id) == canvas_id
    assert repository.read_revision(conversation_id, root_id).source == sources[0]
    assert (
        repository.read_revision(conversation_id, left_id).parent_revision_id == root_id
    )
    assert (
        repository.read_revision(conversation_id, right_id).parent_revision_id
        == root_id
    )
    assert [
        item.sequence for item in repository.list_revision_metadata(conversation_id)
    ] == [
        1,
        2,
        3,
    ]


def test_import_prevalidation_collisions_and_injected_write_failure_are_atomic(
    db,
) -> None:
    """Malformed/colliding batches and mid-import SQL failures leave zero imported state."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)

    def batch(canvas_id: str, *, digest: str | None = None) -> CanvasImportBatch:
        root_id = str(uuid4())
        child_id = str(uuid4())
        source = "root"
        return CanvasImportBatch(
            conversation_id=conversation_id,
            documents=(
                CanvasImportDocument(
                    canvas_id=canvas_id,
                    conversation_id=conversation_id,
                    created_at="2026-09-03T12:00:00.000Z",
                    deleted_at=None,
                ),
            ),
            revisions=(
                CanvasImportRevision(
                    revision_id=root_id,
                    canvas_id=canvas_id,
                    parent_revision_id=None,
                    sequence=1,
                    title="root",
                    runtime_profile="canvas-v1",
                    source=source,
                    content_sha256=digest or _sha256(source),
                    source_bytes=4,
                    actor_kind="user_import",
                    origin_message_id=message_id,
                    origin_turn_id="import-root",
                    created_at="2026-09-03T12:00:00.000Z",
                    deleted_at=None,
                ),
                CanvasImportRevision(
                    revision_id=child_id,
                    canvas_id=canvas_id,
                    parent_revision_id=root_id,
                    sequence=2,
                    title="child",
                    runtime_profile="canvas-v1",
                    source="child",
                    content_sha256=_sha256("child"),
                    source_bytes=5,
                    actor_kind="user_import",
                    origin_message_id=message_id,
                    origin_turn_id="import-child",
                    created_at="2026-09-03T12:00:00.000Z",
                    deleted_at=None,
                ),
            ),
            reopen_canvas_id=canvas_id,
        )

    with pytest.raises(CanvasValidationError) as bad_digest:
        repository.import_batch(batch(str(uuid4()), digest="0" * 64))
    assert bad_digest.value.code == "digest_mismatch"
    assert repository.list_identities(conversation_id) == ()

    retryable = batch(str(uuid4()))
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_canvas_import_child BEFORE INSERT ON canvas_revisions "
        "WHEN NEW.sequence = 2 BEGIN SELECT RAISE(ABORT, 'injected import'); END"
    )
    with pytest.raises(CanvasRepositoryError):
        repository.import_batch(retryable)
    assert repository.list_identities(conversation_id) == ()
    connection.execute("DROP TRIGGER fail_canvas_import_child")

    repository.import_batch(retryable)
    with pytest.raises(CanvasConflictError) as collision:
        repository.import_batch(retryable)
    assert collision.value.code == "identity_collision"
    assert len(repository.list_identities(conversation_id)) == 1
    assert len(repository.list_revision_metadata(conversation_id)) == 2


def test_parameter_values_remain_inert_and_query_plans_use_canvas_indexes(db) -> None:
    """Source/title values cannot become SQL and active-path support avoids table scans."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    sentinel = "x'); DROP TABLE conversations; --"
    created = _create(
        repository,
        conversation_id,
        message_id,
        title=sentinel,
        source=f"<main>{sentinel}</main>",
    )
    assert (
        repository.read_revision(conversation_id, created.revision.revision_id).title
        == sentinel
    )
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,))
        .fetchone()[0]
        == 1
    )

    document_plan = (
        db.get_connection()
        .execute(
            "EXPLAIN QUERY PLAN SELECT id FROM canvas_documents "
            "WHERE conversation_id = ? AND deleted = 0",
            (conversation_id,),
        )
        .fetchall()
    )
    revision_plan = (
        db.get_connection()
        .execute(
            "EXPLAIN QUERY PLAN SELECT id FROM canvas_revisions "
            "WHERE origin_message_id = ? ORDER BY canvas_id, sequence",
            (message_id,),
        )
        .fetchall()
    )
    assert "idx_canvas_documents_conversation" in " ".join(
        str(row[3]) for row in document_plan
    )
    assert "idx_canvas_revisions_origin_message" in " ".join(
        str(row[3]) for row in revision_plan
    )
