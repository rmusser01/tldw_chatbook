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


def _message(
    db: CharactersRAGDB,
    conversation_id: str,
    content: str,
    *,
    parent_message_id: str | None = None,
) -> str:
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": parent_message_id,
            "sender": "assistant",
            "role": "assistant",
            "content": content,
        }
    )
    assert message_id is not None
    return message_id


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


def _turn_batch(
    conversation_id: str,
    message_id: str,
    *,
    canvas_id: str,
    revision_id: str,
    source: str,
    sequence: int,
    parent_revision_id: str | None,
) -> CanvasImportBatch:
    """Build one exact assistant-turn batch for the transaction-aware seam."""

    created_at = "2026-09-04T00:00:00+00:00"
    return CanvasImportBatch(
        conversation_id=conversation_id,
        documents=(
            (
                CanvasImportDocument(
                    canvas_id=canvas_id,
                    conversation_id=conversation_id,
                    created_at=created_at,
                ),
            )
            if parent_revision_id is None
            else ()
        ),
        revisions=(
            CanvasImportRevision(
                revision_id=revision_id,
                canvas_id=canvas_id,
                parent_revision_id=parent_revision_id,
                sequence=sequence,
                title="Turn batch",
                runtime_profile="canvas-v1",
                source=source,
                content_sha256=_sha256(source),
                source_bytes=len(source.encode("utf-8")),
                actor_kind="assistant",
                origin_message_id=message_id,
                origin_turn_id="turn-batch",
                created_at=created_at,
            ),
        ),
    )


_RAW_REVISION_INSERT_SQL = (
    "INSERT INTO canvas_revisions "
    "(id, canvas_id, parent_revision_id, sequence, title, runtime_profile, "
    "html, content_sha256, html_bytes, actor_kind, origin_message_id, "
    "origin_turn_id, created_at, deleted_at) "
    "VALUES (?, ?, ?, 2, 'raw revision', 'canvas-v1', ?, ?, ?, 'assistant', "
    "?, 'turn-raw', '2026-09-03T00:00:00.000Z', NULL)"
)
_RAW_REVISION_CAST_TEXT_INSERT_SQL = _RAW_REVISION_INSERT_SQL.replace(
    "'canvas-v1', ?, ?, ?",
    "'canvas-v1', CAST(? AS TEXT), ?, ?",
)


def _insert_raw_revision(
    connection: sqlite3.Connection,
    *,
    canvas_id: str,
    parent_revision_id: str,
    origin_message_id: str,
    source: object,
    content_sha256: str,
    source_bytes: int,
    cast_source_as_text: bool = False,
) -> None:
    """Attempt one otherwise-valid sequence-two insert outside the repository."""

    statement = (
        _RAW_REVISION_CAST_TEXT_INSERT_SQL
        if cast_source_as_text
        else _RAW_REVISION_INSERT_SQL
    )
    connection.execute(
        statement,
        (
            str(uuid4()),
            canvas_id,
            parent_revision_id,
            source,
            content_sha256,
            source_bytes,
            origin_message_id,
        ),
    )


def _seed_untraced_message(
    connection: sqlite3.Connection,
    *,
    message_id: str,
    conversation_id: str,
    parent_message_id: str | None = None,
) -> None:
    """Insert a legacy message with no semantic-revision protection."""

    connection.execute(
        "INSERT INTO messages "
        "(id, conversation_id, parent_message_id, sender, content, client_id) "
        "VALUES (?, ?, ?, 'assistant', 'legacy canvas origin', 'canvas-tests')",
        (message_id, conversation_id, parent_message_id),
    )


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


def test_owner_origin_and_parent_checks_fail_without_mutation(db) -> None:
    """Foreign origins and parents must never create repository rows."""

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


def test_database_rejects_blob_revision_source_with_matching_digest_and_size(
    db,
) -> None:
    """TEXT affinity must not accept a BLOB payload through otherwise-valid SQL."""

    conversation_id, message_id = _owner(db)
    root = _create(CanvasRepository(db), conversation_id, message_id).revision
    source = b"<main>blob</main>"

    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_revision(
            db.get_connection(),
            canvas_id=root.canvas_id,
            parent_revision_id=root.revision_id,
            origin_message_id=message_id,
            source=sqlite3.Binary(source),
            content_sha256=hashlib.sha256(source).hexdigest(),
            source_bytes=len(source),
        )
    assert len(CanvasRepository(db).list_revision_metadata(conversation_id)) == 1


def test_database_rejects_invalid_utf8_stored_with_text_storage_class(db) -> None:
    """A CAST-created invalid UTF-8 TEXT value must fail at the database boundary."""

    conversation_id, message_id = _owner(db)
    root = _create(CanvasRepository(db), conversation_id, message_id).revision
    source = b"\x80"

    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_revision(
            db.get_connection(),
            canvas_id=root.canvas_id,
            parent_revision_id=root.revision_id,
            origin_message_id=message_id,
            source=sqlite3.Binary(source),
            content_sha256=hashlib.sha256(source).hexdigest(),
            source_bytes=len(source),
            cast_source_as_text=True,
        )
    assert len(CanvasRepository(db).list_revision_metadata(conversation_id)) == 1


def test_database_rejects_revision_source_byte_count_mismatch(db) -> None:
    """A unique-sequence raw insert cannot forge the encoded source byte count."""

    conversation_id, message_id = _owner(db)
    root = _create(CanvasRepository(db), conversation_id, message_id).revision
    source = "<main>λ</main>"

    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_revision(
            db.get_connection(),
            canvas_id=root.canvas_id,
            parent_revision_id=root.revision_id,
            origin_message_id=message_id,
            source=source,
            content_sha256=_sha256(source),
            source_bytes=len(source.encode("utf-8")) + 1,
        )
    assert len(CanvasRepository(db).list_revision_metadata(conversation_id)) == 1


def test_database_rejects_revision_digest_mismatch_at_unique_sequence(db) -> None:
    """A well-shaped forged digest must fail independently of sequence uniqueness."""

    conversation_id, message_id = _owner(db)
    root = _create(CanvasRepository(db), conversation_id, message_id).revision
    source = "<main>forged</main>"

    with pytest.raises(sqlite3.IntegrityError):
        _insert_raw_revision(
            db.get_connection(),
            canvas_id=root.canvas_id,
            parent_revision_id=root.revision_id,
            origin_message_id=message_id,
            source=source,
            content_sha256="0" * 64,
            source_bytes=len(source.encode("utf-8")),
        )
    assert len(CanvasRepository(db).list_revision_metadata(conversation_id)) == 1


def test_database_without_canvas_payload_function_fails_closed_on_insert(db) -> None:
    """A generic SQLite connection cannot bypass connection-local validation."""

    conversation_id, message_id = _owner(db)
    root = _create(CanvasRepository(db), conversation_id, message_id).revision
    source = "<main>valid</main>"
    external = sqlite3.connect(db.db_path_str, isolation_level=None)
    try:
        external.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.OperationalError, match="function"):
            _insert_raw_revision(
                external,
                canvas_id=root.canvas_id,
                parent_revision_id=root.revision_id,
                origin_message_id=message_id,
                source=source,
                content_sha256=_sha256(source),
                source_bytes=len(source.encode("utf-8")),
            )
    finally:
        external.close()
    assert len(CanvasRepository(db).list_revision_metadata(conversation_id)) == 1


def test_database_rejects_reassigning_a_canvas_origin_message(db) -> None:
    """A legacy untraced origin cannot move away from its Canvas owner."""

    first_conversation = db.add_conversation({"title": "first"})
    second_conversation = db.add_conversation({"title": "second"})
    assert first_conversation is not None
    assert second_conversation is not None
    origin_message_id = str(uuid4())
    ordinary_message_id = str(uuid4())
    connection = db.get_connection()
    _seed_untraced_message(
        connection,
        message_id=origin_message_id,
        conversation_id=first_conversation,
    )
    _seed_untraced_message(
        connection,
        message_id=ordinary_message_id,
        conversation_id=first_conversation,
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_trace_semantic_revisions "
            "WHERE live_message_id IN (?, ?)",
            (origin_message_id, ordinary_message_id),
        ).fetchone()[0]
        == 0
    )
    _create(CanvasRepository(db), first_conversation, origin_message_id)

    with pytest.raises(sqlite3.IntegrityError, match="origin owner"):
        connection.execute(
            "UPDATE messages SET conversation_id = ? WHERE id = ?",
            (second_conversation, origin_message_id),
        )
    assert (
        connection.execute(
            "SELECT conversation_id FROM messages WHERE id = ?", (origin_message_id,)
        ).fetchone()[0]
        == first_conversation
    )

    connection.execute(
        "UPDATE messages SET conversation_id = ? WHERE id = ?",
        (second_conversation, ordinary_message_id),
    )
    assert (
        connection.execute(
            "SELECT conversation_id FROM messages WHERE id = ?", (ordinary_message_id,)
        ).fetchone()[0]
        == second_conversation
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


def test_write_rejects_managed_outer_deferred_transaction_before_mutation(db) -> None:
    """A caught inner failure must not commit an identity through a deferred owner."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_canvas_root_in_managed_outer "
        "BEFORE INSERT ON canvas_revisions WHEN NEW.sequence = 1 "
        "BEGIN SELECT RAISE(ABORT, 'injected managed outer failure'); END"
    )

    observed_code: str | None = None
    with db.transaction() as cursor:
        try:
            _create(repository, conversation_id, message_id)
        except CanvasRepositoryError as exc:
            observed_code = exc.code
        cursor.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            ("managed outer committed", conversation_id),
        )

    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 0
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_revisions").fetchone()[0] == 0
    )
    assert observed_code == "transaction_ownership_required"
    assert (
        connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == "managed outer committed"
    )


def test_write_rejects_native_transaction_before_partial_commit(db) -> None:
    """A caught failure inside native BEGIN must leave no Canvas identity to commit."""

    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_canvas_root_in_native_outer "
        "BEFORE INSERT ON canvas_revisions WHEN NEW.sequence = 1 "
        "BEGIN SELECT RAISE(ABORT, 'injected native outer failure'); END"
    )

    observed_code: str | None = None
    connection.execute("BEGIN")
    try:
        try:
            _create(repository, conversation_id, message_id)
        except CanvasRepositoryError as exc:
            observed_code = exc.code
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            ("native outer committed", conversation_id),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise

    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 0
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_revisions").fetchone()[0] == 0
    )
    assert observed_code == "transaction_ownership_required"
    assert (
        connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == "native outer committed"
    )


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


@pytest.mark.parametrize(
    ("limits", "new_canvas", "source", "expected_code"),
    (
        (
            CanvasRepositoryLimits(max_canvases_per_conversation=1),
            True,
            "x",
            "canvas_count",
        ),
        (
            CanvasRepositoryLimits(max_revisions_per_canvas=1),
            False,
            "x",
            "revision_count",
        ),
        (
            CanvasRepositoryLimits(
                max_revisions_per_canvas=2,
                max_source_bytes_per_conversation=8,
            ),
            False,
            "12345",
            "conversation_source_bytes",
        ),
    ),
)
def test_transaction_append_counts_existing_rows_before_any_write(
    db, limits, new_canvas, source, expected_code
) -> None:
    conversation_id, message_id = _owner(db)
    repository = CanvasRepository(db)
    root = _create(repository, conversation_id, message_id, source="1234")
    canvas_id = str(uuid4()) if new_canvas else root.identity.canvas_id
    batch = _turn_batch(
        conversation_id,
        message_id,
        canvas_id=canvas_id,
        revision_id=str(uuid4()),
        source=source,
        sequence=1 if new_canvas else 2,
        parent_revision_id=None if new_canvas else root.revision.revision_id,
    )

    with (
        pytest.raises(CanvasQuotaError) as refused,
        db.transaction(immediate=True) as cursor,
    ):
        CanvasRepository.append_batch_in_transaction(
            cursor,
            batch,
            anchor_message_id=message_id,
            limits=limits,
        )

    assert refused.value.code == expected_code
    assert len(repository.list_identities(conversation_id)) == 1
    assert len(repository.list_revision_metadata(conversation_id)) == 1


def test_transaction_append_rejects_off_path_parent_and_invalid_anchor(db) -> None:
    conversation_id, root_message_id = _owner(db)
    left_message_id = _message(
        db, conversation_id, "left", parent_message_id=root_message_id
    )
    right_message_id = _message(
        db, conversation_id, "right", parent_message_id=root_message_id
    )
    repository = CanvasRepository(db)
    root = _create(repository, conversation_id, left_message_id)
    batch = _turn_batch(
        conversation_id,
        right_message_id,
        canvas_id=root.identity.canvas_id,
        revision_id=str(uuid4()),
        source="<main>right</main>",
        sequence=2,
        parent_revision_id=root.revision.revision_id,
    )

    with (
        pytest.raises(CanvasValidationError) as off_path,
        db.transaction(immediate=True) as cursor,
    ):
        CanvasRepository.append_batch_in_transaction(
            cursor, batch, anchor_message_id=right_message_id
        )
    with (
        pytest.raises(CanvasValidationError) as invalid_anchor,
        db.transaction(immediate=True) as cursor,
    ):
        CanvasRepository.append_batch_in_transaction(
            cursor, batch, anchor_message_id="missing-message"
        )

    assert off_path.value.code == "parent_owner_mismatch"
    assert invalid_anchor.value.code == "invalid_active_path"
    assert len(repository.list_revision_metadata(conversation_id)) == 1


def test_transaction_append_serializes_duplicate_sequences_across_connections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "canvas-turn-race.sqlite"
    first_db = CharactersRAGDB(database_path, client_id="canvas-turn-race")
    second_db = CharactersRAGDB(database_path, client_id="canvas-turn-race")
    try:
        conversation_id, message_id = _owner(first_db)
        root = _create(CanvasRepository(first_db), conversation_id, message_id)

        def append(database: CharactersRAGDB, title: str) -> str:
            batch = _turn_batch(
                conversation_id,
                message_id,
                canvas_id=root.identity.canvas_id,
                revision_id=str(uuid4()),
                source=f"<main>{title}</main>",
                sequence=2,
                parent_revision_id=root.revision.revision_id,
            )
            try:
                with database.transaction(immediate=True) as cursor:
                    CanvasRepository.append_batch_in_transaction(
                        cursor, batch, anchor_message_id=message_id
                    )
            except CanvasValidationError as exc:
                return exc.code
            return "committed"

        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(append, first_db, "left")
            right = executor.submit(append, second_db, "right")
            outcomes = {left.result(timeout=10), right.result(timeout=10)}

        assert outcomes == {"committed", "invalid_revision_sequence"}
        assert (
            len(CanvasRepository(first_db).list_revision_metadata(conversation_id)) == 2
        )
    finally:
        first_db.close_connection()
        second_db.close_connection()


def test_transaction_append_failure_rolls_back_caller_and_canvas_writes(db) -> None:
    conversation_id, message_id = _owner(db)
    canvas_id = str(uuid4())
    first_revision_id = str(uuid4())
    created_at = "2026-09-04T00:00:00+00:00"
    first = _turn_batch(
        conversation_id,
        message_id,
        canvas_id=canvas_id,
        revision_id=first_revision_id,
        source="one",
        sequence=1,
        parent_revision_id=None,
    )
    second_source = "two"
    batch = CanvasImportBatch(
        conversation_id=conversation_id,
        documents=first.documents,
        revisions=first.revisions
        + (
            CanvasImportRevision(
                revision_id=str(uuid4()),
                canvas_id=canvas_id,
                parent_revision_id=first_revision_id,
                sequence=2,
                title="Turn batch",
                runtime_profile="canvas-v1",
                source=second_source,
                content_sha256=_sha256(second_source),
                source_bytes=len(second_source),
                actor_kind="assistant",
                origin_message_id=message_id,
                origin_turn_id="turn-batch",
                created_at=created_at,
            ),
        ),
    )
    connection = db.get_connection()
    connection.execute(
        "CREATE TRIGGER fail_canvas_turn_second BEFORE INSERT ON canvas_revisions "
        "WHEN NEW.sequence = 2 BEGIN SELECT RAISE(ABORT, 'injected turn failure'); END"
    )

    with (
        pytest.raises(sqlite3.IntegrityError),
        db.transaction(immediate=True) as cursor,
    ):
        cursor.execute(
            "UPDATE conversations SET title = 'must roll back' WHERE id = ?",
            (conversation_id,),
        )
        CanvasRepository.append_batch_in_transaction(
            cursor, batch, anchor_message_id=message_id
        )

    assert (
        connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == "owner"
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 0
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_revisions").fetchone()[0] == 0
    )


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


def test_hard_purge_iteratively_deletes_a_deep_message_chain_child_first(db) -> None:
    """A valid chain deeper than Python's recursion limit must still purge."""

    conversation_id = db.add_conversation({"title": "deep owner"})
    assert conversation_id is not None
    message_ids = tuple(f"deep-message-{index:04d}" for index in range(1_205))
    rows = []
    parent_message_id = None
    for message_id in message_ids:
        rows.append(
            (
                message_id,
                conversation_id,
                parent_message_id,
                "assistant",
                "deep node",
                "canvas-tests",
            )
        )
        parent_message_id = message_id

    connection = db.get_connection()
    with db.transaction(immediate=True) as cursor:
        cursor.executemany(
            "INSERT INTO messages "
            "(id, conversation_id, parent_message_id, sender, content, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    repository = CanvasRepository(db)
    _create(repository, conversation_id, message_ids[-1])
    connection.execute(
        "CREATE TABLE canvas_purge_message_order ("
        "position INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TRIGGER audit_canvas_message_purge AFTER DELETE ON messages "
        "BEGIN INSERT INTO canvas_purge_message_order(message_id) VALUES (OLD.id); END"
    )

    purged = repository.hard_purge_conversation(conversation_id)

    assert (purged.canvases_deleted, purged.revisions_deleted) == (1, 1)
    deleted_message_ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT message_id FROM canvas_purge_message_order ORDER BY position"
        ).fetchall()
    ]
    assert deleted_message_ids == list(reversed(message_ids))
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 0
    )


def test_hard_purge_rejects_a_message_cycle_without_partial_deletion(db) -> None:
    """Iterative traversal must keep cycle detection and transaction atomicity."""

    conversation_id = db.add_conversation({"title": "cyclic owner"})
    assert conversation_id is not None
    connection = db.get_connection()
    root_message_id = "cycle-root"
    child_message_id = "cycle-child"
    with db.transaction(immediate=True) as cursor:
        _seed_untraced_message(
            cursor.connection,
            message_id=root_message_id,
            conversation_id=conversation_id,
        )
        _seed_untraced_message(
            cursor.connection,
            message_id=child_message_id,
            conversation_id=conversation_id,
            parent_message_id=root_message_id,
        )
        cursor.execute(
            "UPDATE messages SET parent_message_id = ? WHERE id = ?",
            (child_message_id, root_message_id),
        )

    repository = CanvasRepository(db)
    _create(repository, conversation_id, child_message_id)

    with pytest.raises(CanvasConflictError) as cycle:
        repository.hard_purge_conversation(conversation_id)

    assert cycle.value.code == "message_graph_cycle"
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == 1
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        == 2
    )
    assert (
        connection.execute("SELECT COUNT(*) FROM canvas_documents").fetchone()[0] == 1
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


def test_repository_scoped_mutations_accept_one_exact_durable_path(db) -> None:
    """Removing the scoped mutation parameter must break this repository contract."""

    conversation_id, root_message_id = _owner(db)
    leaf_message_id = _message(
        db,
        conversation_id,
        "leaf",
        parent_message_id=root_message_id,
    )
    active_message_ids = (root_message_id, leaf_message_id)
    repository = CanvasRepository(db)

    created = repository.create_canvas(
        conversation_id,
        title="Scoped create",
        source="<main>create</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=leaf_message_id,
        origin_turn_id="run-create",
        active_message_ids=active_message_ids,
    )
    appended = repository.append_revision(
        conversation_id,
        created.identity.canvas_id,
        parent_revision_id=created.revision.revision_id,
        title="Scoped append",
        source="<main>append</main>",
        runtime_profile="canvas-v1",
        actor_kind="assistant",
        origin_message_id=leaf_message_id,
        origin_turn_id="run-append",
        active_message_ids=active_message_ids,
    )

    assert created.revision.origin_message_id == leaf_message_id
    assert appended.parent_revision_id == created.revision.revision_id
    assert appended.origin_message_id == leaf_message_id


@pytest.mark.parametrize(
    "invalid_path_kind",
    ("duplicate", "foreign", "deleted", "reparented"),
)
def test_repository_scoped_create_rejects_invalid_path_before_any_canvas_write(
    db,
    invalid_path_kind,
) -> None:
    """Skipping any complete-path invariant must authorize one malformed case."""

    conversation_id, root_message_id = _owner(db)
    other_root_id = _message(db, conversation_id, "other root")
    middle_message_id = _message(
        db,
        conversation_id,
        "middle",
        parent_message_id=root_message_id,
    )
    leaf_message_id = _message(
        db,
        conversation_id,
        "leaf",
        parent_message_id=middle_message_id,
    )
    foreign_conversation_id, foreign_message_id = _owner(db, "foreign")
    assert foreign_conversation_id != conversation_id

    if invalid_path_kind == "duplicate":
        active_message_ids = (
            root_message_id,
            middle_message_id,
            middle_message_id,
            leaf_message_id,
        )
    elif invalid_path_kind == "foreign":
        active_message_ids = (
            root_message_id,
            middle_message_id,
            foreign_message_id,
        )
    else:
        active_message_ids = (
            root_message_id,
            middle_message_id,
            leaf_message_id,
        )
        with db.transaction(immediate=True) as cursor:
            if invalid_path_kind == "deleted":
                cursor.execute(
                    "UPDATE messages SET deleted = 1 WHERE id = ?",
                    (middle_message_id,),
                )
        if invalid_path_kind == "reparented":
            assert db.update_message(
                middle_message_id,
                {"parent_message_id": other_root_id},
                expected_version=1,
                preserve_descendants=True,
            )

    repository = CanvasRepository(db)
    with pytest.raises(CanvasValidationError) as invalid_path:
        repository.create_canvas(
            conversation_id,
            title="Must not exist",
            source="<main>blocked</main>",
            runtime_profile="canvas-v1",
            actor_kind="assistant",
            origin_message_id=leaf_message_id,
            origin_turn_id="run-blocked",
            active_message_ids=active_message_ids,
        )

    assert invalid_path.value.code == "invalid_active_path"
    assert repository.list_identities(conversation_id) == ()
