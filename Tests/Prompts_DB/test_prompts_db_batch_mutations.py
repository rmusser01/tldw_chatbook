"""Atomic Prompt/Recipe batch delete and restore integration coverage."""

from __future__ import annotations

import inspect
import queue
import sqlite3
import threading

import pytest
from loguru import logger

from tldw_chatbook.DB import Prompts_DB as prompts_db_module
from tldw_chatbook.DB.Prompts_DB import (
    DatabaseError,
    ExpectedVersionConflictError,
    PromptsDatabase,
)
from tldw_chatbook.Prompt_Management.prompt_batch_models import PromptBatchTarget
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


@pytest.fixture()
def db(tmp_path) -> PromptsDatabase:
    """Use a real file-backed SQLite database for every mutation test."""
    database = PromptsDatabase(tmp_path / "batch-prompts.db", client_id="batch-test")
    try:
        yield database
    finally:
        database.close_connection()


def _add_artifact(
    db: PromptsDatabase,
    *,
    name: str,
    artifact_type: str = "prompt",
    keywords: list[str] | None = None,
    body: str = "unique body",
) -> tuple[int, str]:
    local_id, artifact_uuid, _message = db.add_prompt(
        name=name,
        author="Batch Author",
        details=body,
        system_prompt=f"system {body}",
        user_prompt=f"user {body}",
        keywords=keywords,
        artifact_type=artifact_type,
    )
    assert local_id is not None
    assert artifact_uuid is not None
    return local_id, artifact_uuid


def _row_state(db: PromptsDatabase, local_id: int) -> tuple[int, int]:
    row = (
        db.get_connection()
        .execute("SELECT deleted, version FROM Prompts WHERE id = ?", (local_id,))
        .fetchone()
    )
    assert row is not None
    return int(row["deleted"]), int(row["version"])


def _set_next_prompt_id(db: PromptsDatabase, next_id: int) -> None:
    db.get_connection().execute(
        "INSERT INTO sqlite_sequence(name, seq) VALUES ('Prompts', ?)",
        (next_id - 1,),
    )
    db.get_connection().commit()


def _assert_no_active_batch_rows(
    db: PromptsDatabase, local_ids: tuple[int, ...]
) -> None:
    assert all(_row_state(db, local_id)[0] == 1 for local_id in local_ids)


def _assert_all_active_batch_rows(
    db: PromptsDatabase, local_ids: tuple[int, ...]
) -> None:
    assert all(_row_state(db, local_id)[0] == 0 for local_id in local_ids)


def _complete_mutation_state(db: PromptsDatabase) -> dict[str, tuple[tuple, ...]]:
    """Capture every row a prompt mutation is allowed to change."""
    conn = db.get_connection()
    return {
        table: tuple(
            tuple(row)
            for row in conn.execute(
                f'SELECT rowid, * FROM "{table}" ORDER BY rowid'
            ).fetchall()
        )
        for table in (
            "Prompts",
            "PromptKeywordsTable",
            "PromptKeywordLinks",
            "sync_log",
            "prompts_fts",
            "prompt_keywords_fts",
            "LocalPromptCollections",
            "LocalPromptCollectionItems",
        )
    }


def _add_collection_membership(db: PromptsDatabase, prompt_id: int, name: str) -> None:
    LocalPromptService(db).create_prompt_collection(
        {"name": name, "prompt_ids": [prompt_id]}
    )


def _insert_prompt_with_uuid(db: PromptsDatabase, malformed_uuid: str) -> int:
    """Insert schema-admitted legacy data that public writers reject."""
    keyword_id = db.add_keyword("canonical recovery keyword")
    assert keyword_id is not None
    now = db._get_current_utc_timestamp_str()
    conn = db.get_connection()
    cursor = conn.execute(
        """
        INSERT INTO Prompts (
            name, author, details, system_prompt, user_prompt, uuid,
            last_modified, version, client_id, deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0)
        """,
        (
            "Malformed UUID prompt",
            "Private author",
            "private malformed UUID body",
            "private system",
            "private user",
            malformed_uuid,
            now,
            "legacy-import",
        ),
    )
    prompt_id = int(cursor.lastrowid)
    conn.execute(
        """
        INSERT INTO prompts_fts (
            rowid, name, author, details, system_prompt, user_prompt
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            prompt_id,
            "Malformed UUID prompt",
            "Private author",
            "private malformed UUID body",
            "private system",
            "private user",
        ),
    )
    conn.execute(
        "INSERT INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)",
        (prompt_id, keyword_id),
    )
    conn.commit()
    _add_collection_membership(db, prompt_id, "Malformed UUID collection")
    return prompt_id


def _corrupt_active_keyword(db: PromptsDatabase, malformed_keyword: str) -> int:
    prompt_id, _ = _add_artifact(
        db,
        name="Malformed keyword prompt",
        keywords=["canonical recovery keyword"],
        body="private malformed keyword body",
    )
    _add_collection_membership(db, prompt_id, "Malformed keyword collection")
    conn = db.get_connection()
    keyword = conn.execute(
        """
        SELECT pkw.id, pkw.version
        FROM PromptKeywordLinks AS pkl
        JOIN PromptKeywordsTable AS pkw ON pkw.id = pkl.keyword_id
        WHERE pkl.prompt_id = ?
        """,
        (prompt_id,),
    ).fetchone()
    assert keyword is not None
    conn.execute(
        """
        UPDATE PromptKeywordsTable
        SET keyword = ?, last_modified = ?, version = ?, client_id = ?
        WHERE id = ?
        """,
        (
            malformed_keyword,
            db._get_current_utc_timestamp_str(),
            int(keyword["version"]) + 1,
            "legacy-import",
            keyword["id"],
        ),
    )
    conn.commit()
    return prompt_id


@pytest.mark.parametrize("surface", ["batch", "legacy"])
@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_public_mutations_reject_caller_owned_transactions_before_writes(
    db, surface, operation
):
    prompt_id, _ = _add_artifact(
        db,
        name=f"Ambient {surface} {operation}",
        keywords=["ambient keyword"],
        body="private ambient body",
    )
    _add_collection_membership(
        db, prompt_id, f"Ambient {surface} {operation} collection"
    )
    if operation == "restore":
        targets = db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),)).targets
        expected_version = 2
    else:
        targets = (PromptBatchTarget(prompt_id, 1),)
        expected_version = 1
    baseline = _complete_mutation_state(db)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    conn = db.get_connection()
    conn.execute("BEGIN")
    try:
        with pytest.raises(DatabaseError) as caught:
            if surface == "batch" and operation == "delete":
                db.soft_delete_prompts(targets)
            elif surface == "batch":
                db.restore_deleted_prompts(targets)
            elif operation == "delete":
                db.soft_delete_prompt(prompt_id, expected_version=expected_version)
            else:
                db.restore_deleted_prompt(prompt_id, expected_version=expected_version)

        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True
        assert _complete_mutation_state(db) == baseline
    finally:
        conn.rollback()
        logger.remove(sink)

    assert "committed operation" not in "\n".join(messages)


@pytest.mark.parametrize("surface", ["batch", "legacy"])
@pytest.mark.parametrize(
    "malformed_uuid",
    ["", "PRIVATE INVALID UUID VALUE", "123E4567-E89B-42D3-A456-426614174000"],
)
def test_delete_rejects_noncanonical_prompt_uuid_before_first_write(
    db, monkeypatch, surface, malformed_uuid
):
    prompt_id = _insert_prompt_with_uuid(db, malformed_uuid)
    baseline = _complete_mutation_state(db)
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as caught:
            if surface == "batch":
                db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),))
            else:
                db.soft_delete_prompt(prompt_id, expected_version=1)
    finally:
        logger.remove(sink)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert helper_calls == []
    assert _complete_mutation_state(db) == baseline
    rendered = "\n".join(messages)
    assert "committed operation" not in rendered
    assert malformed_uuid not in rendered or malformed_uuid == ""


@pytest.mark.parametrize("surface", ["batch", "legacy"])
@pytest.mark.parametrize("malformed_keyword", ["", " PRIVATE Keyword VALUE "])
def test_delete_rejects_noncanonical_keyword_recovery_before_first_write(
    db, monkeypatch, surface, malformed_keyword
):
    prompt_id = _corrupt_active_keyword(db, malformed_keyword)
    baseline = _complete_mutation_state(db)
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as caught:
            if surface == "batch":
                db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),))
            else:
                db.soft_delete_prompt(prompt_id, expected_version=1)
    finally:
        logger.remove(sink)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert helper_calls == []
    assert _complete_mutation_state(db) == baseline
    rendered = "\n".join(messages)
    assert "committed operation" not in rendered
    assert malformed_keyword not in rendered or malformed_keyword == ""


@pytest.mark.parametrize("corruption", ["uuid", "keyword"])
def test_batch_delete_preflights_all_recovery_rows_before_first_mutation(
    db, monkeypatch, corruption
):
    first_id, _ = _add_artifact(
        db, name=f"First recovery preflight {corruption}", keywords=["first keyword"]
    )
    second_id = (
        _insert_prompt_with_uuid(db, "PRIVATE INVALID SECOND UUID")
        if corruption == "uuid"
        else _corrupt_active_keyword(db, " PRIVATE Second Keyword ")
    )
    baseline = _complete_mutation_state(db)
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)

    with pytest.raises(DatabaseError):
        db.soft_delete_prompts(
            (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
        )

    assert helper_calls == []
    assert _complete_mutation_state(db) == baseline


@pytest.mark.parametrize("surface", ["batch", "legacy"])
def test_delete_requires_prompt_sync_event_before_success(db, monkeypatch, surface):
    prompt_id, _ = _add_artifact(
        db,
        name=f"Required sync {surface}",
        keywords=["required sync keyword"],
        body="private required sync body",
    )
    _add_collection_membership(db, prompt_id, f"Required sync {surface} collection")
    baseline = _complete_mutation_state(db)
    original = db._log_sync_event

    def drop_prompt_delete(conn, entity, entity_uuid, operation, version, payload=None):
        if entity == "Prompts" and operation == "delete":
            return None
        return original(conn, entity, entity_uuid, operation, version, payload)

    monkeypatch.setattr(db, "_log_sync_event", drop_prompt_delete)

    with pytest.raises(DatabaseError) as caught:
        if surface == "batch":
            db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),))
        else:
            db.soft_delete_prompt(prompt_id, expected_version=1)

    assert caught.value.__cause__ is None
    assert _complete_mutation_state(db) == baseline


def test_begin_immediate_preflights_after_competing_recovery_metadata_commit(
    db, monkeypatch
):
    prompt_id, _ = _add_artifact(
        db,
        name="Writer race prompt",
        keywords=["writer race keyword"],
        body="writer race body",
    )
    _add_collection_membership(db, prompt_id, "Writer race collection")
    keyword = (
        db.get_connection()
        .execute(
            """
        SELECT keyword_table.*
        FROM PromptKeywordLinks AS link
        JOIN PromptKeywordsTable AS keyword_table
          ON keyword_table.id = link.keyword_id
        WHERE link.prompt_id = ?
        """,
            (prompt_id,),
        )
        .fetchone()
    )
    assert keyword is not None
    link = (
        db.get_connection()
        .execute("SELECT * FROM PromptKeywordLinks WHERE prompt_id = ?", (prompt_id,))
        .fetchone()
    )
    assert link is not None
    baseline = _complete_mutation_state(db)
    competitor = PromptsDatabase(db.db_path, client_id="competing-writer")
    metadata_staged = threading.Event()
    batch_begin_attempted = threading.Event()
    allow_writer_commit = threading.Event()
    writer_committed = threading.Event()
    outcomes: queue.Queue[BaseException | object] = queue.Queue()
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)

    def writer() -> None:
        try:
            with competitor.transaction(immediate=True) as conn:
                conn.execute(
                    "DELETE FROM PromptKeywordsTable WHERE id = ?", (keyword["id"],)
                )
                conn.execute(
                    """
                    INSERT INTO PromptKeywordsTable (
                        id, keyword, uuid, last_modified, version, client_id,
                        deleted, prev_version, merge_parent_uuid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        keyword["id"],
                        keyword["keyword"],
                        "PRIVATE INVALID COMPETING UUID",
                        keyword["last_modified"],
                        keyword["version"],
                        keyword["client_id"],
                        keyword["deleted"],
                        keyword["prev_version"],
                        keyword["merge_parent_uuid"],
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO PromptKeywordLinks (id, prompt_id, keyword_id)
                    VALUES (?, ?, ?)
                    """,
                    (link["id"], link["prompt_id"], link["keyword_id"]),
                )
                metadata_staged.set()
                if not allow_writer_commit.wait(5):
                    raise AssertionError("batch never attempted BEGIN IMMEDIATE")
            writer_committed.set()
        except BaseException as exc:
            outcomes.put(exc)
        finally:
            competitor.close_connection()

    def batch_delete() -> None:
        conn = db.get_connection()

        def trace(statement: str) -> None:
            if statement.strip().upper() == "BEGIN IMMEDIATE":
                batch_begin_attempted.set()

        conn.set_trace_callback(trace)
        try:
            outcomes.put(db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),)))
        except BaseException as exc:
            outcomes.put(exc)
        finally:
            conn.set_trace_callback(None)
            db.close_connection()

    writer_thread = threading.Thread(target=writer, name="prompt-competing-writer")
    batch_thread = threading.Thread(target=batch_delete, name="prompt-batch-writer")
    writer_thread.start()
    assert metadata_staged.wait(5)
    batch_thread.start()
    assert batch_begin_attempted.wait(5)
    assert not writer_committed.is_set()
    assert helper_calls == []
    allow_writer_commit.set()
    writer_thread.join(5)
    batch_thread.join(5)
    assert not writer_thread.is_alive()
    assert not batch_thread.is_alive()

    observed = []
    while not outcomes.empty():
        observed.append(outcomes.get_nowait())
    assert len(observed) == 1
    assert isinstance(observed[0], DatabaseError)
    assert str(observed[0]) == "Prompt batch delete failed."
    assert observed[0].__cause__ is None
    assert helper_calls == []
    current = _complete_mutation_state(db)
    mutated_keyword = (
        db.get_connection()
        .execute(
            "SELECT * FROM PromptKeywordsTable WHERE id = ?",
            (keyword["id"],),
        )
        .fetchone()
    )
    assert mutated_keyword is not None
    assert dict(mutated_keyword) == {
        **dict(keyword),
        "uuid": "PRIVATE INVALID COMPETING UUID",
    }
    for table in baseline.keys() - {"PromptKeywordsTable"}:
        assert current[table] == baseline[table]


def test_batch_delete_and_restore_preserve_exact_database_state(db):
    prompt_id, _ = _add_artifact(db, name="Batch Prompt", keywords=["Alpha", "Shared"])
    recipe_id, _ = _add_artifact(
        db,
        name="Batch Recipe",
        artifact_type="recipe",
        keywords=["Beta", "Shared"],
    )
    collection_service = LocalPromptService(db)
    collection_id = collection_service.create_prompt_collection(
        {"name": "Batch Collection", "prompt_ids": [prompt_id, recipe_id]}
    )["collection_id"]
    baseline_change_id = (
        db.get_connection().execute("SELECT MAX(change_id) FROM sync_log").fetchone()[0]
    )

    result = db.soft_delete_prompts(
        (PromptBatchTarget(prompt_id, 1), PromptBatchTarget(recipe_id, 1))
    )

    assert [entry.local_id for entry in result.entries] == sorted(
        (prompt_id, recipe_id)
    )
    assert [entry.artifact_type for entry in result.entries] == ["prompt", "recipe"]
    assert db.search_prompts("unique body")[0] == []
    assert _row_state(db, prompt_id) == (1, 2)
    assert _row_state(db, recipe_id) == (1, 2)
    assert db.fetch_keywords_for_prompt(prompt_id) == []
    assert db.fetch_keywords_for_prompt(recipe_id) == []
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        .fetchone()[0]
        == 2
    )
    assert db.browse_prompts(collection_id=collection_id)[3] == 0

    delete_events = db.get_sync_log_entries(since_change_id=baseline_change_id)
    prompt_deletes = [
        event
        for event in delete_events
        if event["entity"] == "Prompts" and event["operation"] == "delete"
    ]
    assert [event["version"] for event in prompt_deletes] == [2, 2]
    assert [event["payload"]["keywords"] for event in prompt_deletes] == [
        ["alpha", "shared"],
        ["beta", "shared"],
    ]
    assert (
        sum(
            event["entity"] == "PromptKeywordLinks" and event["operation"] == "unlink"
            for event in delete_events
        )
        == 4
    )

    restored = db.restore_deleted_prompts(result.targets)

    assert tuple(entry.local_id for entry in restored.entries) == tuple(
        entry.local_id for entry in result.entries
    )
    assert [entry.restored_version for entry in restored.entries] == [3, 3]
    assert {row["id"] for row in db.search_prompts("unique body")[0]} == {
        prompt_id,
        recipe_id,
    }
    assert db.fetch_keywords_for_prompt(prompt_id) == ["alpha", "shared"]
    assert db.fetch_keywords_for_prompt(recipe_id) == ["beta", "shared"]
    assert db.browse_prompts(collection_id=collection_id)[3] == 2
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        .fetchone()[0]
        == 2
    )

    restore_events = db.get_sync_log_entries(
        since_change_id=delete_events[-1]["change_id"]
    )
    prompt_restores = [
        event
        for event in restore_events
        if event["entity"] == "Prompts" and event["operation"] == "update"
    ]
    assert [event["version"] for event in prompt_restores] == [3, 3]
    assert [event["payload"]["keywords"] for event in prompt_restores] == [
        ["alpha", "shared"],
        ["beta", "shared"],
    ]
    assert (
        sum(
            event["entity"] == "PromptKeywordLinks" and event["operation"] == "link"
            for event in restore_events
        )
        == 4
    )


def test_batch_results_are_canonical_for_reverse_ordered_targets(db):
    first_id, _ = _add_artifact(db, name="First canonical")
    second_id, _ = _add_artifact(db, name="Second canonical", artifact_type="recipe")

    deleted = db.soft_delete_prompts(
        (PromptBatchTarget(second_id, 1), PromptBatchTarget(first_id, 1))
    )
    restored = db.restore_deleted_prompts(tuple(reversed(deleted.targets)))

    assert [entry.local_id for entry in deleted.entries] == [first_id, second_id]
    assert [entry.local_id for entry in restored.entries] == [first_id, second_id]


def test_batch_restore_recovers_keyword_soft_deleted_after_tombstone(db):
    prompt_id, _ = _add_artifact(
        db, name="Recover deleted keyword", keywords=["Recover Later"]
    )
    receipt = db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),))
    assert db.soft_delete_keyword("recover later") is True

    restored = db.restore_deleted_prompts(receipt.targets)

    assert restored.entries[0].restored_version == 3
    assert db.fetch_keywords_for_prompt(prompt_id) == ["recover later"]
    keyword = db.get_active_keyword_by_text("recover later")
    assert keyword is not None
    assert keyword["version"] == 3
    assert [
        row["id"] for row in db.search_prompts("recover later", ["keywords"])[0]
    ] == [prompt_id]


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
@pytest.mark.parametrize(
    "container_factory", [list, lambda values: TupleSubclass(values)]
)
def test_batch_methods_require_an_exact_tuple(db, method_name, container_factory):
    local_id, _ = _add_artifact(db, name=f"Exact tuple {method_name}")
    target = PromptBatchTarget(local_id, 1)

    with pytest.raises(TypeError, match="tuple"):
        getattr(db, method_name)(container_factory((target,)))

    assert _row_state(db, local_id) == (0, 1)


class TupleSubclass(tuple):
    """Prove that the DB boundary requires the exact immutable container type."""


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
def test_batch_methods_reject_empty_tuple_before_sql(db, method_name):
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            getattr(db, method_name)(())
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
def test_batch_methods_reject_duplicate_local_ids_before_sql(db, method_name):
    local_id, _ = _add_artifact(db, name=f"Duplicate target {method_name}")
    targets = (PromptBatchTarget(local_id, 1), PromptBatchTarget(local_id, 2))
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match="unique"):
            getattr(db, method_name)(targets)
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []
    assert _row_state(db, local_id) == (0, 1)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("local_id", True),
        ("local_id", 0),
        ("local_id", -1),
        ("local_id", 2**63),
        ("expected_version", True),
        ("expected_version", 0),
        ("expected_version", -1),
        ("expected_version", 2**63),
    ],
)
def test_batch_target_rejects_noncanonical_integer_values(field_name, value):
    values = {"local_id": 1, "expected_version": 1}
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        PromptBatchTarget(**values)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("local_id", True),
        ("local_id", 0),
        ("local_id", -1),
        ("local_id", 2**63),
        ("expected_version", True),
        ("expected_version", 0),
        ("expected_version", -1),
        ("expected_version", 2**63),
    ],
)
def test_batch_boundary_revalidates_tampered_targets_before_sql(db, field_name, value):
    target = object.__new__(PromptBatchTarget)
    object.__setattr__(target, "local_id", 1)
    object.__setattr__(target, "expected_version", 1)
    object.__setattr__(target, field_name, value)
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match=field_name):
            db.soft_delete_prompts((target,))
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_batch_delete_validates_every_target_before_first_write(
    db, monkeypatch, failure
):
    first_id, _ = _add_artifact(db, name=f"First delete {failure}")
    second_id, _ = _add_artifact(db, name=f"Second delete {failure}")
    second_target = (
        PromptBatchTarget(second_id + 999_999, 1)
        if failure == "missing"
        else PromptBatchTarget(second_id, 2)
    )
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)

    with pytest.raises(ExpectedVersionConflictError):
        db.soft_delete_prompts((PromptBatchTarget(first_id, 1), second_target))

    assert helper_calls == []
    _assert_all_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_batch_restore_rejects_missing_or_stale_targets_atomically(db, failure):
    first_id, _ = _add_artifact(db, name=f"First restore {failure}")
    second_id, _ = _add_artifact(db, name=f"Second restore {failure}")
    receipt = db.soft_delete_prompts(
        (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    )
    second_target = (
        PromptBatchTarget(second_id + 999_999, 2)
        if failure == "missing"
        else PromptBatchTarget(second_id, 1)
    )

    with pytest.raises(ExpectedVersionConflictError):
        db.restore_deleted_prompts((receipt.targets[0], second_target))

    _assert_no_active_batch_rows(db, (first_id, second_id))


def test_batch_restore_validates_every_recovery_payload_before_first_write(
    db, monkeypatch
):
    first_id, _ = _add_artifact(db, name="First restore payload", keywords=["one"])
    second_id, second_uuid = _add_artifact(
        db, name="Second restore payload", keywords=["two"]
    )
    receipt = db.soft_delete_prompts(
        (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    )
    db.get_connection().execute(
        """
        UPDATE sync_log SET payload = ?
        WHERE entity = 'Prompts' AND entity_uuid = ?
          AND operation = 'delete' AND version = 2
        """,
        ('{"keywords": ["private two"]}', second_uuid),
    )
    db.get_connection().commit()
    helper_calls: list[int] = []
    original = db._restore_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_restore_prompt_in_transaction", record_helper)

    with pytest.raises(DatabaseError, match="batch restore"):
        db.restore_deleted_prompts(receipt.targets)

    assert helper_calls == []
    _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_forced_second_mutation_failure_rolls_back_every_row(
    db, monkeypatch, operation
):
    first_id, _ = _add_artifact(db, name=f"First forced {operation}")
    second_id, _ = _add_artifact(db, name=f"Second forced {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        helper_name = "_delete_prompt_in_transaction"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        helper_name = "_restore_prompt_in_transaction"
    original = getattr(db, helper_name)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.OperationalError("private second mutation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, helper_name, fail_second)

    with pytest.raises(DatabaseError) as caught:
        getattr(
            db,
            f"{'soft_delete' if operation == 'delete' else 'restore_deleted'}_prompts",
        )(targets)

    assert caught.value.__cause__ is None
    if operation == "delete":
        _assert_all_active_batch_rows(db, (first_id, second_id))
    else:
        _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_result_construction_failure_rolls_back_before_commit(
    db, monkeypatch, operation
):
    first_id, _ = _add_artifact(db, name=f"First result {operation}")
    second_id, _ = _add_artifact(db, name=f"Second result {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        result_name = "PromptBatchDeleteResult"
        method_name = "soft_delete_prompts"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        result_name = "PromptBatchRestoreResult"
        method_name = "restore_deleted_prompts"

    class ExplodingResult:
        def __init__(self, **_kwargs):
            raise ValueError("private result-constructor failure")

    monkeypatch.setattr(prompts_db_module, result_name, ExplodingResult, raising=False)

    with pytest.raises(DatabaseError) as caught:
        getattr(db, method_name)(targets)

    assert caught.value.__cause__ is None
    if operation == "delete":
        _assert_all_active_batch_rows(db, (first_id, second_id))
    else:
        _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_uses_exactly_one_begin_immediate(db, operation):
    first_id, _ = _add_artifact(db, name=f"First begin {operation}")
    second_id, _ = _add_artifact(db, name=f"Second begin {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        method_name = "soft_delete_prompts"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        method_name = "restore_deleted_prompts"
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        getattr(db, method_name)(targets)
    finally:
        db.get_connection().set_trace_callback(None)

    begins = [
        statement.strip().upper()
        for statement in statements
        if statement.lstrip().upper().startswith("BEGIN")
    ]
    assert begins == ["BEGIN IMMEDIATE"]


@pytest.mark.parametrize("lookup", ["integer", "name", "uuid"])
def test_batch_refactor_preserves_legacy_single_lookup_and_mapping(db, lookup):
    local_id, artifact_uuid = _add_artifact(
        db,
        name=f"Legacy {lookup}",
        artifact_type="recipe",
        keywords=["Legacy Keyword"],
    )
    identifier: int | str = {
        "integer": local_id,
        "name": f"Legacy {lookup}",
        "uuid": artifact_uuid,
    }[lookup]

    assert db.soft_delete_prompt(identifier, expected_version=1) is True
    assert _row_state(db, local_id) == (1, 2)

    restored = db.restore_deleted_prompt(identifier, expected_version=2)

    assert isinstance(restored, dict)
    assert restored["id"] == local_id
    assert restored["version"] == 3
    assert restored["deleted"] == 0
    assert restored["artifact_type"] == "recipe"
    assert restored["keywords"] == ["legacy keyword"]


def test_batch_refactor_preserves_legacy_single_edge_contracts(db):
    local_id, _ = _add_artifact(db, name="Legacy edge")
    assert db.soft_delete_prompt("missing legacy prompt") is False

    with pytest.raises(ExpectedVersionConflictError):
        db.soft_delete_prompt(local_id, expected_version=2)

    assert _row_state(db, local_id) == (0, 1)
    assert db.soft_delete_prompt(local_id) is True
    assert _row_state(db, local_id) == (1, 2)
    assert list(inspect.signature(PromptsDatabase.soft_delete_prompt).parameters) == [
        "self",
        "prompt_id_or_name_or_uuid",
        "expected_version",
    ]
    assert list(
        inspect.signature(PromptsDatabase.restore_deleted_prompt).parameters
    ) == ["self", "prompt_id_or_name_or_uuid", "expected_version"]


def test_batch_refactor_routes_legacy_single_wrappers_through_shared_helpers(
    db, monkeypatch
):
    local_id, _ = _add_artifact(db, name="Legacy shared helper")
    delete_original = db._delete_prompt_in_transaction
    restore_original = db._restore_prompt_in_transaction
    calls: list[str] = []

    def delete_spy(*args, **kwargs):
        calls.append("delete")
        return delete_original(*args, **kwargs)

    def restore_spy(*args, **kwargs):
        calls.append("restore")
        return restore_original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", delete_spy)
    monkeypatch.setattr(db, "_restore_prompt_in_transaction", restore_spy)

    assert db.soft_delete_prompt(local_id) is True
    restored = db.restore_deleted_prompt(local_id, expected_version=2)

    assert calls == ["delete", "restore"]
    assert restored["version"] == 3


@pytest.mark.parametrize("restore_mode", ["single", "batch"])
def test_restore_reuses_prevalidated_keyword_recovery_rows(
    db, monkeypatch, restore_mode
):
    local_id, _ = _add_artifact(
        db,
        name=f"Recovery rows {restore_mode}",
        keywords=["recovery keyword"],
    )
    deleted = db.soft_delete_prompts((PromptBatchTarget(local_id, 1),))
    original = db._restore_prompt_keyword_rows
    calls = 0

    def recovery_spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_restore_prompt_keyword_rows", recovery_spy)

    if restore_mode == "single":
        db.restore_deleted_prompt(local_id, expected_version=2)
    else:
        db.restore_deleted_prompts(deleted.targets)

    assert calls == 1


def test_batch_success_diagnostics_are_aggregate_and_privacy_safe(db):
    _set_next_prompt_id(db, 71_234_567)
    first_id, first_uuid = _add_artifact(
        db,
        name="PRIVATE NAME ALPHA",
        keywords=["PRIVATE KEYWORD ALPHA"],
        body="PRIVATE BODY ALPHA",
    )
    second_id, second_uuid = _add_artifact(
        db,
        name="PRIVATE NAME BETA",
        keywords=["PRIVATE KEYWORD BETA"],
        body="PRIVATE BODY BETA",
    )
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        deleted = db.soft_delete_prompts(
            (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
        )
        db.restore_deleted_prompts(deleted.targets)
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert "operation=delete count=2" in rendered
    assert "operation=restore count=2" in rendered
    for forbidden in (
        "PRIVATE NAME",
        "PRIVATE KEYWORD",
        "PRIVATE BODY",
        str(first_id),
        str(second_id),
        first_uuid,
        second_uuid,
    ):
        assert forbidden not in rendered


def test_batch_failure_is_bounded_and_privacy_safe(db, monkeypatch):
    _set_next_prompt_id(db, 81_234_567)
    first_id, first_uuid = _add_artifact(
        db, name="PRIVATE FAILURE NAME", keywords=["PRIVATE FAILURE KEYWORD"]
    )
    second_id, second_uuid = _add_artifact(db, name="Second failure row")
    private_exception = (
        f"PRIVATE EXCEPTION {first_id} {second_id} {first_uuid} {second_uuid}"
    )

    def fail_without_mutating(*_args, **_kwargs):
        raise sqlite3.OperationalError(private_exception)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", fail_without_mutating)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as caught:
            db.soft_delete_prompts(
                (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
            )
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert str(caught.value) == "Prompt batch delete failed."
    assert caught.value.__cause__ is None
    assert "operation=delete count=2 category=OperationalError" in rendered
    assert "Traceback" not in rendered
    for forbidden in (
        "PRIVATE FAILURE NAME",
        "PRIVATE FAILURE KEYWORD",
        private_exception,
        str(first_id),
        str(second_id),
        first_uuid,
        second_uuid,
    ):
        assert forbidden not in rendered
