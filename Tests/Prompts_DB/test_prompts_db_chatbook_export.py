from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import pytest
from loguru import logger

from tldw_chatbook.DB.Prompts_DB import DatabaseError, PromptsDatabase


@pytest.fixture()
def database(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="chatbook-export-test")
    try:
        yield db
    finally:
        db.close_connection()


def _insert_prompt(
    database: PromptsDatabase,
    *,
    name: str,
    deleted: int = 0,
    artifact_type: str = "prompt",
) -> int:
    cursor = database.get_connection().execute(
        """
        INSERT INTO Prompts (
            name, author, details, system_prompt, user_prompt, uuid,
            last_modified, version, client_id, deleted, prompt_format,
            prompt_schema_version, prompt_definition, artifact_type
        ) VALUES (?, 'Author', 'Details', 'System', 'User', ?,
                  '2026-08-12T00:00:00.000Z', 1, ?, ?, 'legacy',
                  NULL, NULL, ?)
        """,
        (name, str(uuid.uuid4()), database.client_id, deleted, artifact_type),
    )
    database.get_connection().commit()
    return int(cursor.lastrowid)


def test_get_all_active_prompt_ids_is_uncapped_ordered_and_excludes_deleted(
    database: PromptsDatabase,
) -> None:
    expected = [
        _insert_prompt(
            database,
            name=f"Prompt {index:03d}",
            artifact_type="recipe" if index % 2 else "prompt",
        )
        for index in range(207)
    ]
    deleted_id = _insert_prompt(database, name="Deleted", deleted=1)

    result = database.get_all_active_prompt_ids()

    assert result == expected
    assert deleted_id not in result
    assert all(type(prompt_id) is int for prompt_id in result)


def test_prompt_chatbook_snapshot_returns_exact_portable_fields_and_keywords(
    database: PromptsDatabase,
) -> None:
    definition = '{"kind":"block_prompt","version":2,"literal":"[bold]研究🙂"}'
    prompt_id, _, _ = database.add_prompt(
        name="Structured Recipe",
        author=None,
        details="",
        system_prompt="System\nline",
        user_prompt="User\nمرحبا",
        keywords=["Zulu", "alpha"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    assert prompt_id is not None

    snapshot = database.fetch_prompt_chatbook_snapshot(prompt_id)

    assert snapshot == {
        "name": "Structured Recipe",
        "author": None,
        "details": "",
        "system_prompt": "System\nline",
        "user_prompt": "User\nمرحبا",
        "keywords": ["alpha", "zulu"],
        "artifact_type": "recipe",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": definition,
    }


def test_prompt_chatbook_snapshot_returns_none_for_missing_or_deleted(
    database: PromptsDatabase,
) -> None:
    deleted_id = _insert_prompt(database, name="Deleted", deleted=1)

    assert database.fetch_prompt_chatbook_snapshot(deleted_id) is None
    assert database.fetch_prompt_chatbook_snapshot(deleted_id + 999) is None


@pytest.mark.parametrize("prompt_id", [True, False, 0, -1, 1.0, "1", None, 2**63])
def test_prompt_chatbook_snapshot_rejects_invalid_ids_before_sql(
    database: PromptsDatabase, prompt_id: Any
) -> None:
    traced: list[str] = []
    database.get_connection().set_trace_callback(traced.append)
    try:
        with pytest.raises(ValueError, match="positive integer"):
            database.fetch_prompt_chatbook_snapshot(prompt_id)
    finally:
        database.get_connection().set_trace_callback(None)

    assert traced == []


def test_prompt_chatbook_snapshot_uses_one_explicit_read_transaction(
    database: PromptsDatabase,
) -> None:
    prompt_id, _, _ = database.add_prompt(
        name="Transaction",
        author="Author",
        details="Details",
        keywords=["alpha"],
    )
    assert prompt_id is not None
    traced: list[str] = []
    database.get_connection().set_trace_callback(traced.append)
    try:
        database.fetch_prompt_chatbook_snapshot(prompt_id)
    finally:
        database.get_connection().set_trace_callback(None)

    normalized = [statement.strip().upper() for statement in traced]
    begin_index = normalized.index("BEGIN")
    commit_index = normalized.index("COMMIT")
    select_indices = [
        index
        for index, statement in enumerate(normalized)
        if statement.startswith("SELECT")
    ]
    assert len(select_indices) == 2
    assert begin_index < select_indices[0] < select_indices[1] < commit_index


def test_prompt_chatbook_snapshot_uses_shared_transaction_context(
    database: PromptsDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id = _insert_prompt(database, name="Shared transaction")
    transaction_calls: list[bool] = []
    original_transaction = database.transaction

    @contextmanager
    def recording_transaction(*, immediate: bool = False):
        transaction_calls.append(immediate)
        with original_transaction(immediate=immediate) as connection:
            yield connection

    monkeypatch.setattr(database, "transaction", recording_transaction)

    snapshot = database.fetch_prompt_chatbook_snapshot(prompt_id)

    assert snapshot is not None
    assert transaction_calls == [False]


def test_prompt_chatbook_snapshot_keeps_row_and_keywords_on_one_wal_snapshot(
    database: PromptsDatabase,
) -> None:
    prompt_id, _, _ = database.add_prompt(
        name="Snapshot",
        author="Author",
        details="Before",
        keywords=["before"],
    )
    assert prompt_id is not None
    path = database.db_path_str
    reader = database.get_connection()
    reader.execute("PRAGMA journal_mode=WAL")
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    mutation_done = False

    def mutate_before_keyword_select(statement: str) -> None:
        nonlocal mutation_done
        if (
            mutation_done
            or "FROM PromptKeywordsTable AS keyword_table" not in statement
        ):
            return
        mutation_done = True
        keyword_cursor = writer.execute(
            """
            INSERT INTO PromptKeywordsTable (
                keyword, uuid, last_modified, version, client_id, deleted
            ) VALUES ('after', ?, '2026-08-12T00:01:00.000Z', 1, 'writer', 0)
            """,
            (str(uuid.uuid4()),),
        )
        writer.execute(
            "INSERT INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)",
            (prompt_id, int(keyword_cursor.lastrowid)),
        )
        writer.commit()

    reader.set_trace_callback(mutate_before_keyword_select)
    try:
        snapshot = database.fetch_prompt_chatbook_snapshot(prompt_id)
    finally:
        reader.set_trace_callback(None)
        writer.close()

    assert mutation_done is True
    assert snapshot is not None
    assert snapshot["keywords"] == ["before"]
    assert database.fetch_keywords_for_prompt(prompt_id) == ["after", "before"]


class _Cursor:
    def __init__(self, *, one: Any = None, many: list[Any] | None = None) -> None:
        self._one = one
        self._many = [] if many is None else many

    def fetchone(self) -> Any:
        return self._one

    def fetchall(self) -> list[Any]:
        return self._many


class _ShapeConnection:
    in_transaction = False

    def __init__(
        self, row: Mapping[str, Any], keywords: list[Mapping[str, Any]]
    ) -> None:
        self.row = row
        self.keywords = keywords
        self.select_count = 0
        self.committed = False
        self.rolled_back = False

    def execute(self, statement: str, params: tuple[Any, ...] = ()) -> _Cursor:
        if statement == "BEGIN":
            return _Cursor()
        self.select_count += 1
        if self.select_count == 1:
            return _Cursor(one=self.row)
        return _Cursor(many=self.keywords)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


@pytest.mark.parametrize(
    ("row", "keywords"),
    [
        ({"name": "Missing all other fields"}, []),
        (
            {
                "name": "Complete",
                "author": None,
                "details": None,
                "system_prompt": None,
                "user_prompt": None,
                "artifact_type": "prompt",
                "prompt_format": "legacy",
                "prompt_schema_version": None,
                "prompt_definition": None,
            },
            [{}],
        ),
    ],
)
def test_prompt_chatbook_snapshot_shape_failures_rollback_before_return(
    database: PromptsDatabase,
    monkeypatch: pytest.MonkeyPatch,
    row: Mapping[str, Any],
    keywords: list[Mapping[str, Any]],
) -> None:
    fake = _ShapeConnection(row, keywords)
    monkeypatch.setattr(database, "get_connection", lambda: fake)

    with pytest.raises(DatabaseError, match="Failed to read Prompt export snapshot"):
        database.fetch_prompt_chatbook_snapshot(1)

    assert fake.rolled_back is True
    assert fake.committed is False


class _FailingConnection:
    in_transaction = False

    def execute(self, statement: str, params: tuple[Any, ...] = ()) -> _Cursor:
        if statement == "BEGIN":
            return _Cursor()
        raise sqlite3.DatabaseError("TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK")

    def commit(self) -> None:
        raise AssertionError("commit must not run")

    def rollback(self) -> None:
        raise sqlite3.DatabaseError("TASK197_ROLLBACK_EXCEPTION_MUST_NOT_LEAK")


def test_prompt_chatbook_snapshot_database_failure_is_bounded_and_unlogged(
    database: PromptsDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    monkeypatch.setattr(database, "get_connection", _FailingConnection)
    try:
        with pytest.raises(DatabaseError) as raised:
            database.fetch_prompt_chatbook_snapshot(1)
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert str(raised.value) == "Failed to read Prompt export snapshot."
    assert "TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK" not in str(raised.value)
    assert "TASK197_ROLLBACK_EXCEPTION_MUST_NOT_LEAK" not in repr(raised.value)
    assert "TASK197_DATABASE_EXCEPTION_MUST_NOT_LEAK" not in rendered
    assert "TASK197_ROLLBACK_EXCEPTION_MUST_NOT_LEAK" not in rendered
    assert "Traceback" not in rendered
