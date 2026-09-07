"""Retained Prompt history database contract tests for TASK-196."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import pytest

import tldw_chatbook.DB.Prompts_DB as prompts_db_module
from tldw_chatbook.DB.Prompts_DB import (
    ConflictError,
    DatabaseError,
    InputError,
    PromptsDatabase,
)


HISTORY_INDEX = "idx_sync_log_prompt_history"
TARGET_UUID = "00000000-0000-4000-8000-000000000196"
OTHER_UUID = "00000000-0000-4000-8000-000000000197"


def _seed_v3_database(
    database_path: Any, *, create_wrong_history_index: bool = False
) -> list[tuple[Any, ...]]:
    """Create a real v3 database with retained rows but no history index."""
    conn = sqlite3.connect(database_path)
    try:
        conn.executescript(
            f"""
            {PromptsDatabase._TABLES_SQL_V1}
            {PromptsDatabase._INDICES_SQL_V1}
            {PromptsDatabase._TRIGGERS_SQL_V1}
            ALTER TABLE Prompts
                ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy';
            ALTER TABLE Prompts ADD COLUMN prompt_schema_version INTEGER;
            ALTER TABLE Prompts ADD COLUMN prompt_definition TEXT;
            ALTER TABLE Prompts
                ADD COLUMN artifact_type TEXT NOT NULL DEFAULT 'prompt'
                CHECK(artifact_type IN ('prompt', 'recipe'));
            UPDATE schema_version SET version = 3;
            {PromptsDatabase._FTS_TABLES_SQL}
            """
        )
        rows = [
            (
                7,
                "Prompts",
                TARGET_UUID,
                "create",
                "2026-08-08T00:00:00.000Z",
                "v3-client",
                1,
                '{"name":"Retained v1","version":1}',
            ),
            (
                21,
                "Prompts",
                TARGET_UUID,
                "update",
                "2026-08-08T00:01:00.000Z",
                "v3-client",
                2,
                '{"name":"Retained v2","version":2}',
            ),
            (
                42,
                "Prompts",
                TARGET_UUID,
                "delete",
                "2026-08-08T00:02:00.000Z",
                "v3-client",
                3,
                '{"name":"Retained v2","version":3}',
            ),
        ]
        conn.executemany(
            """
            INSERT INTO sync_log (
                change_id, entity, entity_uuid, operation, timestamp, client_id,
                version, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        if create_wrong_history_index:
            conn.execute(f"CREATE INDEX {HISTORY_INDEX} ON sync_log(timestamp)")
        conn.commit()
        return rows
    finally:
        conn.close()


def _insert_sync_event(
    conn: sqlite3.Connection,
    *,
    entity: str = "Prompts",
    entity_uuid: str = TARGET_UUID,
    operation: str,
    version: int,
    payload: str | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO sync_log (
            entity, entity_uuid, operation, timestamp, client_id, version, payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entity,
            entity_uuid,
            operation,
            f"2026-08-08T00:{version:02d}:00.000Z",
            "history-test",
            version,
            payload,
        ),
    )
    return int(cursor.lastrowid)


def _seed_history_rows(database: PromptsDatabase) -> dict[str, int]:
    with database.transaction() as conn:
        ids = {
            "create": _insert_sync_event(
                conn,
                operation="create",
                version=1,
                payload=json.dumps({"name": "v1", "version": 1}),
            ),
            "other_entity": _insert_sync_event(
                conn,
                entity="PromptKeywordsTable",
                operation="create",
                version=1,
                payload=json.dumps({"keyword": "unrelated"}),
            ),
            "update_2": _insert_sync_event(
                conn,
                operation="update",
                version=2,
                payload=json.dumps({"name": "v2", "version": 2}),
            ),
            "delete": _insert_sync_event(
                conn,
                operation="delete",
                version=3,
                payload=json.dumps({"name": "deleted", "version": 3}),
            ),
            "other_uuid": _insert_sync_event(
                conn,
                entity_uuid=OTHER_UUID,
                operation="update",
                version=20,
                payload=json.dumps({"name": "other", "version": 20}),
            ),
            "malformed_update": _insert_sync_event(
                conn,
                operation="update",
                version=3,
                payload="{not valid json",
            ),
            "update_4": _insert_sync_event(
                conn,
                operation="update",
                version=4,
                payload=json.dumps({"name": "v4", "version": 4}),
            ),
        }
    return ids


def _index_sql(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (HISTORY_INDEX,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def _normalized_index_predicate(conn: sqlite3.Connection) -> str:
    normalized_sql = " ".join(_index_sql(conn).lower().split())
    _prefix, separator, predicate = normalized_sql.partition(" where ")
    assert separator
    return predicate


def _trace_prompt_history_query_plans(
    database: PromptsDatabase, *, page_size: int = 10
) -> dict[str, list[str]]:
    traced: list[str] = []
    conn = database.get_connection()
    conn.set_trace_callback(traced.append)
    database.get_prompt_history_entries(TARGET_UUID, page_size=page_size)
    conn.set_trace_callback(None)

    production_queries = [
        statement
        for statement in traced
        if statement.lstrip().upper().startswith("SELECT")
        and "FROM sync_log" in statement
    ]
    assert len(production_queries) == 2
    return {
        "count": [
            row[3]
            for row in conn.execute("EXPLAIN QUERY PLAN " + production_queries[0])
        ],
        "rows": [
            row[3]
            for row in conn.execute("EXPLAIN QUERY PLAN " + production_queries[1])
        ],
    }


def _assert_prompt_history_query_plans_use_index(
    plans: dict[str, list[str]],
) -> None:
    assert any(
        f"USING COVERING INDEX {HISTORY_INDEX}" in detail for detail in plans["count"]
    )
    assert any(f"USING INDEX {HISTORY_INDEX}" in detail for detail in plans["rows"])
    assert all(
        "SCAN sync_log" not in detail and "USE TEMP B-TREE" not in detail
        for plan in plans.values()
        for detail in plan
    )


def _sync_events_after(
    database: PromptsDatabase, change_id: int = 0
) -> list[dict[str, Any]]:
    rows = database.get_connection().execute(
        """
        SELECT change_id, entity, entity_uuid, operation, version, payload
        FROM sync_log
        WHERE change_id > ?
        ORDER BY change_id
        """,
        (change_id,),
    )
    events = []
    for row in rows:
        event = dict(row)
        event["decoded_payload"] = json.loads(event["payload"])
        events.append(event)
    return events


def _latest_change_id(database: PromptsDatabase) -> int:
    row = (
        database.get_connection()
        .execute("SELECT COALESCE(MAX(change_id), 0) FROM sync_log")
        .fetchone()
    )
    return int(row[0])


def _prompt_storage_state(
    database: PromptsDatabase,
) -> dict[str, list[tuple[Any, ...]]]:
    conn = database.get_connection()
    queries = {
        "prompts": "SELECT * FROM Prompts ORDER BY id",
        "prompt_fts": """
            SELECT rowid, name, author, details, system_prompt, user_prompt
            FROM prompts_fts
            ORDER BY rowid
        """,
        "keywords": "SELECT * FROM PromptKeywordsTable ORDER BY id",
        "keyword_fts": "SELECT rowid, keyword FROM prompt_keywords_fts ORDER BY rowid",
        "links": """
            SELECT prompt_id, keyword_id
            FROM PromptKeywordLinks
            ORDER BY prompt_id, keyword_id
        """,
        "sync_log": "SELECT * FROM sync_log ORDER BY change_id",
    }
    return {
        name: [tuple(row) for row in conn.execute(query)]
        for name, query in queries.items()
    }


def _install_link_insert_failure(database: PromptsDatabase) -> None:
    conn = database.get_connection()
    conn.execute(
        """
        CREATE TRIGGER fail_prompt_keyword_link_insert
        BEFORE INSERT ON PromptKeywordLinks
        BEGIN
            SELECT RAISE(ABORT, 'forced PromptKeywordLinks insert failure');
        END
        """
    )
    conn.commit()


def _install_snapshot_finalize_failure(database: PromptsDatabase) -> None:
    conn = database.get_connection()
    conn.execute(
        """
        CREATE TRIGGER fail_prompt_snapshot_finalize
        BEFORE UPDATE OF payload ON sync_log
        WHEN OLD.entity = 'Prompts'
        BEGIN
            SELECT RAISE(ABORT, 'forced Prompt snapshot finalize failure');
        END
        """
    )
    conn.commit()


def test_fresh_v4_schema_has_partial_covering_prompt_history_index():
    database = PromptsDatabase(":memory:", client_id="fresh-v4")
    try:
        conn = database.get_connection()

        assert database._get_db_version(conn) == 4
        assert [
            row[2] for row in conn.execute(f"PRAGMA index_info({HISTORY_INDEX})")
        ] == ["entity", "entity_uuid", "change_id", "operation"]

        assert _normalized_index_predicate(conn) == (
            "entity = 'prompts' and operation in ('create', 'update')"
        )
    finally:
        database.close_connection()


def test_v3_to_v4_migration_preserves_every_retained_sync_row(tmp_path):
    database_path = tmp_path / "prompts-v3.db"
    expected_rows = _seed_v3_database(database_path)

    database = PromptsDatabase(database_path, client_id="upgrade-v4")
    try:
        conn = database.get_connection()
        actual_rows = [
            tuple(row)
            for row in conn.execute(
                """
                SELECT change_id, entity, entity_uuid, operation,
                       CAST(timestamp AS TEXT), client_id, version, payload
                FROM sync_log
                ORDER BY change_id
                """
            )
        ]

        assert database._get_db_version(conn) == 4
        assert actual_rows == expected_rows
        assert HISTORY_INDEX in _index_sql(conn)
    finally:
        database.close_connection()


def test_v3_to_v4_migration_replaces_wrong_reserved_index(tmp_path):
    database_path = tmp_path / "prompts-v3-wrong-index.db"
    _seed_v3_database(database_path, create_wrong_history_index=True)

    database = PromptsDatabase(database_path, client_id="replace-wrong-index")
    try:
        conn = database.get_connection()

        assert database._get_db_version(conn) == 4
        assert [
            row[2] for row in conn.execute(f"PRAGMA index_info({HISTORY_INDEX})")
        ] == ["entity", "entity_uuid", "change_id", "operation"]
        assert _normalized_index_predicate(conn) == (
            "entity = 'prompts' and operation in ('create', 'update')"
        )
        _assert_prompt_history_query_plans_use_index(
            _trace_prompt_history_query_plans(database)
        )
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "bad_index_sql",
    [
        f"CREATE INDEX {HISTORY_INDEX} ON sync_log(timestamp)",
        f"""
        CREATE INDEX {HISTORY_INDEX}
        ON sync_log(entity, entity_uuid, change_id ASC, operation)
        WHERE entity = 'Prompts' AND operation IN ('create', 'update')
        """,
        f"""
        CREATE INDEX {HISTORY_INDEX}
        ON sync_log(entity, entity_uuid, change_id DESC, operation)
        WHERE entity = 'Prompts' AND operation = 'create'
        """,
    ],
    ids=["wrong-columns", "wrong-sort-order", "wrong-partial-predicate"],
)
def test_v3_to_v4_migration_rolls_back_index_when_validation_fails(
    tmp_path, monkeypatch, bad_index_sql
):
    database_path = tmp_path / "prompts-v3-invalid-created-index.db"
    _seed_v3_database(database_path)
    monkeypatch.setattr(
        PromptsDatabase,
        "_PROMPT_HISTORY_INDEX_SQL",
        bad_index_sql,
    )

    with pytest.raises(DatabaseError, match="initialization"):
        PromptsDatabase(database_path, client_id="invalid-created-index")

    conn = sqlite3.connect(database_path)
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                (HISTORY_INDEX,),
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_prompt_history_count_is_exact_and_decodes_no_payload(monkeypatch):
    database = PromptsDatabase(":memory:", client_id="history-count")
    try:
        _seed_history_rows(database)

        def fail_json_decode(_raw: str) -> Any:
            raise AssertionError("history count must not decode payloads")

        monkeypatch.setattr(prompts_db_module.json, "loads", fail_json_decode)

        assert database.get_prompt_history_count(TARGET_UUID) == 4
        assert database.get_prompt_history_count(OTHER_UUID) == 1
    finally:
        database.close_connection()


def test_prompt_history_page_is_filtered_bounded_and_cursor_paged(monkeypatch):
    database = PromptsDatabase(":memory:", client_id="history-page")
    try:
        ids = _seed_history_rows(database)
        real_loads = prompts_db_module.json.loads
        decoded_payloads: list[str] = []

        def recording_loads(raw: str) -> Any:
            decoded_payloads.append(raw)
            return real_loads(raw)

        monkeypatch.setattr(prompts_db_module.json, "loads", recording_loads)
        statements: list[str] = []
        database.get_connection().set_trace_callback(statements.append)

        first_page = database.get_prompt_history_entries(TARGET_UUID, page_size=2)

        assert set(first_page) == {
            "items",
            "predecessor",
            "total_count",
            "has_more",
            "next_before_change_id",
        }
        assert [item["change_id"] for item in first_page["items"]] == [
            ids["update_4"],
            ids["malformed_update"],
        ]
        assert first_page["predecessor"]["change_id"] == ids["update_2"]
        assert first_page["total_count"] == 4
        assert first_page["has_more"] is True
        assert first_page["next_before_change_id"] == ids["malformed_update"]
        assert len(decoded_payloads) == 3

        malformed = first_page["items"][1]
        assert malformed["entity"] == "Prompts"
        assert malformed["entity_uuid"] == TARGET_UUID
        assert malformed["operation"] == "update"
        assert malformed["version"] == 3
        assert malformed["payload"] is None
        assert malformed["payload_error"] == "malformed_json"
        assert malformed["raw_payload"] == "{not valid json"

        transaction_statements = [
            " ".join(statement.upper().split()) for statement in statements
        ]
        begin_position = transaction_statements.index("BEGIN")
        count_position = next(
            index
            for index, statement in enumerate(transaction_statements)
            if statement.startswith("SELECT COUNT(*) FROM SYNC_LOG")
        )
        rows_position = next(
            index
            for index, statement in enumerate(transaction_statements)
            if statement.startswith("SELECT * FROM SYNC_LOG")
        )
        commit_position = transaction_statements.index("COMMIT")
        assert begin_position < count_position < rows_position < commit_position

        decoded_payloads.clear()
        second_page = database.get_prompt_history_entries(
            TARGET_UUID,
            page_size=2,
            before_change_id=first_page["next_before_change_id"],
        )

        assert [item["change_id"] for item in second_page["items"]] == [
            ids["update_2"],
            ids["create"],
        ]
        assert second_page["predecessor"] is None
        assert second_page["total_count"] == 4
        assert second_page["has_more"] is False
        assert second_page["next_before_change_id"] is None
        assert len(decoded_payloads) == 2
    finally:
        database.close_connection()


def test_prompt_history_preserves_valid_non_object_json_as_decoded_data():
    database = PromptsDatabase(":memory:", client_id="history-json-shape")
    try:
        with database.transaction() as conn:
            _insert_sync_event(
                conn,
                operation="create",
                version=1,
                payload='["valid", "non-object"]',
            )
            _insert_sync_event(
                conn,
                operation="update",
                version=2,
                payload="{malformed",
            )

        page = database.get_prompt_history_entries(TARGET_UUID, page_size=2)

        malformed, non_object = page["items"]
        assert malformed["payload"] is None
        assert malformed["payload_error"] == "malformed_json"
        assert malformed["raw_payload"] == "{malformed"
        assert non_object["payload"] == ["valid", "non-object"]
        assert non_object["payload_error"] is None
        assert non_object["raw_payload"] is None
    finally:
        database.close_connection()


@pytest.mark.parametrize("entity_uuid", [None, "", "   ", 196, True])
def test_prompt_history_rejects_invalid_entity_uuid(entity_uuid):
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        with pytest.raises(InputError, match="entity_uuid"):
            database.get_prompt_history_count(entity_uuid)
        with pytest.raises(InputError, match="entity_uuid"):
            database.get_prompt_history_entries(entity_uuid, page_size=1)
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "page_size",
    [None, True, False, 0, -1, 1.0, math.inf, "2", 101, 10**100],
)
def test_prompt_history_rejects_non_positive_or_non_integer_page_size(page_size):
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        with pytest.raises(InputError, match="page_size"):
            database.get_prompt_history_entries(TARGET_UUID, page_size=page_size)
    finally:
        database.close_connection()


def test_prompt_history_accepts_maximum_page_size():
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        assert database.get_prompt_history_entries(TARGET_UUID, page_size=100) == {
            "items": [],
            "predecessor": None,
            "total_count": 0,
            "has_more": False,
            "next_before_change_id": None,
        }
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "before_change_id",
    [True, False, 0, -1, 1.0, math.inf, "2", 2**63, 10**100],
)
def test_prompt_history_rejects_invalid_before_change_id(before_change_id):
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        with pytest.raises(InputError, match="before_change_id"):
            database.get_prompt_history_entries(
                TARGET_UUID,
                page_size=1,
                before_change_id=before_change_id,
            )
    finally:
        database.close_connection()


def test_prompt_history_accepts_largest_sqlite_change_id():
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        _seed_history_rows(database)

        page = database.get_prompt_history_entries(
            TARGET_UUID,
            page_size=1,
            before_change_id=(2**63) - 1,
        )

        assert page["items"]
    finally:
        database.close_connection()


def test_prompt_history_production_queries_use_index_without_scan_or_temp_sort():
    database = PromptsDatabase(":memory:", client_id="history-query-plan")
    try:
        with database.transaction() as conn:
            for version in range(1, 501):
                _insert_sync_event(
                    conn,
                    entity="PromptKeywordsTable",
                    entity_uuid=f"unrelated-{version}",
                    operation="update",
                    version=version,
                    payload=json.dumps({"version": version}),
                )
            _insert_sync_event(
                conn,
                operation="create",
                version=1,
                payload=json.dumps({"name": "target", "version": 1}),
            )

        _assert_prompt_history_query_plans_use_index(
            _trace_prompt_history_query_plans(database)
        )
    finally:
        database.close_connection()


def test_new_prompt_snapshot_captures_canonical_keywords_before_link_events():
    database = PromptsDatabase(":memory:", client_id="snapshot-create")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Canonical create",
            author="Author",
            details="Created details",
            system_prompt="System",
            user_prompt="User",
            keywords=["  Beta   Tag ", "alpha", "ALPHA", "\t"],
        )

        assert database.fetch_keywords_for_prompt(prompt_id) == ["alpha", "beta tag"]
        prompt_row = (
            database.get_connection()
            .execute("SELECT name, details FROM Prompts WHERE id = ?", (prompt_id,))
            .fetchone()
        )
        assert tuple(prompt_row) == ("Canonical create", "Created details")
        fts_row = (
            database.get_connection()
            .execute(
                "SELECT name, details FROM prompts_fts WHERE rowid = ?", (prompt_id,)
            )
            .fetchone()
        )
        assert tuple(fts_row) == ("Canonical create", "Created details")

        events = _sync_events_after(database)
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        link_events = [
            event for event in events if event["entity"] == "PromptKeywordLinks"
        ]
        assert len(prompt_events) == 1
        assert prompt_events[0]["operation"] == "create"
        assert prompt_events[0]["entity_uuid"] == prompt_uuid
        assert prompt_events[0]["decoded_payload"]["keywords"] == [
            "alpha",
            "beta tag",
        ]
        assert prompt_events[0]["decoded_payload"]["keywords_captured"] is True
        assert len(link_events) == 2
        assert all(
            prompt_events[0]["change_id"] < event["change_id"] for event in link_events
        )
    finally:
        database.close_connection()


def test_new_prompt_snapshot_with_omitted_keywords_captures_empty_membership():
    database = PromptsDatabase(":memory:", client_id="snapshot-create-empty")
    try:
        _prompt_id, prompt_uuid, _message = database.add_prompt(
            name="No keywords",
            author=None,
            details=None,
        )

        events = _sync_events_after(database)
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        assert len(prompt_events) == 1
        assert prompt_events[0]["entity_uuid"] == prompt_uuid
        assert prompt_events[0]["decoded_payload"]["keywords"] == []
        assert prompt_events[0]["decoded_payload"]["keywords_captured"] is True
    finally:
        database.close_connection()


def test_add_prompt_overwrite_without_keywords_captures_unchanged_membership():
    database = PromptsDatabase(":memory:", client_id="snapshot-overwrite")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Overwrite target",
            author="Before",
            details="Before details",
            keywords=[" Beta ", "alpha"],
        )
        before_change_id = _latest_change_id(database)

        updated_id, updated_uuid, _message = database.add_prompt(
            name="Overwrite target",
            author="After",
            details="After details",
            overwrite=True,
        )

        assert (updated_id, updated_uuid) == (prompt_id, prompt_uuid)
        assert database.fetch_keywords_for_prompt(prompt_id) == ["alpha", "beta"]
        events = _sync_events_after(database, before_change_id)
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        assert len(prompt_events) == 1
        assert prompt_events[0]["operation"] == "update"
        assert prompt_events[0]["decoded_payload"]["keywords"] == ["alpha", "beta"]
        assert prompt_events[0]["decoded_payload"]["keywords_captured"] is True
        assert not [
            event for event in events if event["entity"] == "PromptKeywordLinks"
        ]

        before_change_id = _latest_change_id(database)
        database.add_prompt(
            name="Overwrite target",
            author="Final",
            details="Final details",
            keywords=["  Gamma   Tag ", "ALPHA", "alpha"],
            overwrite=True,
        )

        assert database.fetch_keywords_for_prompt(prompt_id) == ["alpha", "gamma tag"]
        events = _sync_events_after(database, before_change_id)
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        link_events = [
            event for event in events if event["entity"] == "PromptKeywordLinks"
        ]
        assert len(prompt_events) == 1
        assert prompt_events[0]["decoded_payload"]["keywords"] == [
            "alpha",
            "gamma tag",
        ]
        assert prompt_events[0]["decoded_payload"]["keywords_captured"] is True
        assert link_events
        assert all(
            prompt_events[0]["change_id"] < event["change_id"] for event in link_events
        )
    finally:
        database.close_connection()


def test_update_prompt_snapshot_captures_final_keywords_without_rewriting_legacy_row():
    database = PromptsDatabase(":memory:", client_id="snapshot-update")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Update target",
            author="Before",
            details="Before details",
            keywords=["legacy", "remove me"],
        )
        conn = database.get_connection()
        legacy_row = conn.execute(
            """
            SELECT change_id
            FROM sync_log
            WHERE entity = 'Prompts' AND entity_uuid = ?
            ORDER BY change_id
            LIMIT 1
            """,
            (prompt_uuid,),
        ).fetchone()
        legacy_payload = '{"name":"legacy snapshot","version":1}'
        conn.execute(
            "UPDATE sync_log SET payload = ? WHERE change_id = ?",
            (legacy_payload, legacy_row["change_id"]),
        )
        conn.commit()
        before_change_id = _latest_change_id(database)

        updated_uuid, _message = database.update_prompt_by_id(
            prompt_id,
            {
                "name": "Updated target",
                "details": "Updated details",
                "keywords": ["  Final   Tag ", "alpha", "ALPHA"],
            },
        )

        assert updated_uuid == prompt_uuid
        assert database.fetch_keywords_for_prompt(prompt_id) == ["alpha", "final tag"]
        assert (
            conn.execute(
                "SELECT payload FROM sync_log WHERE change_id = ?",
                (legacy_row["change_id"],),
            ).fetchone()[0]
            == legacy_payload
        )
        assert tuple(
            conn.execute(
                "SELECT name, details FROM prompts_fts WHERE rowid = ?", (prompt_id,)
            ).fetchone()
        ) == ("Updated target", "Updated details")

        events = _sync_events_after(database, before_change_id)
        prompt_events = [event for event in events if event["entity"] == "Prompts"]
        link_events = [
            event for event in events if event["entity"] == "PromptKeywordLinks"
        ]
        assert len(prompt_events) == 1
        assert prompt_events[0]["operation"] == "update"
        assert prompt_events[0]["decoded_payload"]["keywords"] == [
            "alpha",
            "final tag",
        ]
        assert prompt_events[0]["decoded_payload"]["keywords_captured"] is True
        assert link_events
        assert all(
            prompt_events[0]["change_id"] < event["change_id"] for event in link_events
        )
    finally:
        database.close_connection()


def test_retained_restore_re_resolves_snapshot_and_uses_conditional_update():
    database = PromptsDatabase(":memory:", client_id="retained-restore")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Original",
            author="Author",
            details="Original details",
            system_prompt="Original system",
            user_prompt="Original user",
            keywords=["original"],
        )
        database.update_prompt_by_id(
            prompt_id,
            {
                "name": "Current",
                "author": "Current author",
                "details": "Current details",
                "system_prompt": "Current system",
                "user_prompt": "Current user",
                "keywords": ["current"],
            },
            expected_version=1,
        )
        history = database.get_prompt_history_entries(prompt_uuid, page_size=10)
        source = next(item for item in history["items"] if item["version"] == 1)

        def validate_snapshot(snapshot):
            payload = snapshot["payload"]
            return {
                "update_data": {
                    field: payload[field]
                    for field in (
                        "name",
                        "author",
                        "details",
                        "system_prompt",
                        "user_prompt",
                        "prompt_format",
                        "prompt_schema_version",
                        "prompt_definition",
                        "artifact_type",
                    )
                },
                "keywords": payload["keywords"],
                "keywords_captured": True,
            }

        result = database.restore_prompt_history_entry(
            prompt_uuid,
            change_id=source["change_id"],
            version=1,
            expected_version=2,
            snapshot_validator=validate_snapshot,
        )

        restored = database.fetch_prompt_details(prompt_uuid)
        assert result == {
            "outcome": "restored",
            "snapshot_unavailable": False,
            "no_change": False,
            "source_version": 1,
            "current_version": 2,
            "new_version": 3,
            "retained_current_keywords": False,
        }
        assert restored["name"] == "Original"
        assert restored["version"] == 3
        assert restored["keywords"] == ["original"]
        assert database.get_prompt_history_count(prompt_uuid) == 3
    finally:
        database.close_connection()


def test_retained_restore_returns_snapshot_unavailable_without_writing():
    database = PromptsDatabase(":memory:", client_id="retained-pruned")
    try:
        _prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Current", author=None, details="Current"
        )
        selected = database.get_prompt_history_entries(prompt_uuid, page_size=1)[
            "items"
        ][0]
        database.get_connection().execute(
            "DELETE FROM sync_log WHERE change_id = ?", (selected["change_id"],)
        )
        database.get_connection().commit()
        before = _prompt_storage_state(database)

        result = database.restore_prompt_history_entry(
            prompt_uuid,
            change_id=selected["change_id"],
            version=1,
            expected_version=1,
            snapshot_validator=lambda _snapshot: pytest.fail("must not validate"),
        )

        assert result == {
            "outcome": "snapshot_unavailable",
            "snapshot_unavailable": True,
            "no_change": False,
            "source_version": 1,
            "current_version": None,
            "new_version": None,
            "retained_current_keywords": False,
        }
        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_retained_restore_stale_expected_version_keeps_conflict_error_path():
    database = PromptsDatabase(":memory:", client_id="retained-stale")
    try:
        _prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Current", author=None, details="Current"
        )
        source = database.get_prompt_history_entries(prompt_uuid, page_size=1)["items"][
            0
        ]
        before = _prompt_storage_state(database)

        with pytest.raises(
            ConflictError, match="Prompt changed after it was opened"
        ) as caught:
            database.restore_prompt_history_entry(
                prompt_uuid,
                change_id=source["change_id"],
                version=1,
                expected_version=99,
                snapshot_validator=lambda _snapshot: pytest.fail("must not validate"),
            )

        assert caught.value.code == "expected_version"

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_retained_restore_refuses_deleted_current_prompt_without_writing():
    database = PromptsDatabase(":memory:", client_id="retained-deleted")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Deleted", author=None, details="Original"
        )
        source = database.get_prompt_history_entries(prompt_uuid, page_size=1)["items"][
            0
        ]
        assert database.soft_delete_prompt(prompt_id) is True
        before = _prompt_storage_state(database)

        result = database.restore_prompt_history_entry(
            prompt_uuid,
            change_id=source["change_id"],
            version=1,
            expected_version=2,
            snapshot_validator=lambda _snapshot: pytest.fail("must not validate"),
        )

        assert result["outcome"] == "current_unavailable"
        assert result["snapshot_unavailable"] is False
        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_retained_restore_no_change_does_not_append_sync_history():
    database = PromptsDatabase(":memory:", client_id="retained-no-change")
    try:
        _prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Unchanged",
            author="Author",
            details="Details",
            system_prompt="System",
            user_prompt="User",
            keywords=["same"],
        )
        source = database.get_prompt_history_entries(prompt_uuid, page_size=1)["items"][
            0
        ]
        before = _prompt_storage_state(database)

        result = database.restore_prompt_history_entry(
            prompt_uuid,
            change_id=source["change_id"],
            version=1,
            expected_version=1,
            snapshot_validator=lambda snapshot: {
                "update_data": {
                    field: snapshot["payload"][field]
                    for field in (
                        "name",
                        "author",
                        "details",
                        "system_prompt",
                        "user_prompt",
                        "prompt_format",
                        "prompt_schema_version",
                        "prompt_definition",
                        "artifact_type",
                    )
                },
                "keywords": snapshot["payload"]["keywords"],
                "keywords_captured": True,
            },
        )

        assert result["outcome"] == "no_change"
        assert result["no_change"] is True
        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_retained_restore_duplicate_name_rolls_back_prompt_and_history():
    database = PromptsDatabase(":memory:", client_id="retained-duplicate")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Restore me", author=None, details="Original"
        )
        database.update_prompt_by_id(prompt_id, {"name": "Current"}, expected_version=1)
        database.add_prompt(name="Restore me", author=None, details="Other")
        source = next(
            item
            for item in database.get_prompt_history_entries(prompt_uuid, page_size=10)[
                "items"
            ]
            if item["version"] == 1
        )
        before = _prompt_storage_state(database)

        with pytest.raises(ConflictError, match="already exists") as caught:
            database.restore_prompt_history_entry(
                prompt_uuid,
                change_id=source["change_id"],
                version=1,
                expected_version=2,
                snapshot_validator=lambda snapshot: {
                    "update_data": {
                        field: snapshot["payload"][field]
                        for field in (
                            "name",
                            "author",
                            "details",
                            "system_prompt",
                            "user_prompt",
                            "prompt_format",
                            "prompt_schema_version",
                            "prompt_definition",
                            "artifact_type",
                        )
                    },
                    "keywords": snapshot["payload"]["keywords"],
                    "keywords_captured": True,
                },
            )

        assert caught.value.code == "name_conflict"
        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_retained_restore_keyword_failure_rolls_back_prompt_links_and_history():
    database = PromptsDatabase(":memory:", client_id="retained-keyword-failure")
    try:
        prompt_id, prompt_uuid, _message = database.add_prompt(
            name="Restore keywords", author=None, details="Original", keywords=["old"]
        )
        database.update_prompt_by_id(
            prompt_id,
            {"details": "Current", "keywords": ["current"]},
            expected_version=1,
        )
        source = next(
            item
            for item in database.get_prompt_history_entries(prompt_uuid, page_size=10)[
                "items"
            ]
            if item["version"] == 1
        )
        _install_link_insert_failure(database)
        before = _prompt_storage_state(database)

        with pytest.raises(DatabaseError, match="Keyword update failed"):
            database.restore_prompt_history_entry(
                prompt_uuid,
                change_id=source["change_id"],
                version=1,
                expected_version=2,
                snapshot_validator=lambda snapshot: {
                    "update_data": {
                        field: snapshot["payload"][field]
                        for field in (
                            "name",
                            "author",
                            "details",
                            "system_prompt",
                            "user_prompt",
                            "prompt_format",
                            "prompt_schema_version",
                            "prompt_definition",
                            "artifact_type",
                        )
                    },
                    "keywords": snapshot["payload"]["keywords"],
                    "keywords_captured": True,
                },
            )

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "bad_keywords",
    [["valid", 196], [None], "not-a-list"],
    ids=["non-string-member", "none-member", "non-list"],
)
def test_add_prompt_keyword_validation_raises_input_error_without_partial_state(
    bad_keywords,
):
    database = PromptsDatabase(":memory:", client_id="snapshot-create-validation")
    try:
        before = _prompt_storage_state(database)

        with pytest.raises(InputError, match="keyword"):
            database.add_prompt(
                name="Invalid create",
                author=None,
                details="Must roll back",
                keywords=bad_keywords,
            )

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


def test_update_prompt_keyword_validation_raises_input_error_without_partial_state():
    database = PromptsDatabase(":memory:", client_id="snapshot-update-validation")
    try:
        prompt_id, _prompt_uuid, _message = database.add_prompt(
            name="Validation target",
            author=None,
            details="Original details",
            keywords=["original"],
        )
        before = _prompt_storage_state(database)

        with pytest.raises(InputError, match="keyword"):
            database.update_prompt_by_id(
                prompt_id,
                {
                    "name": "Mutated name",
                    "details": "Mutated details",
                    "keywords": ["valid", object()],
                },
            )

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


@pytest.mark.parametrize("operation", ["create", "update"])
def test_prompt_and_keyword_state_rolls_back_on_link_persistence_failure(operation):
    database = PromptsDatabase(":memory:", client_id=f"snapshot-link-{operation}")
    try:
        prompt_id = None
        if operation == "update":
            prompt_id, _prompt_uuid, _message = database.add_prompt(
                name="Link failure target",
                author=None,
                details="Original details",
                keywords=["original"],
            )
        _install_link_insert_failure(database)
        before = _prompt_storage_state(database)

        with pytest.raises(DatabaseError, match="Keyword update failed"):
            if operation == "create":
                database.add_prompt(
                    name="Link failure create",
                    author=None,
                    details="Must roll back",
                    keywords=["new keyword"],
                )
            else:
                database.update_prompt_by_id(
                    prompt_id,
                    {
                        "name": "Mutated target",
                        "details": "Mutated details",
                        "keywords": ["new keyword"],
                    },
                )

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()


@pytest.mark.parametrize("operation", ["create", "overwrite", "update"])
def test_all_prompt_state_rolls_back_on_snapshot_finalize_failure(operation):
    database = PromptsDatabase(":memory:", client_id=f"snapshot-finalize-{operation}")
    try:
        prompt_id = None
        if operation != "create":
            prompt_id, _prompt_uuid, _message = database.add_prompt(
                name="Finalize failure target",
                author="Before",
                details="Original details",
                keywords=["original"],
            )
        _install_snapshot_finalize_failure(database)
        before = _prompt_storage_state(database)

        with pytest.raises(DatabaseError, match="finalize Prompt sync snapshot"):
            if operation == "create":
                database.add_prompt(
                    name="Finalize failure create",
                    author=None,
                    details="Must roll back",
                    keywords=["new keyword"],
                )
            elif operation == "overwrite":
                database.add_prompt(
                    name="Finalize failure target",
                    author="After",
                    details="Mutated details",
                    keywords=["new keyword"],
                    overwrite=True,
                )
            else:
                database.update_prompt_by_id(
                    prompt_id,
                    {
                        "name": "Mutated target",
                        "details": "Mutated details",
                        "keywords": ["new keyword"],
                    },
                )

        assert _prompt_storage_state(database) == before
    finally:
        database.close_connection()
