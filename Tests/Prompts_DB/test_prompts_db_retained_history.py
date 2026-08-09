"""Retained Prompt history database contract tests for TASK-196."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

import pytest

import tldw_chatbook.DB.Prompts_DB as prompts_db_module
from tldw_chatbook.DB.Prompts_DB import InputError, PromptsDatabase


HISTORY_INDEX = "idx_sync_log_prompt_history"
TARGET_UUID = "00000000-0000-4000-8000-000000000196"
OTHER_UUID = "00000000-0000-4000-8000-000000000197"


def _seed_v3_database(database_path: Any) -> list[tuple[Any, ...]]:
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
                "Prompts",
                TARGET_UUID,
                "create",
                "2026-08-08T00:00:00.000Z",
                "v3-client",
                1,
                '{"name":"Retained v1","version":1}',
            ),
            (
                "Prompts",
                TARGET_UUID,
                "update",
                "2026-08-08T00:01:00.000Z",
                "v3-client",
                2,
                '{"name":"Retained v2","version":2}',
            ),
            (
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
                entity, entity_uuid, operation, timestamp, client_id, version, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
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


def test_fresh_v4_schema_has_partial_covering_prompt_history_index():
    database = PromptsDatabase(":memory:", client_id="fresh-v4")
    try:
        conn = database.get_connection()

        assert database._get_db_version(conn) == 4
        assert [
            row[2] for row in conn.execute(f"PRAGMA index_info({HISTORY_INDEX})")
        ] == ["entity", "entity_uuid", "change_id", "operation"]

        normalized_sql = " ".join(_index_sql(conn).lower().split())
        assert "where entity = 'prompts'" in normalized_sql
        assert "operation in ('create', 'update')" in normalized_sql
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
                SELECT entity, entity_uuid, operation, CAST(timestamp AS TEXT), client_id,
                       version, payload
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
    [None, True, False, 0, -1, 1.0, math.inf, "2"],
)
def test_prompt_history_rejects_non_positive_or_non_integer_page_size(page_size):
    database = PromptsDatabase(":memory:", client_id="history-validation")
    try:
        with pytest.raises(InputError, match="page_size"):
            database.get_prompt_history_entries(TARGET_UUID, page_size=page_size)
    finally:
        database.close_connection()


@pytest.mark.parametrize(
    "before_change_id",
    [True, False, 0, -1, 1.0, math.inf, "2"],
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

        traced: list[str] = []
        conn = database.get_connection()
        conn.set_trace_callback(traced.append)
        database.get_prompt_history_entries(TARGET_UUID, page_size=10)
        conn.set_trace_callback(None)

        production_queries = [
            statement
            for statement in traced
            if statement.lstrip().upper().startswith("SELECT")
            and "FROM sync_log" in statement
        ]
        assert len(production_queries) == 2

        plans = {
            "count": [
                row[3]
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN " + production_queries[0]
                )
            ],
            "rows": [
                row[3]
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN " + production_queries[1]
                )
            ],
        }

        assert any(
            f"USING COVERING INDEX {HISTORY_INDEX}" in detail
            for detail in plans["count"]
        )
        assert any(
            f"USING INDEX {HISTORY_INDEX}" in detail for detail in plans["rows"]
        )
        assert all(
            "SCAN sync_log" not in detail and "USE TEMP B-TREE" not in detail
            for plan in plans.values()
            for detail in plan
        )
    finally:
        database.close_connection()
