# test_evals_db_v3_to_v4_migration.py
# Description: Pins the Evals_DB v3 -> v4 ALTER TABLE migration path.
#
"""TASK-708: PR 2 of the Evals rebuild took Evals_DB.SCHEMA_VERSION from 3 to
4, adding ``eval_runs.run_group_id`` plus its index. Every other test in this
suite builds a fresh database, which always goes through ``_create_schema``
(the v4 shape from scratch) -- none of them exercise the ``ALTER TABLE`` path
real, pre-upgrade user databases actually take on their next launch.

Follows the pattern already used by sibling DBs in this repo for a legacy
schema fixture (e.g. ``Tests/DB/test_agent_runs_db.py``'s
``_LEGACY_V1_AGENT_RUNS_DDL`` / ``test_opening_legacy_v1_db_migrates_column_
and_create_run_works``): build the OLD shape by hand with raw sqlite3, then
open it through the real DB class and assert the migration landed.
"""

from __future__ import annotations

import sqlite3

from tldw_chatbook.DB.Evals_DB import SCHEMA_VERSION, EvalsDB
from tldw_chatbook.DB.sql_validation import validate_identifier

#: The exact v3 shape of Evals_DB._create_schema, minus `run_group_id` and
#: its index (added at v4) -- everything else copied verbatim so the
#: migration is exercised against a schema `_migrate_schema` actually
#: recognises (FKs, CHECK constraints, FTS5 tables/triggers included, since
#: those already existed by v3).
_V3_SCHEMA_DDL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE eval_tasks (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        task_type TEXT NOT NULL CHECK (task_type IN ('question_answer', 'logprob', 'generation', 'classification')),
        config_format TEXT NOT NULL CHECK (config_format IN ('eleuther', 'custom')),
        config_data TEXT NOT NULL,
        dataset_id TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT,
        FOREIGN KEY (dataset_id) REFERENCES eval_datasets (id)
    );

    CREATE TABLE eval_datasets (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        format TEXT NOT NULL CHECK (format IN ('huggingface', 'json', 'csv', 'custom')),
        source_path TEXT NOT NULL,
        metadata TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT
    );

    CREATE TABLE eval_models (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL,
        provider TEXT NOT NULL,
        model_id TEXT NOT NULL,
        config TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT,
        UNIQUE(name, provider, model_id)
    );

    -- v3 shape: no run_group_id column yet (added at v4).
    CREATE TABLE eval_runs (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL,
        task_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
        start_time TEXT,
        end_time TEXT,
        total_samples INTEGER,
        completed_samples INTEGER DEFAULT 0,
        config_overrides TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT,
        FOREIGN KEY (task_id) REFERENCES eval_tasks (id),
        FOREIGN KEY (model_id) REFERENCES eval_models (id)
    );

    CREATE TABLE eval_results (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_id TEXT NOT NULL,
        sample_id TEXT NOT NULL,
        input_data TEXT NOT NULL,
        expected_output TEXT,
        actual_output TEXT,
        logprobs TEXT,
        metrics TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES eval_runs (id),
        UNIQUE(run_id, sample_id)
    );

    CREATE TABLE eval_run_metrics (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_type TEXT NOT NULL CHECK (metric_type IN ('accuracy', 'f1', 'rouge', 'bleu', 'perplexity', 'custom')),
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES eval_runs (id),
        UNIQUE(run_id, metric_name)
    );

    CREATE INDEX idx_eval_tasks_type ON eval_tasks (task_type);
    CREATE INDEX idx_eval_tasks_deleted ON eval_tasks (deleted_at);
    CREATE INDEX idx_eval_runs_status ON eval_runs (status);
    CREATE INDEX idx_eval_runs_task ON eval_runs (task_id);
    CREATE INDEX idx_eval_runs_model ON eval_runs (model_id);
    CREATE INDEX idx_eval_results_run ON eval_results (run_id);
    CREATE INDEX idx_eval_run_metrics_run ON eval_run_metrics (run_id);

    CREATE VIRTUAL TABLE eval_tasks_fts USING fts5(
        id UNINDEXED, name, description,
        content='eval_tasks', content_rowid='rowid'
    );
    CREATE VIRTUAL TABLE eval_datasets_fts USING fts5(
        id UNINDEXED, name, description,
        content='eval_datasets', content_rowid='rowid'
    );

    CREATE TRIGGER eval_tasks_fts_insert AFTER INSERT ON eval_tasks BEGIN
        INSERT INTO eval_tasks_fts (rowid, id, name, description)
        VALUES (new.rowid, new.id, new.name, new.description);
    END;
    CREATE TRIGGER eval_tasks_fts_update AFTER UPDATE ON eval_tasks BEGIN
        INSERT INTO eval_tasks_fts (eval_tasks_fts, rowid, id, name, description)
        VALUES ('delete', old.rowid, old.id, old.name, old.description);
        INSERT INTO eval_tasks_fts (rowid, id, name, description)
        VALUES (new.rowid, new.id, new.name, new.description);
    END;
    CREATE TRIGGER eval_tasks_fts_delete AFTER DELETE ON eval_tasks BEGIN
        INSERT INTO eval_tasks_fts (eval_tasks_fts, rowid, id, name, description)
        VALUES ('delete', old.rowid, old.id, old.name, old.description);
    END;
    CREATE TRIGGER eval_datasets_fts_insert AFTER INSERT ON eval_datasets BEGIN
        INSERT INTO eval_datasets_fts (rowid, id, name, description)
        VALUES (new.rowid, new.id, new.name, new.description);
    END;
    CREATE TRIGGER eval_datasets_fts_update AFTER UPDATE ON eval_datasets BEGIN
        INSERT INTO eval_datasets_fts (eval_datasets_fts, rowid, id, name, description)
        VALUES ('delete', old.rowid, old.id, old.name, old.description);
        INSERT INTO eval_datasets_fts (rowid, id, name, description)
        VALUES (new.rowid, new.id, new.name, new.description);
    END;
    CREATE TRIGGER eval_datasets_fts_delete AFTER DELETE ON eval_datasets BEGIN
        INSERT INTO eval_datasets_fts (eval_datasets_fts, rowid, id, name, description)
        VALUES ('delete', old.rowid, old.id, old.name, old.description);
    END;

    CREATE TABLE ab_tests (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        test_id TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        description TEXT,
        task_id TEXT NOT NULL,
        model_a_id TEXT NOT NULL,
        model_b_id TEXT NOT NULL,
        config TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
        winner TEXT CHECK (winner IN ('model_a', 'model_b', 'tie', NULL)),
        result_data TEXT,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT,
        FOREIGN KEY (task_id) REFERENCES eval_tasks (id),
        FOREIGN KEY (model_a_id) REFERENCES eval_models (id),
        FOREIGN KEY (model_b_id) REFERENCES eval_models (id)
    );

    CREATE TABLE ab_test_runs (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        ab_test_id TEXT NOT NULL,
        run_a_id TEXT NOT NULL,
        run_b_id TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        FOREIGN KEY (ab_test_id) REFERENCES ab_tests (id),
        FOREIGN KEY (run_a_id) REFERENCES eval_runs (id),
        FOREIGN KEY (run_b_id) REFERENCES eval_runs (id)
    );

    CREATE INDEX idx_ab_tests_status ON ab_tests (status);
    CREATE INDEX idx_ab_tests_task ON ab_tests (task_id);
    CREATE INDEX idx_ab_tests_models ON ab_tests (model_a_id, model_b_id);
    CREATE INDEX idx_ab_test_runs_test ON ab_test_runs (ab_test_id);

    PRAGMA user_version = 3;
"""


def _build_v3_database(path: str) -> dict[str, str]:
    """Write a real v3 Evals_DB file by hand and seed it with rows that must
    survive the upgrade. Returns the ids of the rows it created."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V3_SCHEMA_DDL)
        conn.execute(
            "INSERT INTO eval_tasks (id, name, task_type, config_format, config_data, client_id) "
            "VALUES ('task-1', 'pre-upgrade task', 'logprob', 'custom', '{}', 'test')"
        )
        conn.execute(
            "INSERT INTO eval_models (id, name, provider, model_id, config, client_id) "
            "VALUES ('model-1', 'pre-upgrade model', 'llama_cpp', 'm', '{}', 'test')"
        )
        conn.execute(
            "INSERT INTO eval_runs "
            "(id, name, task_id, model_id, status, total_samples, client_id) "
            "VALUES ('run-1', 'pre-upgrade run', 'task-1', 'model-1', 'completed', 2, 'test')"
        )
        conn.execute(
            "INSERT INTO eval_results (id, run_id, sample_id, input_data, client_id) "
            "VALUES ('result-1', 'run-1', 's1', '{}', 'test')"
        )
        conn.commit()
    finally:
        conn.close()
    return {"task_id": "task-1", "model_id": "model-1", "run_id": "run-1"}


#: The tables this module's hand-built _V3_SCHEMA_DDL defines (see above).
#: SQLite's `?` placeholders can only bind values, never identifiers, so
#: `table` below is interpolated into the SQL string -- it must be checked
#: against this explicit allow-list first, same as production code would be
#: required to under Tests/DB/test_sql_validation.py's rule. This applies
#: even in a test: `table` is a plain function argument, not a hardcoded
#: literal, so nothing stops a future caller from passing something else.
_ALLOWED_RAW_COLUMN_TABLES = {
    "eval_tasks", "eval_datasets", "eval_models", "eval_runs",
    "eval_results", "eval_run_metrics", "ab_tests", "ab_test_runs",
}


def _raw_columns(path: str, table: str) -> set[str]:
    if table not in _ALLOWED_RAW_COLUMN_TABLES or not validate_identifier(
        table, "table name"
    ):
        raise ValueError(f"Unexpected table name for _raw_columns: {table!r}")
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_v3_database_really_lacks_run_group_id_before_it_is_opened(tmp_path):
    """Sanity check on the fixture itself: the raw v3 file must NOT already
    carry the v4 column, or the migration below would prove nothing."""
    path = str(tmp_path / "v3.db")
    _build_v3_database(path)

    assert "run_group_id" not in _raw_columns(path, "eval_runs")
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        conn.close()


def test_opening_a_v3_database_migrates_to_v4_and_adds_run_group_id(tmp_path):
    """The ALTER TABLE path: EvalsDB opening a real, hand-built v3 file must
    pass through the v4 step, add `eval_runs.run_group_id`, and create its
    index -- the exact upgrade every existing user's database takes. The
    final PRAGMA user_version is compared against the live SCHEMA_VERSION
    rather than a literal 4: opening a v3 database runs every migration up
    to the current version in one pass (task-1691 added a v5 step after
    this test was written), and this test's own concern is the v3->v4
    ALTER specifically, not freezing the module's overall version."""
    path = str(tmp_path / "v3.db")
    ids = _build_v3_database(path)

    db = EvalsDB(db_path=path, client_id="test")
    conn = db.get_connection()

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    columns = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    assert "run_group_id" in columns

    indexes = {row[1] for row in conn.execute("PRAGMA index_list(eval_runs)")}
    assert "idx_eval_runs_group" in indexes
    index_columns = [
        row[2] for row in conn.execute("PRAGMA index_info(idx_eval_runs_group)")
    ]
    assert index_columns == ["run_group_id"]

    # Pre-existing rows survive the migration intact.
    task = conn.execute(
        "SELECT name, task_type FROM eval_tasks WHERE id = ?", (ids["task_id"],)
    ).fetchone()
    assert tuple(task) == ("pre-upgrade task", "logprob")

    run = conn.execute(
        "SELECT name, status, total_samples, run_group_id FROM eval_runs WHERE id = ?",
        (ids["run_id"],),
    ).fetchone()
    assert tuple(run) == ("pre-upgrade run", "completed", 2, None)

    result = conn.execute(
        "SELECT sample_id FROM eval_results WHERE run_id = ?", (ids["run_id"],)
    ).fetchone()
    assert tuple(result) == ("s1",)

    # The migrated database is still fully functional: a new run can set
    # run_group_id, and it round-trips through the ordinary API.
    new_run_id = db.create_run(
        name="post-upgrade run", task_id=ids["task_id"], model_id=ids["model_id"]
    )
    db.update_run(new_run_id, {"run_group_id": "group-xyz"})
    grouped = db.list_runs(run_group_id="group-xyz")
    assert [r["id"] for r in grouped] == [new_run_id]


def test_reopening_a_migrated_v3_database_is_idempotent(tmp_path):
    """Re-opening the same file a second time (the normal case: the app
    restarts against a database it already migrated) must not raise and
    must not re-run the v3->v4 ALTER (which would fail on an existing
    column) or duplicate the index."""
    path = str(tmp_path / "v3.db")
    ids = _build_v3_database(path)

    first = EvalsDB(db_path=path, client_id="test")
    first_conn = first.get_connection()
    assert first_conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    # A second, independent EvalsDB instance opening the same file must not
    # raise (the guarded `if "run_group_id" not in existing` check in
    # _migrate_schema is what makes the ALTER safe to run twice; opening
    # again also exercises the `current_version == SCHEMA_VERSION` branch of
    # `_init_schema`, which does neither create nor migrate).
    second = EvalsDB(db_path=path, client_id="test")
    second_conn = second.get_connection()

    assert second_conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    columns = {row[1] for row in second_conn.execute("PRAGMA table_info(eval_runs)")}
    assert "run_group_id" in columns
    indexes = [
        row[1] for row in second_conn.execute("PRAGMA index_list(eval_runs)")
        if row[1] == "idx_eval_runs_group"
    ]
    assert len(indexes) == 1, "the index must not be duplicated across reopens"

    # And the pre-existing row is still intact after both opens.
    run = second_conn.execute(
        "SELECT name, status FROM eval_runs WHERE id = ?", (ids["run_id"],)
    ).fetchone()
    assert tuple(run) == ("pre-upgrade run", "completed")
