# test_evals_db_v5_to_v6_migration.py
# Description: Pins the Evals_DB v5 -> v6 DROP COLUMN migration path.
#
"""TASK-19566 F8: the five ``version INTEGER NOT NULL DEFAULT 1`` columns
(eval_tasks, eval_datasets, eval_models, eval_runs, ab_tests) were inert
optimistic-locking residue -- ``expected_version`` appeared in zero callers
and no UPDATE ever carried ``AND version = ?`` -- and were REMOVED at
SCHEMA_VERSION 6 rather than made real. This suite pins both halves of that
removal: the migration every existing v5 database takes on its next launch
(columns dropped, everything else byte-identical), and the fresh-schema
half (a newly created database has no ``version`` column to grow back).

Follows the sibling v3->v4 suite's pattern
(``Tests/Evals/test_evals_db_v3_to_v4_migration.py``): build the OLD shape
by hand with raw sqlite3, then open it through the real DB class and
assert the migration landed. The v5 shape below is the pre-v6
``_create_schema`` verbatim (the probe-review tables included, since a
real v5 database has them).
"""

from __future__ import annotations

import sqlite3

from tldw_chatbook.DB.Evals_DB import SCHEMA_VERSION, EvalsDB
from tldw_chatbook.DB.sql_validation import validate_identifier

#: The exact v5 shape of Evals_DB._create_schema, with the five inert
#: ``version`` columns still present -- everything else copied verbatim so
#: the migration is exercised against a schema ``_migrate_schema`` actually
#: recognises (FKs, CHECK constraints, FTS5 tables/triggers, the probe
#: review tables, and every index included).
_V5_SCHEMA_DDL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE eval_tasks (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        task_type TEXT NOT NULL CHECK (task_type IN ('question_answer', 'logprob', 'generation', 'classification')),
        config_format TEXT NOT NULL CHECK (config_format IN ('eleuther', 'custom')),
        config_data TEXT NOT NULL, -- JSON configuration
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
        metadata TEXT, -- JSON metadata
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
        config TEXT, -- JSON configuration for model parameters
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted_at TEXT,
        UNIQUE(name, provider, model_id)
    );

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
        config_overrides TEXT, -- JSON overrides for task config
        run_group_id TEXT,
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
        input_data TEXT NOT NULL, -- JSON input data
        expected_output TEXT,
        actual_output TEXT,
        logprobs TEXT, -- JSON log probabilities if available
        metrics TEXT, -- JSON metrics for this sample
        metadata TEXT, -- JSON additional metadata
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

    CREATE TABLE eval_probe_turn_annotations (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_group_id TEXT NOT NULL,
        card_id INTEGER NOT NULL,
        probe_index INTEGER NOT NULL,
        sample_index INTEGER NOT NULL,
        target_id TEXT NOT NULL,
        turn_index INTEGER NOT NULL,
        tags TEXT NOT NULL,          -- JSON list of tag slugs
        note TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id, turn_index)
    );

    CREATE TABLE eval_probe_review_state (
        id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
        run_group_id TEXT NOT NULL,
        card_id INTEGER NOT NULL,
        probe_index INTEGER NOT NULL,
        sample_index INTEGER NOT NULL,
        target_id TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        reviewed_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
        client_id TEXT NOT NULL,
        UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id)
    );

    CREATE INDEX idx_eval_tasks_type ON eval_tasks (task_type);
    CREATE INDEX idx_eval_tasks_deleted ON eval_tasks (deleted_at);
    CREATE INDEX idx_eval_runs_status ON eval_runs (status);
    CREATE INDEX idx_eval_runs_task ON eval_runs (task_id);
    CREATE INDEX idx_eval_runs_model ON eval_runs (model_id);
    CREATE INDEX idx_eval_runs_group ON eval_runs (run_group_id);
    CREATE INDEX idx_eval_results_run ON eval_results (run_id);
    CREATE INDEX idx_eval_run_metrics_run ON eval_run_metrics (run_id);
    CREATE INDEX idx_probe_annotations_group ON eval_probe_turn_annotations (run_group_id);
    CREATE INDEX idx_probe_review_group ON eval_probe_review_state (run_group_id);

    CREATE VIRTUAL TABLE eval_tasks_fts USING fts5(
        id UNINDEXED,
        name,
        description,
        content='eval_tasks',
        content_rowid='rowid'
    );

    CREATE VIRTUAL TABLE eval_datasets_fts USING fts5(
        id UNINDEXED,
        name,
        description,
        content='eval_datasets',
        content_rowid='rowid'
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
        config TEXT NOT NULL, -- JSON configuration
        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
        winner TEXT CHECK (winner IN ('model_a', 'model_b', 'tie', NULL)),
        result_data TEXT, -- JSON result data
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

    PRAGMA user_version = 5;
"""

#: Every table this module's hand-built _V5_SCHEMA_DDL defines; used by
#: ``_raw_columns`` below, which interpolates the table name into a PRAGMA
#: (identifiers cannot be bind parameters) exactly like the sibling
#: v3->v4 suite's allow-listed helper.
_ALLOWED_RAW_COLUMN_TABLES = {
    "eval_tasks", "eval_datasets", "eval_models", "eval_runs",
    "eval_results", "eval_run_metrics", "eval_probe_turn_annotations",
    "eval_probe_review_state", "ab_tests", "ab_test_runs",
}

#: The five tables that carried the inert ``version`` column through v5
#: (TASK-19566 F8); mirrors ``Evals_DB._VERSION_COLUMN_TABLES``.
_VERSION_CARRYING_TABLES = (
    "eval_tasks", "eval_datasets", "eval_models", "eval_runs", "ab_tests",
)


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


def _build_v5_database(path: str) -> dict[str, str]:
    """Write a real v5 Evals_DB file by hand and seed one row in every
    version-carrying table (plus a result row), with the ``version`` values
    deliberately NOT left at their default, so the migration's data
    assertions below prove the DROP loses nothing but the dead column.

    Returns the ids of the rows it created.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V5_SCHEMA_DDL)
        conn.execute(
            "INSERT INTO eval_datasets (id, name, format, source_path, metadata, "
            "client_id, version) "
            "VALUES ('ds-1', 'legacy dataset', 'custom', 'inline:x', '{}', 'test', 3)"
        )
        conn.execute(
            "INSERT INTO eval_models (id, name, provider, model_id, config, client_id) "
            "VALUES ('model-1', 'legacy model', 'local', 'm', '{}', 'test')"
        )
        conn.execute(
            "INSERT INTO eval_tasks (id, name, description, task_type, config_format, "
            "config_data, dataset_id, client_id, version) "
            "VALUES ('task-1', 'legacy bench name', 'survives the drop', 'logprob', "
            "'custom', '{\"k\": \"v\"}', 'ds-1', 'test', 7)"
        )
        conn.execute(
            "INSERT INTO eval_runs (id, name, task_id, model_id, status, "
            "total_samples, client_id, version) "
            "VALUES ('run-1', 'legacy run', 'task-1', 'model-1', 'completed', 2, 'test', 5)"
        )
        conn.execute(
            "INSERT INTO eval_results (id, run_id, sample_id, input_data, client_id) "
            "VALUES ('result-1', 'run-1', 's1', '{}', 'test')"
        )
        conn.execute(
            "INSERT INTO ab_tests (id, test_id, name, description, task_id, "
            "model_a_id, model_b_id, config, client_id, version) "
            "VALUES ('ab-1', 'abt-1', 'legacy ab test', NULL, 'task-1', 'model-1', "
            "'model-1', '{}', 'test', 4)"
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "dataset_id": "ds-1",
        "model_id": "model-1",
        "task_id": "task-1",
        "run_id": "run-1",
        "ab_test_id": "ab-1",
    }


def test_v5_database_really_carries_the_version_columns_before_upgrade(tmp_path):
    """Sanity check on the fixture itself: the raw v5 file must carry the
    ``version`` column on all five tables (with non-default values) and
    user_version 5, or the migration below would prove nothing."""
    path = str(tmp_path / "v5.db")
    _build_v5_database(path)

    for table in _VERSION_CARRYING_TABLES:
        assert "version" in _raw_columns(path, table), table
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        assert conn.execute(
            "SELECT version FROM eval_tasks WHERE id = 'task-1'"
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_opening_a_v5_database_drops_the_inert_version_columns(request, tmp_path):
    """The exact upgrade every existing user's database takes on its next
    launch: EvalsDB opening a real, hand-built v5 file must drop the inert
    ``version`` column from all five carrying tables and keep every other
    column's data byte-identical. The seeded ``version`` values (7/3/5/4)
    are asserted GONE -- they were never read by anything, which is the
    finding -- while the real payload columns are asserted intact."""
    path = str(tmp_path / "v5.db")
    ids = _build_v5_database(path)

    db = EvalsDB(db_path=path, client_id="test")
    request.addfinalizer(db.close)
    conn = db.get_connection()

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert SCHEMA_VERSION == 6  # this suite pins the v5 -> v6 hop specifically

    for table in _VERSION_CARRYING_TABLES:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert "version" not in columns, table

    # Data survives the DROP intact, row by row.
    dataset = conn.execute(
        "SELECT name, format, source_path FROM eval_datasets WHERE id = ?",
        (ids["dataset_id"],),
    ).fetchone()
    assert tuple(dataset) == ("legacy dataset", "custom", "inline:x")

    model = conn.execute(
        "SELECT name, provider, model_id FROM eval_models WHERE id = ?",
        (ids["model_id"],),
    ).fetchone()
    assert tuple(model) == ("legacy model", "local", "m")

    task = conn.execute(
        "SELECT name, description, task_type, config_data, dataset_id "
        "FROM eval_tasks WHERE id = ?",
        (ids["task_id"],),
    ).fetchone()
    assert tuple(task) == (
        "legacy bench name",
        "survives the drop",
        "logprob",
        '{"k": "v"}',
        "ds-1",
    )

    run = conn.execute(
        "SELECT name, status, total_samples, run_group_id FROM eval_runs WHERE id = ?",
        (ids["run_id"],),
    ).fetchone()
    assert tuple(run) == ("legacy run", "completed", 2, None)

    ab = conn.execute(
        "SELECT test_id, name, status FROM ab_tests WHERE id = ?",
        (ids["ab_test_id"],),
    ).fetchone()
    assert tuple(ab) == ("abt-1", "legacy ab test", "pending")

    result = conn.execute(
        "SELECT sample_id FROM eval_results WHERE run_id = ?", (ids["run_id"],)
    ).fetchone()
    assert tuple(result) == ("s1",)

    # The migrated database is still fully functional: the FTS index and
    # triggers survived the DROP (DROP COLUMN neither fires them nor drops
    # them), and the row-level API no longer exposes a "version" key at all.
    hits = db.search_tasks("legacy bench")
    assert [h["id"] for h in hits] == [ids["task_id"]]
    assert "version" not in hits[0]

    assert db.update_task(ids["task_id"], description="post-upgrade edit") is True
    updated = db.get_task(ids["task_id"])
    assert updated["description"] == "post-upgrade edit"
    assert "version" not in updated


def test_reopening_a_migrated_v5_database_is_idempotent(request, tmp_path):
    """Re-opening the same file a second time (the normal case: the app
    restarts against a database it already migrated) must not raise and
    must not re-run the DROP (which would fail on a missing column)."""
    path = str(tmp_path / "v5.db")
    _build_v5_database(path)

    first = EvalsDB(db_path=path, client_id="test")
    request.addfinalizer(first.close)
    assert (
        first.get_connection().execute("PRAGMA user_version").fetchone()[0]
        == SCHEMA_VERSION
    )

    second = EvalsDB(db_path=path, client_id="test")
    request.addfinalizer(second.close)
    conn = second.get_connection()
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    for table in _VERSION_CARRYING_TABLES:
        assert "version" not in _raw_columns(path, table), table
    task = conn.execute(
        "SELECT name FROM eval_tasks WHERE id = 'task-1'"
    ).fetchone()
    assert tuple(task) == ("legacy bench name",)


def test_fresh_database_is_created_without_any_version_column(request):
    """The fresh-schema half of the removal: a newly created database must
    not grow the column back, on any of the five tables that used to carry
    it. Pins ``_create_schema`` against a regression that reintroduces
    the inert column (and its false appearance of concurrency protection)
    to new databases."""
    db = EvalsDB(db_path=":memory:", client_id="test")
    request.addfinalizer(db.close)
    conn = db.get_connection()

    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    for table in _VERSION_CARRYING_TABLES:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert "version" not in columns, table
