"""AgentRunsDB against a real on-disk SQLite file."""

import sqlite3
from contextlib import contextmanager

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


def test_create_and_get_run(db):
    run_id = db.create_run(
        conversation_id="conv1", agent_kind="primary", budget={"max_steps": 8}
    )
    run = db.get_run(run_id)
    assert run["conversation_id"] == "conv1"
    assert run["agent_kind"] == "primary"
    assert run["status"] == "running"
    assert run["steps"] == [] and run["parent_run_id"] is None
    assert run["budget"] == {"max_steps": 8}
    assert run["created_at"] and run["updated_at"]


def test_get_missing_run_returns_none(db):
    assert db.get_run("nope") is None


def test_append_steps_accumulates_and_parses(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model", "summary": "hi"}])
    db.append_steps(
        run_id, [{"index": 1, "kind": "tool_call", "tool_name": "calculator"}]
    )
    steps = db.get_run(run_id)["steps"]
    assert [s["index"] for s in steps] == [0, 1]
    assert steps[1]["tool_name"] == "calculator"


def test_set_status_and_result(db):
    run_id = db.create_run(
        conversation_id="c", agent_kind="subagent", task="do x", parent_run_id="p1"
    )
    db.set_status(run_id, "done", result="the answer")
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "the answer"
    assert run["task"] == "do x" and run["parent_run_id"] == "p1"


def test_count_subagents_counts_only_subagent_kind(db):
    db.create_run(conversation_id="c", agent_kind="primary")
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    for i in range(3):
        db.create_run(
            conversation_id="c",
            agent_kind="subagent",
            task=f"t{i}",
            parent_run_id=parent,
        )
    db.create_run(
        conversation_id="other", agent_kind="subagent", task="x", parent_run_id="zzz"
    )
    assert db.count_subagent_runs("c") == 3


# --- Finding A: batched per-conversation sub-agent counts (single query,
# not one connection/query per conversation row per poll tick). ---


def test_count_subagents_by_conversation_batches_single_query(db, monkeypatch):
    parent_a = db.create_run(conversation_id="conv-a", agent_kind="primary")
    for i in range(2):
        db.create_run(
            conversation_id="conv-a",
            agent_kind="subagent",
            task=f"a{i}",
            parent_run_id=parent_a,
        )
    parent_b = db.create_run(conversation_id="conv-b", agent_kind="primary")
    db.create_run(
        conversation_id="conv-b",
        agent_kind="subagent",
        task="b0",
        parent_run_id=parent_b,
    )
    # conv-c has only a primary run -- zero sub-agents, must be absent.
    db.create_run(conversation_id="conv-c", agent_kind="primary")

    executed = []
    original_get_connection = type(db)._get_connection

    def spy_get_connection(self):
        conn = original_get_connection(self)
        conn.set_trace_callback(executed.append)
        return conn

    monkeypatch.setattr(type(db), "_get_connection", spy_get_connection)
    counts = db.count_subagents_by_conversation(["conv-a", "conv-b", "conv-c"])

    assert counts == {"conv-a": 2, "conv-b": 1}
    assert "conv-c" not in counts  # zero-absent, not zero-valued
    select_calls = [c for c in executed if c.strip().upper().startswith("SELECT")]
    assert len(select_calls) == 1  # one batched query, not one per conversation


def test_count_subagents_by_conversation_empty_input_returns_empty_dict(db):
    assert db.count_subagents_by_conversation([]) == {}


def test_count_subagents_by_conversation_dedupes_ids_and_ignores_blanks(db):
    parent = db.create_run(conversation_id="conv-a", agent_kind="primary")
    db.create_run(
        conversation_id="conv-a", agent_kind="subagent", task="x", parent_run_id=parent
    )
    counts = db.count_subagents_by_conversation(["conv-a", "conv-a", "", None])
    assert counts == {"conv-a": 1}


def test_supersede_run_tree_marks_run_and_children(db):
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    child = db.create_run(
        conversation_id="c", agent_kind="subagent", task="t", parent_run_id=parent
    )
    other = db.create_run(conversation_id="c", agent_kind="primary")
    changed = db.supersede_run_tree(parent)
    assert changed == 2
    assert db.get_run(parent)["status"] == "superseded"
    assert db.get_run(child)["status"] == "superseded"
    assert db.get_run(other)["status"] == "running"


def test_list_runs_filters_superseded_when_asked(db):
    a = db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="primary")
    db.supersede_run_tree(a)
    assert len(db.list_runs("c")) == 2
    live = db.list_runs("c", include_superseded=False)
    assert len(live) == 1 and live[0]["status"] == "running"


def test_sql_is_parameterized_against_quotes(db):
    run_id = db.create_run(
        conversation_id="c''; DROP TABLE agent_runs;--",
        agent_kind="primary",
        task="a 'quoted' task",
    )
    assert db.get_run(run_id)["task"] == "a 'quoted' task"


# --- G2: writes must take the write lock up front (BEGIN IMMEDIATE), not
# lazily (plain BEGIN / deferred), to avoid the two-reader-upgrade-deadlock
# hazard when multiple workers write concurrently. ---


def test_transaction_begins_immediate_not_deferred(db, monkeypatch):
    # sqlite3.Connection is a C type — can't monkeypatch .execute on it —
    # so use the module-supported trace callback to observe every SQL
    # statement actually sent to SQLite on the transaction() connection.
    calls = []
    original_get_connection = type(db)._get_connection

    def spy_get_connection(self):
        conn = original_get_connection(self)
        conn.set_trace_callback(calls.append)
        return conn

    monkeypatch.setattr(type(db), "_get_connection", spy_get_connection)
    with db.transaction() as conn:
        conn.execute("SELECT 1")
    begin_calls = [c for c in calls if c.strip().upper().startswith("BEGIN")]
    assert begin_calls == ["BEGIN IMMEDIATE"]


# --- Q3: list_runs pagination. ---


def test_list_runs_limit_returns_newest_only(db):
    for _ in range(3):
        db.create_run(conversation_id="c", agent_kind="primary")
    full = db.list_runs("c")
    limited = db.list_runs("c", limit=1)
    assert len(limited) == 1
    assert limited[0]["id"] == full[0]["id"]


def test_list_runs_default_limit_preserves_behavior(db):
    for _ in range(3):
        db.create_run(conversation_id="c", agent_kind="primary")
    assert len(db.list_runs("c")) == 3
    assert len(db.list_runs("c", limit=None)) == 3


# --- Phase C Task 1: assistant_message_id column (v1->v2) + setter, so a
# run can record the persisted id of the assistant reply it produced. ---


def test_create_run_with_assistant_message_id_round_trips(db):
    run_id = db.create_run(
        conversation_id="c", agent_kind="primary", assistant_message_id="m-9"
    )
    assert db.get_run(run_id)["assistant_message_id"] == "m-9"


def test_create_run_without_assistant_message_id_defaults_to_none(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    assert db.get_run(run_id)["assistant_message_id"] is None


def test_set_run_assistant_message_id_updates_get_and_list(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.set_run_assistant_message_id(run_id, "p-42")
    assert db.get_run(run_id)["assistant_message_id"] == "p-42"
    listed = db.list_runs("c")
    assert listed[0]["assistant_message_id"] == "p-42"


def test_set_run_assistant_message_id_can_clear_with_none(db):
    run_id = db.create_run(
        conversation_id="c", agent_kind="primary", assistant_message_id="m-1"
    )
    db.set_run_assistant_message_id(run_id, None)
    assert db.get_run(run_id)["assistant_message_id"] is None


# There's no migration framework here -- _initialize_schema only runs
# CREATE TABLE IF NOT EXISTS, so a DB file created before this column
# existed keeps its old 11-column table until a guarded ALTER TABLE runs.
# These tests replicate that pre-v2 shape by hand and prove the guarded
# ALTER migrates it (and is idempotent across re-open).

_LEGACY_V1_AGENT_RUNS_DDL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (1);

    CREATE TABLE IF NOT EXISTS agent_runs (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        parent_run_id TEXT,
        agent_kind TEXT NOT NULL,
        task TEXT,
        status TEXT NOT NULL,
        steps TEXT NOT NULL DEFAULT '[]',
        result TEXT,
        budget TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation
        ON agent_runs(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_parent
        ON agent_runs(parent_run_id);
"""


def test_opening_legacy_v1_db_migrates_column_and_create_run_works(tmp_path):
    legacy_path = tmp_path / "legacy_agent_runs.db"
    conn = sqlite3.connect(str(legacy_path))
    try:
        conn.executescript(_LEGACY_V1_AGENT_RUNS_DDL)
        conn.commit()
    finally:
        conn.close()

    # Sanity: the raw file really is the old 11-column shape before it's
    # opened through AgentRunsDB.
    raw = sqlite3.connect(str(legacy_path))
    try:
        cols = {row[1] for row in raw.execute("PRAGMA table_info(agent_runs)")}
    finally:
        raw.close()
    assert "assistant_message_id" not in cols

    migrated = AgentRunsDB(legacy_path, client_id="test")
    run_id = migrated.create_run(
        conversation_id="c", agent_kind="primary", assistant_message_id="y"
    )
    assert migrated.get_run(run_id)["assistant_message_id"] == "y"


def test_reopening_same_file_twice_is_idempotent(tmp_path):
    path = tmp_path / "agent_runs.db"
    first = AgentRunsDB(path, client_id="test")
    first.create_run(
        conversation_id="c", agent_kind="primary", assistant_message_id="a"
    )

    # Re-opening must not raise (guarded ALTER is a no-op once the column
    # already exists) and the second instance must still work correctly.
    second = AgentRunsDB(path, client_id="test")
    run_id = second.create_run(
        conversation_id="c", agent_kind="primary", assistant_message_id="b"
    )
    assert second.get_run(run_id)["assistant_message_id"] == "b"
    assert len(second.list_runs("c")) == 2


# --- TASK-327 Task 2 (AC#2): a hard crash between create_run's 'running'
# insert and the service's finalizing set_status leaves a row stuck
# 'running' forever. On DB open (file-backed only, once per file per
# process) such rows are swept to 'error'. ---


def test_orphaned_running_runs_reconciled_on_open(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    r_run1 = db1.create_run(conversation_id="c1", agent_kind="primary")
    r_run2 = db1.create_run(conversation_id="c2", agent_kind="primary")
    r_done = db1.create_run(conversation_id="c3", agent_kind="primary")
    db1.set_status(r_done, "done", result="the answer")

    # Simulate a fresh process opening the same file: remove only this
    # test's own path from the shared once-guard, not every path any other
    # AgentRunsDB constructed earlier in this pytest process has
    # registered (AgentRunsDB._swept_paths.clear() would be an
    # order-dependent test hazard).
    AgentRunsDB._swept_paths.discard(db1.db_path_str)
    db2 = AgentRunsDB(db_path)

    run1 = db2.get_run(r_run1)
    run2 = db2.get_run(r_run2)
    done = db2.get_run(r_done)
    assert run1["status"] == "error"
    assert run1["result"] == "Interrupted by app restart"
    assert run2["status"] == "error"
    assert done["status"] == "done"  # terminal row untouched
    assert done["result"] == "the answer"


def test_reconcile_preserves_existing_result(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    rid = db1.create_run(conversation_id="c", agent_kind="primary")
    db1.set_status(rid, "running", result="partial output")  # running WITH a result
    # Simulate a fresh process opening the same file (scoped to this
    # test's own path -- see the discard() comment above).
    AgentRunsDB._swept_paths.discard(db1.db_path_str)
    db2 = AgentRunsDB(db_path)
    row = db2.get_run(rid)
    assert row["status"] == "error"
    assert row["result"] == "partial output"  # COALESCE keeps it


def test_reconcile_idempotent_same_process(tmp_path):
    db_path = tmp_path / "agent_runs.db"
    db1 = AgentRunsDB(db_path)
    db1.create_run(conversation_id="c", agent_kind="primary")
    # second open in the SAME process (guard already set by db1) is a no-op
    assert AgentRunsDB(db_path).reconcile_orphaned_runs() == 0


def test_reconcile_skips_memory_db():
    # :memory: must not error and must not register a swept path
    AgentRunsDB._swept_paths.discard(":memory:")
    AgentRunsDB(":memory:")  # must not raise
    assert ":memory:" not in AgentRunsDB._swept_paths


def test_reconcile_failed_sweep_leaves_path_unregistered_for_retry(tmp_path, monkeypatch):
    """A transient failure (e.g. a locked DB) during the sweep must NOT
    register the path -- otherwise no later AgentRunsDB(path) construction
    in this process ever retries, silently defeating AC#2's crash-recovery
    guarantee for the rest of the process (review Finding 1)."""
    db_path = tmp_path / "agent_runs.db"

    # Seed a file with an orphaned 'running' row, as a prior process would
    # have left the table before crashing again.
    setup = AgentRunsDB(db_path)
    rid = setup.create_run(conversation_id="c", agent_kind="primary")
    path_str = setup.db_path_str
    # Simulate a fresh process: this path hasn't been swept yet.
    AgentRunsDB._swept_paths.discard(path_str)

    real_transaction = AgentRunsDB.transaction
    call_count = {"n": 0}

    @contextmanager
    def flaky_transaction(self):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        with real_transaction(self) as conn:
            yield conn

    monkeypatch.setattr(AgentRunsDB, "transaction", flaky_transaction)

    # Construction must not raise (reconcile is best-effort in __init__),
    # but the failed sweep must leave the path unregistered and the row
    # untouched.
    db2 = AgentRunsDB(db_path)
    assert path_str not in AgentRunsDB._swept_paths
    assert db2.get_run(rid)["status"] == "running"

    # A later reconcile (e.g. the next AgentRunsDB(path) construction in
    # this process) retries and actually sweeps the orphaned row.
    assert db2.reconcile_orphaned_runs() == 1
    assert db2.get_run(rid)["status"] == "error"
    assert path_str in AgentRunsDB._swept_paths


def test_file_db_uses_wal_and_busy_timeout(tmp_path):
    db = AgentRunsDB(tmp_path / "agent_runs.db")
    with db.connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_memory_db_skips_wal():
    # :memory: cannot use WAL; must not raise and must stay 'memory'
    db = AgentRunsDB(":memory:")
    with db.connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_latest_primary_run_targets_newest_primary_only(tmp_path):
    """Qodo (PR #872): the Stop-path lookup must be a single bounded query,
    and interleaved newer SUBAGENT runs must not hide the newest primary."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")

    old_primary = db.create_run(conversation_id="c1", agent_kind="primary", task="a")
    newer_primary = db.create_run(conversation_id="c1", agent_kind="primary", task="b")
    db.create_run(
        conversation_id="c1",
        agent_kind="subagent",
        task="sub",
        parent_run_id=newer_primary,
    )

    record = db.latest_primary_run("c1")
    assert record is not None
    assert record["id"] == newer_primary
    assert record["agent_kind"] == "primary"

    # Superseded newest primary falls back to the next non-superseded one.
    db.set_status(newer_primary, "superseded")
    record = db.latest_primary_run("c1")
    assert record is not None and record["id"] == old_primary

    assert db.latest_primary_run("no-such-conversation") is None
