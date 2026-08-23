"""AgentRunsDB against a real on-disk SQLite file."""

import sqlite3
from contextlib import contextmanager

import pytest

from tldw_chatbook.Agents.agent_models import AgentDefinition
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


def test_count_subagents_by_conversation_batches_single_query(db):
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

    # Attach the trace callback directly to the thread's HELD connection
    # (task-3012) rather than monkeypatching _get_connection + db.close():
    # that older approach lost coverage of the held-connection path itself,
    # since it forced a fresh per-call connection through the spy instead of
    # observing the one every real call actually reuses.
    executed = []
    db._held_connection().set_trace_callback(executed.append)
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


def test_supersede_run_tree_marks_run_and_terminal_children(db):
    # A parent and child that have ALREADY finished (terminal) by the time
    # supersede runs are both superseded exactly as before this task --
    # that half must not regress.
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    db.set_status(parent, "done", result="parent finished before supersede")
    child = db.create_run(
        conversation_id="c", agent_kind="subagent", task="t", parent_run_id=parent
    )
    db.set_status(child, "done", result="child finished before supersede")
    other = db.create_run(conversation_id="c", agent_kind="primary")
    changed = db.supersede_run_tree(parent)
    assert changed == 2
    assert db.get_run(parent)["status"] == "superseded"
    assert db.get_run(child)["status"] == "superseded"
    assert db.get_run(other)["status"] == "running"


def test_supersede_run_tree_leaves_live_child_untouched(db):
    # PR3a-1 Task 2 lets a sub-agent outlive its turn. Task 4: superseding
    # the primary (retry/regenerate/variant) must not flip a still-running
    # child to a terminal status out from under its live worker thread --
    # that child is not a dead attempt, it is a real cross-turn survivor
    # still spending tokens. The primary here is put in a terminal status
    # first (the realistic case -- see the coupled primary-liveness test
    # below for the case where the primary itself is still live) so this
    # test isolates the CHILD guard specifically.
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    db.set_status(parent, "done", result="parent finished before supersede")
    live_child = db.create_run(
        conversation_id="c", agent_kind="subagent", task="t", parent_run_id=parent
    )
    changed = db.supersede_run_tree(parent)
    # The already-terminal parent is superseded; the live child is skipped
    # entirely, so exactly one row (the parent) is counted.
    assert changed == 1
    assert db.get_run(parent)["status"] == "superseded"
    assert db.get_run(live_child)["status"] == "running"
    # Lineage stays intact: the child is still parented to the (now
    # superseded) primary -- this only changes status semantics.
    assert db.get_run(live_child)["parent_run_id"] == parent


def test_supersede_run_tree_leaves_a_live_primary_untouched(db):
    # The hole in the first draft of this fix: `run_turn`'s guarantee that
    # ITS OWN primary is persisted terminally before it returns says
    # nothing about a DIFFERENT, earlier run_turn call whose coroutine
    # already returned to the UI (via Stop) while its OS thread -- an
    # `asyncio.to_thread` call, which survives Task cancellation -- keeps
    # running. `_previous_primary_run_id` resolves to the newest
    # non-superseded primary for the WHOLE conversation, not the run tied
    # to the message being retried, so retrying an older failed message
    # can supersede a DIFFERENT, still-live, stopped-but-not-dead primary.
    # That primary's row must be left untouched by supersede, and its own
    # later terminal set_status (e.g. "cancelled" once the thread notices)
    # must still land -- assert the result is READABLE afterwards, not
    # merely that the row isn't 'superseded'.
    live_primary = db.create_run(conversation_id="c", agent_kind="primary")
    changed = db.supersede_run_tree(live_primary)
    assert changed == 0
    assert db.get_run(live_primary)["status"] == "running"
    updated = db.set_status(
        live_primary, "cancelled", result="the primary's real terminal result"
    )
    assert updated is True
    run = db.get_run(live_primary)
    assert run["status"] == "cancelled"
    assert run["result"] == "the primary's real terminal result"


def test_supersede_run_tree_does_not_lose_a_live_childs_real_result(db):
    # The actual defect: superseding used to flip a live child straight to
    # the terminal status 'superseded'. Because 'superseded' is itself a
    # TERMINAL_RUN_STATUSES member, set_status's first-writer-wins guard
    # then silently dropped the child's real terminal write when it
    # finished for real -- the row lied dead while the child was alive, and
    # its genuine result was lost on arrival. Assert the result is
    # READABLE afterwards, not merely that the row isn't 'superseded'.
    parent = db.create_run(conversation_id="c", agent_kind="primary")
    live_child = db.create_run(
        conversation_id="c", agent_kind="subagent", task="t", parent_run_id=parent
    )
    db.supersede_run_tree(parent)  # primary retried while the child runs on
    updated = db.set_status(live_child, "done", result="the child's real answer")
    assert updated is True
    run = db.get_run(live_child)
    assert run["status"] == "done"
    assert run["result"] == "the child's real answer"


def test_list_runs_filters_superseded_when_asked(db):
    a = db.create_run(conversation_id="c", agent_kind="primary")
    db.set_status(a, "done", result="a finished before supersede")
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


def test_transaction_begins_immediate_not_deferred(db):
    # sqlite3.Connection is a C type — can't monkeypatch .execute on it —
    # so use the module-supported trace callback to observe every SQL
    # statement actually sent to SQLite on the transaction() connection.
    # Attach directly to the thread's HELD connection (task-3012) — see the
    # comment in test_count_subagents_by_conversation_batches_single_query.
    calls = []
    db._held_connection().set_trace_callback(calls.append)
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


# --- task-1273 review finding A: list_runs(agent_kind=...) + count_runs(),
# so a caller needing "the N newest PRIMARY runs, plus an exact count of how
# many more exist" never has to fetch every run (of every kind) for a
# conversation just to filter and count client-side. ---


def test_list_runs_agent_kind_filters_in_the_query(db):
    primary = db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="subagent", parent_run_id=primary)

    only_primary = db.list_runs("c", agent_kind="primary")
    assert [r["id"] for r in only_primary] == [primary]
    assert all(r["agent_kind"] == "primary" for r in only_primary)


def test_list_runs_agent_kind_none_preserves_prior_behavior(db):
    db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="subagent")
    assert len(db.list_runs("c")) == 2
    assert len(db.list_runs("c", agent_kind=None)) == 2


def test_list_runs_agent_kind_and_limit_compose(db):
    for _ in range(3):
        db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="subagent")
    limited = db.list_runs("c", agent_kind="primary", limit=2)
    assert len(limited) == 2
    assert all(r["agent_kind"] == "primary" for r in limited)


def test_count_runs_matches_agent_kind_and_conversation(db):
    db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="primary")
    db.create_run(conversation_id="c", agent_kind="subagent")
    db.create_run(conversation_id="other", agent_kind="primary")

    assert db.count_runs("c") == 3
    assert db.count_runs("c", agent_kind="primary") == 2
    assert db.count_runs("c", agent_kind="subagent") == 1
    assert db.count_runs("no-such-conversation") == 0


def test_count_runs_excludes_superseded_when_asked(db):
    a = db.create_run(conversation_id="c", agent_kind="primary")
    db.set_status(a, "done", result="a finished before supersede")
    db.create_run(conversation_id="c", agent_kind="primary")
    db.supersede_run_tree(a)

    assert db.count_runs("c") == 2
    assert db.count_runs("c", include_superseded=False) == 1


def test_count_runs_does_not_materialize_rows_beyond_a_single_count(db):
    """Finding A's own point: `count_runs` must be a single `COUNT(*)`
    query, never `len(list_runs(...))` in disguise -- assert on the ACTUAL
    SQL sent, the same trace-callback technique
    test_transaction_begins_immediate_not_deferred already uses."""
    for _ in range(5):
        db.create_run(conversation_id="c", agent_kind="primary")

    # Attach directly to the thread's HELD connection (task-3012) — see the
    # comment in test_count_subagents_by_conversation_batches_single_query.
    calls = []
    db._held_connection().set_trace_callback(calls.append)
    n = db.count_runs("c", agent_kind="primary")
    assert n == 5
    select_calls = [c for c in calls if c.strip().upper().startswith("SELECT")]
    assert len(select_calls) == 1
    assert "COUNT(*)" in select_calls[0].upper()


# --- agent_definitions CRUD tests (Task 2: fleet spec §4) ---


def _defn(**overrides):
    base = dict(
        name="researcher",
        description="Searches sources.",
        instructions="Research thoroughly.",
        tool_allowlist=("web_search",),
    )
    base.update(overrides)
    return AgentDefinition(**base)


def test_definition_crud_round_trip(db):
    definition_id = db.create_agent_definition(_defn())
    rows = db.list_agent_definitions()
    assert [r["name"] for r in rows] == ["researcher"]
    assert rows[0]["tool_allowlist"] == ["web_search"]
    db.update_agent_definition(definition_id, _defn(description="v2"))
    assert db.get_agent_definition(definition_id)["description"] == "v2"
    db.soft_delete_agent_definition(definition_id)
    assert db.list_agent_definitions() == []


def test_duplicate_name_raises_and_frees_after_soft_delete(db):
    definition_id = db.create_agent_definition(_defn())
    with pytest.raises(ValueError, match="already exists"):
        db.create_agent_definition(_defn())
    db.soft_delete_agent_definition(definition_id)
    db.create_agent_definition(_defn())  # name reusable after soft delete


def test_invalid_definition_rejected_at_db_boundary(db):
    with pytest.raises(ValueError, match="reserved"):
        db.create_agent_definition(_defn(name="subagent"))


def test_update_after_soft_delete_raises_not_found(db):
    # A missing/soft-deleted id used to no-op silently (0-row UPDATE), and
    # the Settings ▸ Agents panel would still report "Saved" -- the caller
    # must be able to tell the edit never landed.
    definition_id = db.create_agent_definition(_defn())
    db.soft_delete_agent_definition(definition_id)
    with pytest.raises(ValueError, match="not found"):
        db.update_agent_definition(definition_id, _defn(description="v2"))


def test_update_unknown_id_raises_not_found(db):
    with pytest.raises(ValueError, match="not found"):
        db.update_agent_definition("does-not-exist", _defn())


def test_enabled_only_filter(db):
    db.create_agent_definition(_defn(name="on-agent"))
    db.create_agent_definition(_defn(name="off-agent", enabled=False))
    assert [r["name"] for r in db.list_agent_definitions(enabled_only=True)] == [
        "on-agent"
    ]
    assert len(db.list_agent_definitions()) == 2


def test_definitions_survive_reopen_and_migration_is_idempotent(tmp_path):
    path = tmp_path / "agent_runs.db"
    first = AgentRunsDB(path, client_id="test")
    first.create_agent_definition(_defn())
    first.close()
    second = AgentRunsDB(path, client_id="test")  # re-runs _initialize_schema
    assert [r["name"] for r in second.list_agent_definitions()] == ["researcher"]
    with second.connection() as conn:
        versions = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_version").fetchall()
        }
    assert 5 in versions


# --- Task 3: agent_definition + definition_fingerprint audit columns ---


def test_create_run_records_definition_audit_fields(db):
    run_id = db.create_run(
        conversation_id="c",
        agent_kind="subagent",
        task="t",
        parent_run_id=None,
        agent_definition="researcher",
        definition_fingerprint="abc123def4567890",
    )
    run = db.get_run(run_id)
    assert run["agent_definition"] == "researcher"
    assert run["definition_fingerprint"] == "abc123def4567890"


def test_create_run_definition_fields_default_none(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    run = db.get_run(run_id)
    assert run["agent_definition"] is None
    assert run["definition_fingerprint"] is None


def test_agent_runs_columns_backfilled_on_old_file(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    # Simulate a pre-v5 file: the v4-era 12-column table, no new columns.
    conn.execute(
        """CREATE TABLE agent_runs (
               id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
               parent_run_id TEXT, agent_kind TEXT NOT NULL, task TEXT,
               status TEXT NOT NULL, steps TEXT NOT NULL DEFAULT '[]',
               result TEXT, budget TEXT, created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL, assistant_message_id TEXT)"""
    )
    conn.commit()
    conn.close()
    db = AgentRunsDB(path, client_id="test")  # open runs the ALTER guards
    with db.connection() as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
    assert {"agent_definition", "definition_fingerprint"} <= columns


# --- Task 2: terminal-status guard on set_status (first-writer-wins) ---


def test_set_status_first_terminal_write_wins(db):
    # A child abandoned after a join timeout can persist LATE; it must not
    # overwrite the terminal status the coordinator already recorded.
    run_id = db.create_run(conversation_id="c", agent_kind="subagent", task="t")
    assert db.set_status(run_id, "cancelled") is True
    assert db.set_status(run_id, "done", result="late answer") is False
    run = db.get_run(run_id)
    assert run["status"] == "cancelled"
    assert run["result"] is None


def test_set_status_still_updates_a_running_run(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    assert db.set_status(run_id, "done", result="ok") is True
    assert db.get_run(run_id)["status"] == "done"
    assert db.get_run(run_id)["result"] == "ok"


def test_set_status_missing_run_returns_false(db):
    assert db.set_status("nope", "done") is False


# --- PR3b Task 4 (fleet continuation): resumed_from_run_id, v11, and the
# task-15669 constant-vs-version-table fold (coordinator ruling #3). ---

#: The pre-v11 shape: agent_runs as every migration through v10 left it --
#: all columns EXCEPT resumed_from_run_id -- with version rows through 10.
_LEGACY_PRE_V11_DDL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (4);
    INSERT OR IGNORE INTO schema_version (version) VALUES (5);
    INSERT OR IGNORE INTO schema_version (version) VALUES (6);
    INSERT OR IGNORE INTO schema_version (version) VALUES (7);
    INSERT OR IGNORE INTO schema_version (version) VALUES (8);
    INSERT OR IGNORE INTO schema_version (version) VALUES (9);
    INSERT OR IGNORE INTO schema_version (version) VALUES (10);

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
        updated_at TEXT NOT NULL,
        assistant_message_id TEXT,
        agent_definition TEXT,
        definition_fingerprint TEXT,
        wake_delivered_at TEXT
    );
"""


def test_schema_version_constant_agrees_with_the_version_table(tmp_path):
    """task-15669 AC#1/#3 (folded into v11 per coordinator ruling #3): the
    constant CLAUDE.md points every schema change at must agree with the
    highest version a freshly created database actually records -- and
    this test fails if the two ever diverge again."""
    db = AgentRunsDB(tmp_path / "fresh.db", client_id="test")
    with db.connection() as conn:
        recorded = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
    assert recorded == AgentRunsDB._CURRENT_SCHEMA_VERSION


def test_pre_v11_db_gains_resumed_from_run_id_and_opens_twice(tmp_path):
    """Migration idempotency (plan red): a pre-v11 file gains the column
    via the guarded ALTER on first open, and a second open is a no-op."""
    path = tmp_path / "legacy_pre_v11.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_LEGACY_PRE_V11_DDL)
        conn.commit()
    finally:
        conn.close()

    raw = sqlite3.connect(str(path))
    try:
        cols = {row[1] for row in raw.execute("PRAGMA table_info(agent_runs)")}
    finally:
        raw.close()
    assert "resumed_from_run_id" not in cols

    first = AgentRunsDB(path, client_id="test")
    run_id = first.create_run(
        conversation_id="c",
        agent_kind="subagent",
        task="t",
        resumed_from_run_id="prior-run",
    )
    assert first.get_run(run_id)["resumed_from_run_id"] == "prior-run"
    with first.connection() as conn:
        versions = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_version")
        }
    assert 11 in versions

    # Open TWICE (the plan's wording): the guarded ALTER must be a no-op.
    second = AgentRunsDB(path, client_id="test")
    second_id = second.create_run(
        conversation_id="c", agent_kind="subagent", task="t2"
    )
    assert second.get_run(second_id)["resumed_from_run_id"] is None
    assert second.get_run(run_id)["resumed_from_run_id"] == "prior-run"


def test_create_run_resumed_from_run_id_round_trips_and_defaults_none(db):
    origin = db.create_run(conversation_id="c", agent_kind="subagent", task="t")
    resumed = db.create_run(
        conversation_id="c",
        agent_kind="subagent",
        task="t",
        resumed_from_run_id=origin,
    )
    assert db.get_run(resumed)["resumed_from_run_id"] == origin
    assert db.get_run(origin)["resumed_from_run_id"] is None
    # The lineage flows through the list read too (SELECT * row dicts).
    listed = {row["id"]: row for row in db.list_runs("c")}
    assert listed[resumed]["resumed_from_run_id"] == origin


def test_pre_v14_db_gains_spawn_event_id_and_opens_twice(tmp_path):
    path = tmp_path / "legacy_pre_v14.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
        INSERT INTO schema_version(version) VALUES
            (4), (5), (6), (7), (8), (9), (10), (11), (12), (13);
        CREATE TABLE agent_runs (
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
            updated_at TEXT NOT NULL,
            assistant_message_id TEXT,
            agent_definition TEXT,
            definition_fingerprint TEXT,
            wake_delivered_at TEXT,
            resumed_from_run_id TEXT
        );
        """
    )
    conn.commit()
    columns_before = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
    conn.close()
    assert "spawn_event_id" not in columns_before

    first = AgentRunsDB(path, client_id="migrate-v14")
    with first.connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
        recorded = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert "spawn_event_id" in columns
    assert recorded == AgentRunsDB._CURRENT_SCHEMA_VERSION == 14
    parent = first.create_run(conversation_id="c", agent_kind="primary")
    child = first.create_run(
        conversation_id="c",
        agent_kind="subagent",
        parent_run_id=parent,
        spawn_event_id=f"agent-step:{parent}:3",
    )
    assert first.get_run(child)["spawn_event_id"] == f"agent-step:{parent}:3"
    first.close()

    second = AgentRunsDB(path, client_id="reopen-v14")
    assert second.get_run(child)["spawn_event_id"] == f"agent-step:{parent}:3"
    second.close()
