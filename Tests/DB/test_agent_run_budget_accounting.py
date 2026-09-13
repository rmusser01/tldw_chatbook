"""Durable, idempotent budget counters and scoped continuation ancestry."""

import sqlite3

import pytest

from Tests.DB.test_agent_runs_db import _LEGACY_PRE_V11_DDL
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_legacy_usage_stays_unknown_across_migration_and_reopen(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(_LEGACY_PRE_V11_DDL)
        conn.execute("ALTER TABLE agent_runs ADD COLUMN resumed_from_run_id TEXT")
        conn.execute("INSERT INTO schema_version VALUES (11)")
        conn.execute("INSERT INTO schema_version VALUES (12)")
        conn.execute(
            "INSERT INTO agent_runs (id, conversation_id, agent_kind, status, created_at, updated_at) "
            "VALUES ('old', 'c', 'subagent', 'done', '2026-01-01', '2026-01-01')"
        )
    for _ in range(2):
        db = AgentRunsDB(path, client_id="budget")
        assert db.get_run("old")["budget_tokens"] is None
        with db.connection() as conn:
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == db._CURRENT_SCHEMA_VERSION
            )
        db.close()


@pytest.fixture
def db(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.db", client_id="budget")
    try:
        yield database
    finally:
        database.close()


def _run(db, count, *, parent=None, conversation="c", kind="subagent", status="done"):
    run_id = db.create_run(
        conversation_id=conversation, agent_kind=kind, resumed_from_run_id=parent
    )
    db.set_status(run_id, status, budget_tokens=count)
    return run_id


def test_zero_is_known_and_late_budget_does_not_replace_cancellation(db):
    zero = _run(db, 0)
    assert db.get_run(zero)["budget_tokens"] == 0
    child = db.create_run(conversation_id="c", agent_kind="subagent")
    assert db.set_status(child, "cancelled", result="stopped")
    assert not db.set_status(child, "done", result="late answer", budget_tokens=75)
    assert not db.set_status(child, "done", result="repeated", budget_tokens=150)
    record = db.get_run(child)
    assert (record["status"], record["result"], record["budget_tokens"]) == (
        "cancelled",
        "stopped",
        75,
    )


@pytest.mark.parametrize("bad", [True, -1, 1.5, "4", 2**63])
def test_invalid_counter_does_not_finalize_the_run(db, bad):
    child = db.create_run(conversation_id="c", agent_kind="subagent")
    with pytest.raises(ValueError):
        db.set_status(child, "done", budget_tokens=bad)
    assert db.get_run(child)["status"] == "running"
    assert db.get_run(child)["budget_tokens"] is None


def test_chain_survives_reopen_and_excludes_sibling_forks(db):
    root = _run(db, 100)
    branch = _run(db, 40, parent=root)
    selected = _run(db, 10, parent=branch)
    _run(db, 900, parent=root)
    path = db.db_path_str
    db.close()
    reopened = AgentRunsDB(path, client_id="reopen")
    try:
        assert reopened.continuation_budget("c", selected) == {
            "budget_tokens": 150,
            "run_count": 3,
            "recorded_run_count": 3,
            "complete": True,
        }
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "ancestor_kind", ["unknown", "foreign", "primary", "missing", "running"]
)
def test_incomplete_or_foreign_ancestry_never_fabricates_a_complete_total(
    db, ancestor_kind
):
    if ancestor_kind == "missing":
        parent = "missing"
    else:
        parent = _run(
            db,
            None if ancestor_kind in {"unknown", "running"} else 900,
            conversation="foreign" if ancestor_kind == "foreign" else "c",
            kind="primary" if ancestor_kind == "primary" else "subagent",
            status="running" if ancestor_kind == "running" else "done",
        )
    selected = _run(db, 40, parent=parent)
    result = db.continuation_budget("c", selected)
    assert result["budget_tokens"] == 40
    assert result["recorded_run_count"] == 1
    assert result["complete"] is False
    assert db.continuation_budget("foreign", selected) is None


def test_cyclic_ancestry_counts_each_run_once_and_reports_partial(db):
    first = _run(db, 10)
    second = _run(db, 20, parent=first)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE agent_runs SET resumed_from_run_id=? WHERE id=?", (second, first)
        )
    assert db.continuation_budget("c", second) == {
        "budget_tokens": 30,
        "run_count": 2,
        "recorded_run_count": 2,
        "complete": False,
    }


def test_chain_sum_does_not_overflow_sqlite_integer_accumulation(db):
    root = _run(db, 2**63 - 1)
    selected = _run(db, 1, parent=root)
    assert db.continuation_budget("c", selected)["budget_tokens"] == 2**63
