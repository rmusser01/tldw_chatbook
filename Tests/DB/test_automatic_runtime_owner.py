"""Explicit native recovery durably revokes completed owners too."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from Tests.Chat.test_automatic_provider_budget import context_for
from Tests.DB.test_automatic_wake_attempts import claim, survivor
from Tests.DB.test_automatic_work_budget import _automatic_work_db, chain  # noqa: F401
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_completed_attempt_cannot_resume_after_owner_replacement(db):
    db.automatic_work.recover(current_owner_id="owner")
    context = context_for(db)
    db.automatic_work.reserve(
        context.chain_id,
        reservation_id="child",
        owner_id="owner",
        kind="child_launch",
        amount=1,
    )
    assert db.automatic_work.commit("child", owner_id="owner")
    tokens = context.begin_call(10, 10)
    context.settle_call(tokens, 5)
    assert db.automatic_work.complete_wake("attempt", owner_id="owner")
    assert db.automatic_work.recover(current_owner_id="replacement") == 0
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        context.check()
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        context.begin_call(1, 1)
    assert db.automatic_work.snapshot(context.chain_id).status == "active"
    assert db.automatic_work.snapshot(context.chain_id).used["model_call"] == 1


@pytest.mark.parametrize("operation", ["reserve", "admit", "claim", "accept", "commit"])
def test_every_dispatch_authority_checks_durable_owner_in_transaction(db, operation):
    ledger = db.automatic_work
    ledger.recover(current_owner_id="owner")
    chain_id = chain(db)
    run_id = survivor(db, chain_id)
    claim(db, chain_id, [run_id])
    ledger.reserve(
        chain_id,
        reservation_id="child",
        owner_id="owner",
        kind="child_launch",
        amount=1,
    )
    ledger.recover(current_owner_id="replacement")
    calls = {
        "reserve": lambda: ledger.reserve(
            chain_id,
            reservation_id="new",
            owner_id="owner",
            kind="child_launch",
            amount=1,
        ),
        "admit": lambda: ledger.admit_call(
            chain_id, call_id="new", owner_id="owner", input_tokens=1, output_tokens=1
        ),
        "claim": lambda: claim(db, chain_id, [run_id]),
        "accept": lambda: ledger.accept_wake("attempt", owner_id="owner"),
        "commit": lambda: ledger.commit("child", owner_id="owner"),
    }
    with pytest.raises(AutomaticWorkRefused, match="runtime_owner_replaced"):
        calls[operation]()


def test_recovery_owner_fence_is_full_atomic_and_old_usage_can_settle(db):
    ledger = db.automatic_work
    ledger.recover(current_owner_id="owner")
    context = context_for(db)
    token_id = context.begin_call(10, 10)
    with db.connection() as conn:
        conn.create_function(
            "is_full", 0, lambda: conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        )
        conn.execute(
            "CREATE TRIGGER owner_must_be_full BEFORE UPDATE ON automatic_work_runtime_owner WHEN is_full() != 1 BEGIN SELECT RAISE(ABORT, 'not full'); END"
        )
        conn.execute(
            "CREATE TRIGGER reject_recovery BEFORE UPDATE ON automatic_wake_attempts BEGIN SELECT RAISE(ABORT, 'rollback recovery'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="rollback recovery"):
        ledger.recover(current_owner_id="replacement")
    context.check()
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER reject_recovery")
    assert ledger.recover(current_owner_id="replacement") == 1
    context.settle_call(token_id, 7)
    assert ledger.snapshot(context.chain_id).used["tokens"] == 7
    assert ledger.snapshot(context.chain_id).status == "review_required"
    db.close()
    with pytest.raises(AutomaticWorkRefused):
        context.check()


@pytest.mark.parametrize("standalone", [False, True])
def test_v17_to_v18_migration_preserves_history_and_does_not_recover(
    tmp_path, standalone
):
    path = tmp_path / "runs.sqlite"
    db = AgentRunsDB(path)
    context = context_for(db)
    assert db.automatic_work.complete_wake("attempt", owner_id="owner")
    with db.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS automatic_work_runtime_owner")
        conn.execute("DELETE FROM schema_version WHERE version>=18")
    db.close()
    if standalone:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        with closing(connect_private_sqlite("db.base", path)) as conn:
            conn.executescript(
                Path(
                    "tldw_chatbook/DB/migrations/agent_runs_v17_to_v18_runtime_owner.sql"
                ).read_text()
            )
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == 18
            )
    reopened = AgentRunsDB(path)
    try:
        with reopened.connection() as conn:
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == AgentRunsDB._CURRENT_SCHEMA_VERSION
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM automatic_work_runtime_owner"
                ).fetchone()[0]
                == 0
            )
        assert (
            reopened.automatic_work.read_attempt("attempt", owner_id="owner").state
            == "completed"
        )
        assert (
            reopened.automatic_work.snapshot(context.chain_id).used["generation"] == 1
        )
    finally:
        reopened.close()
