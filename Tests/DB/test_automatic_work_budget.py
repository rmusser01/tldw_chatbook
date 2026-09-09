"""Durable automatic-work admission, independent of provider billing."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture(name="db")
def _automatic_work_db(tmp_path):
    database = AgentRunsDB(tmp_path / "runs.sqlite", client_id="automatic-test")
    yield database
    database.close()


def chain(db, *, conversation="conversation", submission="submission", **limits):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    return db.automatic_work.create_chain(
        conversation,
        root_submission_id=submission,
        limits=replace(AutomaticWorkLimits(), **limits),
    )


def reserve(
    db, chain_id, reservation_id, *, kind="generation", amount=1, owner="owner"
):
    return db.automatic_work.reserve(
        chain_id,
        reservation_id=reservation_id,
        owner_id=owner,
        kind=kind,
        amount=amount,
    )


def test_automatic_work_commit_is_full_synced_and_rollback_restores_policy(db):
    with db.connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    with (
        pytest.raises(RuntimeError, match="rollback"),
        db.automatic_work.transaction() as conn,
    ):
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
        conn.execute("INSERT INTO schema_version VALUES (999)")
        raise RuntimeError("rollback")
    with db.connection() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert (
            conn.execute("SELECT 1 FROM schema_version WHERE version=999").fetchone()
            is None
        )
    db.close()
    with db.automatic_work.transaction() as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_same_submission_keeps_immutable_limits_and_scope(db):
    first = chain(db, generations=2)
    assert chain(db, generations=99) == first
    assert db.automatic_work.snapshot(first).limits.generations == 2
    with pytest.raises(ValueError, match="scope"):
        chain(db, conversation="elsewhere")
    second = chain(db, submission="new-submission")
    assert second != first


def test_reserved_and_consumed_generations_share_one_finite_allowance(db):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db, generations=2)
    first = reserve(db, chain_id, "one")
    assert reserve(db, chain_id, "one") == first
    assert db.automatic_work.commit("one", owner_id="owner") is True
    assert db.automatic_work.commit("one", owner_id="owner") is False
    assert db.automatic_work.release("one", owner_id="owner") is False
    reserve(db, chain_id, "two")
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        reserve(db, chain_id, "three")
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.used["generation"] == 1
    assert snapshot.reserved["generation"] == 1
    assert snapshot.available["generation"] == 0
    assert db.automatic_work.release("two", owner_id="owner") is True
    assert db.automatic_work.release("two", owner_id="owner") is False
    reserve(db, chain_id, "three")


@pytest.mark.parametrize(
    "kind,limit_name",
    [
        ("child_launch", "child_launches"),
        ("model_call", "model_calls"),
        ("tokens", "budget_tokens"),
    ],
)
def test_other_resource_kinds_are_finite(db, kind, limit_name):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db, **{limit_name: 2})
    reserve(db, chain_id, "whole", kind=kind, amount=2)
    with pytest.raises(AutomaticWorkRefused):
        reserve(db, chain_id, "excess", kind=kind)
    assert db.automatic_work.snapshot(chain_id).available[kind] == 0


@pytest.mark.parametrize("amount", [True, False, -1, 0, 1.0, "1", 2**63])
def test_invalid_reservation_amount_never_changes_balance(db, amount):
    chain_id = chain(db)
    with pytest.raises(ValueError):
        reserve(db, chain_id, "invalid", amount=amount)
    assert db.automatic_work.snapshot(chain_id).available["generation"] == 3


def test_duplicate_id_cannot_change_amount_chain_kind_or_owner(db):
    first = chain(db)
    second = chain(db, submission="second")
    reserve(db, first, "existing")
    for changes in ({"amount": 2}, {"kind": "tokens"}, {"owner": "another"}):
        with pytest.raises(ValueError, match="conflict"):
            reserve(db, first, "existing", **changes)
    with pytest.raises(ValueError, match="conflict"):
        reserve(db, second, "existing")
    with pytest.raises(ValueError, match="owner"):
        db.automatic_work.commit("existing", owner_id="another")
    assert db.automatic_work.snapshot(first).reserved["generation"] == 1
    assert db.automatic_work.snapshot(second).reserved["generation"] == 0


def test_actual_tokens_release_estimate_without_changing_run_accounting(db):
    chain_id = chain(db, budget_tokens=100)
    run_id = db.create_run(conversation_id="conversation", agent_kind="primary")
    db.set_status(run_id, "done", budget_tokens=17)
    reserve(db, chain_id, "call", kind="tokens", amount=80)
    db.automatic_work.commit("call", owner_id="owner")
    assert db.automatic_work.settle("call", owner_id="owner", actual_amount=30)
    assert not db.automatic_work.settle("call", owner_id="owner", actual_amount=30)
    with pytest.raises(ValueError, match="conflict"):
        db.automatic_work.settle("call", owner_id="owner", actual_amount=31)
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.used["tokens"] == 30
    assert snapshot.available["tokens"] == 70
    assert db.get_run(run_id)["budget_tokens"] == 17


def test_legacy_child_attachment_cannot_conflict_with_parent_chain(db):
    first = chain(db)
    second = chain(db, submission="second")
    parent = db.create_run(conversation_id="conversation", agent_kind="primary")
    child = db.create_run(
        conversation_id="conversation", agent_kind="subagent", parent_run_id=parent
    )
    db.automatic_work.attach_run(parent, first)
    with pytest.raises(ValueError, match="parent"):
        db.automatic_work.attach_run(child, second)
    assert db.get_run(child)["work_chain_id"] is None
    db.automatic_work.attach_run(child, first)
    assert db.get_run(child)["work_chain_id"] == first


def test_unknown_usage_stays_reserved_and_pauses_chain_after_reopen(db):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db, budget_tokens=100)
    reserve(db, chain_id, "call", kind="tokens", amount=80)
    db.automatic_work.commit("call", owner_id="owner")
    db.automatic_work.settle("call", owner_id="owner", actual_amount=None)
    db.close()
    snapshot = db.automatic_work.snapshot(chain_id)
    assert snapshot.used["tokens"] + snapshot.reserved["tokens"] == 80
    assert snapshot.available["tokens"] == 20
    assert snapshot.uncertain
    assert snapshot.pause_reason == "usage_unknown"
    with pytest.raises(AutomaticWorkRefused, match="usage_unknown"):
        reserve(db, chain_id, "next", kind="model_call")
    db.automatic_work.settle("call", owner_id="owner", actual_amount=30)
    assert db.automatic_work.snapshot(chain_id).used["tokens"] == 30
    # Late measurement does not grant unattended recovery authority.
    assert db.automatic_work.snapshot(chain_id).status == "review_required"


def test_last_slot_is_atomic_across_separate_sqlite_connections(db):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db, generations=1)
    barrier = threading.Barrier(2, timeout=5)

    def compete(identity):
        try:
            barrier.wait()
            reserve(db, chain_id, identity)
            return True
        except AutomaticWorkRefused:
            return False
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(compete, ["first", "second"]))
    assert sorted(results) == [False, True]
    assert db.automatic_work.snapshot(chain_id).reserved["generation"] == 1


def test_run_lineage_inherits_parent_and_cannot_be_reassigned(db):
    original = chain(db)
    newer = chain(db, submission="newer")
    parent = db.create_run(
        conversation_id="conversation", agent_kind="primary", work_chain_id=original
    )
    child = db.create_run(
        conversation_id="conversation", agent_kind="subagent", parent_run_id=parent
    )
    assert db.get_run(child)["work_chain_id"] == original
    with pytest.raises(ValueError, match="chain"):
        db.automatic_work.attach_run(child, newer)
    with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
        conn.execute("UPDATE agent_runs SET work_chain_id=? WHERE id=?", (newer, child))
    with pytest.raises(ValueError, match="scope"):
        db.create_run(
            conversation_id="other", agent_kind="primary", work_chain_id=original
        )
    legacy = db.create_run(conversation_id="conversation", agent_kind="primary")
    assert db.get_run(legacy)["work_chain_id"] is None


@pytest.mark.parametrize(
    "configured", [False, True, -1, 1.5, float("nan"), float("inf"), "bad"]
)
def test_invalid_config_falls_back_to_finite_default(db, monkeypatch, configured):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    monkeypatch.setattr(run_log, "_setting", lambda key, default: configured)
    chain_id = db.automatic_work.create_chain(
        "conversation", root_submission_id="config"
    )
    for index in range(3):
        reserve(db, chain_id, str(index))
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        reserve(db, chain_id, "fourth")


def test_env_limit_overrides_config_and_zero_disables_generations(db, monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_MAX_AUTOWAKE_GENERATIONS", "0")
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = db.automatic_work.create_chain(
        "conversation", root_submission_id="disabled"
    )
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        reserve(db, chain_id, "one")


def test_usage_over_estimate_stops_further_admission_of_every_resource(db):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    chain_id = chain(db, budget_tokens=10)
    reserve(db, chain_id, "call", kind="tokens", amount=10)
    db.automatic_work.commit("call", owner_id="owner")
    db.automatic_work.settle("call", owner_id="owner", actual_amount=12)
    assert db.automatic_work.snapshot(chain_id).pause_reason == "tokens_budget"
    for kind in ("generation", "child_launch", "model_call", "tokens"):
        with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
            reserve(db, chain_id, kind, kind=kind)
    assert db.automatic_work.snapshot(chain_id).used["tokens"] == 12


def test_legacy_parent_cannot_bridge_conversation_scope(db):
    parent = db.create_run(conversation_id="elsewhere", agent_kind="primary")
    chain_id = chain(db)
    with pytest.raises(ValueError, match="scope"):
        db.create_run(
            conversation_id="conversation",
            agent_kind="subagent",
            parent_run_id=parent,
            work_chain_id=chain_id,
        )
