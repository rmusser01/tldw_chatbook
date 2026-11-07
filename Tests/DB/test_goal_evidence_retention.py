"""Capacity is reserved before dispatch; checkpoint settlement is transactional."""

import sqlite3
from dataclasses import replace

import pytest

from Tests.Agents.test_goal_iteration_report import report
from Tests.Agents.test_goal_models import request
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def test_launch_payloads_count_toward_aggregate_and_removed_goal_is_tombstone(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db")
    db.goal_runs.payload_limit = 2500
    first = db.goal_runs.create(request(), launch_id="one")
    with pytest.raises(ValueError, match="payload_capacity"):
        for i in range(10):
            db.goal_runs.create(request(), launch_id=f"other-{i}")
    with pytest.raises(ValueError, match="settled"):
        db.goal_runs.remove_payloads(first.id)
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table",
    ["goal_reports", "goal_checkpoints", "goal_runs", "automatic_wake_attempts"],
)
async def test_checkpoint_atomic_rollback_and_exact_replay(stores, monkeypatch, table):
    def provider(**kwargs):
        return {
            "choices": [{"message": {"content": report(candidate_draft="work")}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _, _, _, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    db = stores[0]
    try:
        result = await coordinator.dispatch_once(goal.id)
        used = db.automatic_work.snapshot(goal.chain_id).used
        operation = (
            "UPDATE" if table in ("goal_runs", "automatic_wake_attempts") else "INSERT"
        )
        with db.transaction() as conn:
            conn.execute(
                f"CREATE TRIGGER fail_checkpoint BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'injected checkpoint'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            coordinator.service.checkpoint(result)
        with db.connection() as conn:
            assert conn.execute("SELECT count(*) FROM goal_reports").fetchone()[0] == 0
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (result.attempt_id,),
                ).fetchone()[0]
                == "accepted"
            )
        assert len(calls) == 1
        with db.transaction() as conn:
            conn.execute("DROP TRIGGER fail_checkpoint")
        saved = coordinator.service.checkpoint(result)
        assert coordinator.service.checkpoint(result).revision == saved.revision
        with pytest.raises(ValueError, match="checkpoint_conflict"):
            coordinator.service.checkpoint(
                replace(
                    result,
                    outcome=replace(
                        result.outcome, final_text=report(summary="different")
                    ),
                )
            )
        assert db.automatic_work.snapshot(goal.chain_id).used == used
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_settled_removal_retains_accounting_and_cannot_be_restarted(
    stores, monkeypatch
):
    def provider(**kwargs):
        return {
            "choices": [{"message": {"content": report()}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _, _, _, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    try:
        for _ in range(2):
            result = await coordinator.dispatch_once(goal.id)
            saved = coordinator.service.checkpoint(result)
        assert saved.status == "paused"
        before = saved.accounting.used
        removed = coordinator.service.remove_payloads(goal.id)
        assert removed.status == "removed" and removed.request is None
        assert removed.accounting.used == before
        assert removed.reports == () and removed.checkpoints == ()
        with pytest.raises(ValueError, match="removed"):
            coordinator.service.create(goal.request, launch_id=goal.launch_id)
        assert coordinator.service.get(goal.id).status == "removed"
        assert len(calls) == 2
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_admission_reserves_full_result_capacity_before_any_provider(
    stores, monkeypatch
):
    goal, _, _, _, coordinator, gateway, calls = build_goal_rig(stores, monkeypatch)
    stores[0].goal_runs.payload_limit = 2000
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.native_run_id is None and calls == []
        assert stores[0].automatic_work.snapshot(goal.chain_id).used["generation"] == 0
    finally:
        await gateway.aclose()


def test_source_era_goal_schema_is_not_a_current_version_18_archive(tmp_path):
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    path = tmp_path / "source-era.db"
    db = AgentRunsDB(path)
    db.goal_runs.create(request(), launch_id="source-era")
    with db.transaction() as conn:
        conn.execute("DELETE FROM schema_version WHERE version>18")
    db.close()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=True)
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_evidence_insert_failure_keeps_actual_cli_effects_and_accepted_attempt(
    stores, monkeypatch, tmp_path
):
    from Tests.Agents.test_goal_progress import run_verifier

    coordinator, _goal, result, fixture, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, mode="during"
    )
    db = stores[0]
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER reject_evidence AFTER INSERT ON goal_evidence BEGIN SELECT RAISE(ABORT, 'injected evidence'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected evidence"):
        coordinator.service.checkpoint(result)
    assert fixture.read_text() == "changed during check"
    assert len(calls) == 2
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM goal_evidence").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM goal_checkpoints").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT state FROM automatic_wake_attempts WHERE id=?",
                (result.attempt_id,),
            ).fetchone()[0]
            == "accepted"
        )


@pytest.mark.asyncio
async def test_nearly_full_goal_refuses_before_an_iteration_lacks_one_record_slot(
    stores, monkeypatch, tmp_path
):
    from Tests.Agents.test_goal_progress import run_verifier

    coordinator, goal, result, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True, mode="repeat"
    )
    coordinator.service.checkpoint(result)
    db = stores[0]
    original = db.goal_runs.evidence(goal.id)[0]
    with db.transaction() as conn:
        for index in range(31):
            item = original.model_copy(update={"id": f"copied{index}", "stdout": ""})
            item = item.model_copy(
                update={
                    "stdout": "x" * (128 * 1024 - len(item.model_dump_json().encode()))
                }
            )
            conn.execute(
                "INSERT INTO goal_evidence VALUES (?,?,?,?)",
                (item.id, goal.id, result.attempt_id, item.model_dump_json()),
            )
    with pytest.raises(ValueError, match="goal_evidence_capacity"):
        db.automatic_work.prepare_goal_iteration(
            goal.id, owner_id=coordinator._owner_id
        )
    assert db.automatic_work.snapshot(goal.chain_id).used["generation"] == 1


def test_generic_copy_bounds_encoded_json_not_only_raw_output(stores):
    from tldw_chatbook.Agents.goal_iteration import resolve_runtime_evidence
    from tldw_chatbook.Agents.goal_models import (
        GoalIterationResult,
        GoalToolObservation,
    )

    db, _, _, req = stores
    goal = db.goal_runs.create(req, launch_id="encoded")
    result = GoalIterationResult(
        goal.id,
        1,
        "attempt",
        "run",
        None,
        "done",
        observations=(
            GoalToolObservation(
                "record",
                goal.id,
                "attempt",
                "run",
                "fs_read",
                "a" * 64,
                "\x00" * 48000,
                True,
            ),
        ),
    )
    evidence = resolve_runtime_evidence(goal, result)
    assert len(evidence[0].model_dump_json().encode()) <= 128 * 1024
    assert not evidence[0].passed
