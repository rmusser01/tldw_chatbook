"""Native captured Goal transforms preserve provenance and erased bodies."""

import hashlib
import importlib

import pytest

from Tests.Backup_Recovery.test_mobile_content_graph import content_reads


@pytest.mark.parametrize("stale_removed_child", [False, True])
def test_captured_goal_request_mapping_keeps_native_hash_and_removed_tombstone(
    tmp_path, monkeypatch, stale_removed_child
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.goal_models import GoalRequest
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    original_seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, identities):
        goal = db.goal_runs.get(identities["goal_id"])
        if legacy:
            source["goal"] = goal
        removed = db.goal_runs.create(goal.request, launch_id="removed-launch")
        db.goal_runs.control(removed.id)
        removed = db.goal_runs.remove_payloads(removed.id)
        identities["removed_conversation"] = removed.conversation_id
        if legacy:
            source["removed"] = removed
            if stale_removed_child:
                # Malformed historical fixture before preview/capture: native removal
                # deleted this family, so it cannot become a restored body again.
                with db.transaction() as connection:
                    connection.execute(
                        "INSERT INTO goal_waits VALUES (?, ?, ?)",
                        (removed.id, 1.0, "old private wait"),
                    )

    def linked(db, legacy, identities):
        db.add_conversation(
            {
                "id": identities["removed_conversation"],
                "title": "Removed Goal",
                "workspace_id": identities["workspace_id"],
            }
        )

    def seeded(stores, **kwargs):
        return original_seed(
            stores, **kwargs, runs_seed=native, linked_chat_seed=linked
        )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None
    ) as reads:
        identities = reads.identity_plan()
        name = "tldw_chatbook.Backup_Recovery.mobile_runs_goals"
        assert importlib.util.find_spec(name) is not None, (
            "captured Goal rows need a native pure transformation"
        )
        build = importlib.import_module(name)._plan_goal_rows
        plan = build(reads, identities, _read_budget=_TraceReadBudget())
        if stale_removed_child:
            assert any(
                b.table == "goal_waits" and b.reason == "native_goal_content_invalid"
                for b in plan.blockers
            )
        else:
            assert not plan.blockers
        rows = {(r.table, r.original): dict(r.values) for r in plan.rows}
        original = source["goal"]
        row = rows["goal_runs", (original.id,)]
        request = GoalRequest.model_validate_json(row["request_json"])
        assert request.binding.workspace_id == identities.mapped(
            "workspace", "conflicting-workspace"
        )
        assert request.binding.binding_id == identities.mapped("binding", "binding")
        assert request.binding.locator == original.request.binding.locator
        assert request.objective == original.request.objective
        assert (
            row["payload_hash"]
            == hashlib.sha256(request.canonical_json().encode()).hexdigest()
        )
        assert row["payload_hash"] != original.payload_hash
        assert row["launch_id"] != original.launch_id
        provenance = next(
            p
            for p in plan.provenance
            if p.table == "goal_runs" and p.original == (original.id,)
        )
        assert provenance.original_hash == original.payload_hash
        assert provenance.mapped_hash == row["payload_hash"]
        removed = source["removed"]
        erased = rows["goal_runs", (removed.id,)]
        assert erased["status"] == "removed"
        assert erased["request_json"] == ""
        assert erased["payload_hash"] == removed.payload_hash
        assert erased["revision"] == removed.revision
        assert erased["pause_reason"] == "payload_removed"
        assert not any(dict(r.values).get("goal_id") == erased["id"] for r in plan.rows)
        with pytest.raises(ValueError, match="pending"):
            plan.require_complete()
    with pytest.raises(ValueError, match="inactive"):
        build(reads, identities, _read_budget=_TraceReadBudget())


def test_captured_checkpoint_keeps_full_result_and_review_digests(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.agent_models import RunOutcome, RunTerminationReason
    from tldw_chatbook.Agents.goal_models import (
        GoalCheckpoint,
        GoalEvidence,
        GoalIterationResult,
        GoalToolObservation,
        IterationReport,
    )
    from tldw_chatbook.Backup_Recovery.mobile_runs_goals import _plan_goal_rows
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    original_seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, identities):
        goal = db.goal_runs.get(identities["goal_id"])
        goal = db.goal_runs.set_provisioning(goal, status="ready")
        attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
        assert db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
        run = db.create_run(
            conversation_id=goal.conversation_id,
            agent_kind="primary",
            work_chain_id=goal.chain_id,
        )
        with db.transaction() as conn:
            conn.execute(
                "UPDATE goal_iterations SET run_id=? WHERE attempt_id=?",
                (run, attempt.id),
            )
        report = IterationReport(
            summary="Observed obs in body",
            evidence_ids=("obs",),
            completion_recommended=True,
        )
        result = GoalIterationResult(
            goal.id,
            attempt.ordinal,
            attempt.id,
            run,
            RunOutcome(status="done", steps=[], final_text=report.model_dump_json()),
            RunTerminationReason.DONE,
            observations=(
                GoalToolObservation(
                    "obs",
                    goal.id,
                    attempt.id,
                    run,
                    "fs_read",
                    "a" * 64,
                    "original body names obs",
                    True,
                ),
            ),
        )
        db.set_status(run, "done", result="retained result")
        saved = db.goal_runs.checkpoint(result)
        if legacy:
            with db.connection() as conn:
                source["checkpoint"] = dict(
                    conn.execute(
                        "SELECT * FROM goal_checkpoints WHERE attempt_id=?",
                        (attempt.id,),
                    ).fetchone()
                )
            source.update(
                goal=goal,
                run=run,
                attempt=attempt.id,
                saved=saved,
                evidence=db.goal_runs.evidence(goal.id),
            )

    def seeded(stores, **kwargs):
        return original_seed(stores, **kwargs, runs_seed=native)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(
        tmp_path, monkeypatch, lambda connection, purpose: None
    ) as reads:
        identities = reads.identity_plan()
        plan = _plan_goal_rows(reads, identities, _read_budget=_TraceReadBudget())
        assert not plan.blockers
        rows = {(r.table, r.original): dict(r.values) for r in plan.rows}
        original = source["checkpoint"]
        row = rows["goal_checkpoints", (source["attempt"],)]
        checkpoint = GoalCheckpoint.model_validate_json(row["payload_json"])
        assert row["payload_hash"] == original["payload_hash"]
        assert (
            checkpoint.artifact_digest == source["saved"].checkpoints[0].artifact_digest
        )
        assert checkpoint.decision.artifact_digest == checkpoint.artifact_digest
        assert checkpoint.id == identities.mapped("wake_attempt", source["attempt"])
        assert checkpoint.decision.checkpoint_id == checkpoint.id
        assert checkpoint.report.summary == "Observed obs in body"
        assert checkpoint.report.evidence_ids == (
            identities.mapped("goal_evidence", "obs"),
        )
        evidence_row = rows["goal_evidence", ("obs",)]
        evidence = GoalEvidence.model_validate_json(evidence_row["payload_json"])
        assert evidence.id == identities.mapped("goal_evidence", "obs")
        assert evidence.goal_id == identities.mapped("goal", source["goal"].id)
        assert evidence.attempt_id == checkpoint.id
        assert evidence.run_id == identities.mapped("run", source["run"])
        assert evidence.stdout == source["evidence"][0].stdout
        assert evidence.source_digest == source["evidence"][0].source_digest
        provenance = next(p for p in plan.provenance if p.table == "goal_checkpoints")
        assert provenance.original_hash == original["payload_hash"]
        assert provenance.mapped_hash == original["payload_hash"]
        assert provenance.artifact_digest == checkpoint.artifact_digest


def test_captured_goal_binding_requires_its_actual_workspace(tmp_path, monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery.mobile_runs_goals import _plan_goal_rows
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    original_seed = mobile_consolidation_fixture._seed
    source = {}

    def native(db, legacy, ids):
        goal = db.goal_runs.get(ids["goal_id"])
        request = goal.request.model_copy(
            update={
                "binding": goal.request.binding.model_copy(
                    update={"workspace_id": "workspace-default"}
                )
            }
        )
        added = db.goal_runs.create(request, launch_id="cross-owner-launch")
        ids["cross_conversation"] = added.conversation_id
        if legacy:
            source["goal"] = added.id

    def linked(db, legacy, ids):
        db.add_conversation(
            {
                "id": ids["cross_conversation"],
                "title": "Cross owner",
                "workspace_id": ids["workspace_id"],
            }
        )

    def seeded(stores, **kwargs):
        return original_seed(
            stores, **kwargs, runs_seed=native, linked_chat_seed=linked
        )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seeded)
    with content_reads(tmp_path, monkeypatch, lambda c, p: None) as reads:
        plan = _plan_goal_rows(
            reads, reads.identity_plan(), _read_budget=_TraceReadBudget()
        )
        assert any(
            b.table == "goal_runs" and b.original == (source["goal"],)
            for b in plan.blockers
        )
        assert not any(
            r.table == "goal_runs" and r.original == (source["goal"],)
            for r in plan.rows
        )
