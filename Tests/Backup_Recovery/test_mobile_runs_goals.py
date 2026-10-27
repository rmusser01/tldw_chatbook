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


@pytest.mark.parametrize("kind", ["request", "checkpoint"])
def test_actual_goal_internal_serializers_are_admitted_on_one_consumed_budget(
    tmp_path, monkeypatch, kind
):
    import json
    import time
    from uuid import uuid4

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Agents.agent_models import RunOutcome, RunTerminationReason
    from tldw_chatbook.Agents.goal_models import (
        GoalIterationResult,
        GoalRequest,
        IterationReport,
    )
    from tldw_chatbook.Backup_Recovery.mobile_runs_goals import _plan_goal_rows
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )

    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    seed = mobile_consolidation_fixture._seed
    marker = "FIX1-INTERNAL-"

    def native(db, legacy, ids):
        goal = db.goal_runs.get(ids["goal_id"])
        if kind == "request":
            data = goal.request.model_dump(mode="json")
            data["objective"] = marker + "x" * (8192 - len(marker))
            data["criteria"] = "y" * 8192
            data["verifiers"] = [
                {
                    "id": "native-check",
                    "executor_tool_id": "run_skill_script",
                    "verifier_path": "/trusted/native-verifier.py",
                    "verifier_sha256": "a" * 64,
                    "input_paths": ["input.txt"],
                    "arguments": ["arg" + "x" * 4090] * 20,
                }
            ]
            # Native historical JSON may omit defaults; immutable launch rows are
            # inserted before preview, with exact native validation/hash/chain.
            for key in ("policy", "source_bindings", "human_review_required"):
                data.pop(key)
            data["tool_scope"].pop("mcp_bindings")
            raw = json.dumps(data)
            request = GoalRequest.model_validate_json(raw)
            goal_id, conversation = str(uuid4()), str(uuid4())
            chain = db.automatic_work.create_chain(
                conversation,
                root_submission_id="goal:" + goal_id,
                limits=request.policy.chain_limits(),
            )
            with db.transaction() as connection:
                connection.execute(
                    "INSERT INTO goal_runs (id,launch_id,payload_hash,request_json,conversation_id,chain_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        goal_id,
                        "large-native-request",
                        hashlib.sha256(raw.encode()).hexdigest(),
                        raw,
                        conversation,
                        chain,
                        time.time(),
                        time.time(),
                    ),
                )
            ids["extra_goal_conversation"] = conversation
        else:
            goal = db.goal_runs.set_provisioning(goal, status="ready")
            attempt = db.automatic_work.prepare_goal_iteration(
                goal.id, owner_id="owner"
            )
            assert db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
            run = db.create_run(
                conversation_id=goal.conversation_id,
                agent_kind="primary",
                work_chain_id=goal.chain_id,
            )
            with db.transaction() as connection:
                connection.execute(
                    "UPDATE goal_iterations SET run_id=? WHERE attempt_id=?",
                    (run, attempt.id),
                )
            report = IterationReport(summary=marker + "z" * 60000)
            db.set_status(run, "done", result="retained")
            db.goal_runs.checkpoint(
                GoalIterationResult(
                    goal.id,
                    attempt.ordinal,
                    attempt.id,
                    run,
                    RunOutcome(
                        status="done", steps=[], final_text=report.model_dump_json()
                    ),
                    RunTerminationReason.DONE,
                )
            )

    def linked(db, legacy, ids):
        if "extra_goal_conversation" in ids:
            db.add_conversation(
                {
                    "id": ids["extra_goal_conversation"],
                    "title": "Native large Goal",
                    "workspace_id": ids["workspace_id"],
                }
            )

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        lambda stores, **kw: seed(
            stores, **kw, runs_seed=native, linked_chat_seed=linked
        ),
    )
    with content_reads(tmp_path, monkeypatch, lambda c, p: None) as reads:
        identities = reads.identity_plan()
        budget = _TraceReadBudget()
        json_charge = _TraceReadBudget.json
        pending = []
        observations = []
        shortfalls = []
        method = (
            GoalRequest.canonical_json
            if kind == "request"
            else IterationReport.model_dump_json
        )

        def charged(self, raw):
            if (
                self is budget
                and marker in (raw.decode() if isinstance(raw, bytes) else raw)
                and not pending
            ):
                pending.append(self._remaining)
            return json_charge(self, raw)

        def internal(self, *args, **kwargs):
            text = (
                self.objective
                + self.criteria
                + "".join(
                    argument
                    for verifier in self.verifiers
                    for argument in verifier.arguments
                )
                if kind == "request"
                else self.summary
            )
            if marker in text and pending:
                consumed = pending.pop() - budget._remaining
                # Even one model string clone, serialized Unicode and UTF-8 output
                # require 4+4+1 units per ASCII character before the native call.
                if consumed < 9 * len(text):
                    shortfalls.append((consumed, 9 * len(text)))
                observations.append(budget._remaining)
            return method(self, *args, **kwargs)

        monkeypatch.setattr(_TraceReadBudget, "json", charged)
        monkeypatch.setattr(
            GoalRequest if kind == "request" else IterationReport,
            "canonical_json" if kind == "request" else "model_dump_json",
            internal,
        )
        # Real captured transformations consume this one original allowance.
        # Neither a replacement budget nor an artificial smaller cap is injected.
        with pytest.raises(_TraceReadLimit):
            for _ in range(256):
                plan = _plan_goal_rows(reads, identities, _read_budget=budget)
                assert any(
                    marker
                    in dict(row.values).get(
                        "request_json" if kind == "request" else "payload_json", ""
                    )
                    for row in plan.rows
                ), "native valid Goal must survive admitted validation"
                assert not shortfalls, (
                    "native internal serialization entered before admission"
                )
        assert not shortfalls, "native internal serialization entered before admission"
        assert observations and min(observations) < 4 * 1024 * 1024
    assert not reads.active and not reads.roots


def test_goal_remap_json_admits_native_escape_width_before_encoding(monkeypatch):
    from tldw_chatbook.Agents.goal_models import GoalCheck
    from tldw_chatbook.Backup_Recovery import mobile_runs_goal_budget
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    data = GoalCheck(criterion_id="\x7f" * 50_000).model_dump(mode="json")
    codec = mobile_runs_goal_budget.json.dumps
    expected = codec(data)
    budget = _TraceReadBudget()
    before = budget._remaining
    observations = []

    def encode(value):
        observations.append(before - budget._remaining)
        return codec(value)

    monkeypatch.setattr(mobile_runs_goal_budget.json, "dumps", encode)
    assert mobile_runs_goal_budget._goal_json(data, budget) == expected
    assert observations and observations[0] >= 64 + 4 * len(expected)
