"""Exact CLI identity selects one approved input scope, including reused scripts."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from Tests.Agents.test_goal_iteration_report import report
from Tests.Agents.test_goal_models import request
from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_cli_verification import trusted_skill
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Agents.goal_models import (
    GoalRequest,
    GoalToolScope,
    VerificationSpec,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

stores = _stores_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("both", [False, True])
async def test_actual_reused_script_selects_second_input_scope(
    stores, monkeypatch, tmp_path, both
):
    db, persistence, registry, req = stores
    root = Path(req.binding.locator)
    first, second = root / "first.txt", root / "second.txt"
    first.write_text("first")
    second.write_text("second")
    scope, script, trust = trusted_skill(
        tmp_path,
        "from pathlib import Path\nimport sys\nprint(Path(sys.argv[1]).read_text())\n",
    )
    verifiers = tuple(
        VerificationSpec(
            id=label,
            executor_tool_id="run_skill_script",
            verifier_path=str(script),
            verifier_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            skill_trust_ref=trust.current_fingerprint_digest("verifier"),
            arguments=(str(target),),
            input_paths=(target.name,),
        )
        for label, target in (("first", first), ("second", second))
    )
    req = GoalRequest.model_validate_json(
        req.model_copy(
            update={"verifiers": verifiers, "human_review_required": False}
        ).canonical_json()
    )
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        if count == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": v.id,
                        "type": "function",
                        "function": {
                            "name": "run_skill_script",
                            "arguments": json.dumps(
                                {
                                    "skill_name": "verifier",
                                    "script_path": "scripts/check.py",
                                    "args": list(v.arguments),
                                }
                            ),
                        },
                    }
                    for v in (verifiers if both else verifiers[1:])
                ],
            }
        else:
            ids = re.findall(r"goal_evidence_id: ([a-f0-9]{32})", str(kwargs))
            message = {
                "content": report(
                    evidence_ids=list(dict.fromkeys(ids)), completion_recommended=True
                )
            }
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _, _, controller, coordinator, gateway, calls = build_goal_rig(
        (db, persistence, registry, req), monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    try:
        replay = coordinator.service.create(goal.request, launch_id=goal.launch_id)
        assert replay.payload_hash == goal.payload_hash and replay.id == goal.id
        result = await coordinator.dispatch_once(goal.id)
        saved = coordinator.service.checkpoint(result)
        checks = saved.checkpoints[-1].decision.checks
        assert checks[1].satisfied, checks
        assert checks[0].satisfied is both
        assert saved.status == ("completed" if both else "ready")
        assert result.tool_records[-1].invocation.verifier_id == "second"
        copied = db.goal_runs.evidence(goal.id)[-1]
        assert copied.verifier_id == "second" and copied.passed and copied.fresh
        assert copied.stdout == "second\n"
        assert len(calls) == 2
    finally:
        await gateway.aclose()


def ambiguous_request(alias):
    first = VerificationSpec(
        id="first",
        executor_tool_id="run_skill_script",
        verifier_path="/tmp/trusted-verifier.py",
        verifier_sha256="a" * 64,
        arguments=("check",),
        input_paths=("first.txt",),
    )
    second = first.model_copy(
        update={
            "id": "second",
            "input_paths": ("second.txt",),
            "executor_tool_id": "runtime:run_skill_script"
            if alias
            else "run_skill_script",
        }
    )
    return request(
        verifiers=(first, second),
        tool_scope=GoalToolScope(
            runtime_tools=("run_skill_script", "runtime:run_skill_script")
        ),
    )


@pytest.mark.parametrize("alias", [False, True])
def test_ambiguous_new_launch_refused_after_structural_roundtrip(tmp_path, alias):
    db = AgentRunsDB(tmp_path / "runs.db")
    req = ambiguous_request(alias)
    try:
        hydrated = GoalRequest.model_validate_json(req.canonical_json())
        assert hydrated == req  # Older stored requests must stay inspectable.
        with pytest.raises(ValueError, match="ambiguous_verifier_invocation"):
            db.goal_runs.create(hydrated, launch_id="ambiguous")
        with db.connection() as conn:
            assert conn.execute("SELECT count(*) FROM goal_runs").fetchone()[0] == 0
            assert (
                conn.execute("SELECT count(*) FROM automatic_work_chains").fetchone()[0]
                == 0
            )
    finally:
        db.close()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_legacy_ambiguity_reopens_and_replays_but_cannot_prepare_or_accept(
    tmp_path, monkeypatch, prepared, alias
):
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path)
    req = ambiguous_request(alias)
    unique = req.model_copy(update={"verifiers": req.verifiers[:1]})
    goal = db.goal_runs.create(unique, launch_id="legacy")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = (
        db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
        if prepared
        else None
    )
    payload = req.canonical_json()
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER goal_launch_immutable")
        conn.execute(
            "UPDATE goal_runs SET request_json=?,payload_hash=? WHERE id=?",
            (payload, digest, goal.id),
        )
    db.close()
    reopened = AgentRunsDB(path)
    try:
        hydrated = reopened.goal_runs.get(goal.id)
        replay = reopened.goal_runs.create(
            GoalRequest.model_validate_json(payload), launch_id="legacy"
        )
        assert (
            hydrated.request == req
            and replay.id == goal.id
            and replay.payload_hash == digest
        )
        with pytest.raises(
            AutomaticWorkRefused, match="ambiguous_verifier_invocation"
        ) as caught:
            if prepared:
                reopened.automatic_work.accept_goal_iteration(
                    attempt.id, owner_id="owner"
                )
            else:
                reopened.automatic_work.prepare_goal_iteration(
                    goal.id, owner_id="owner"
                )
        assert caught.value.reason == "ambiguous_verifier_invocation"
        with reopened.connection() as conn:
            row = conn.execute(
                "SELECT request_json,payload_hash FROM goal_runs WHERE id=?", (goal.id,)
            ).fetchone()
            assert tuple(row) == (payload, digest)
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_wake_attempts WHERE state='accepted'"
                ).fetchone()[0]
                == 0
            )
        assert reopened.automatic_work.snapshot(goal.chain_id).used["generation"] == 0
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_legacy_unselected_invocation_keeps_checkpoint_hash_and_replay(
    stores, monkeypatch, tmp_path
):
    from dataclasses import asdict, replace

    from Tests.Agents.test_goal_progress import run_verifier

    coordinator, _goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path)
    old_record = replace(
        result.tool_records[0],
        invocation=replace(result.tool_records[0].invocation, verifier_id=None),
    )
    legacy = replace(result, tool_records=(old_record,))
    old_payload = asdict(legacy)
    for record in old_payload["tool_records"]:
        record["invocation"].pop("verifier_id")
    old_hash = hashlib.sha256(
        json.dumps(old_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    saved = coordinator.service.checkpoint(legacy)
    assert saved.status == "completed"
    with stores[0].connection() as conn:
        stored_hash = conn.execute(
            "SELECT payload_hash FROM goal_checkpoints WHERE attempt_id=?",
            (result.attempt_id,),
        ).fetchone()[0]
    assert stored_hash == old_hash
    assert coordinator.service.checkpoint(legacy).revision == saved.revision
    with pytest.raises(ValueError, match="checkpoint_conflict"):
        coordinator.service.checkpoint(result)


@pytest.mark.asyncio
async def test_explicit_wrong_spec_id_cannot_certify_actual_cli_result(
    stores, monkeypatch, tmp_path
):
    from dataclasses import replace

    from Tests.Agents.test_goal_progress import run_verifier

    coordinator, _goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path)
    wrong = replace(
        result.tool_records[0],
        invocation=replace(result.tool_records[0].invocation, verifier_id="other"),
    )
    saved = coordinator.service.checkpoint(replace(result, tool_records=(wrong,)))
    assert saved.status == "ready"
    assert not saved.checkpoints[-1].decision.checks[0].satisfied


def test_repeated_start_reconciles_exact_legacy_ambiguous_launch(stores):
    from tldw_chatbook.Agents.goal_run_service import GoalRunService

    db, persistence, _registry, req = stores
    verifiers = ambiguous_request(False).verifiers
    req = req.model_copy(update={"verifiers": verifiers[:1]})
    service = GoalRunService(db, persistence)
    goal = db.goal_runs.create(req, launch_id="legacy-service")
    assert goal.status == "starting"
    legacy = req.model_copy(update={"verifiers": verifiers})
    payload = legacy.canonical_json()
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER goal_launch_immutable")
        conn.execute(
            "UPDATE goal_runs SET request_json=?,payload_hash=? WHERE id=?",
            (payload, digest, goal.id),
        )
    replay = service.create(
        GoalRequest.model_validate_json(payload), launch_id=goal.launch_id
    )
    assert replay.id == goal.id and replay.payload_hash == digest
    assert replay.conversation_id == goal.conversation_id and replay.status == "ready"
    repeated = service.create(
        GoalRequest.model_validate_json(payload), launch_id=goal.launch_id
    )
    assert repeated.revision == replay.revision and repeated.payload_hash == digest
    with db.connection() as conn:
        assert (
            conn.execute(
                "SELECT request_json FROM goal_runs WHERE id=?", (goal.id,)
            ).fetchone()[0]
            == payload
        )
        assert conn.execute("SELECT count(*) FROM goal_iterations").fetchone()[0] == 0
