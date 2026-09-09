"""Observed progress and human judgment are separate from model assertions."""

from Tests.Agents.test_goal_iteration_report import iteration, report


def test_model_success_without_resolved_evidence_cannot_complete():
    m = iteration()
    decision = m.evaluate_iteration(
        m.parse_iteration_report(
            report(completion_recommended=True, evidence_ids=["foreign"])
        ),
        (),
        None,
        (m.GoalCriterion(id="check", verifier_id="check"),),
    )
    assert decision.action == "continue"
    assert not any(c.satisfied for c in decision.checks)
    assert decision.evidence_errors == ("foreign",)


def test_reworded_summary_does_not_reset_two_no_progress_pause():
    m = iteration()
    first = m.parse_iteration_report(report(summary="one"))
    d = m.evaluate_iteration(first, (), None, ())
    previous = m.GoalCheckpoint(ordinal=1, report=first, decision=d)
    second = m.evaluate_iteration(
        m.parse_iteration_report(report(summary="new words")), (), previous, ()
    )
    assert second.action == "pause"
    assert second.no_progress_count == 2


import hashlib
import json
from pathlib import Path

import pytest

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from Tests.Chat.test_goal_cli_verification import trusted_skill
from tldw_chatbook.Agents.goal_models import VerificationSpec


async def run_verifier(
    stores,
    monkeypatch,
    tmp_path,
    *,
    human=False,
    mode="pass",
    completion=True,
    repeats=1,
):
    runs, persistence, registry, req = stores
    fixture = Path(req.binding.locator) / "fixture.txt"
    fixture.write_text("valid")
    script = "print('typed check')\n"
    if mode in ("pass_then_fail", "pass_fail_same_turn"):
        marker = tmp_path / "check-counter"
        script += f"from pathlib import Path\nm=Path({str(marker)!r})\nfailed=m.exists()\nm.write_text('checked')\nraise SystemExit(7 if failed else 0)\n"
    if mode == "during":
        script += f"from pathlib import Path\nPath({str(fixture)!r}).write_text('changed during check')\n"
    if mode == "fail":
        script += "raise SystemExit(7)\n"
    if mode == "artifact":
        script += "from pathlib import Path\nPath('proof.txt').write_text('retained artifact')\n"
    if mode == "oversized":
        script += "print('x'*200000)\n"
    scope, path, trust = trusted_skill(tmp_path, script)
    verifier = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        input_paths=("fixture.txt",),
    )
    req = req.model_copy(
        update={"verifiers": (verifier,), "human_review_required": human}
    )
    count = 0
    observed_ids = []

    def provider(**kwargs):
        nonlocal count
        count += 1
        if mode == "pass_then_fail" and count >= 5:
            message = {
                "content": report(
                    evidence_ids=observed_ids[:1], completion_recommended=True
                )
            }
        elif mode == "agent_after" and count == 2:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "edit",
                        "type": "function",
                        "function": {
                            "name": "fs_edit",
                            "arguments": json.dumps(
                                {
                                    "path": "fixture.txt",
                                    "old_string": "valid",
                                    "new_string": "agent edit after check",
                                }
                            ),
                        },
                    }
                ],
            }
        elif count % 2 == 1 and not (mode == "agent_after" and count > 1):
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "check",
                        "type": "function",
                        "function": {
                            "name": "run_skill_script",
                            "arguments": json.dumps(
                                {
                                    "skill_name": "verifier",
                                    "script_path": "scripts/check.py",
                                    "args": [],
                                }
                            ),
                        },
                    }
                ],
            }
        else:
            import re

            ids = re.findall(r"goal_evidence_id: ([a-f0-9]{32})", str(kwargs))
            observed_ids.extend(ids)
            if mode == "after":
                fixture.write_text("changed after check")
            if mode == "verifier_changed":
                path.write_text("print('different checker')")
            message = {
                "content": report(
                    evidence_ids=[]
                    if mode in ("repeat", "pass_then_fail")
                    else ids[:1],
                    completion_recommended=completion,
                )
            }
        if mode == "pass_fail_same_turn" and count == 1:
            from copy import deepcopy

            second = deepcopy(message["tool_calls"][0])
            second["id"] = "second-check"
            message["tool_calls"].append(second)
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    rig = build_goal_rig((runs, persistence, registry, req), monkeypatch, provider)
    goal, _, _, controller, coordinator, gateway, calls = rig
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    if mode == "agent_after":
        from types import SimpleNamespace

        import tldw_chatbook.Chat.console_chat_controller as controller_module
        from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService

        controller.app = SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
        setting = controller_module.get_cli_setting
        monkeypatch.setattr(
            controller_module,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                True
                if (section, key) == ("console", "local_tools_enabled")
                else setting(section, key, default)
            ),
        )
    try:
        for index in range(repeats):
            result = await coordinator.dispatch_once(goal.id)
            if index < repeats - 1:
                coordinator.service.checkpoint(result)
        assert observed_ids, "exact runtime evidence ID must arrive before model report"
        return coordinator, goal, result, fixture, path, calls
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["pass", "after", "during", "fail", "oversized", "verifier_changed"]
)
async def test_real_cli_checks_require_stable_inputs_and_exact_verifier(
    stores, monkeypatch, tmp_path, mode
):
    coordinator, _goal, result, _fixture, _path, calls = await run_verifier(
        stores, monkeypatch, tmp_path, mode=mode
    )
    snapshot = coordinator.service.checkpoint(result)
    assert snapshot.status == ("completed" if mode == "pass" else "ready"), (
        snapshot.checkpoints[-1].model_dump_json()
    )
    assert snapshot.checkpoints[-1].decision.checks[0].satisfied is (mode == "pass")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_human_review_rechecks_manual_edits_and_retained_copy_survives_pruning(
    stores, monkeypatch, tmp_path
):
    coordinator, goal, result, fixture, path, _ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True
    )
    snapshot = coordinator.service.checkpoint(result)
    assert snapshot.status == "awaiting_result_review"
    copied = coordinator.service.db.goal_runs.evidence(goal.id)
    assert copied[0].stdout == "typed check\n"
    fixture.write_text("manual edit before review")
    assert not coordinator.service.completion_check(goal.id).checks[0].satisfied
    path.unlink()
    assert (
        coordinator.service.db.goal_runs.evidence(goal.id)[0].stdout == "typed check\n"
    )
    assert not coordinator.service.completion_check(goal.id).checks[0].satisfied


@pytest.mark.asyncio
async def test_unknown_effect_saves_evidence_but_retains_recovery_attempt(
    stores, monkeypatch, tmp_path
):
    from dataclasses import replace

    from tldw_chatbook.Agents.agent_models import RunTerminationReason

    coordinator, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path)
    result = replace(result, termination_reason=RunTerminationReason.UNKNOWN_EFFECT)
    saved = coordinator.service.checkpoint(result)
    assert saved.status == "recovery_required"
    assert coordinator.service.db.goal_runs.evidence(goal.id)[0].passed
    with coordinator.service.db.connection() as conn:
        row = conn.execute(
            "SELECT state,completed_at FROM automatic_wake_attempts WHERE id=?",
            (result.attempt_id,),
        ).fetchone()
        assert tuple(row) == ("review_required", None)
    with pytest.raises(ValueError, match="settled"):
        coordinator.service.remove_payloads(goal.id)


@pytest.mark.asyncio
async def test_foreign_references_and_reused_unchanged_observation_do_not_complete_or_reset_progress(
    stores, monkeypatch, tmp_path
):
    from dataclasses import replace

    coordinator, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path)
    foreign = replace(
        result,
        outcome=replace(
            result.outcome,
            final_text=report(
                evidence_ids=["another_conversation_record"],
                completion_recommended=True,
            ),
        ),
    )
    saved = coordinator.service.checkpoint(foreign)
    decision = saved.checkpoints[-1].decision
    assert decision.action == "continue"
    assert not decision.checks[0].satisfied
    evidence = coordinator.service.db.goal_runs.evidence(goal.id)
    m = iteration()
    repeated = m.evaluate_iteration(
        saved.checkpoints[-1].report,
        evidence,
        saved.checkpoints[-1],
        m.goal_criteria(saved),
    )
    assert repeated.no_progress_count == 1
    previous = m.GoalCheckpoint(
        ordinal=2, report=saved.checkpoints[-1].report, decision=repeated
    )
    again = m.evaluate_iteration(
        m.parse_iteration_report(report(summary="changed wording")),
        evidence,
        previous,
        m.goal_criteria(saved),
    )
    assert again.action == "pause" and again.no_progress_count == 2


def test_manifest_refuses_escape_missing_oversized_and_changed_during_read(
    tmp_path, monkeypatch
):
    m = iteration()
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    spec = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path="/tmp/trusted.py",
        verifier_sha256="a" * 64,
        input_paths=("input",),
    )
    assert m.capture_manifest(str(root), spec) is None
    (root / "input").symlink_to(outside)
    assert m.capture_manifest(str(root), spec) is None
    (root / "input").unlink()
    (root / "input").write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    assert m.capture_manifest(str(root), spec) is None
    (root / "input").write_text("valid")
    assert m.capture_manifest(str(root), spec, seconds=0) is None
    before = m.capture_manifest(str(root), spec)
    assert before
    (root / "input").write_text("valid")
    assert m.capture_manifest(str(root), spec) != before


def test_manifest_does_not_open_replaced_symlink_leaf_or_fifo(tmp_path, monkeypatch):
    import os

    m = iteration()
    root = tmp_path / "bound"
    root.mkdir()
    source = root / "input"
    source.write_text("original")
    outside = tmp_path / "outside"
    outside.write_text("private")
    spec = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path="/tmp/trusted.py",
        verifier_sha256="a" * 64,
        input_paths=("input",),
    )
    original = os.open

    def swap(path, *args, **kwargs):
        if path == "input":
            source.unlink()
            source.symlink_to(outside)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    assert m.capture_manifest(str(root), spec) is None
    assert source.is_symlink()
    monkeypatch.setattr(os, "open", original)
    source.unlink()
    os.mkfifo(source)
    assert m.capture_manifest(str(root), spec) is None


@pytest.mark.asyncio
async def test_repeated_actual_cli_with_new_ids_is_not_progress(
    stores, monkeypatch, tmp_path
):
    coordinator, goal, result, _, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, human=True, mode="repeat", repeats=3
    )
    saved = coordinator.service.checkpoint(result)
    assert len(calls) == 6
    evidence = coordinator.service.db.goal_runs.evidence(goal.id)
    assert len(evidence) == 3 and len({e.id for e in evidence}) == 3
    assert len({e.source_digest for e in evidence}) == 1
    assert (
        saved.status == "paused"
        and saved.checkpoints[-1].decision.no_progress_count == 2
    )


@pytest.mark.asyncio
async def test_completion_check_cannot_review_over_an_unsettled_successor(
    stores, monkeypatch, tmp_path
):
    coordinator, goal, result, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True, mode="repeat"
    )
    coordinator.service.checkpoint(result)
    attempt = coordinator.service.db.automatic_work.prepare_goal_iteration(
        goal.id, owner_id=coordinator._owner_id
    )
    coordinator.service.db.automatic_work.accept_goal_iteration(
        attempt.id, owner_id=coordinator._owner_id
    )
    assert coordinator.service.completion_check(goal.id).action == "recovery_required"


@pytest.mark.asyncio
async def test_review_identity_is_runtime_checkpoint_and_checked_version(
    stores, monkeypatch, tmp_path
):
    coordinator, goal, result, fixture, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True
    )
    saved = coordinator.service.checkpoint(result)
    checkpoint = saved.checkpoints[-1]
    assert checkpoint.id == result.attempt_id
    assert len(checkpoint.artifact_digest) == 64
    decision = coordinator.service.completion_check(
        goal.id, checkpoint_id=checkpoint.id, artifact_digest=checkpoint.artifact_digest
    )
    assert (
        decision.checkpoint_id == checkpoint.id
        and decision.artifact_digest == checkpoint.artifact_digest
    )
    assert decision.action == "awaiting_result_review"
    with pytest.raises(ValueError, match="stale_result_review"):
        coordinator.service.completion_check(
            goal.id, checkpoint_id="old", artifact_digest=checkpoint.artifact_digest
        )
    with pytest.raises(ValueError, match="stale_result_review"):
        coordinator.service.completion_check(
            goal.id, checkpoint_id=checkpoint.id, artifact_digest="0" * 64
        )
    fixture.write_text("changed after result review opened")
    decision = coordinator.service.completion_check(
        goal.id, checkpoint_id=checkpoint.id, artifact_digest=checkpoint.artifact_digest
    )
    assert decision.action not in ("completed", "awaiting_result_review")
    assert not decision.checks[0].satisfied


@pytest.mark.asyncio
async def test_three_real_failed_checks_pause_with_failure_counter(
    stores, monkeypatch, tmp_path
):
    coordinator, _goal, result, _, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, human=True, mode="fail", repeats=3
    )
    saved = coordinator.service.checkpoint(result)
    assert saved.status == "paused"
    assert saved.checkpoints[-1].decision.failed_count == 3
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_actual_agent_edit_after_cli_invalidates_previous_check(
    stores, monkeypatch, tmp_path
):
    coordinator, _goal, result, fixture, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, mode="agent_after"
    )
    assert fixture.read_text() == "agent edit after check"
    saved = coordinator.service.checkpoint(result)
    assert len(calls) == 3
    assert len(result.tool_records) == 1 and len(result.observations) == 1
    assert (
        saved.status != "completed"
        and not saved.checkpoints[-1].decision.checks[0].satisfied
    )


@pytest.mark.asyncio
async def test_private_evidence_survives_actual_script_artifact_pruning(
    stores, monkeypatch, tmp_path
):
    import shutil

    coordinator, goal, result, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, mode="artifact"
    )
    saved = coordinator.service.checkpoint(result)
    assert saved.status == "completed"
    output = Path(result.tool_records[0].result.output_dir)
    assert (output / "proof.txt").read_text() == "retained artifact"
    shutil.rmtree(output)
    retained = coordinator.service.db.goal_runs.evidence(goal.id)
    assert retained[0].stdout == "typed check\n"
    assert coordinator.service.completion_check(goal.id).action == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["arguments", "run", "goal"])
async def test_runtime_result_requires_exact_original_run_goal_and_arguments(
    stores, monkeypatch, tmp_path, change
):
    from dataclasses import replace

    coordinator, goal, result, *_ = await run_verifier(stores, monkeypatch, tmp_path)
    original = result.tool_records[0]
    values = {
        "arguments": {"arguments": ("other-target",)},
        "run": {"run_id": "another-run"},
        "goal": {"goal_id": "another-goal"},
    }
    changed = replace(
        original, invocation=replace(original.invocation, **values[change])
    )
    saved = coordinator.service.checkpoint(replace(result, tool_records=(changed,)))
    assert saved.status != "completed"
    assert not saved.checkpoints[-1].decision.checks[0].satisfied
    assert coordinator.service.db.goal_runs.evidence(goal.id) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,repeats,model_calls",
    [("pass_then_fail", 3, 5), ("pass_fail_same_turn", 1, 2)],
)
async def test_newer_real_failed_check_supersedes_retained_pass(
    stores, monkeypatch, tmp_path, mode, repeats, model_calls
):
    coordinator, goal, result, _, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, mode=mode, repeats=repeats
    )
    saved = coordinator.service.checkpoint(result)
    evidence = coordinator.service.db.goal_runs.evidence(goal.id)
    assert len(calls) == model_calls
    assert len(evidence) == 2
    assert [item.passed for item in evidence] == [True, False]
    assert evidence[0].checked_manifest == evidence[1].checked_manifest
    assert saved.checkpoints[-1].report.evidence_ids == (evidence[0].id,)
    assert not saved.checkpoints[-1].decision.checks[0].satisfied
    assert saved.status != "completed"


def test_equivalent_unicode_and_newline_drafts_do_not_reset_progress():
    m = iteration()
    initial = m.parse_iteration_report(report(candidate_draft="é\n"))
    first = m.evaluate_iteration(initial, (), None, ())
    previous = m.GoalCheckpoint(ordinal=1, report=initial, decision=first)
    equivalent = m.parse_iteration_report(report(candidate_draft="e\u0301\r\n"))
    assert m.evaluate_iteration(equivalent, (), previous, ()).no_progress_count == 1
