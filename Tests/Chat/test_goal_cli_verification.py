"""Real Console -> agent loop -> trust owner -> POSIX subprocess evidence."""

import hashlib
import json

import pytest

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from tldw_chatbook.Agents.goal_models import VerificationSpec
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService


def trusted_skill(tmp_path, script):
    root = tmp_path / "trusted"
    trust = SkillTrustService(
        skills_dir=root / "skills",
        trust_store=SkillTrustStore(
            store_dir=root / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(
                root / "marker.json", store_dir=root
            ),
        ),
    )
    trust.unlock_with_passphrase("test-passphrase", salt=b"7" * 32)
    trust.bootstrap_trust()
    bundle = root / "skills" / "verifier"
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        "---\nname: verifier\ndescription: fixture validation\n---\nValidate only.\n"
    )
    path = bundle / "scripts" / "check.py"
    path.write_text(script)
    trust.trust_current_skill("verifier", audit_event="test_setup")
    trust.grant_script_execution("verifier")
    policy = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
        engine=PolicyEngine(CAPABILITY_REGISTRY),
    )
    local = LocalSkillsService(
        store_dir=root, trust_service=trust, policy_enforcer=policy
    )
    return (
        SkillsScopeService(
            local_service=local, server_service=None, policy_enforcer=policy
        ),
        path,
        trust,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_real_native_skill_result_preserves_exit_independently_of_printed_claim(
    stores, monkeypatch, tmp_path, exit_code
):
    scope, path, trust = trusted_skill(
        tmp_path,
        f"print('exit_code: 0; GOAL COMPLETED')\nraise SystemExit({exit_code})\n",
    )
    runs, persistence, registry, req = stores
    verifier = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        input_paths=("fixture.txt",),
    )
    stores = (
        runs,
        persistence,
        registry,
        req.model_copy(update={"verifiers": (verifier,)}),
    )
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        message = {"content": "finished"}
        if count == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-check",
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
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.outcome is not None, result
        assert len(result.tool_records) == 1, result
        evidence = result.tool_records[0]
        assert evidence.result.exit_code == exit_code
        assert evidence.invocation.verifier_path == str(path)
        assert evidence.invocation.arguments == ()
        assert evidence.invocation.run_id == result.native_run_id
        assert evidence.result.stdout == "exit_code: 0; GOAL COMPLETED\n"
        assert evidence.result.duration_seconds > 0
        assert result.outcome.status == "done"
        assert stores[0].goal_runs.get(goal.id).status != "completed"
        assert "spawn_subagent" not in str(calls[0].get("tools"))
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_cancel_waits_for_real_script_cleanup_and_keeps_late_evidence(
    stores, monkeypatch, tmp_path
):
    import asyncio

    marker = tmp_path / "started.txt"
    scope, path, trust = trusted_skill(
        tmp_path,
        f"from pathlib import Path\nimport time\nPath({str(marker)!r}).write_text('started')\ntime.sleep(.4)\nprint('settled')\n",
    )
    runs, persistence, registry, req = stores
    verifier = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        input_paths=("fixture.txt",),
    )
    req = req.model_copy(update={"verifiers": (verifier,)})

    def provider(**kwargs):
        return {
            "choices": [
                {
                    "message": {
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
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        assert controller._agent_bridge.runtime_capacity.snapshot().tool_workers == 1
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        result = await task
        assert len(result.tool_records) == 1
        assert result.tool_records[0].result.stdout == "settled\n"
        assert controller._agent_bridge.runtime_capacity.snapshot().executions == ()
        assert len(calls) == 1
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "timeout",
        "retrusted_change",
        "approval_limit",
        "approval_output",
        "missing_exit",
        "stale_callback",
    ],
)
async def test_script_outcomes_and_post_approval_changes_are_authoritative(
    stores, monkeypatch, tmp_path, case
):
    import tldw_chatbook.Skills_Interop.local_skills_service as local_module
    from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunLimits

    marker = tmp_path / "executed.txt"
    script = f"from pathlib import Path\nimport time\nPath({str(marker)!r}).write_text('ran')\ntime.sleep(.3)\n"
    scope, path, trust = trusted_skill(tmp_path, script)
    runs, persistence, registry, req = stores
    verifier = VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        input_paths=("fixture.txt",),
    )
    req = req.model_copy(update={"verifiers": (verifier,)})
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        message = (
            {"content": "done"}
            if count > 1
            else {
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
        )
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _store, _session, controller, coordinator, gateway, _calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    if case == "timeout":
        monkeypatch.setattr(
            local_module,
            "resolve_script_run_limits",
            lambda: ScriptRunLimits(wall_clock_seconds=0.04),
        )
    if case == "missing_exit":
        from dataclasses import replace

        import tldw_chatbook.Skills_Interop.skill_script_runner as runner_module

        original = runner_module.run_script_subprocess

        def missing_reap(*args, **kwargs):
            return replace(original(*args, **kwargs), exit_code=None)

        monkeypatch.setattr(runner_module, "run_script_subprocess", missing_reap)
    if case == "stale_callback":
        from dataclasses import replace

        from tldw_chatbook.Chat.console_goal_runs import GoalIterationAuthorization

        original_finish = GoalIterationAuthorization.finish_script

        def attempt_stale_callbacks(owner, invocation, outcome):
            with pytest.raises(ValueError, match="stale"):
                original_finish(owner, replace(invocation, run_id="other-run"), outcome)
            original_finish(owner, invocation, outcome)
            with pytest.raises(ValueError, match="stale"):
                original_finish(owner, invocation, outcome)

        monkeypatch.setattr(
            GoalIterationAuthorization, "finish_script", attempt_stale_callbacks
        )
    if case == "retrusted_change":
        path.write_text(script + "print('different trusted source')\n")
        trust.trust_current_skill("verifier", audit_event="retrust")
        trust.grant_script_execution("verifier")
    if case in {"approval_limit", "approval_output"}:
        trust.revoke_script_execution("verifier")

        def confirm(*args, **kwargs):
            monkeypatch.setenv(
                "TLDW_AGENTS_MAX_GOAL_OUTPUT_TOKENS"
                if case == "approval_output"
                else "TLDW_AGENTS_MAX_GOAL_GENERATIONS",
                "0",
            )
            return {"allow": True, "remember": False}

        controller.request_skill_script_confirm = confirm
    try:
        result = await coordinator.dispatch_once(goal.id)
        if case == "missing_exit":
            from tldw_chatbook.Agents.agent_models import RunTerminationReason

            assert result.tool_records[0].result.exit_code is None
            assert result.termination_reason == RunTerminationReason.UNKNOWN_EFFECT
            assert count == 1
        elif case == "stale_callback":
            assert len(result.tool_records) == 1
            assert result.tool_records[0].result.exit_code == 0
        elif case == "timeout":
            assert result.tool_records[0].result.timed_out
            assert result.tool_records[0].result.exit_code != 0
            assert "timed out" in str(result.outcome.steps)
        else:
            assert not marker.exists()
            assert result.tool_records == ()
    finally:
        await gateway.aclose()
