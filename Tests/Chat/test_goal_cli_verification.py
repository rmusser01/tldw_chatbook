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
@pytest.mark.parametrize("scheduled", [False, True])
async def test_cancel_waits_for_real_script_cleanup_and_keeps_late_evidence(
    stores, monkeypatch, tmp_path, scheduled
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
    task = (
        coordinator.start(goal.id)
        if scheduled
        else asyncio.create_task(coordinator.dispatch_once(goal.id))
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        assert controller._agent_bridge.runtime_capacity.snapshot().tool_workers == 1
        if scheduled:
            coordinator.service.stop(goal.id)
        else:
            task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        result = await task
        if scheduled:
            assert result.status == "stopped"
            evidence = coordinator.service.db.goal_runs.evidence(goal.id)
            assert len(evidence) == 1 and evidence[0].stdout == "settled\n"
        else:
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
        "script_denied",
        "policy_denied",
        "trust_changed",
        "approval_trust_changed",
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
    if case == "script_denied":
        trust.revoke_script_execution("verifier")
        controller.request_skill_script_confirm = lambda *a, **k: {"allow": False}
    if case == "policy_denied":
        from tldw_chatbook.runtime_policy.types import PolicyDeniedError

        def denied():
            raise PolicyDeniedError(
                action_id="skills.scripts.execute.local",
                reason_code="denied",
                user_message="Script policy denied",
                effective_source="local",
                authority_owner="test",
            )

        monkeypatch.setattr(scope, "enforce_run_script", denied)
    if case == "approval_trust_changed":
        trust.revoke_script_execution("verifier")

        def change_after_approval(*args, **kwargs):
            path.write_text(script + "# changed during approval\n")
            return {"allow": True}

        controller.request_skill_script_confirm = change_after_approval
    if case == "trust_changed":
        path.write_text(script + "# changed without renewed trust\n")
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
            if case in {
                "script_denied",
                "policy_denied",
                "trust_changed",
                "approval_trust_changed",
            }:
                from tldw_chatbook.Agents.agent_models import RunTerminationReason

                expected = (
                    RunTerminationReason.AUTHORITY_CHANGED
                    if case in {"trust_changed", "approval_trust_changed"}
                    else RunTerminationReason.PERMISSION_REFUSED
                )
                assert result.termination_reason == expected
                if case == "trust_changed":
                    # Mandatory invocation projection refuses stale trust before
                    # any model work; later approval changes still use the native gate.
                    assert result.outcome is None
                    assert count == 0
                else:
                    assert result.outcome.status == "stuck"
                    assert count == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["user_denied", "policy_denied", "execution_failed"])
async def test_native_install_refusal_is_typed_and_denial_never_fetches(
    stores, monkeypatch, tmp_path, case
):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as remote
    from tldw_chatbook.Agents.goal_models import GoalToolScope
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError

    scope, _, _ = trusted_skill(tmp_path, 'print("unused")\n')
    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={"tool_scope": GoalToolScope(runtime_tools=("install_skill",))}
    )
    calls = []
    fetches = []

    async def fetch(*args, **kwargs):
        fetches.append(args)
        raise RuntimeError("fetch outcome unavailable")

    monkeypatch.setattr(remote, "fetch_zip_bytes", fetch)
    if case == "policy_denied":

        def denied():
            raise PolicyDeniedError(
                action_id="skills.install.remote",
                reason_code="denied",
                user_message="Install policy denied",
                effective_source="local",
                authority_owner="test",
            )

        monkeypatch.setattr(scope, "enforce_install_remote", denied)

    def provider(**kwargs):
        calls.append(kwargs)
        message = (
            {"content": "done"}
            if len(calls) > 1
            else {
                "content": None,
                "tool_calls": [
                    {
                        "id": "install",
                        "type": "function",
                        "function": {
                            "name": "install_skill",
                            "arguments": json.dumps({"url": "https://github.com/o/r"}),
                        },
                    }
                ],
            }
        )
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _, _, controller, coordinator, gateway, _ = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.request_skill_install_confirm = lambda *a, **k: case != "user_denied"
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.termination_reason.value == (
            "unknown_effect" if case == "execution_failed" else "permission_refused"
        )
        assert result.outcome.status == "stuck"
        assert len(calls) == 1
        assert len(fetches) == (1 if case == "execution_failed" else 0)
        assert controller._agent_bridge.runtime_capacity.snapshot().executions == ()
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_two_increment_real_cli_repairs_file_with_model_visible_evidence(
    stores, monkeypatch, tmp_path
):
    """POSIX qualification: real subprocess, real approved fs_edit, actual native loop."""
    import difflib
    import re
    from pathlib import Path
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    from Tests.Agents.test_goal_iteration_report import report
    from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService

    runs, persistence, registry, req = stores
    project = Path(req.binding.locator)
    artifact = project / "fixture.txt"
    artifact.write_text("invalid\n")
    sentinel = tmp_path / "external-sentinel.txt"
    sentinel.write_text("must remain unchanged")
    original = artifact.read_text()
    script = """from pathlib import Path
import sys
project = Path(sys.argv[1])
value = (project / 'fixture.txt').read_text()
print('checked fixture.txt: ' + value.strip())
print('validation provenance: trusted check.py', file=sys.stderr)
raise SystemExit(0 if value == 'valid\\n' else 7)
"""
    scope, path, trust = trusted_skill(tmp_path, script)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verifier = VerificationSpec(
        id="fixture-check",
        executor_tool_id="run_skill_script",
        verifier_path=str(path),
        verifier_sha256=digest,
        skill_trust_ref=trust.current_fingerprint_digest("verifier"),
        arguments=(str(project),),
        input_paths=("fixture.txt",),
    )
    req = req.model_copy(
        update={"verifiers": (verifier,), "human_review_required": False}
    )
    n = 0
    seen_ids = []

    def tool(name, args):
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{n}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }

    def provider(**kwargs):
        nonlocal n
        n += 1
        user_context = next(
            m["content"] for m in kwargs["messages_payload"] if m["role"] == "user"
        )
        resources = json.loads(
            user_context.split("Selected launch resources (JSON):\n", 1)[1].split(
                "\n\nPrivate checkpoint memory", 1
            )[0]
        )
        selected = resources["verifiers"][0]
        if n in (1, 4):
            message = tool(selected["tool_name"], selected["arguments"])
        elif n == 3:
            message = tool(
                "fs_edit",
                {
                    "path": str(
                        Path(resources["primary_target"]["locator"])
                        / selected["input_paths"][0]
                    ),
                    "old_string": "invalid",
                    "new_string": "valid",
                },
            )
        else:
            refs = re.findall(r"goal_evidence_id: ([a-f0-9]{32})", str(kwargs))
            seen_ids.extend(refs)
            message = {
                "content": report(
                    summary="first failed check" if n == 2 else "corrected and checked",
                    next_action="Fix fixture then rerun the unchanged check"
                    if n == 2
                    else "",
                    evidence_ids=list(dict.fromkeys(refs)),
                    completion_recommended=n == 5,
                )
            }
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _store, _session, controller, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
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
    from tldw_chatbook.Skills_Interop import skill_script_runner

    commands = []
    subprocess_run = skill_script_runner.run_script_subprocess

    def record_process(target_argv, **kwargs):
        result = subprocess_run(target_argv, **kwargs)
        commands.append({"argv": target_argv, "exit_code": result.exit_code})
        return result

    monkeypatch.setattr(skill_script_runner, "run_script_subprocess", record_process)
    native_results = []
    dispatch = coordinator.dispatch_once

    async def record_dispatch(goal_id):
        result = await dispatch(goal_id)
        native_results.append(result)
        return result

    monkeypatch.setattr(coordinator, "dispatch_once", record_dispatch)
    try:
        saved = await coordinator.start(goal.id)
        assert saved.status == "completed", (
            saved.status,
            saved.pause_reason,
            str(calls[-1]),
        )
        process_records = [
            record for result in native_results for record in result.tool_records
        ]
        assert [record.result.exit_code for record in process_records] == [7, 0]
        assert all(
            record.invocation.arguments == (str(project),) for record in process_records
        )
        assert saved.iteration_count == 2
        assert saved.accounting.used["generation"] == 2
        assert saved.accounting.used["model_call"] == 5
        evidence = tuple(
            e
            for e in runs.goal_runs.evidence(goal.id)
            if e.verifier_id == "fixture-check"
        )
        assert len(evidence) == 2
        assert [item.passed for item in evidence] == [False, True]
        assert all(
            item.stderr == "validation provenance: trusted check.py\n"
            for item in evidence
        )
        assert [item.stdout for item in evidence] == [
            "checked fixture.txt: invalid\n",
            "checked fixture.txt: valid\n",
        ]
        assert evidence[0].source_digest != evidence[1].source_digest
        assert all(item.id in seen_ids for item in evidence)
        assert artifact.read_text() == "valid\n"
        assert (
            path.read_text() == script
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        )
        assert sentinel.read_text() == "must remain unchanged"
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(True),
                artifact.read_text().splitlines(True),
                fromfile="fixture.txt.before",
                tofile="fixture.txt.after",
            )
        )
        assert "-invalid\n+valid\n" in diff
        # Actual execution evidence over synthetic fixtures; no user data or credentials.
        import os

        target = os.environ.get("TLDW_GOAL_QUALIFICATION_ARTIFACT")
        if target:
            Path(target).write_text(
                json.dumps(
                    {
                        "qualification": "POSIX local trusted skill; process retains host executor authority",
                        "provider": "deterministic recording adapter",
                        "actual_exit_codes": [
                            r.result.exit_code for r in process_records
                        ],
                        "commands": commands,
                        "verifier_source": script,
                        "verifier_sha256": digest,
                        "model_visible_requests": [
                            c["messages_payload"] for c in calls
                        ],
                        "iterations": saved.iteration_count,
                        "model_calls": len(calls),
                        "checks": [
                            {
                                "passed": e.passed,
                                "stdout": e.stdout,
                                "stderr": e.stderr,
                                "source_digest": e.source_digest,
                                "evidence_id": e.id,
                            }
                            for e in evidence
                        ],
                        "diff": diff,
                        "final_artifact": artifact.read_text(),
                        "sentinel_unchanged": True,
                    },
                    indent=2,
                )
            )
    finally:
        await coordinator.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_configured_local_goal_live(stores, monkeypatch, tmp_path):
    """Explicit opt-in, finite local OpenAI-compatible Qwen endpoint qualification."""
    import asyncio
    import difflib
    import os
    from pathlib import Path
    from types import SimpleNamespace

    import httpx

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    from Tests.Chat.test_automatic_provider_budget import resolution
    from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService
    from tldw_chatbook.Agents.goal_models import GoalPolicy

    if os.environ.get("TLDW_GOAL_LIVE_LOCAL") != "1":
        pytest.skip(
            "Set TLDW_GOAL_LIVE_LOCAL=1 for the explicitly configured loopback endpoint"
        )
    runs, persistence, registry, req = stores
    project = Path(req.binding.locator)
    artifact = project / "fixture.txt"
    artifact.write_text("invalid\n")
    before = artifact.read_text()
    sentinel = tmp_path / "external-sentinel.txt"
    sentinel.write_text("unchanged")
    script = "from pathlib import Path\nimport sys\nv=(Path(sys.argv[1])/'fixture.txt').read_text()\nprint('checked value: '+v.strip())\nprint('trusted verifier',file=sys.stderr)\nraise SystemExit(0 if v == 'valid\\n' else 7)\n"
    scope, path, _trust = trusted_skill(tmp_path, script)
    verifier = await scope.goal_verifier_reference(
        "verifier",
        "scripts/check.py",
        arguments=(str(project),),
        input_paths=("fixture.txt",),
    )
    req = req.model_copy(
        update={
            "objective": f"Repair fixture.txt in {project}. First run the trusted verifier skill verifier, scripts/check.py, with args [{json.dumps(str(project))}]. After observing the failure, change invalid to valid using fs_edit. Rerun the same check. Cite runtime goal_evidence_id references in the required JSON report.",
            "criteria": "fixture.txt contains exactly valid followed by a newline; the unchanged trusted verifier exits zero for that current file.",
            "verifiers": (verifier,),
            "human_review_required": False,
            "policy": GoalPolicy(
                iterations=2,
                model_calls=8,
                budget_tokens=50000,
                output_tokens=1024,
                wall_seconds=60,
                iteration_model_turns=4,
                iteration_steps=32,
                iteration_wall_seconds=30,
            ),
        }
    )
    selected = resolution(
        provider="llama_cpp",
        execution_key="llama_cpp",
        readiness_key="llama_cpp",
        model="Qwen2.5-0.5B-Instruct",
        base_url="http://127.0.0.1:9099/v1",
        max_tokens=1024,
        streaming=False,
    )
    goal, _store, _session, controller, co, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, resolved=selected
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
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
    exchanges = []
    send = httpx.AsyncClient.send

    async def recording_send(client, request, *args, **kwargs):
        assert request.url.host == "127.0.0.1" and request.url.port == 9099
        response = await send(client, request, *args, **kwargs)
        await response.aread()
        exchanges.append(
            {
                "method": request.method,
                "url": str(request.url),
                "request": json.loads(request.content) if request.content else None,
                "status": response.status_code,
                "response": response.text,
            }
        )
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", recording_send)
    try:
        saved = await asyncio.wait_for(co.start(goal.id), 90)
        assert exchanges, (
            "No actual HTTP/provider operation occurred",
            saved.status,
            saved.pause_reason,
            [c.model_dump() for c in saved.checkpoints],
            calls,
        )
        assert saved.iteration_count <= 2 and saved.accounting.used["model_call"] <= 8
        assert path.read_text() == script and sentinel.read_text() == "unchanged"
        evidence = runs.goal_runs.evidence(goal.id)
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                artifact.read_text().splitlines(True),
                fromfile="fixture.txt.before",
                tofile="fixture.txt.after",
            )
        )
        output = {
            "endpoint": "http://127.0.0.1:9099/v1",
            "advertised_model": selected.model,
            "adapter": "Chatbook llama_cpp (existing fence tool protocol)",
            "limits": req.policy.model_dump(),
            "status": saved.status,
            "reason": saved.pause_reason,
            "iterations": saved.iteration_count,
            "model_calls": saved.accounting.used["model_call"],
            "quality_success": saved.status == "completed",
            "exchanges": exchanges,
            "checkpoints": [c.model_dump() for c in saved.checkpoints],
            "evidence": [e.model_dump() for e in evidence],
            "diff": diff,
            "final_artifact": artifact.read_text(),
            "sentinel_unchanged": True,
        }
        target = Path(os.environ["TLDW_GOAL_LIVE_ARTIFACT"])
        target.write_text(json.dumps(output, indent=2))
        print(
            json.dumps(
                {
                    k: output[k]
                    for k in (
                        "status",
                        "reason",
                        "iterations",
                        "model_calls",
                        "quality_success",
                        "final_artifact",
                    )
                }
            )
        )
    finally:
        await co.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_existing_fence_agent_protocol_runs_real_trusted_cli(
    stores, monkeypatch, tmp_path
):
    from Tests.Chat.test_automatic_provider_budget import resolution

    scope, path, trust = trusted_skill(tmp_path, "print('real fence check')\n")
    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={
            "verifiers": (
                VerificationSpec(
                    id="check",
                    executor_tool_id="run_skill_script",
                    verifier_path=str(path),
                    verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    skill_trust_ref=trust.current_fingerprint_digest("verifier"),
                    input_paths=("fixture.txt",),
                ),
            )
        }
    )
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        content = (
            "```tool_call\n"
            + json.dumps(
                {
                    "name": "run_skill_script",
                    "arguments": {
                        "skill_name": "verifier",
                        "script_path": "scripts/check.py",
                        "args": [],
                    },
                }
            )
            + "\n```"
            if count == 1
            else "finished"
        )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }

    goal, _, _, controller, co, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req),
        monkeypatch,
        provider,
        resolved=resolution(provider="huggingface", execution_key="huggingface"),
    )
    controller._agent_bridge._skills_service = scope
    controller.set_pending_skill_script = lambda *args, **kwargs: None
    try:
        result = await co.dispatch_once(goal.id)
        assert len(result.tool_records) == 1, result
        assert result.tool_records[0].result.exit_code == 0
        assert result.tool_records[0].result.stdout == "real fence check\n"
        assert "```tool_call" in str(calls[0])
    finally:
        await co.shutdown()
        await gateway.aclose()
