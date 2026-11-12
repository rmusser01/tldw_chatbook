"""Native workflow checks must survive correction, review, and Stop in one worktree."""

import hashlib
import json
import os
import shlex
import subprocess
import threading
from collections import Counter
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

pytestmark = pytest.mark.bootstrap_profile

_PRIVATE_JOURNEY = """
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="true", TLDW_AGENTS_MAX_LIVE_SUBAGENTS="2", TLDW_AGENTS_RUN_LOG_ENABLED="false", OPENAI_API_KEY="private-native-workflow-test")
from Tests.Chat.test_workflow_native_journeys import correction_journey
with user_fixture_default_owner():
    asyncio.run(correction_journey(Path(os.environ["XDG_DATA_HOME"]).parent, sys.argv[1]))
"""


class CorrectionProvider(WorkflowProvider):
    """Only provider output is scripted. Every mutation and check goes through its tool."""

    def __init__(self):
        super().__init__()
        self.commands = {}
        self.barrier = threading.Barrier(2)
        self.counts = Counter()

    def __call__(self, **kwargs):
        messages = kwargs.get("messages", kwargs.get("messages_payload", ()))
        key = next(
            (
                key
                for key in self.commands
                if any(row.get("content") == key for row in messages)
            ),
            None,
        )
        if key is None:
            from tldw_chatbook.Agents.goal_iteration import verifier_digest
            from tldw_chatbook.Agents.workflow_models import WorkflowPlan

            seed = next(
                row["content"]
                for row in messages
                if str(row.get("content", "")).startswith(
                    "Produce exactly one strict JSON WorkflowPlan"
                )
            )
            contract = json.loads(seed.split("\n", 1)[1])["workflow_planning"]
            plan = WorkflowPlan.model_validate_json(json.dumps(self.plan))
            print(
                "JOURNEY_PLANNER_INPUT",
                sorted(contract["prospective_native_tools"]),
                flush=True,
            )
            assert set(plan.tools) <= set(contract["prospective_native_tools"])
            assert plan.request.model_dump(mode="json") == contract["request"]
            assert plan.root_identity == contract["root_identity"]
            for phase in plan.phases:
                for task in phase.tasks:
                    for spec in task.verifiers:
                        assert verifier_digest(spec) == spec.verifier_sha256
            return super().__call__(**kwargs)
        from Tests.Chat.test_automatic_provider_budget import response

        with self._lock:
            self.calls.append((key, kwargs))
            index = self.counts[key]
            self.counts[key] += 1
        if key in {"fix-source", "update-note"} and index == 0:
            self.barrier.wait(timeout=12)
        steps = self.commands[key]
        assert index <= len(steps), (key, index)
        text = "Handoff: " + key
        if index < len(steps):
            name, arguments = steps[index]
            text = (
                "```tool_call\n"
                + json.dumps({"name": name, "arguments": arguments})
                + "\n```"
            )
        reply = response()
        reply["choices"][0]["message"]["content"] = text
        return reply


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-C",
            str(root),
            *args,
        ],
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
    )


async def correction_journey(private, mode="correction"):
    provider = CorrectionProvider()
    rig = await build_workflow_rig(private, None, provider, native_fleet_consumers=True)
    root = Path(rig.workflow.request.binding.locator)
    app = rig.controller.app
    session = rig.store.sessions()[0]
    raw = app.raw_cli_runtime
    assert rig.runtime.ensure_run_hooks() is rig.runtime.run_hooks_engine
    retained_refusal = False
    try:
        app.app_config.setdefault("console", {})["raw_cli_permitted"] = True
        assert rig.runtime.raw_cli_source.raw is raw
        assert raw.arm().armed
        raw.grant_model_session(session.id)
        assert raw.model_session_granted(session.id)
        for tool in ("fs_read", "fs_edit", "shell_exec"):
            app.unified_mcp_service.approve_for_session("local:__local__", tool)
        git(root, "init", "-q")
        source, note, wrapper = (
            root / name for name in ("source.txt", "note.txt", "check.sh")
        )
        source.write_text("original\n")
        note.write_text("unreviewed\n")
        wrapper.write_text(
            '#!/bin/sh\ncd "$(dirname "$0")" || exit 2\ngrep -Fxq corrected source.txt || exit 7\nprintf "verified\\n"\n'
        )
        wrapper.chmod(0o700)
        git(root, "add", "--", "source.txt", "note.txt", "check.sh")
        git(root, "commit", "-qm", "private qualification inputs")
        source.write_text("user edit\noriginal\n")
        head, index = git(root, "rev-parse", "HEAD"), git(root, "ls-files", "--stage")
        assert git(root, "diff", "--", "source.txt")
        alias = rig.workflow.request.binding.binding_id
        verifier = {
            "id": "verify-source",
            "executor_tool_id": "shell_exec",
            "verifier_path": str(wrapper),
            "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "input_paths": ["source.txt"],
        }
        target = provider.plan["request"]["targets"][0]

        def task(key, tools=(), inputs=(), outputs=(), dependencies=(), check=False):
            paths = dict.fromkeys((p.split(":")[-1] for p in inputs), "read")
            paths.update(dict.fromkeys(outputs, "write"))
            if check:
                paths["check.sh"] = "read"
            return {
                "id": key,
                "instructions": key,
                "target": target,
                "role": "worker",
                "allowed_tools": list(tools),
                "inputs": list(inputs),
                "outputs": list(outputs),
                "dependencies": list(dependencies),
                "footprint": [{"path": p, "access": mode} for p, mode in paths.items()],
                "execution_mode": "exclusive" if check else "scoped",
                "expected_checks": ["verify-source"] if check else [],
                "verifiers": [verifier] if check else [],
            }

        def check(key, *, inputs=("source.txt",), dependencies=()):
            provider.commands[key] = [
                ("shell_exec", {"command": shlex.quote(str(wrapper)), "shell": "bash"})
            ]
            return task(
                key,
                ("shell_exec",),
                inputs=inputs,
                dependencies=dependencies,
                check=True,
            )

        async def execute(phase_id, *, expected="awaiting_result_review"):
            decision = await rig.service.call(
                rig.service.prepare_decision, rig.workflow.id, phase_id=phase_id
            )
            assert decision.evidence.ready, decision.reason
            await rig.coordinator.approve(decision, command_id="approve-" + phase_id)
            snapshot = await rig.wait_for_status(expected, timeout=35)
            return snapshot

        async def review(phase_id):
            decision = await rig.service.call(
                rig.service.prepare_quality_review, rig.workflow.id, phase_id=phase_id
            )
            assert decision.evidence.ready, decision.reason
            return await rig.coordinator.approve(
                decision, command_id="quality-" + phase_id
            )

        async def evidence(result, status, code):
            checkpoint = await rig.service.call(
                rig.service.read_checkpoint, rig.workflow.id, result.attempt_id
            )
            (record,) = checkpoint.verifiers
            (observed,) = result.checks
            assert (observed.status, observed.exit_code) == (status, code)
            assert record.complete and record.cleanup_proven and record.fresh
            assert (
                record.exit_code == code
                and record.input_manifest == observed.input_manifest
            )
            assert (
                record.input_manifest
                and record.verifier_sha256 == verifier["verifier_sha256"]
            )
            return record

        provider.plan["tools"] = ["fs_read", "fs_edit", "shell_exec"]
        provider.plan["phases"] = [{"id": "diagnose", "tasks": [check("first-check")]}]
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        if mode in {"disarm", "kill", "source"}:
            import asyncio

            decision = await rig.service.call(
                rig.service.prepare_decision, rig.workflow.id, phase_id="diagnose"
            )
            if mode == "disarm":
                raw.disarm()
            elif mode == "kill":
                app.unified_mcp_service.set_kill_switch(True)
            else:
                rig.runtime._raw_cli_source = object()
            try:
                await rig.coordinator.approve(decision, command_id="revoked-shell")
            except PermissionError as error:
                print(
                    "SHELL_BEFORE_CHILD_REFUSED",
                    type(error).__name__,
                    str(error),
                    flush=True,
                )
            else:
                run = rig.coordinator._runs.get(rig.workflow.id)
                assert run is not None
                await asyncio.wait_for(asyncio.shield(run.driver), 15)
            snapshot = await rig.coordinator.get(rig.workflow.id)
            assert (
                provider.counts == {} and snapshot.accounting.used["child_launch"] == 0
            )
            assert source.read_text() == "user edit\noriginal\n"
            with rig.service.db.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT count(*) FROM agent_runs WHERE agent_kind!='primary'"
                    ).fetchone()[0]
                    == 0
                )
            print("SHELL_NATIVE_REVOKED_NO_CHILD_EFFECT_" + mode, flush=True)
            retained_refusal = bool(rig.coordinator._runs)
            return
        initial = await execute("diagnose", expected="paused")
        assert initial.results[0].status == "failed"
        failed = await evidence(initial.results[0], "failed", 7)
        rejected = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="diagnose"
        )
        assert not rejected.evidence.ready
        edits = []
        for key, file, old, new in (
            ("fix-source", source, "original", "corrected"),
            ("update-note", note, "unreviewed", "reviewed"),
        ):
            args = {"path": file.name, "root_alias": alias}
            provider.commands[key] = [
                ("fs_read", args),
                ("fs_edit", {**args, "old_string": old, "new_string": new}),
            ]
            edits.append(task(key, ("fs_read", "fs_edit"), (file.name,), (file.name,)))
        provider.commands["independent-review"] = [
            ("fs_read", {"path": name, "root_alias": alias})
            for name in ("source.txt", "note.txt")
        ]
        provider.plan["phases"] = [
            {"id": "correct", "tasks": edits},
            {
                "id": "check",
                "tasks": [
                    check(
                        "fresh-check",
                        inputs=("fix-source:source.txt",),
                        dependencies=("fix-source",),
                    )
                ],
            },
            {
                "id": "review",
                "tasks": [
                    task(
                        "independent-review",
                        ("fs_read",),
                        inputs=("source.txt", "note.txt"),
                        dependencies=("fresh-check", "update-note"),
                    )
                ],
            },
        ]
        await rig.coordinator.revise(
            rig.workflow.id,
            expected_control_revision=initial.control_revision,
            instruction="Authorize a fresh plan to correct source.txt. Preserve the user's edit.",
            command_id="revise-after-failed-check",
        )
        revised = await rig.wait_for_status("awaiting_phase_review")
        assert len(revised.plans) == 2 and revised.results == initial.results
        corrected = await execute("correct")
        assert (
            source.read_text() == "user edit\ncorrected\n"
            and note.read_text() == "reviewed\n"
        )
        assert len(corrected.results) == 3
        await review("correct")
        checked = await execute("check")
        passed = await evidence(checked.results[-1], "passed", 0)
        assert passed.input_manifest != failed.input_manifest
        await review("check")
        reviewed = await execute("review")
        assert all(r.handoff for r in reviewed.results)
        assert (await review("review")).status == "completed"
        assert dict(provider.counts) == {
            "first-check": 2,
            "fresh-check": 2,
            "fix-source": 3,
            "update-note": 3,
            "independent-review": 3,
        }
        assert reviewed.accounting.used["model_call"] == len(provider.calls)
        assert reviewed.accounting.used["child_launch"] == 5
        with rig.service.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_wake_attempts WHERE attempt_kind='fleet_wake'"
                ).fetchone()[0]
                == 0
            )
        # A separate accepted planner can be stopped without rolling back either worktree edit.
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id="stop-preserves-edits"
        )
        stopped_review = await rig.wait_for_status("awaiting_phase_review")
        stopped = await rig.coordinator.stop(
            rig.workflow.id, expected_control_revision=stopped_review.control_revision
        )
        assert stopped.status == "stopped"
        assert (
            source.read_text() == "user edit\ncorrected\n"
            and note.read_text() == "reviewed\n"
        )
        assert (git(root, "rev-parse", "HEAD"), git(root, "ls-files", "--stage")) == (
            head,
            index,
        )
        (private / "native-journey.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "pgid": os.getpgid(0) if hasattr(os, "getpgid") else None,
                    "counts": dict(provider.counts),
                    "failed": failed.model_dump(),
                    "passed": passed.model_dump(),
                    "outcomes": ["failed", "passed"],
                    "handoffs": [r.handoff for r in reviewed.results],
                    "git_diff": git(root, "diff").decode(),
                    "accounting": dict(reviewed.accounting.used),
                },
                indent=2,
            )
        )
        assert not tuple(root.rglob("logs.*.txt"))
        from Tests.Chat.workflow_wave_archive import capture_profile, durable_facts
        from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            facts = durable_facts(conn, reviewed.id)
        assert len(facts["checkpoints"]) == len(reviewed.results)
        (Path.home() / "wave.json").write_text(
            json.dumps({"workflow_id": reviewed.id, "facts": facts})
        )
        await capture_profile(app, Path.home() / "retained-temporary.tldw-backup.zip")
    finally:
        if retained_refusal:
            with pytest.raises(
                RuntimeError, match="workflow_original_shutdown_unresolved"
            ):
                await rig.close()
            print("SHELL_RETAINED_ORIGINAL_NO_CREATOR_" + mode, flush=True)
        else:
            await rig.close()


def test_native_correction_keeps_failed_check_and_fresh_passing_handoff(tmp_path):
    from Tests.Chat.workflow_wave_archive import restore_and_rebackup

    original = tmp_path / "original"
    original.mkdir()
    _run(original, "correction", "native", script=_PRIVATE_JOURNEY, timeout=130)
    assert json.loads((original / "native-journey.json").read_text())["outcomes"] == [
        "failed",
        "passed",
    ]
    restore_and_rebackup(original)


@pytest.mark.parametrize("mode", ["disarm", "kill", "source"])
def test_public_workflow_shell_revocation_prevents_child_effect(tmp_path, mode):
    from Tests.Chat.test_console_workflow_scheduling import (
        _expect_retained_planning_child,
    )

    _expect_retained_planning_child(
        (tmp_path, mode, "native"),
        {"script": _PRIVATE_JOURNEY, "timeout": 60},
        "SHELL_NATIVE_REVOKED_NO_CHILD_EFFECT_" + mode,
        "SHELL_RETAINED_ORIGINAL_NO_CREATOR_" + mode,
    )


async def hook_validation_journey(private, mode):
    """Hold original capture return while another original native caller enters."""
    import asyncio
    import sys

    from tldw_chatbook import config
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkRefused
    from tldw_chatbook.Agents.legacy_hook_ownership import _NativeLegacyHookSource
    from tldw_chatbook.Agents.workflow_task_scope import _check_hooks

    rig = await build_workflow_rig(private, None, WorkflowProvider())
    capacity = rig.runtime.execution_capacity
    assert not capacity.snapshot().executions
    engine = rig.runtime.ensure_run_hooks()
    issuer = rig.runtime._legacy_hooks_source
    session = rig.store.sessions()[0]
    held, release, contender_entered, contender_done, initial_policy_read = (
        threading.Event() for _ in range(5)
    )
    outcomes, returns = [], []
    capture_code = _NativeLegacyHookSource.capture.__wrapped__.__code__
    policy_code = config.locked_hooks_config_snapshot.__wrapped__.__code__
    monitor = sys.monitoring
    monitor.use_tool_id(5, "workflow-hook-source-overlap")

    def yielded(code, offset, value):
        frame = sys._getframe(1)
        if frame.f_locals["self"] is issuer and threading.current_thread() is first:
            assert issuer._admissions == 1 and not issuer._events
            held.set()
            assert release.wait(5), "original source capture was never released"

    def returned(code, offset, value):
        if code is policy_code:
            if threading.current_thread() is second:
                initial_policy_read.set()
            return
        if threading.current_thread() is first:
            assert not issuer.cleanup_pending
            returns.append("original capture returned")

    def entered(code, offset):
        if threading.current_thread() is second:
            contender_entered.set()

    def check():
        try:
            _check_hooks(rig.runtime, session.id)
            outcomes.append(
                (threading.current_thread().name, "accepted", bool(returns))
            )
        except AutomaticWorkRefused as error:
            outcomes.append(
                (threading.current_thread().name, str(error), bool(returns))
            )
        finally:
            if threading.current_thread() is second:
                contender_done.set()

    def event():
        engine.notify("ApprovalRequested", session_id=session.id)

    first = threading.Thread(target=event if mode == "event" else check, name="first")
    second = threading.Thread(target=check, name="second")
    monitor.register_callback(5, monitor.events.PY_YIELD, yielded)
    monitor.register_callback(5, monitor.events.PY_RETURN, returned)
    monitor.register_callback(5, monitor.events.PY_START, entered)
    monitor.set_local_events(
        5, capture_code, monitor.events.PY_YIELD | monitor.events.PY_RETURN
    )
    monitor.set_local_events(5, _check_hooks.__code__, monitor.events.PY_START)
    monitor.set_local_events(5, policy_code, monitor.events.PY_RETURN)
    try:
        first.start()
        assert await asyncio.to_thread(held.wait, 4)
        second.start()
        assert await asyncio.to_thread(contender_entered.wait, 4)
        if mode != "event":
            assert await asyncio.to_thread(initial_policy_read.wait, 4)
            assert not await asyncio.to_thread(contender_done.wait, 0.15), outcomes
            if mode == "policy":
                changed = config.replace_hooks_config_snapshot(
                    config.read_hooks_config_snapshot(),
                    {
                        "enabled": True,
                        "hook": [
                            {
                                "id": "after-wait",
                                "event": "PreToolUse",
                                "command": [sys.executable, "-c", "pass"],
                            }
                        ],
                    },
                )
                assert changed.file_replaced and changed.failure_phase is None
            elif mode == "source":
                engine.close()
            else:
                assert mode == "overlap"
        else:
            assert mode == "event"
            assert await asyncio.to_thread(contender_done.wait, 4)
            assert outcomes == [
                ("second", "workflow_task_effectful_hooks_unsupported", False)
            ]
    finally:
        release.set()
        await asyncio.to_thread(first.join, 5)
        if second.ident is not None:
            await asyncio.to_thread(second.join, 5)
        monitor.set_local_events(5, capture_code, 0)
        monitor.set_local_events(5, policy_code, 0)
        monitor.set_local_events(5, _check_hooks.__code__, 0)
        monitor.register_callback(5, monitor.events.PY_YIELD, None)
        monitor.register_callback(5, monitor.events.PY_RETURN, None)
        monitor.register_callback(5, monitor.events.PY_START, None)
        monitor.free_tool_id(5)
        await rig.close()
    assert not first.is_alive() and not second.is_alive()
    assert returns == ["original capture returned"]
    if mode == "overlap":
        assert outcomes == [("first", "accepted", True), ("second", "accepted", True)]
    elif mode in {"source", "policy"}:
        assert outcomes == [
            ("first", "accepted", True),
            ("second", "workflow_task_effectful_hooks_unsupported", True),
        ]
    assert not issuer.cleanup_pending and not issuer.permissions._published_targets
    assert not rig.provider.calls
    print("NATIVE_HOOK_VALIDATION", mode, outcomes, returns, flush=True)


@pytest.mark.parametrize("mode", ["overlap", "event", "source", "policy"])
def test_original_workflow_hook_validation_preserves_capture_return(tmp_path, mode):
    script = _PRIVATE_JOURNEY.replace("correction_journey", "hook_validation_journey")
    _run(tmp_path, mode, "native", script=script, timeout=40)
