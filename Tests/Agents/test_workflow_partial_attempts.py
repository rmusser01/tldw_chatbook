"""Exact native partial history; remaining acceptance has separate Console custody."""

import pytest

from Tests.Agents.test_workflow_run_service import _native_result
from Tests.Agents.test_workflow_task_scope import (
    task_host as task_host,  # noqa: PLC0414
)
from tldw_chatbook.Agents.workflow_models import WorkflowDecision

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("mode", ["clean", "stale", "precreated"])
def test_drained_native_partial_closes_only_never_run_sibling(task_host, mode):
    def add_pending(data, _root):
        task = dict(data["phases"][0]["tasks"][0])
        task.update(
            id="task-2",
            outputs=["outside.py"],
            footprint=[{"path": "outside.py", "access": "write", "subtree": False}],
        )
        data["phases"][0]["tasks"].append(task)

    host = task_host(
        tools=("fs_read",),
        plan_edit=add_pending if mode != "precreated" else None,
        sibling=mode == "precreated",
    )
    with (
        pytest.raises(RuntimeError, match="workflow_original_fence_producer_required"),
        host.phase._fence("child_commit"),
    ):
        pytest.fail("direct fence entry created native dispatch authority")
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service, receipt, result = _native_result(host)
    service.checkpoint_task(result, drained=True)
    before = service.get(result.workflow_id)
    control = WorkflowDecision(
        id="pause-complete-subset",
        workflow_id=before.id,
        plan_revision=1,
        control_revision=before.control_revision,
        phase_id="phase-1",
        action="pause",
        reason="user_pause",
    )
    if mode == "clean":
        import sqlite3

        with host.db.connection() as conn:
            conn.execute(
                "CREATE TRIGGER refuse_control BEFORE INSERT ON workflow_decisions BEGIN SELECT RAISE(ABORT,'control rollback'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="control rollback"):
            host.phase._control(control)
        assert not host.phase._paused_tasks
        assert service.get(before.id).control_revision == before.control_revision
        with host.db.connection() as conn:
            conn.execute("DROP TRIGGER refuse_control")
    host.phase._control(control)
    settle = getattr(service, "settle_execution_phase", None)
    assert callable(settle), "missing original positively drained partial closure"
    if mode == "precreated":
        with pytest.raises(ValueError, match="native_retirement_unproven"):
            settle(host.phase, outcome="cancelled")
        assert service.get(before.id).results == (result,)
        return
    if mode == "clean":
        import sqlite3

        with host.db.connection() as conn:
            conn.execute(
                "CREATE TRIGGER refuse_partial BEFORE UPDATE OF status ON workflow_phase_attempts BEGIN SELECT RAISE(ABORT,'partial rollback'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="partial rollback"):
            settle(host.phase, outcome="cancelled")
        assert host.phase._retired
        assert service.get(before.id).accounting.used == before.accounting.used
        with host.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (host.phase.attempt.id,),
                ).fetchone()[0]
                == "accepted"
            )
            conn.execute("DROP TRIGGER refuse_partial")
    partial = settle(host.phase, outcome="cancelled")
    assert partial.status == "awaiting_resume"
    assert partial.results == (result,)
    assert partial.accounting.used == before.accounting.used
    assert receipt.owner._drain_outcome is True
    with host.db.connection() as conn:
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT t.task_id,t.ordinal,t.status,t.retirement,p.state FROM workflow_task_attempts t "
                "JOIN workflow_payload_reservations p ON p.attempt_id=t.id ORDER BY t.task_id"
            )
        ] == [
            ("task-1", 1, "completed", "drained", "published"),
            ("task-2", 1, "closed", "drained", "removed"),
        ]
    if mode == "stale":
        (host.root / "source.py").write_text("changed after the native checkpoint")
        with pytest.raises(ValueError, match="stale_task_output"):
            service.prepare_remaining_decision(before.id, phase_id="phase-1")
        return
    decision = service.prepare_remaining_decision(before.id, phase_id="phase-1")
    assert decision.result_refs == (result.attempt_id,)
    service.validate_remaining_decision(decision)
    # Passive preparation and saved IDs cannot bypass the new physical receiver.
    attempt = host.db.automatic_work._prepare_workflow(
        before.id,
        revision_id=1,
        phase_id="phase-1",
        purpose="execution",
        owner_id="owner",
        session_id="session",
        command_id="passive-id",
        _remaining=decision,
    )
    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase

    with pytest.raises(RuntimeError, match="workflow_owned_remaining_review_required"):
        _accept_workflow_phase(host.controller, attempt.id, session_id="session")
    assert service.get(before.id).accounting.used == partial.accounting.used


@pytest.mark.parametrize("failure", ["provider", "check"])
def test_actual_failed_or_unverified_result_requires_repair(task_host, failure):
    import hashlib
    import json
    import shlex
    from types import SimpleNamespace

    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime

    raw = RawCliRuntime(lambda: True)
    raw.arm()
    raw.grant_model_session("session")

    def pending(data, _root):
        first = data["phases"][0]["tasks"][0]
        data["phases"][0]["tasks"].append(
            dict(
                first,
                id="task-2",
                inputs=[],
                outputs=["outside.py"],
                footprint=[{"path": "outside.py", "access": "write"}],
            )
        )
        if failure == "check":
            wrapper = _root / "verify.sh"
            wrapper.write_text("#!/bin/bash\nexit 7\n")
            wrapper.chmod(0o700)
            first["expected_checks"] = ["required-check"]
            first["verifiers"] = [
                {
                    "id": "required-check",
                    "executor_tool_id": "shell_exec",
                    "verifier_path": str(wrapper),
                    "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                    "input_paths": ["source.py"],
                }
            ]
            first["footprint"].append({"path": "verify.sh", "access": "read"})

    host = task_host(
        mode="exclusive",
        tools=("fs_read", "shell_exec"),
        plan_edit=pending,
        configure=lambda registry, root, _binding: registry.register_provider(
            RawShellToolProvider(
                runtime=raw,
                console_session_id="session",
                initial_directory=lambda: root,
            )
        ),
    )
    host.controller.app.raw_cli_runtime = raw
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    seed = service.prepare_task_inputs(host.authorization)

    def refused_provider(**_kwargs):
        raise RuntimeError("actual native provider failure")

    script = (
        "```tool_call\n"
        + json.dumps(
            {
                "name": "shell_exec",
                "arguments": {
                    "command": shlex.join((str(host.root / "verify.sh"),)),
                    "shell": "bash",
                },
            }
        )
        + "\n```"
    )
    native, _ = _prepared_service(
        host,
        refused_provider if failure == "provider" else ScriptedChat([script, "done"]),
        seed=seed,
    )
    receipt = _launch(native, host)
    _join(host)
    service.bind_native_receipt(host.authorization, receipt)
    result = WorkflowTaskResult(
        workflow_id=host.authorization.claim.workflow_id,
        attempt_id=receipt.attempt_id,
        run_id=receipt.run_id,
        status="failed" if failure == "provider" else "completed",
        summary="Native failed or unverified outcome",
        handoff="Explicit repair is required",
    )
    saved = service.checkpoint_task(result, drained=True)
    host.phase._control(
        WorkflowDecision(
            id="pause-failed",
            workflow_id=saved.id,
            plan_revision=1,
            control_revision=saved.control_revision,
            phase_id="phase-1",
            action="pause",
            reason="user_pause",
        )
    )
    partial = service.settle_execution_phase(host.phase, outcome="cancelled")
    if failure == "check":
        assert saved.results[0].checks[0].status == "failed"
        assert saved.results[0].checks[0].exit_code == 7
    with pytest.raises(
        ValueError, match="remaining_requires_repair|native_check_missing_or_stale"
    ):
        service.prepare_remaining_decision(partial.id, phase_id="phase-1")
    assert service.get(partial.id).accounting.used == partial.accounting.used
    raw.shutdown()


def test_original_launch_marker_excludes_unused_retirement(task_host):
    import threading
    from types import SimpleNamespace

    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    host = task_host(tools=("fs_read",))
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    seed = service.prepare_task_inputs(host.authorization)
    native, _ = _prepared_service(host, ScriptedChat(["done"]), seed=seed)
    entered, release = threading.Event(), threading.Event()
    result = []

    def observe(frame, event, _arg):
        if (
            event == "call"
            and frame.f_code.co_qualname
            == "AgentService.launch_workflow_task.<locals>.launch"
        ):
            assert host.authorization._native_launch == [None]
            entered.set()
            assert release.wait(10)

    threading.setprofile(observe)
    worker = threading.Thread(target=lambda: result.append(_launch(native, host)))
    try:
        worker.start()
        assert entered.wait(10)
        with pytest.raises(ValueError, match="native_retirement_unproven"):
            service.settle_execution_phase(host.phase, outcome="cancelled")
        assert not host.phase._retired
        assert service.get(host.phase.attempt.workflow_id).results == ()
    finally:
        release.set()
        worker.join(10)
        threading.setprofile(None)
    assert not worker.is_alive()
    _join(host)
    receipt = result[0]
    service.bind_native_receipt(host.authorization, receipt)
    service.checkpoint_task(
        WorkflowTaskResult(
            workflow_id=host.phase.attempt.workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed",
            summary="Original launch completed",
            handoff="Marker held through the settlement refusal",
        ),
        drained=True,
    )
