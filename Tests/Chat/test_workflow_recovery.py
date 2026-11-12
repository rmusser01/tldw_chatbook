"""Workflow recovery consumes one startup audit and never dispatches history."""

from contextlib import closing
from types import SimpleNamespace

import pytest

from Tests.DB.test_workflow_runs import planned
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


def test_service_constructor_is_passive_and_rejects_untrusted_audit(tmp_path):
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snapshot = planned(db, tmp_path)
        service = WorkflowRunService(db, SimpleNamespace(workspace_registry=None))
        assert service.get(snapshot.id) == snapshot
        with pytest.raises(ValueError, match="untrusted_recovery_audit"):
            service.project_recovery(object())
        assert service.get(snapshot.id) == snapshot
        assert db.list_runs(snapshot.request.conversation_id) == []


@pytest.mark.asyncio
async def test_service_offload_dispatches_the_owned_named_operation(
    tmp_path, monkeypatch
):
    import functools
    from types import MethodType

    from Tests.Agents.test_workflow_run_service import _history_persistence
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    with closing(AgentRunsDB(tmp_path / "dispatch.db")) as db:
        snapshot = planned(db, tmp_path)
        service = WorkflowRunService(db, _history_persistence(tmp_path))
        original = service.get
        entered = []

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            entered.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "get", MethodType(wrapped, service))
        assert await service.call(service.get, snapshot.id) == snapshot
        assert entered == [True]

        @functools.wraps(original)
        def unrelated(self, *args, **kwargs):
            pytest.fail("an unregistered callback received owned storage")

        assert (
            await service.call(MethodType(unrelated, service), snapshot.id) == snapshot
        )
        assert entered == [True, True]


@pytest.mark.asyncio
async def test_finite_cancelled_workflow_callback_keeps_actual_db_owner_until_drain(
    tmp_path,
    monkeypatch,
):
    import asyncio
    import threading
    import time

    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    db = AgentRunsDB(tmp_path / "finite.db")
    from Tests.Agents.test_workflow_run_service import _history_persistence

    snapshot = planned(db, tmp_path)
    service = WorkflowRunService(db, _history_persistence(tmp_path))
    db.close()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    original = service._source_manifest

    def preflight(snapshot, paths):
        with db.connection() as conn:
            entered.set()
            assert release.wait(5)
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            result = original(snapshot, paths)
        finished.set()
        return result

    monkeypatch.setattr(service, "_source_manifest", preflight)
    with pytest.raises(ValueError, match="unsupported_workflow_operation"):
        await service.call(lambda: None)
    task = asyncio.create_task(
        service.call(service.prepare_decision, snapshot.id, phase_id="phase-1")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        participant = db._maintenance_participant
        participant.close_admission()
        assert not participant.drain(time.monotonic() + 0.02)
        release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        assert await asyncio.to_thread(participant.drain, time.monotonic() + 3)
    finally:
        release.set()
        db._maintenance_participant.resume()
        db.close()


def test_clean_resume_is_cas_idempotent_inert_and_preserves_spend(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents import run_log

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    from Tests.Agents.test_workflow_run_service import _history_persistence
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    with closing(AgentRunsDB(tmp_path / "resume.db")) as db:
        (tmp_path / "source.py").write_text("original")
        snapshot = planned(db, tmp_path)
        import json

        from tldw_chatbook.Agents.workflow_models import WorkflowPlan

        data = snapshot.plans[0].model_dump(mode="json")
        data["phases"][0]["tasks"][0]["inputs"] = ["source.py"]
        snapshot = db.workflow_runs.save_plan(
            snapshot.id,
            WorkflowPlan.model_validate_json(json.dumps(data)),
            expected_control_revision=snapshot.control_revision,
        )
        db.automatic_work.recover(current_owner_id="old")
        db.automatic_work.prepare_workflow_phase(
            snapshot.id,
            revision_id=2,
            phase_id="phase-1",
            owner_id="old",
            session_id="s",
            command_id="old-approval",
        )
        db.automatic_work.recover(current_owner_id="new")
        service = WorkflowRunService(db, _history_persistence(tmp_path))
        service.project_recovery(db.automatic_work.recovery_result)
        before = service.get(snapshot.id)
        assert before.status == "awaiting_resume"
        with pytest.raises(ValueError, match="revision_conflict"):
            service.resume(snapshot.id, expected_control_revision=1, command_id="stale")
        (tmp_path / "source.py").unlink()
        with pytest.raises(ValueError, match="resume_inputs_unavailable"):
            service.resume(
                snapshot.id,
                expected_control_revision=before.control_revision,
                command_id="missing-input",
            )
        (tmp_path / "source.py").write_text("original")
        resumed = service.resume(
            snapshot.id,
            expected_control_revision=before.control_revision,
            command_id="resume",
        )
        assert resumed.status == "awaiting_approval"
        assert resumed.accounting.used == before.accounting.used
        assert (
            service.resume(
                snapshot.id,
                expected_control_revision=before.control_revision,
                command_id="resume",
            )
            == resumed
        )
        with pytest.raises(ValueError, match="resume_command_conflict"):
            service.resume(
                snapshot.id,
                expected_control_revision=before.control_revision + 1,
                command_id="resume",
            )
        assert db.list_runs(snapshot.request.conversation_id) == []
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM workflow_phase_attempts WHERE status IN ('prepared','accepted')"
                ).fetchone()[0]
                == 0
            )


@pytest.mark.asyncio
async def test_actual_startup_owner_projects_workflows_once_before_ready(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    db = AgentRunsDB(tmp_path / "startup.db")
    snapshot = planned(db, tmp_path)
    controller = SimpleNamespace(
        _agent_bridge=SimpleNamespace(runs_db=db),
        store=SimpleNamespace(
            persistence=SimpleNamespace(workspace_registry=None), sessions=list
        ),
    )
    runtime = ConsoleRuntime(None)
    runtime.set_chat_controller(controller)
    wake = ConsoleFleetWakeCoordinator(controller)
    wake._recovery_projection = runtime._project_workflow_recovery
    original = db.automatic_work.recover
    calls = []

    def recover(**kwargs):
        calls.append(kwargs["current_owner_id"])
        return original(**kwargs)

    monkeypatch.setattr(db.automatic_work, "recover", recover)
    try:
        assert (
            runtime.ensure_workflow_service().get(snapshot.id).status
            == "awaiting_approval"
        )
        assert not calls
        wake.start_recovery()
        wake.start_recovery()
        assert await wake.wait_for_recovery()
        assert calls == [wake.runtime_owner_id]
        assert (
            runtime.ensure_workflow_service().get(snapshot.id).status
            == "awaiting_resume"
        )
        assert db.list_runs(snapshot.request.conversation_id) == []
    finally:
        wake.dispose()
        await wake.drain_recovery()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "closed,opaque,replaced_root,legacy_closed",
    [
        (False, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, "symlink", False),
        (True, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (False, False, False, True),
        (False, True, False, True),
    ],
)
async def test_startup_scopes_historical_protection_and_reconciliation_releases_it(
    tmp_path, monkeypatch, closed, opaque, replaced_root, legacy_closed
):
    import json

    from Tests.Agents.test_workflow_run_service import _history_persistence
    from Tests.Agents.workflow_support import make_workflow_request, plan_data
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.workflow_models import (
        WorkflowExecutionRecovery,
        WorkflowPlan,
    )
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import (
        WorktreeFootprint,
        WorktreeScope,
    )

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    root = tmp_path / "project"
    root.mkdir()
    db = AgentRunsDB(tmp_path / "recovery-protection.db")
    snapshot = db.workflow_runs.create(make_workflow_request(root), launch_id="start")
    data = plan_data(root)
    from tldw_chatbook.Agents.workflow_run_service import _workflow_root_identity

    data["root_identity"] = _workflow_root_identity(capture_directory_chain(root))
    if opaque:
        data["phases"][0]["tasks"][0]["execution_mode"] = "exclusive"
    snapshot = db.workflow_runs.save_plan(
        snapshot.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=1,
    )
    db.automatic_work.recover(current_owner_id="old")
    attempt = db.automatic_work.prepare_workflow_phase(
        snapshot.id,
        revision_id=1,
        phase_id="phase-1",
        owner_id="old",
        session_id="session",
        command_id="approve",
    )
    assert db.automatic_work.accept_workflow_phase(attempt.id, owner_id="old")
    persistence = _history_persistence(root)
    service = WorkflowRunService(db, persistence)
    if closed:
        db.automatic_work.recover(current_owner_id="intermediate")
        snapshot = service.get(snapshot.id)
        service.resolve_recovery(
            snapshot.id,
            expected_control_revision=snapshot.control_revision,
            decision=WorkflowExecutionRecovery(reason="acknowledged_unknown"),
        )
    if legacy_closed:
        db.automatic_work.recover(current_owner_id="intermediate")
        # This is the post-upgrade state: old closed history has no review marker.
        with db.connection() as conn:
            conn.execute(
                "UPDATE workflow_runs SET status='closed' WHERE id=?", (snapshot.id,)
            )
        assert service.get(snapshot.id).accounting.uncertain
    controller = SimpleNamespace(
        _agent_bridge=SimpleNamespace(runs_db=db),
        store=SimpleNamespace(persistence=persistence, sessions=list),
    )
    runtime = ConsoleRuntime(None)
    runtime.set_chat_controller(controller)
    wake = ConsoleFleetWakeCoordinator(controller)
    wake._recovery_projection = runtime._project_workflow_recovery
    if replaced_root:
        root.rename(tmp_path / "old-project")
        if replaced_root == "symlink":
            target = tmp_path / "unrelated-target"
            target.mkdir()
            root.symlink_to(target, target_is_directory=True)
        else:
            root.mkdir()
    try:
        assert not runtime.worktree_activity.recovery_blockers(
            WorktreeFootprint(opaque=True)
        )
        wake.start_recovery()
        assert await wake.wait_for_recovery()
        chain = capture_directory_chain(root)
        footprint = WorktreeFootprint(
            reads=(WorktreeScope(chain.canonical_root / "source.py", False, chain),)
        )
        blocked = not closed and not replaced_root
        assert runtime.worktree_activity.recovery_blockers(footprint) == (
            (snapshot.id,) if blocked else ()
        )
        with runtime.worktree_activity.prelaunch(footprint=footprint) as claim:
            assert (claim is None) is blocked
        unrelated = tmp_path / "scratch"
        unrelated.mkdir()
        scratch = WorktreeFootprint(
            writes=(WorktreeScope(unrelated, True, capture_directory_chain(unrelated)),)
        )
        with runtime.worktree_activity.prelaunch(footprint=scratch) as claim:
            assert claim is not None
        assert db.list_runs(snapshot.request.conversation_id) == []
        assert runtime.execution_capacity.snapshot().executions == ()
        assert service.get(snapshot.id).status == (
            "closed" if closed or legacy_closed else "recovery_required"
        )
        if not closed:
            live_service = runtime.ensure_workflow_service()
            before = await live_service.call(live_service.get, snapshot.id)
            resolved = await live_service.call(
                live_service.resolve_recovery,
                snapshot.id,
                expected_control_revision=before.control_revision,
                decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            )
            assert resolved.status == "closed"
            assert resolved.accounting == before.accounting
            assert runtime.worktree_activity.recovery_blockers(footprint) == ()
            with runtime.worktree_activity.prelaunch(footprint=footprint) as claim:
                assert claim is not None
    finally:
        wake.dispose()
        await wake.drain_recovery()
        db.close()
