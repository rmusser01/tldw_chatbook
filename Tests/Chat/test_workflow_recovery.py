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


@pytest.mark.parametrize(
    "boundary",
    [
        "prepared",
        "accepted",
        "quiesced",
        "quiesced_result",
        "thread_start",
        "effect",
        "result_commit",
        "settled",
    ],
)
def test_fresh_process_recovery_never_replays_native_work(tmp_path, boundary):
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    cwd = Path(__file__).resolve().parents[2]
    profile = tmp_path / "profile"
    profile.mkdir()
    env = os.environ.copy()
    env.update(
        TLDW_CONFIG_PATH=str(profile / "config.toml"),
        TLDW_TEST_CONFIG_ROOT=str(profile),
        XDG_DATA_HOME=str(profile / "data"),
        XDG_CONFIG_HOME=str(profile / "config"),
        PYTHONPATH=str(cwd),
    )
    env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
    command = [sys.executable, "-m", "Tests.DB.workflow_restart_child", str(tmp_path)]
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(
            [*command, "produce", boundary], cwd=cwd, env=env, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 20
            while (
                not (tmp_path / "boundary.json").exists()
                and child.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (tmp_path / "boundary.json").exists(), (
                tmp_path / "child.log"
            ).read_text()[-7000:]
            counter = tmp_path / "provider-operations"
            before = counter.read_bytes() if counter.exists() else b""
            child.kill()
            child.wait(5)
            restart = subprocess.run(
                [*command, "restart", boundary],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=log,
                timeout=20,
                check=False,
            )
            assert restart.returncode == 0, (tmp_path / "child.log").read_text()[-7000:]
            assert (counter.read_bytes() if counter.exists() else b"") == before
            data = json.loads((tmp_path / "restarted.json").read_text())
            assert data["status"] == (
                "awaiting_resume"
                if boundary in {"prepared", "quiesced", "quiesced_result", "settled"}
                else "recovery_required"
            )
            if boundary in {"effect", "result_commit", "quiesced_result", "settled"}:
                assert (tmp_path / "0" / "source.py").read_text() == "actual effect"
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(5)


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
