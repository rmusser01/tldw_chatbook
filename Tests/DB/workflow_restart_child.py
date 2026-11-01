"""Owned fresh-process workflow crash/reopen driver; no recovery dispatch."""

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

root, mode, boundary = Path(sys.argv[1]), sys.argv[2], sys.argv[3]


def stamp(snapshot, path):
    data = {
        "id": snapshot.id,
        "path": str(path),
        "status": snapshot.status,
        "used": dict(snapshot.accounting.used),
        "reserved": dict(snapshot.accounting.reserved),
        "results": [result.canonical_json() for result in snapshot.results],
    }
    with (root / "boundary.json").open("w") as file:
        json.dump(data, file)
        file.flush()
        os.fsync(file.fileno())
    threading.Event().wait(60)


if mode == "produce":
    patch = pytest.MonkeyPatch()
    from tldw_chatbook.Agents import automatic_work_runtime, run_log

    setting = lambda key, default: True if key == "workflow_runs_enabled" else default
    patch.setattr(run_log, "_setting", setting)
    patch.setattr(automatic_work_runtime, "_setting", setting)
    if boundary == "prepared":
        from Tests.DB.test_workflow_runs import planned

        path = root / "runs.db"
        db = AgentRunsDB(path)
        db.automatic_work.recover(current_owner_id="old")
        snapshot = planned(db, root)
        db.automatic_work.prepare_workflow_phase(
            snapshot.id,
            revision_id=1,
            phase_id="phase-1",
            owner_id="old",
            session_id="session",
            command_id="approve",
        )
        stamp(db.workflow_runs.get(snapshot.id), path)
    else:
        from Tests.Agents.test_agent_service import ScriptedChat, fence
        from Tests.Agents.test_workflow_child_launch import (
            _join,
            _launch,
            _prepared_service,
        )
        from Tests.Agents.test_workflow_task_scope import task_host
        from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult

        fixture = task_host.__wrapped__(root, patch)
        host = next(fixture)()
        db, path = host.db, root / "runs-0.db"
        workflow_id = host.authorization.claim.workflow_id
        if boundary in {"accepted", "quiesced"}:
            if boundary == "quiesced":
                host.owner.finish_root()
                host.claim.settle(durable=True)
                db.automatic_work.quiesce_workflow_phase(
                    host.phase.attempt.id, owner_id="owner", drained=True
                )
            stamp(db.workflow_runs.get(workflow_id), path)
        host.owner.finish_root()
        host.claim.settle(durable=True)
        service = WorkflowRunService(db, SimpleNamespace(workspace_registry=None))
        seed = service.prepare_task_inputs(host.authorization)
        steps = ScriptedChat(
            [
                fence("fs_read", {"path": "source.py"}),
                fence("fs_write", {"path": "source.py", "content": "actual effect"}),
                "done",
            ]
        )
        calls = 0

        def chat(**kwargs):
            global calls
            calls += 1
            with (root / "provider-operations").open("a") as file:
                file.write("call\n")
                file.flush()
                os.fsync(file.fileno())
            (root / "last-messages.json").write_text(
                json.dumps(kwargs.get("messages_payload"), default=str)
            )
            if boundary == "thread_start" or boundary == "effect" and calls == 3:
                stamp(service.get(workflow_id), path)
            return steps(**kwargs)

        native, _ = _prepared_service(host, chat, seed=seed)
        receipt = _launch(native, host)
        service.bind_native_receipt(host.authorization, receipt)
        _join(host)
        result = WorkflowTaskResult(
            workflow_id=workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed",
            summary="settled",
            handoff="durable required handoff",
        )
        service.checkpoint_task(result, drained=True)
        if boundary == "quiesced_result":
            db.automatic_work.quiesce_workflow_phase(
                host.phase.attempt.id, owner_id="owner", drained=True
            )
        if boundary == "settled":
            db.automatic_work.settle_workflow_phase(
                host.phase.attempt.id, owner_id="owner", drained=True
            )
        stamp(service.get(workflow_id), path)
else:
    saved = json.loads((root / "boundary.json").read_text())
    db = AgentRunsDB(saved["path"])
    service = WorkflowRunService(db, SimpleNamespace(workspace_registry=None))
    assert service.get(saved["id"]).results == tuple(
        __import__(
            "tldw_chatbook.Agents.workflow_models", fromlist=["WorkflowTaskResult"]
        ).WorkflowTaskResult.model_validate_json(value)
        for value in saved["results"]
    )
    db.automatic_work.recover(current_owner_id="new")
    service.project_recovery(db.automatic_work.recovery_result)
    current = service.get(saved["id"])
    assert dict(current.accounting.used) == saved["used"]
    expected = (
        "awaiting_resume"
        if boundary in {"prepared", "quiesced", "quiesced_result", "settled"}
        else "recovery_required"
    )
    assert current.status == expected, current
    assert [result.canonical_json() for result in current.results] == saved["results"]
    (root / "restarted.json").write_text(
        json.dumps({"status": current.status, "used": dict(current.accounting.used)})
    )
    db.close()
