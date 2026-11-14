"""Simulated qualification policy; these are not live Windows cleanup proofs."""

import json
import sys

import pytest
import toml

from Tests.Agents.test_hook_permissions import _approve
from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Agents.workflow_support import make_workflow_request, plan_data
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat import console_workflow_runs
from tldw_chatbook.Utils import platform_files
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

hook_file = _hook_file
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("enabled", [False, True])
async def test_unqualified_policy_keeps_actual_granted_ordinary_hooks(
    tmp_path, monkeypatch, hook_file, enabled
):
    monkeypatch.setattr(
        platform_files, "supports_native_process_groups", lambda: False
    )
    marker = tmp_path / "ordinary-ran"
    raw = toml.loads(hook_file.read_text())
    raw.setdefault("agents", {})["workflow_runs_enabled"] = enabled
    raw["hooks"] = {
        "hook": [
            {
                "id": "ordinary",
                "event": "PreToolUse",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
            }
        ]
    }
    hook_file.write_text(toml.dumps(raw))
    runtime = ConsoleRuntime(None)
    engine = runtime.ensure_run_hooks()
    _approve(runtime.ensure_hook_permissions())
    try:
        result = engine.fire("PreToolUse", session_id="", cwd=str(tmp_path))
        assert not result.blocked and marker.read_text() == "ran"
        assert not engine._native_attached, (
            "unqualified ordinary hook acquired native custody"
        )
        assert (
            runtime._execution_capacity is None and runtime._worktree_activity is None
        )
        engine.close()
        assert engine.fire("PreToolUse", session_id="", cwd=str(tmp_path)).blocked
    finally:
        await runtime.dispose()


async def test_enabled_flag_cannot_accept_unqualified_native_authority_but_history_reads(
    tmp_path, monkeypatch, owned_console_databases
):
    from tldw_chatbook.Agents import automatic_work_runtime, run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    monkeypatch.setattr(
        platform_files, "supports_native_process_groups", lambda: False
    )
    setting = lambda key, default: True if key == "workflow_runs_enabled" else default
    monkeypatch.setattr(run_log, "_setting", setting)
    monkeypatch.setattr(automatic_work_runtime, "_setting", setting)
    chat, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(chat, controller)
    runs = AgentRunsDB(tmp_path / "runs.db")
    runtime = ConsoleRuntime(None)
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    controller._agent_bridge = bridge
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    runtime.set_agent_bridge(bridge)
    session = store.sessions()[0]
    owner_id = controller._fleet_wake.runtime_owner_id
    runs.automatic_work.recover(current_owner_id=owner_id)
    conversation_id = store.persist_session_if_needed(session.id)
    request = make_workflow_request(tmp_path).model_copy(
        update={"conversation_id": conversation_id}
    )
    snapshot = runs.workflow_runs.create(request, launch_id="unqualified")
    data = plan_data(tmp_path)
    data["request"] = request.model_dump(mode="json")
    snapshot = runs.workflow_runs.save_plan(
        snapshot.id,
        WorkflowPlan.model_validate_json(json.dumps(data)),
        expected_control_revision=1,
    )
    attempt = runs.automatic_work.prepare_workflow_phase(
        snapshot.id,
        revision_id=1,
        phase_id="phase-1",
        owner_id=owner_id,
        session_id=session.id,
        command_id="approve",
    )
    accounting = runs.automatic_work.snapshot(snapshot.chain_id)
    accept = runs.automatic_work.accept_workflow_phase
    accepted = []

    def observed_accept(*args, **kwargs):
        accepted.append(args)
        return accept(*args, **kwargs)

    monkeypatch.setattr(runs.automatic_work, "accept_workflow_phase", observed_accept)
    try:
        with pytest.raises(AutomaticWorkRefused, match="workflow_platform_unqualified"):
            console_workflow_runs._accept_workflow_phase(
                controller, attempt.id, session_id=session.id
            )
        assert accepted == []
        assert (
            runs.automatic_work.read_workflow_attempt(
                attempt.id, owner_id=owner_id
            ).state
            == "prepared"
        )
        assert runs.automatic_work.snapshot(snapshot.chain_id) == accounting
        assert runs.list_runs(conversation_id) == []
        assert gateway.calls == 0
        assert (
            runtime._execution_capacity is None and runtime._worktree_activity is None
        )
        service = runtime.ensure_workflow_service()
        assert service.get(snapshot.id).id == snapshot.id
        assert gateway.calls == 0
    finally:
        await runtime.dispose()
        runs.close()
