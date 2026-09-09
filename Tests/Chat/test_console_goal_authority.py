"""Scope narrows discovery and dispatch even when other permissions allow work."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Agents.goal_models import GoalToolScope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool", ["run_skill_script", "install_skill", "spawn_subagent"]
)
async def test_unadvertised_runtime_calls_never_reach_their_executor(
    stores, monkeypatch, tool
):
    runs, persistence, registry, req = stores
    stores = (
        runs,
        persistence,
        registry,
        req.model_copy(
            update={
                "tool_scope": GoalToolScope(
                    catalog_tools=("builtin:get_current_datetime",)
                )
            }
        ),
    )
    calls = 0

    def provider(**kwargs):
        nonlocal calls
        calls += 1
        message = {"content": "stopped"}
        if calls == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "denied",
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(
                                {
                                    "task": "spawn",
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

    goal, _store, _session, _controller, coordinator, gateway, requests = (
        build_goal_rig(stores, monkeypatch, provider)
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert result.outcome.subagents_spawned == 0
        assert result.tool_records == ()
        assert "Tool outside goal scope" in str(result.outcome.steps)
        assert tool not in str(requests[0].get("tools"))
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_connected_mcp_target_cannot_follow_a_retargeted_profile(
    stores, monkeypatch, tmp_path
):
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
    from tldw_chatbook.MCP.tool_naming import llm_tool_name

    class Client:
        def __init__(self):
            self.sessions = {}
            self.calls = []

        async def connect_to_server(self, server_id, command, args, env):
            self.sessions[server_id] = object()
            return True

        async def describe_server(self, server_id):
            return {
                "tools": self.get_server_tools(server_id),
                "resources": [],
                "prompts": [],
            }

        def get_server_tools(self, server_id):
            return [
                {
                    "name": "check",
                    "description": "check fixture",
                    "inputSchema": {"type": "object"},
                }
            ]

        async def call_tool(self, server_id, name, arguments):
            self.calls.append((server_id, name, arguments))
            return {"content": [{"type": "text", "text": "done"}]}

    client = Client()
    local_store = LocalMCPStore(tmp_path / "mcp.json")
    profile = LocalExternalMCPProfile(
        profile_id="verifier", command="python", args=("trusted.py",)
    )
    local_store.save_profile(profile)
    local = LocalMCPControlService(store=local_store, client=client)
    assert callable(getattr(local, "goal_tool_binding", None)), (
        "MCP launch binding absent"
    )
    await local.connect_profile("verifier")
    tool_id = llm_tool_name("local:verifier", "check")
    binding = local.goal_tool_binding("verifier", "check", tool_id=tool_id)
    runs, persistence, registry, req = stores
    scope = GoalToolScope(catalog_tools=(tool_id,), mcp_bindings=(binding,))
    stores = (runs, persistence, registry, req.model_copy(update={"tool_scope": scope}))
    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        release.wait(3)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _store, _session, _controller, coordinator, gateway, _requests = (
        build_goal_rig(stores, monkeypatch, provider)
    )
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        with coordinator._active.context.scope():
            await local.execute_external_tool("verifier", "check", {})
            local_store.save_profile(replace(profile, args=("different.py",)))
            with pytest.raises(AutomaticWorkRefused, match="mcp_binding_changed"):
                await local.execute_external_tool("verifier", "check", {})
            # Putting the launch config back cannot adopt an untracked replacement session.
            local_store.save_profile(profile)
            client.sessions["verifier"] = object()
            with pytest.raises(AutomaticWorkRefused, match="mcp_connection_changed"):
                await local.execute_external_tool("verifier", "check", {})
        assert len(client.calls) == 1
    finally:
        release.set()
        await task
        await gateway.aclose()


@pytest.mark.asyncio
async def test_find_and_load_cannot_expand_goal_catalog_scope(stores, monkeypatch):
    runs, persistence, registry, req = stores
    req = req.model_copy(
        update={
            "tool_scope": GoalToolScope(
                catalog_tools=("builtin:get_current_datetime",),
                runtime_tools=("find_tools", "load_tools"),
            )
        }
    )
    sequence = [
        ("find_tools", {"query": "calculator"}),
        ("load_tools", {"ids": ["builtin:calculator"]}),
        ("calculator", {"expression": "2+2"}),
    ]
    index = 0

    def provider(**kwargs):
        nonlocal index
        name, args = sequence[index]
        index += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(index),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _store, _session, _controller, coordinator, gateway, requests = (
        build_goal_rig((runs, persistence, registry, req), monkeypatch, provider)
    )
    try:
        result = await coordinator.dispatch_once(goal.id)
        assert len(requests) == 3
        assert all(
            "calculator" not in str(request.get("tools")) for request in requests
        )
        results = [
            step.result for step in result.outcome.steps if step.kind == "tool_result"
        ]
        assert not any("builtin:calculator" in (text or "") for text in results)
        assert "Tool outside goal scope" in str(results)
        assert result.termination_reason.value == "permission_refused"
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions_enabled", [True, False])
async def test_goal_binding_controls_actual_instructions_and_file_read_after_navigation(
    stores, monkeypatch, tmp_path, instructions_enabled
):
    from pathlib import Path
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService

    runs, persistence, registry, req = stores
    selected = Path(req.binding.locator)
    (selected / "AGENTS.md").write_text("GOAL ROOT INSTRUCTIONS: read fixture.txt.\n")
    (selected / "fixture.txt").write_text("GOAL ROOT CONTENT")
    other = tmp_path / "foreground"
    other.mkdir()
    (other / "AGENTS.md").write_text("UNRELATED ROOT INSTRUCTIONS")
    (other / "fixture.txt").write_text("UNRELATED ROOT CONTENT")
    req = req.model_copy(
        update={"tool_scope": GoalToolScope(catalog_tools=("local:fs_read",))}
    )
    entered, release = threading.Event(), threading.Event()
    count = 0

    def provider(**kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(3)
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "read",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "fixture.txt"}),
                        },
                    }
                ],
            }
        else:
            message = {"content": "done"}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, store, session, controller, coordinator, gateway, calls = build_goal_rig(
        (runs, persistence, registry, req), monkeypatch, provider
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    store.set_session_project_instruction_state(
        session.id,
        ProjectInstructionControlState(
            project_instructions_enabled=instructions_enabled
        ),
    )
    controller.app = SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    original_setting = controller_module.get_cli_setting
    monkeypatch.setattr(
        controller_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if (section, key) == ("console", "local_tools_enabled")
            else str(other)
            if (section, key) == ("console", "workspace_root")
            else original_setting(section, key, default)
        ),
    )
    # Explicit approval still belongs to the normal instruction-disclosure owner.
    controller._confirm_project_instruction_dispatch = lambda notice: "proceed"
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        foreground = store.create_session(workspace_id="other-workspace")
        store.switch_session(foreground.id)
        release.set()
        result = await task
        assert result.outcome.status == "done", result
        assert ("GOAL ROOT INSTRUCTIONS" in str(calls[0])) is instructions_enabled
        assert len(calls) == 2, (calls, result.outcome)
        assert "GOAL ROOT CONTENT" in str(calls[1])
        assert "UNRELATED ROOT" not in str(calls)
        assert session.project_instruction_state.working_folder_binding_id == "binding"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_restored_call_is_rechecked_against_live_goal_scope(stores, monkeypatch):
    from Tests.Agents.test_provider_continuation_runtime import (
        CALCULATOR,
        CONFIG,
        _checkpoint,
        _deps,
        _pending_call,
    )
    from tldw_chatbook.Agents.agent_models import RunTerminationReason
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop
    from tldw_chatbook.Chat.provider_continuation import ContinuationRestoreTarget

    entered, release = threading.Event(), threading.Event()

    def provider(**kwargs):
        entered.set()
        assert release.wait(3)
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    goal, _store, _session, _controller, coordinator, gateway, _calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    task = asyncio.create_task(coordinator.dispatch_once(goal.id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        order, events = [], []
        deps = _deps(
            [],
            order=order,
            persist=events.append,
            invoke=lambda call: pytest.fail(
                "restored out-of-scope call reached executor"
            ),
            expand=lambda checkpoint: [],
            review=lambda calls: {},
        )
        with coordinator._active.context.scope():
            outcome = run_agent_loop(
                CONFIG,
                [],
                [CALCULATOR],
                deps,
                restore_provider_continuation=_checkpoint(_pending_call()),
                restore_provider_target=ContinuationRestoreTarget(
                    "deepseek",
                    "deepseek-v4-flash",
                    "responses",
                    "https://api.deepseek.com/v1",
                ),
                resume_provider_continuation=True,
            )
        assert outcome.termination_reason == RunTerminationReason.PERMISSION_REFUSED
        assert "model" not in order
        assert events[-1].result.value == "Tool outside goal scope."
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await gateway.aclose()
