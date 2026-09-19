"""Permission controls must act on exactly what their displayed view reviewed."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_inspector import InspectorApp, _tool
from Tests.UI.test_mcp_session_revoke_ownership import CONTEXT, held_press
from Tests.UI.test_mcp_workbench import PermissionsApp
from tldw_chatbook.MCP.permission_store import EffectiveToolState, definition_hash
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import PermissionProfileContext
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

SELECTORS = {
    "remove": "#mcp-inspector-arg-rule-remove-0",
    "reallow": "#mcp-inspector-reallow",
}
RULES = [
    {
        "rule_id": '{"query":"one"}',
        "args_json": '{"query":"one"}',
        "profile_id": "parent",
    },
    {
        "rule_id": '{"query":"two"}',
        "args_json": '{"query":"two"}',
        "profile_id": "research",
    },
]


async def show(inspector, *, tool=None, rules=RULES, context=CONTEXT):
    await inspector.show_permission(
        tool or _tool(),
        EffectiveToolState(state="ask", origin="tool_override", config_changed=True),
        profile_context=context,
        arg_rules=rules,
    )


def actions(app):
    return [
        e
        for e in app.events
        if isinstance(
            e, (MCPInspector.RemoveArgRuleRequested, MCPInspector.ReallowRequested)
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", SELECTORS)
@pytest.mark.parametrize(
    "transition", ["same", "reorder", "tool", "profile", "clear", "roundtrip"]
)
@private_profile_test
async def test_obsolete_permission_controls_cannot_target_a_replacement(
    request, monkeypatch, action, transition
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        event = held_press(app.query_one(SELECTORS[action], Button), monkeypatch)
        if transition == "clear":
            await inspector.show_tool(None)
        elif transition == "reorder":
            await show(inspector, rules=list(reversed(RULES)))
        elif transition == "tool":
            await show(inspector, tool=_tool(name="fetch"))
        elif transition == "profile":
            await show(
                inspector, context=PermissionProfileContext("other", 8, "c" * 64, 4)
            )
        elif transition == "roundtrip":
            await show(inspector, tool=_tool(name="fetch"))
            await show(inspector)
        else:
            await show(inspector)
        inspector.post_message(event)
        await pilot.pause()
        assert actions(app) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", SELECTORS)
@private_profile_test
async def test_live_permission_controls_are_consumed_once_with_the_displayed_target(
    request, action
):
    app = InspectorApp()
    async with app.run_test(size=(100, 60)) as pilot:
        inspector = app.query_one(MCPInspector)
        await show(inspector)
        await pilot.pause()
        button = app.query_one(SELECTORS[action], Button)
        button.press()
        button.press()
        await pilot.pause()
        assert len(actions(app)) == 1
        event = actions(app)[0]
        assert (event.server_key, event.tool_name, event.profile_context) == (
            "local:docs",
            "search",
            CONTEXT,
        )
        if action == "remove":
            assert (event.rule_id, event.owner_profile_id) == (
                '{"query":"one"}',
                "parent",
            )
        await show(inspector, tool=_tool(name="fetch"))
        await pilot.pause()
        app.query_one(SELECTORS[action], Button).press()
        await pilot.pause()
        assert len(actions(app)) == 2
        assert actions(app)[1].tool_name == "fetch"


async def open_permissions(app, pilot):
    workbench = app.query_one(MCPWorkbench)
    async with asyncio.timeout(5):
        while workbench.is_loading or workbench._reloading:
            await pilot.pause()
    tool = workbench._tool_for("local:docs", "search")
    tool = replace(
        tool,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    workbench._last_hub_tools = [
        tool if t.tool_id == tool.tool_id else t for t in workbench._last_hub_tools
    ]
    store = app.unified_mcp_service.permission_store
    store.set_tool_state(tool.server_key, tool.name, "allow", definition_hash="a" * 64)
    store.add_tool_arg_rule(
        tool.server_key,
        tool.name,
        args={"query": "one"},
        definition_hash=definition_hash(tool.description, tool.input_schema),
    )
    workbench.set_mode("permissions")
    await pilot.pause()
    await workbench._sync_permissions_mode()
    inspector = app.query_one(MCPInspector)
    context = workbench._tool_policy_profile_context
    await inspector.show_permission(
        tool,
        workbench._effective_for_display(tool),
        profile_context=context,
        arg_rules=workbench._arg_rules_for_row(tool, context.profile_id),
    )
    await pilot.pause()
    return workbench, inspector, tool


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["description", "schema"])
@private_profile_test
async def test_reallow_refuses_a_definition_changed_since_rendering(
    request, tmp_path, monkeypatch, drift
):
    app = PermissionsApp(tmp_path / "permissions.json")
    async with app.run_test(size=(100, 60)) as pilot:
        workbench, inspector, tool = await open_permissions(app, pilot)
        event = held_press(app.query_one(SELECTORS["reallow"], Button), monkeypatch)
        if drift == "description":
            changed = replace(tool, description="A newly changed definition")
            workbench._last_hub_tools = [
                changed if t.tool_id == tool.tool_id else t
                for t in workbench._last_hub_tools
            ]
        else:
            tool.input_schema["description"] = "A newly changed schema"
        before = app.unified_mcp_service.permission_store.load()
        inspector.post_message(event)
        await pilot.pause()
        assert app.unified_mcp_service.permission_store.load() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("action", SELECTORS)
@private_profile_test
async def test_failed_permission_action_is_retryable(
    request, tmp_path, monkeypatch, action
):
    app = PermissionsApp(tmp_path / "permissions.json")
    async with app.run_test(size=(100, 60)) as pilot:
        workbench, _inspector, tool = await open_permissions(app, pilot)
        service = app.unified_mcp_service
        method = "remove_tool_arg_rule" if action == "remove" else "set_tool_state"
        original = getattr(service, method)
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("synthetic permission write failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(service, method, fail_once)
        app.query_one(SELECTORS[action], Button).press()
        await pilot.pause()
        assert failed
        button = app.query_one(SELECTORS[action], Button)
        assert not button.disabled
        button.press()
        await pilot.pause()
        if action == "remove":
            assert (
                service.permission_store.list_tool_arg_rules(tool.server_key, tool.name)
                == []
            )
        else:
            assert workbench._effective_for_display(tool).state == "allow"
        assert not list(app.query(SELECTORS[action]))


@pytest.mark.asyncio
@pytest.mark.parametrize("action", SELECTORS)
@pytest.mark.parametrize("transition", ["tool", "clear", "roundtrip"])
@private_profile_test
async def test_permission_completion_preserves_a_newer_selection(
    request, tmp_path, monkeypatch, action, transition
):
    app = PermissionsApp(tmp_path / "permissions.json")
    async with app.run_test(size=(100, 60)) as pilot:
        workbench, inspector, tool = await open_permissions(app, pilot)
        entered, release = asyncio.Event(), asyncio.Event()
        original = workbench._sync_permissions_mode

        async def hold_sync(*args, **kwargs):
            entered.set()
            await release.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(workbench, "_sync_permissions_mode", hold_sync)
        app.query_one(SELECTORS[action], Button).press()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if transition == "clear":
                await inspector.show_tool(None)
            else:
                await show(
                    inspector,
                    tool=_tool(name="fetch"),
                    context=workbench._tool_policy_profile_context,
                )
                if transition == "roundtrip":
                    await show(
                        inspector,
                        tool=tool,
                        rules=list(reversed(RULES)),
                        context=workbench._tool_policy_profile_context,
                    )
            expected_tool = inspector._current_permission_tool
            expected_rules = inspector._current_permission_arg_rules.copy()
        finally:
            release.set()
        await pilot.pause()
        assert inspector._current_permission_tool == expected_tool
        assert inspector._current_permission_arg_rules == expected_rules


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", ["retry", "session", "success"])
@private_profile_test
async def test_cached_permission_refresh_does_not_reapprove_an_unseen_definition(
    request, tmp_path, monkeypatch, refresh
):
    app = PermissionsApp(tmp_path / "permissions.json")
    async with app.run_test(size=(100, 60)) as pilot:
        workbench, inspector, tool = await open_permissions(app, pilot)
        service = app.unified_mcp_service
        before = service.permission_store.load()
        container = app.query_one("#mcp-inspector-permission")
        original_remove = container.remove_children
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_removal():
            entered.set()
            await release.wait()
            await original_remove()

        if refresh == "success":
            original_sync = workbench._sync_permissions_mode

            async def hold_sync(*args, **kwargs):
                entered.set()
                await release.wait()
                await original_sync(*args, **kwargs)

            monkeypatch.setattr(workbench, "_sync_permissions_mode", hold_sync)
        else:
            monkeypatch.setattr(container, "remove_children", hold_removal)
        original_set = service.set_tool_state

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic permission write failure")

        pending = None
        if refresh == "retry":
            monkeypatch.setattr(service, "set_tool_state", fail)
            app.query_one(SELECTORS["reallow"], Button).press()
        elif refresh == "success":
            app.query_one(SELECTORS["remove"], Button).press()
        else:
            pending = asyncio.create_task(
                inspector.refresh_permission_session_approvals(
                    [], profile_context=workbench._tool_policy_profile_context
                )
            )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if refresh == "success":
                before = service.permission_store.load()
            tool.input_schema["description"] = (
                "Unseen definition after the write failed"
            )
        finally:
            release.set()
        if pending is not None:
            await pending
        await pilot.pause()
        monkeypatch.setattr(service, "set_tool_state", original_set)
        app.query_one(SELECTORS["reallow"], Button).press()
        await pilot.pause()
        assert service.permission_store.load() == before
