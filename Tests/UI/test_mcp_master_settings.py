"""Explicit MCP master choices retain order, lifetime and truthful outcomes."""

import asyncio
import threading

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_root_settings import _open
from Tests.UI.test_mcp_workbench import WorkbenchAppWithBundledCSS
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.MCP.local_config_saves import get_mcp_local_config_saves
from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


def _persistence(monkeypatch, write):
    values = {"cached": True, "saved": True, "generation": 0}

    def mutate(payload, **kwargs):
        precondition = kwargs.get("mutation_precondition")
        if precondition is not None and not precondition():
            return ConfigMutationResult(False, False, None, conflict=True)
        enabled = payload["console"]["local_tools_enabled"]
        result = write(enabled)
        if result.file_replaced:
            values["saved"] = enabled
        if result.caches_reloaded:
            values["cached"] = enabled
            values["generation"] += 1
        return result

    monkeypatch.setattr(
        module,
        "current_config_identity",
        lambda: (values["generation"], str(module.get_cli_config_path())),
    )
    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda section, key, default=None: (
            values["cached"]
            if (section, key) == ("console", "local_tools_enabled")
            else default
        ),
    )
    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", mutate)
    monkeypatch.setattr(
        module,
        "save_setting_to_cli_config",
        lambda section, key, value: mutate({section: {key: value}}).fully_applied,
    )
    return values


@private_profile_test
async def test_master_reversal_waits_for_older_admitted_write(request, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def write(enabled):
        calls.append(enabled)
        if not enabled:
            entered.set()
            assert release.wait(10)
            finished.set()
        return ConfigMutationResult(True, True, None)

    values = _persistence(monkeypatch, write)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        _, canvas = await _open(pilot)
        toggle = canvas.query_one("#mcp-tools-local-enabled", Button)
        try:
            toggle.press()
            await _wait(pilot, entered.is_set)
            toggle.press()
            await _settle(pilot)
            assert calls == [False], "the later choice must wait for the older write"
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            await app.workers.wait_for_complete()
        await _wait(pilot, lambda: values["saved"] is True)
        assert calls == [False, True]
        assert "on" in str(toggle.label)


@private_profile_test
async def test_pending_master_choice_survives_read_only_refresh(request, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def write(enabled):
        entered.set()
        assert release.wait(10)
        return ConfigMutationResult(True, True, None)

    _persistence(monkeypatch, write)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        toggle = canvas.query_one("#mcp-tools-local-enabled", Button)
        try:
            toggle.press()
            await _wait(pilot, entered.is_set)
            await workbench._sync_children()
            await _settle(pilot)
            assert "off" in str(toggle.label)
            assert "Saving" in str(
                canvas.query_one("#mcp-tools-local-config-status").renderable
            )
        finally:
            release.set()
            await app.workers.wait_for_complete()


@private_profile_test
async def test_committed_master_with_failed_cache_is_not_reported_unsaved(
    request, monkeypatch
):
    values = _persistence(
        monkeypatch, lambda enabled: ConfigMutationResult(True, False, "cache_reload")
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        toggle = canvas.query_one("#mcp-tools-local-enabled", Button)
        toggle.press()
        await _settle(pilot)
        await app.workers.wait_for_complete()
        await workbench._sync_children()
        await _settle(pilot)
        assert values["saved"] is False and values["cached"] is True
        assert "off" in str(toggle.label)
        assert "saved to file" in str(
            canvas.query_one("#mcp-tools-local-config-status").renderable
        )


@private_profile_test
async def test_servers_reversal_uses_pending_tools_choice_and_same_queue(
    request, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def write(enabled):
        calls.append(enabled)
        if not enabled:
            entered.set()
            assert release.wait(10)
        return ConfigMutationResult(True, True, None)

    values = _persistence(monkeypatch, write)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        try:
            canvas.query_one("#mcp-tools-local-enabled", Button).press()
            await _wait(pilot, entered.is_set)
            workbench.set_mode("servers")
            workbench._selected_server_key = next(
                s.server_key for s in workbench._snapshots if s.source == "builtin"
            )
            await workbench._sync_children()
            await _settle(pilot)
            servers = workbench.query_one(MCPServersMode)
            toggle = servers.query_one("#mcp-gate-local_tools_enabled", Button)
            assert "off" in str(toggle.label)
            assert not servers.query_one("#mcp-gate-local-master-off-note").display
            toggle.press()
            await _settle(pilot)
            assert calls == [False]
            assert servers._tool_gates_by_id[toggle.id].enabled is True
            assert not servers.query_one("#mcp-gate-local-master-off-note").display
            assert not servers.query_one(
                "#mcp-gate-web_deep_search_enabled", Button
            ).disabled
            assert canvas._local_tools_enabled is True
        finally:
            release.set()
            await app.workers.wait_for_complete()
        assert calls == [False, True]
        assert values["saved"] is True


@pytest.mark.parametrize("succeeds", [False, True])
@private_profile_test
async def test_master_outcome_survives_workbench_recreation(
    request, monkeypatch, succeeds
):
    entered, release = threading.Event(), threading.Event()

    def write(enabled):
        entered.set()
        assert release.wait(10)
        return (
            ConfigMutationResult(True, True, None)
            if succeeds
            else ConfigMutationResult(False, False, "before_replace")
        )

    values = _persistence(monkeypatch, write)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        try:
            canvas.query_one("#mcp-tools-local-enabled", Button).press()
            await _wait(pilot, entered.is_set)
            await workbench.remove()
            await app.mount(MCPWorkbench(app_instance=app, id="replacement"))
            _, canvas = await _open(pilot)
            assert canvas._local_tools_enabled is False
            assert "Saving" in str(
                canvas.query_one("#mcp-tools-local-config-status").renderable
            )
        finally:
            release.set()
            await get_mcp_local_config_saves(app).close_and_drain()
        await _wait(
            pilot,
            lambda: (
                ("Tools off saved" if succeeds else "Save failed")
                in str(canvas.query_one("#mcp-tools-local-config-status").renderable)
            ),
        )
        assert values["saved"] is (not succeeds)
        assert canvas._local_tools_enabled is (not succeeds)


@private_profile_test
async def test_old_configuration_events_from_both_controls_are_refused(
    request, monkeypatch, tmp_path
):
    calls = []
    _persistence(monkeypatch, lambda value: calls.append(value))
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        previous = canvas._workspace_root_config
        monkeypatch.setattr(
            module, "get_cli_config_path", lambda: tmp_path / "other.toml"
        )
        workbench.on_mcp_tools_mode_local_tools_enabled_changed(
            MCPToolsMode.LocalToolsEnabledChanged(False, config_path=previous)
        )
        workbench.on_mcp_servers_mode_tool_gate_changed(
            MCPServersMode.ToolGateChanged(
                "console", "local_tools_enabled", False, config_path=previous
            )
        )
        await _settle(pilot)
        assert not calls
        assert not hasattr(app, "_mcp_local_config_saves")


@private_profile_test
async def test_master_catalog_failure_remains_distinct_from_persistence(
    request, monkeypatch
):
    values = _persistence(
        monkeypatch, lambda value: ConfigMutationResult(True, True, None)
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)

        collect = workbench._collect_snapshots

        async def fail():
            raise RuntimeError("private failure detail")

        monkeypatch.setattr(workbench, "_collect_snapshots", fail)
        canvas.query_one("#mcp-tools-local-enabled", Button).press()
        await app.workers.wait_for_complete()
        await pilot.pause(0.6)
        assert values["saved"] is False
        status = str(canvas.query_one("#mcp-tools-local-config-status").renderable)
        assert "Tools off saved" in status and "Catalog refresh failed" in status
        assert "private failure" not in status
        monkeypatch.setattr(workbench, "_collect_snapshots", collect)
        await workbench.reload()
        assert "Catalog refresh failed" not in str(
            canvas.query_one("#mcp-tools-local-config-status").renderable
        )


@pytest.mark.parametrize("second_publishes", [False, True])
@private_profile_test
async def test_real_atomic_master_write_preserves_or_repairs_partial_root(
    request, monkeypatch, tmp_path, second_publishes
):
    from pathlib import Path

    import toml

    from tldw_chatbook import config
    from tldw_chatbook.MCP.local_config_saves import (
        ConfigSaveRequest,
        MCPLocalConfigSaves,
        config_file_revision,
    )

    root = tmp_path / "saved-root"
    root.mkdir()
    path = module.get_cli_config_path()
    owner = MCPLocalConfigSaves()
    publish = config._publish_runtime_config_unlocked

    def fail_publication(**kwargs):
        raise RuntimeError("controlled cache publication failure")

    monkeypatch.setattr(config, "_publish_runtime_config_unlocked", fail_publication)
    root_request = ConfigSaveRequest(str(root), (None, 0), path, Path.cwd())
    first = await owner.submit(
        root_request, lambda: module._persist_mcp_workspace_root(root_request)
    )
    assert first.result.phase == "cache_refresh"
    if second_publishes:
        monkeypatch.setattr(config, "_publish_runtime_config_unlocked", publish)
    master = ConfigSaveRequest(False, None, path, key="local_tools_enabled")
    second = await owner.submit(
        master, lambda: module._persist_mcp_local_setting(master, False)
    )
    data = toml.loads(path.read_text())
    assert data["console"]["workspace_root"] == str(root)
    assert data["console"]["local_tools_enabled"] is False
    assert second.result.previous_file_revision == first.result.file_revision
    assert second.result.file_revision != first.result.file_revision
    current = owner.state_for("workspace_root")
    assert current.result.phase == ("saved" if second_publishes else "cache_refresh")
    generation = module.current_config_identity()[0]
    assert current.result.cache_generation == generation
    cached = config.get_cli_setting("console", "workspace_root", "")
    assert owner.known_root(
        path, cached, generation, config_file_revision(path)
    ) == str(root)
    assert (
        owner.known_value(
            "local_tools_enabled",
            path,
            config.get_cli_setting("console", "local_tools_enabled", True),
            generation,
            config_file_revision(path),
        )
        is False
    )


@private_profile_test
async def test_queued_master_write_refuses_retargeted_configuration(
    request, monkeypatch, tmp_path
):
    from tldw_chatbook.MCP.local_config_saves import (
        ConfigSaveRequest,
        ConfigSaveResult,
        MCPLocalConfigSaves,
    )

    entered, release = threading.Event(), threading.Event()
    owner = MCPLocalConfigSaves()
    path = module.get_cli_config_path()
    writes = []
    _persistence(monkeypatch, lambda value: writes.append(value))

    def hold():
        entered.set()
        assert release.wait(5)
        return ConfigSaveResult("failed")

    first = owner.submit(ConfigSaveRequest("", None, path), hold)
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        requested = ConfigSaveRequest(False, None, path, key="local_tools_enabled")
        queued = owner.submit(
            requested, lambda: module._persist_mcp_local_setting(requested, False)
        )
        monkeypatch.setattr(
            module, "get_cli_config_path", lambda: tmp_path / "other.toml"
        )
    finally:
        release.set()
        await owner.close_and_drain()
    assert first.done() and queued.result().result.phase == "changed"
    assert writes == []


@pytest.mark.parametrize("control", ["tools", "servers"])
@private_profile_test
async def test_projection_between_two_master_presses_cannot_lose_reversal(
    request, monkeypatch, control
):
    calls = []
    _persistence(
        monkeypatch,
        lambda value: calls.append(value) or ConfigMutationResult(True, True, None),
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        if control == "servers":
            await workbench._select_server_key(
                next(
                    s.server_key for s in workbench._snapshots if s.source == "builtin"
                )
            )
            canvas = workbench.query_one(MCPServersMode)
        button = canvas.query_one(
            "#mcp-tools-local-enabled"
            if control == "tools"
            else "#mcp-gate-local_tools_enabled",
            Button,
        )

        async def activate(event):
            handled = canvas.on_button_pressed(event)
            if control == "servers":
                await handled

        await activate(Button.Pressed(button))
        workbench._sync_master_save_status()
        await activate(Button.Pressed(button))
        await _wait(pilot, lambda: len(calls) == 2)
        await app.workers.wait_for_complete()
        assert calls == [False, True]
        assert workbench.query_one(MCPToolsMode)._local_tools_enabled is True


@pytest.mark.parametrize("control", ["tools", "servers"])
@private_profile_test
async def test_queued_master_button_does_not_retarget_after_projection(
    request, monkeypatch, tmp_path, control
):
    calls = []
    _persistence(
        monkeypatch,
        lambda value: calls.append(value) or ConfigMutationResult(True, True, None),
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        if control == "servers":
            await workbench._select_server_key(
                next(
                    s.server_key for s in workbench._snapshots if s.source == "builtin"
                )
            )
            canvas = workbench.query_one(MCPServersMode)
        button = canvas.query_one(
            "#mcp-tools-local-enabled"
            if control == "tools"
            else "#mcp-gate-local_tools_enabled",
            Button,
        )

        async def activate(event):
            handled = canvas.on_button_pressed(event)
            if control == "servers":
                await handled

        messages = []
        original = Button.post_message

        def capture(self, message):
            if self is button and isinstance(message, Button.Pressed):
                messages.append(message)
                return True
            return original(self, message)

        monkeypatch.setattr(Button, "post_message", capture)
        button.press()
        assert len(messages) == 1
        monkeypatch.setattr(
            module, "get_cli_config_path", lambda: tmp_path / "other.toml"
        )
        workbench._sync_master_save_status()
        await activate(messages[0])
        await _settle(pilot)
        assert calls == []
        assert not hasattr(app, "_mcp_local_config_saves")


@private_profile_test
async def test_master_poll_does_not_overwrite_unprocessed_root_edit(
    request, monkeypatch
):
    from textual.widgets import Input

    _persistence(monkeypatch, lambda value: ConfigMutationResult(True, True, None))
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        field.value = "first new root edit"
        workbench._sync_master_save_status()
        await _settle(pilot)
        assert field.value == "first new root edit"
        assert canvas._workspace_root_dirty


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_master_controls_and_receipts_paint_completely(
    request, monkeypatch, theme, size
):
    _persistence(monkeypatch, lambda value: ConfigMutationResult(True, True, None))
    app = WorkbenchAppWithBundledCSS()
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        workbench, _ = await _open(pilot)
        await workbench._select_server_key(
            next(s.server_key for s in workbench._snapshots if s.source == "builtin")
        )
        for mode, toggle_id, receipt_id in (
            ("tools", "mcp-tools-local-enabled", "mcp-tools-local-config-status"),
            ("servers", "mcp-gate-local_tools_enabled", "mcp-gate-local-master-status"),
        ):
            workbench.set_mode(mode)
            await _settle(pilot)
            toggle = workbench.query_one("#" + toggle_id, Button)
            toggle.focus()
            await _settle(pilot)
            region, clip = app.screen._compositor.visible_widgets[toggle]
            assert region.intersection(clip) == region
            painted = "".join(
                s.crop(region.x, region.right).text
                for s in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )
            assert "".join(toggle.label.plain.split()) in "".join(painted.split())
            status = workbench.query_one("#" + receipt_id)
            status.scroll_visible(animate=False, immediate=True)
            await _settle(pilot)
            region, clip = app.screen._compositor.visible_widgets[status]
            assert region.intersection(clip) == region
            painted = "".join(
                s.crop(region.x, region.right).text
                for s in app.screen._compositor.render_strips()[
                    region.y : region.bottom
                ]
            )
            assert "".join(str(status.renderable).split()) in "".join(painted.split())
