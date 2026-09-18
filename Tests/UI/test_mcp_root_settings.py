"""MCP root drafts and explicit writes retain their own lifetime and meaning."""

import asyncio
import threading
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_mcp_compact_readability import _settle
from Tests.UI.test_mcp_workbench import WorkbenchAppWithBundledCSS
from Tests.UI.test_tool_profile_review_lifetime import _wait
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.MCP.root_settings import (
    RootSaveRequest,
    RootSaveUnavailable,
    get_mcp_root_saves,
)
from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench


async def _open(pilot):
    await pilot.app.workers.wait_for_complete()
    workbench = pilot.app.query_one(MCPWorkbench)
    await workbench._mount_deferred_canvases()
    await workbench._sync_children()
    workbench.set_mode("tools")
    await _settle(pilot)
    return workbench, workbench.query_one(MCPToolsMode)


def _persistence(monkeypatch, write):
    values = {("console", "workspace_root"): "", ("test", "generation"): 0}

    def mutate(payload, **kwargs):
        precondition = kwargs.get("mutation_precondition")
        if precondition is not None and not precondition():
            return ConfigMutationResult(False, False, None, conflict=True)
        root = payload["console"]["workspace_root"]
        result = write(root)
        if result.caches_reloaded:
            values[("test", "generation")] += 1
        if result.file_replaced:
            values[("console", "workspace_root")] = root
        return result

    monkeypatch.setattr(
        module,
        "current_config_identity",
        lambda: (values[("test", "generation")], str(module.get_cli_config_path())),
    )
    monkeypatch.setattr(
        module,
        "get_cli_setting",
        lambda section, key, default=None: values.get((section, key), default),
    )
    monkeypatch.setattr(
        module, "apply_settings_mutation_to_cli_config", mutate, raising=False
    )
    # Exercise the old implementation for red evidence with the same writer.
    monkeypatch.setattr(
        module,
        "save_setting_to_cli_config",
        lambda section, key, value: mutate({section: {key: value}}).fully_applied,
    )
    return values


@private_profile_test
async def test_read_only_refresh_preserves_root_draft_and_validation_receipt(
    request, monkeypatch, tmp_path
):
    writes = []
    _persistence(monkeypatch, lambda root: writes.append(root))
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test(size=(80, 24)) as pilot:
        workbench, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        field.value = str(tmp_path / "not-a-directory")
        await _settle(pilot)
        canvas.query_one("#mcp-tools-workspace-save", Button).press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        await workbench._sync_children()
        await _settle(pilot)
        assert field.value == str(tmp_path / "not-a-directory")
        assert not writes
        status = canvas.query_one("#mcp-tools-workspace-status")
        assert status.has_class("is-error")
        assert "not saved" in str(status.renderable).lower()
        region, clip = app.screen._compositor.visible_widgets[status]
        assert region.intersection(clip) == region


@private_profile_test
async def test_explicit_root_saves_finish_in_submission_order(
    request, monkeypatch, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def write(root):
        calls.append(root)
        if root == str(first):
            entered.set()
            assert release.wait(10)
            finished.set()
        return ConfigMutationResult(True, True, None)

    values = _persistence(monkeypatch, write)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        _, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        button = canvas.query_one("#mcp-tools-workspace-save", Button)
        try:
            field.value = str(first)
            await _settle(pilot)
            button.press()
            await _wait(pilot, entered.is_set)
            field.value = str(second)
            await _settle(pilot)
            button.press()
            await pilot.pause(0.1)
            assert calls == [str(first)], "new save must wait for admitted older write"
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            await app.workers.wait_for_complete()
        await _wait(pilot, lambda: values[("console", "workspace_root")] == str(second))
        assert calls == [str(first), str(second)]
        assert field.value == str(second)


@pytest.mark.parametrize("return_to_saved", [False, True])
@private_profile_test
async def test_older_save_does_not_canonicalize_a_newer_aba_draft(
    request, monkeypatch, tmp_path, return_to_saved
):
    root = tmp_path / "root"
    root.mkdir()
    raw = str(root) + "/."
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def write(value):
        entered.set()
        assert release.wait(10)
        finished.set()
        return ConfigMutationResult(True, True, None)

    values = _persistence(monkeypatch, write)
    if return_to_saved:
        values[("console", "workspace_root")] = "prior saved root"
    expected = "prior saved root" if return_to_saved else raw
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        _, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        try:
            field.value = raw
            await _settle(pilot)
            canvas.query_one("#mcp-tools-workspace-save", Button).press()
            await _wait(pilot, entered.is_set)
            field.value = "another draft"
            await _settle(pilot)
            field.value = expected
            await _settle(pilot)
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            await app.workers.wait_for_complete()
        await _settle(pilot)
        assert field.value == expected
        status = canvas.query_one("#mcp-tools-workspace-status")
        assert "not saved" in str(status.renderable).lower()


@private_profile_test
async def test_failed_root_save_keeps_draft_for_retry(request, monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    results = [
        ConfigMutationResult(False, False, "before_replace"),
        ConfigMutationResult(True, True, None),
    ]
    values = _persistence(monkeypatch, lambda root: results.pop(0))
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        button = canvas.query_one("#mcp-tools-workspace-save", Button)
        field.value = str(root)
        await _settle(pilot)
        button.press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        await workbench._sync_children()
        assert field.value == str(root)
        assert values[("console", "workspace_root")] == ""
        status = canvas.query_one("#mcp-tools-workspace-status")
        assert status.has_class("is-error")
        button.press()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert values[("console", "workspace_root")] == str(root)
        assert not status.has_class("is-error")
        assert "Hub" in str(status.renderable)


@pytest.mark.parametrize("succeeds", [True, False])
@private_profile_test
async def test_root_save_survives_recreated_workbench(
    request, monkeypatch, tmp_path, succeeds
):
    root = tmp_path / "root"
    root.mkdir()
    entered, release = threading.Event(), threading.Event()

    def write(value):
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
            canvas.query_one("#mcp-tools-workspace-root", Input).value = str(root)
            canvas.query_one("#mcp-tools-workspace-save", Button).press()
            await _wait(pilot, entered.is_set)
            await workbench.remove()
            await app.mount(MCPWorkbench(app_instance=app, id="mcp-workbench-new"))
            _, canvas = await _open(pilot)
            await _wait(
                pilot,
                lambda: (
                    "Saving"
                    in str(canvas.query_one("#mcp-tools-workspace-status").renderable)
                ),
            )
        finally:
            release.set()
            await get_mcp_root_saves(app).close_and_drain()
        await _wait(
            pilot,
            lambda: (
                ("Saved" if succeeds else "Review the current root")
                in str(canvas.query_one("#mcp-tools-workspace-status").renderable)
            ),
        )
        expected = str(root) if succeeds else ""
        assert values[("console", "workspace_root")] == expected
        assert canvas.query_one("#mcp-tools-workspace-root", Input).value == expected
        if not succeeds:
            assert "Your edits are kept" not in str(
                canvas.query_one("#mcp-tools-workspace-status").renderable
            )


@private_profile_test
async def test_catalog_refresh_failure_does_not_report_save_failure(
    request, monkeypatch, tmp_path
):
    root = tmp_path / "root"
    root.mkdir()
    values = _persistence(
        monkeypatch, lambda value: ConfigMutationResult(True, True, None)
    )
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)

        async def fail():
            raise RuntimeError("private failure body")

        monkeypatch.setattr(workbench, "_collect_snapshots", fail)
        canvas.query_one("#mcp-tools-workspace-root", Input).value = str(root)
        canvas.query_one("#mcp-tools-workspace-save", Button).press()
        await _wait(
            pilot,
            lambda: (
                "Catalog refresh failed"
                in str(canvas.query_one("#mcp-tools-workspace-status").renderable)
            ),
        )
        assert values[("console", "workspace_root")] == str(root)
        status = str(canvas.query_one("#mcp-tools-workspace-status").renderable)
        assert "Saved local MCP" in status and "private failure" not in status


@private_profile_test
async def test_root_shutdown_drains_and_fences_both_config_owners(request, monkeypatch):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.MCP.root_settings import RootSaveResult
    from tldw_chatbook.Tool_Packs.operations import ToolProfileWriteUnavailable

    app = object.__new__(TldwCli)
    owner = get_mcp_root_saves(app)
    profiles = app._get_tool_profile_operations()
    entered, release = threading.Event(), threading.Event()
    later = []

    async def fail_later(self):
        later.append(True)
        raise RuntimeError("later owner")

    monkeypatch.setattr(TldwCli, "_shutdown_recovery_service", fail_later)

    def write():
        entered.set()
        assert release.wait(5)
        return RootSaveResult("saved", "", True)

    task = owner.submit(
        RootSaveRequest("", (None, 0), Path("config"), Path.cwd()), write
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        shutdown = asyncio.create_task(app._shutdown_app_owned_lifecycles())
        await asyncio.sleep(0)
        assert not shutdown.done() and not later
        with pytest.raises(RootSaveUnavailable):
            get_mcp_root_saves(app)
        with pytest.raises(ToolProfileWriteUnavailable):
            profiles.start("import", "test", lambda cancelled: None)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert not task.done()
    finally:
        release.set()
        await task
    with pytest.raises(RuntimeError, match="later owner"):
        await app._shutdown_app_owned_lifecycles()
    assert later == [True]


@private_profile_test
def test_root_writer_uses_captured_cwd_and_locked_config_fence(
    request, monkeypatch, tmp_path
):
    root = tmp_path / "captured" / "relative"
    root.mkdir(parents=True)
    profile = tmp_path / "config.toml"
    current = [profile]
    writes = []
    monkeypatch.setattr(module, "get_cli_config_path", lambda: current[0])

    def mutate(payload, **kwargs):
        if not kwargs["mutation_precondition"]():
            return ConfigMutationResult(False, False, None, conflict=True)
        writes.append(payload)
        return ConfigMutationResult(False, False, None)

    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", mutate)
    submitted = RootSaveRequest("relative", (None, 0), profile, root.parent)
    outcome = module._persist_mcp_workspace_root(submitted)
    assert outcome.phase == "saved" and outcome.stored == str(root)
    current[0] = tmp_path / "other.toml"
    assert module._persist_mcp_workspace_root(submitted).phase == "changed"
    assert len(writes) == 1


@pytest.mark.parametrize("phase", ["saved", "cache_refresh"])
@pytest.mark.parametrize("changed", ["generation", "file"])
@private_profile_test
async def test_newer_publication_cannot_be_hidden_by_old_save_receipt(
    request, monkeypatch, phase, changed
):
    from tldw_chatbook.MCP.root_settings import RootSaveResult, RootSaveState

    values = _persistence(monkeypatch, lambda root: None)
    app = WorkbenchAppWithBundledCSS()
    async with app.run_test() as pilot:
        workbench, canvas = await _open(pilot)
        field = canvas.query_one("#mcp-tools-workspace-root", Input)
        field.value = "submitted B"
        await _settle(pilot)
        old = RootSaveState(
            1,
            RootSaveRequest(
                field.value,
                canvas.workspace_root_draft_identity,
                module.get_cli_config_path(),
                Path.cwd(),
            ),
            RootSaveResult(phase, "canonical B", phase == "saved", 1, (1, 1, 1, 1)),
        )
        values[("test", "generation")] = 2 if changed == "generation" else 1
        values[("console", "workspace_root")] = "independent C"
        await workbench._sync_children()
        canvas.project_workspace_root_save(
            old,
            generation=values[("test", "generation")],
            file_revision=(1, 1, 2 if changed == "file" else 1, 1),
        )
        assert field.value == "submitted B"
        status = str(canvas.query_one("#mcp-tools-workspace-status").renderable)
        assert "since changed" in status and "not saved" in status


@private_profile_test
def test_real_atomic_root_save_pins_publication_generation_on_repeat(request, tmp_path):
    root = tmp_path / "real-root"
    root.mkdir()
    before = module.current_config_identity()[0]
    submitted = RootSaveRequest(
        str(root), (None, 0), module.get_cli_config_path(), Path.cwd()
    )
    result = module._persist_mcp_workspace_root(submitted)
    assert result.phase == "saved" and result.caches_reloaded
    assert result.cache_generation == before + 1 == module.current_config_identity()[0]
    repeated = module._persist_mcp_workspace_root(submitted)
    assert repeated.phase == "saved" and repeated.caches_reloaded
    assert (
        repeated.cache_generation
        == result.cache_generation + 1
        == module.current_config_identity()[0]
    )


@private_profile_test
async def test_external_config_reload_supersedes_partial_save_without_generation_bump(
    request, monkeypatch, tmp_path
):
    import toml

    from tldw_chatbook import config
    from tldw_chatbook.MCP.root_settings import MCPRootSaves, root_config_file_revision

    root, external = tmp_path / "saved", tmp_path / "external"
    root.mkdir()
    external.mkdir()
    owner = MCPRootSaves()
    submitted = RootSaveRequest(
        str(root), (None, 0), module.get_cli_config_path(), Path.cwd()
    )
    publish = config._publish_runtime_config_unlocked

    def fail_publication(**kwargs):
        raise RuntimeError("controlled cache publication failure")

    monkeypatch.setattr(config, "_publish_runtime_config_unlocked", fail_publication)
    outcome = await owner.submit(
        submitted, lambda: module._persist_mcp_workspace_root(submitted)
    )
    assert outcome.result.phase == "cache_refresh"
    generation = module.current_config_identity()[0]
    assert owner.known_root(
        submitted.config_path,
        "stale",
        generation,
        root_config_file_revision(submitted.config_path),
    ) == str(root)
    monkeypatch.setattr(config, "_publish_runtime_config_unlocked", publish)
    data = toml.loads(submitted.config_path.read_text())
    data["console"]["workspace_root"] = str(external)
    submitted.config_path.write_text(toml.dumps(data))
    config.load_cli_config_and_ensure_existence(force_reload=True)
    assert module.current_config_identity()[0] == generation
    cached = module.get_cli_setting("console", "workspace_root")
    assert cached == str(external)
    assert owner.known_root(
        submitted.config_path,
        cached,
        generation,
        root_config_file_revision(submitted.config_path),
    ) == str(external)
