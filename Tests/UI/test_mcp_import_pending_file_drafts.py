"""MCP file completion must preserve the originating panel's newer draft."""

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp

FILE_TEXT = '{"mcpServers": {"selected": {"command": "npx"}}}'


@asynccontextmanager
async def mounted_import(monkeypatch, tmp_path):
    from Tests.UI.test_mcp_workbench import ImportHubService
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module
    from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel
    from tldw_chatbook.UI.Screens.settings_tools_hub import ToolsHubPanel

    monkeypatch.setattr(module, "_mcp_import_home", lambda: str(tmp_path))

    class HubApp(ConsolidatedCSSApp):
        def __init__(self):
            super().__init__()
            self.unified_mcp_service = ImportHubService()
            self.notices = []

        def compose(self):
            yield ToolsHubPanel(self)

        def notify(self, message, **kwargs):
            self.notices.append((str(message), kwargs.get("severity")))

    host = HubApp()
    async with host.run_test(size=(140, 55)) as pilot:
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#mcp-import-server")
        await pilot.pause()
        panel = host.query_one(MCPImportPanel)
        yield host, pilot, panel


async def pick_file(host, pilot, selected):
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    await pilot.click("#mcp-import-file")
    await pilot.pause()
    assert isinstance(host.screen, EnhancedFileOpen)
    host.screen.dismiss(selected)
    await pilot.pause()


async def wait_for(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def held_file_read(monkeypatch, selected):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = Path.read_text

    def read(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == selected:
            entered.set()
            try:
                assert release.wait(20), "MCP file read gate was not released"
            finally:
                finished.set()
        return result

    monkeypatch.setattr(Path, "read_text", read)
    try:
        yield entered, release, finished
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["untouched", "typing", "clear", "empty-again"])
@private_profile_test
async def test_pending_mcp_file_read_preserves_newer_draft(
    request, tmp_path, monkeypatch, later
):
    selected = tmp_path / "selected.json"
    selected.write_text(FILE_TEXT, encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        editor = panel.query_one(TextArea)
        if later == "clear":
            editor.load_text("seed")
            await pilot.pause()
        async with held_file_read(monkeypatch, selected) as (
            entered,
            release,
            finished,
        ):
            await pick_file(host, pilot, selected)
            await wait_for(entered.is_set)
            expected = FILE_TEXT
            if later != "untouched":
                editor.focus()
                await pilot.press("x")
                if later == "clear":
                    await pilot.press("end", *(["backspace"] * 5))
                elif later == "empty-again":
                    await pilot.press("backspace")
                expected = "x" if later == "typing" else ""
                assert editor.text == expected
            release.set()
            await wait_for(finished.is_set)
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert editor.text == expected
            if later == "typing":
                await pilot.click("#mcp-import-preview")
                await pilot.pause()
                assert editor.text == "x"
                assert str(panel.query_one("#mcp-import-error", Static).renderable)
                assert panel.query_one("#mcp-import-apply", Button).disabled


@pytest.mark.asyncio
@private_profile_test
async def test_pending_mcp_file_read_cannot_replace_reopened_panel(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel

    selected = tmp_path / "selected.json"
    selected.write_text(FILE_TEXT, encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        async with held_file_read(monkeypatch, selected) as (
            entered,
            release,
            finished,
        ):
            await pick_file(host, pilot, selected)
            await wait_for(entered.is_set)
            await pilot.click("#mcp-import-cancel")
            await pilot.pause()
            await pilot.click("#mcp-import-server")
            await pilot.pause()
            replacement = host.query_one(MCPImportPanel)
            assert replacement is not panel
            replacement.query_one(TextArea).focus()
            await pilot.press("x")
            release.set()
            await wait_for(finished.is_set)
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert replacement.query_one(TextArea).text == "x"


@pytest.mark.asyncio
@private_profile_test
async def test_newer_mcp_file_selection_supersedes_pending_read(
    request, tmp_path, monkeypatch
):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(FILE_TEXT, encoding="utf-8")
    second.write_text('{"mcpServers": {}}', encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        async with held_file_read(monkeypatch, first) as (entered, release, finished):
            await pick_file(host, pilot, first)
            await wait_for(entered.is_set)
            await pick_file(host, pilot, second)
            await host.workers.wait_for_complete()
            assert panel.query_one(TextArea).text == '{"mcpServers": {}}'
            release.set()
            await wait_for(finished.is_set)
            await pilot.pause()
            assert panel.query_one(TextArea).text == '{"mcpServers": {}}'


@pytest.mark.asyncio
@private_profile_test
async def test_mcp_file_selection_captures_draft_before_worker_admission(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    selected = tmp_path / "selected.json"
    selected.write_text(FILE_TEXT, encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        await pilot.click("#mcp-import-file")
        await pilot.pause()
        picker = host.screen
        assert isinstance(picker, EnhancedFileOpen)
        # Invoke the actual registered picker callback synchronously, leaving the
        # scheduled coroutine unadmitted until the subsequent accepted edit.
        callback = picker._result_callbacks[-1].callback
        picker.dismiss(None)
        await pilot.pause()
        callback(selected)
        editor = panel.query_one(TextArea)
        editor.insert("x")
        editor.clear()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert editor.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("later", ["typing", "empty-again"])
@private_profile_test
async def test_mcp_pending_read_sees_keys_before_parent_changed_delivery(
    request, tmp_path, monkeypatch, later
):
    selected = tmp_path / "selected.json"
    selected.write_text(FILE_TEXT, encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        parent_entered, parent_release = asyncio.Event(), asyncio.Event()

        async def hold_parent():
            parent_entered.set()
            await asyncio.wait_for(parent_release.wait(), 10)

        async with held_file_read(monkeypatch, selected) as (
            entered,
            release,
            finished,
        ):
            await pick_file(host, pilot, selected)
            await wait_for(entered.is_set)
            editor = panel.query_one(TextArea)
            editor.focus()
            panel.call_next(hold_parent)
            await asyncio.wait_for(parent_entered.wait(), 3)
            keys = asyncio.create_task(pilot.press("x"))
            try:
                await wait_for(lambda: editor.text == "x")
                if later == "empty-again":
                    editor.clear()
                expected = "x" if later == "typing" else ""
                release.set()
                await wait_for(finished.is_set)
                await host.workers.wait_for_complete()
                assert editor.text == expected
            finally:
                parent_release.set()
                release.set()
                await keys
            await pilot.pause()
            assert editor.text == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["outside", "missing", "size", "utf8"])
@private_profile_test
async def test_mcp_picker_file_validation_preserves_raw_draft(
    request, tmp_path, monkeypatch, failure
):
    from tldw_chatbook.UI.MCP_Modules import mcp_workbench as module

    selected = tmp_path / "bad.json"
    if failure == "size":
        selected.write_bytes(b"x" * (module.MAX_MCP_IMPORT_FILE_BYTES + 1))
    elif failure == "utf8":
        selected.write_bytes(b"\xff\xfe")
    elif failure == "outside":
        selected.write_text(FILE_TEXT, encoding="utf-8")
    async with mounted_import(monkeypatch, tmp_path) as (host, pilot, panel):
        if failure == "outside":
            home = tmp_path / "home"
            home.mkdir()
            monkeypatch.setattr(module, "_mcp_import_home", lambda: str(home))
        editor = panel.query_one(TextArea)
        editor.focus()
        await pilot.press("x")
        await pick_file(host, pilot, selected)
        await host.workers.wait_for_complete()
        await pilot.pause()
        expected = {
            "outside": "failed validation",
            "missing": "Could not read",
            "size": "too large",
            "utf8": "Could not read",
        }[failure]
        assert any(
            expected in message and severity == "error"
            for message, severity in host.notices
        )
        assert editor.text == "x"
