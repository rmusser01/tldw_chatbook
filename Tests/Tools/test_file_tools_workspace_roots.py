"""File tools honor workspace folder roots (spec §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Tools import file_operation_tools as fot
from tldw_chatbook.Tools import workspace_file_roots as wfr
from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture()
def bound_workspace(tmp_path, monkeypatch):
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="tool-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    ro_folder = tmp_path / "ro-project"
    rw_folder = tmp_path / "rw-project"
    ro_folder.mkdir()
    rw_folder.mkdir()
    registry.add_folder_binding("ws-a", ro_folder)
    registry.add_folder_binding("ws-a", rw_folder, allow_write=True)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(sandbox))
    return {"ro": ro_folder, "rw": rw_folder, "sandbox": sandbox}


@pytest.mark.asyncio
async def test_read_allowed_in_bound_folder(bound_workspace) -> None:
    target = bound_workspace["ro"] / "notes.md"
    target.write_text("hello")
    with wfr.run_workspace("ws-a"):
        result = await ReadFileTool().execute(file_path=str(target))
    assert result.get("error") is None
    assert result["content"] == "hello"


@pytest.mark.asyncio
async def test_write_denied_in_ro_folder_allowed_in_rw(bound_workspace) -> None:
    ro_target = bound_workspace["ro"] / "out.txt"
    rw_target = bound_workspace["rw"] / "out.txt"
    with wfr.run_workspace("ws-a"):
        denied = await WriteFileTool().execute(
            file_path=str(ro_target), content="x"
        )
        allowed = await WriteFileTool().execute(
            file_path=str(rw_target), content="x"
        )
    assert denied.get("error")
    assert allowed.get("error") is None
    assert rw_target.read_text() == "x"


@pytest.mark.asyncio
async def test_denial_names_roots_and_other_workspace_is_denied(
    bound_workspace, tmp_path
) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x")
    with wfr.run_workspace("ws-a"):
        result = await ReadFileTool().execute(file_path=str(outside))
    assert result.get("error")
    assert str(bound_workspace["sandbox"]) in result["error"]

    with wfr.run_workspace("workspace-default"):
        default_denied = await ReadFileTool().execute(
            file_path=str(bound_workspace["ro"] / "notes.md")
        )
    assert default_denied.get("error")


@pytest.mark.asyncio
async def test_zero_bindings_parity_with_sandbox(bound_workspace) -> None:
    inside = bound_workspace["sandbox"] / "kept.txt"
    inside.write_text("sandboxed")
    with wfr.run_workspace("workspace-default"):
        listed = await ListDirectoryTool().execute(
            directory_path=str(bound_workspace["sandbox"])
        )
        read = await ReadFileTool().execute(file_path=str(inside))
    assert listed.get("error") is None
    assert read.get("error") is None
