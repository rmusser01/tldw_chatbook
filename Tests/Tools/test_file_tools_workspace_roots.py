"""File tools honor workspace folder roots (spec §3)."""

from __future__ import annotations

import os
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


@pytest.mark.asyncio
async def test_recursive_listing_descends_into_bound_folder(bound_workspace) -> None:
    """Pins the containment_root fix: recursion must not cap at depth 0 for
    a directory that resolves into a bound workspace folder rather than the
    sandbox (pre-fix, `_is_within(item, sandbox_root)` was hardcoded and
    every child of a workspace folder failed it, silently stopping descent).
    """
    ro_folder = bound_workspace["ro"]
    deep_dir = ro_folder / "sub" / "deeper"
    deep_dir.mkdir(parents=True)
    deep_file = deep_dir / "file.txt"
    deep_file.write_text("nested")

    with wfr.run_workspace("ws-a"):
        result = await ListDirectoryTool().execute(
            directory_path=str(ro_folder), recursive=True, max_depth=5
        )

    assert result.get("error") is None
    deep_entries = [e for e in result["entries"] if e["name"] == "file.txt"]
    assert deep_entries, f"expected file.txt to be listed, got: {result['entries']}"
    assert deep_entries[0]["depth"] >= 2


@pytest.mark.asyncio
async def test_symlink_inside_bound_folder_cannot_escape(
    bound_workspace, tmp_path
) -> None:
    """A symlink planted inside a bound folder must not let a recursive
    listing (or a direct listing of the link itself) reach outside every
    allowed root.
    """
    ro_folder = bound_workspace["ro"]
    legit_file = ro_folder / "other.txt"
    legit_file.write_text("legit")

    target_dir = tmp_path / "loot"
    target_dir.mkdir()
    marker = target_dir / "secret.txt"
    marker.write_text("marker")

    link = ro_folder / "link"
    os.symlink(target_dir, link)

    with wfr.run_workspace("ws-a"):
        result = await ListDirectoryTool().execute(
            directory_path=str(ro_folder), recursive=True, max_depth=5
        )
        direct_link_result = await ListDirectoryTool().execute(
            directory_path=str(link)
        )

    # (a) Nothing from inside the symlink target leaks into the recursive
    # listing, but legitimate sibling content is still present.
    assert result.get("error") is None
    names = {e["name"] for e in result["entries"]}
    paths = {e["path"] for e in result["entries"]}
    assert "secret.txt" not in names
    assert not any("loot" in p for p in paths)
    assert "other.txt" in names

    # (b) Listing the symlink directly is denied outright — the resolved
    # target sits outside every allowed root.
    assert direct_link_result.get("error")
