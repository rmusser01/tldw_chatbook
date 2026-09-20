"""Builtin file-tool family enforces per-binding exclusions (spec 2026-09-20 §2).

Family 2 (``Tools/file_operation_tools.py``): a folder binding whose metadata
carries user exclusions (``registry.add_binding_exclusion``) must keep those
paths from the builtin ``read_file``/``list_directory``/``write_file``/
``glob_files`` tools, exactly as the system denylist already does -- the
refusal is the DENYLIST's own ("protected path"), reached by folding the
exclusions into the per-call sensitive context
(``Utils.sensitive_paths.merge_sensitive_context``), never by a second
message or a second code path.

The harness is copied from ``Tests/Tools/test_file_tools_workspace_roots.py``'s
``bound_workspace`` fixture (real registry service, monkeypatched
``_registry_factory``/``_resolve_sandbox_config``, tools invoked inside
``run_workspace``) so these tests exercise the same path production takes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Tools import file_operation_tools as fot
from tldw_chatbook.Tools import workspace_file_roots as wfr
from tldw_chatbook.Tools.file_operation_tools import (
    GlobFiles,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from tldw_chatbook.Tools.workspace_file_roots import current_folder_binding_exclusions

KEY_MARKER = "SYNTHETIC-NOT-A-REAL-PRIVATE-KEY-EXCL"
WORKSPACE_ID = "ws-x"


def _make_workspace(tmp_path: Path, monkeypatch, *, exclude_secrets: bool) -> Path:
    """The ``bound_workspace`` harness, plus one optional binding exclusion.

    Returns the bound (rw) folder root. The sandbox stays reachable so the
    tools' confinement never explains a refusal: every denied target below
    sits INSIDE the bound root, which ``validate_path_multi`` admits.
    """
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="excl-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id=WORKSPACE_ID, name="Client X")
    root = tmp_path / "rw-project"
    root.mkdir()
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text(
        f"-----BEGIN PRIVATE KEY-----\n{KEY_MARKER}\n"
    )
    (root / "notes.md").write_text("hello\n")
    binding = registry.add_folder_binding(WORKSPACE_ID, root, allow_write=True)
    if exclude_secrets:
        registry.add_binding_exclusion(WORKSPACE_ID, binding.binding_id, "secrets")
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(sandbox))
    return root


@pytest.fixture()
def excluded_root(tmp_path, monkeypatch) -> Path:
    return _make_workspace(tmp_path, monkeypatch, exclude_secrets=True)


@pytest.fixture()
def plain_root(tmp_path, monkeypatch) -> Path:
    """Same workspace, no exclusions -- the behavior-must-not-change control."""
    return _make_workspace(tmp_path, monkeypatch, exclude_secrets=False)


# ---------------------------------------------------------------------------
# The roots helper: exclusions of exactly the bindings allowed_file_roots
# would admit, absolute, and () outside a run workspace.
# ---------------------------------------------------------------------------


def test_exclusions_empty_outside_a_run_workspace(excluded_root) -> None:
    assert current_folder_binding_exclusions() == ()


def test_exclusions_resolve_bound_binding_entries(excluded_root: Path) -> None:
    with wfr.run_workspace(WORKSPACE_ID):
        paths = current_folder_binding_exclusions()
    assert (excluded_root / "secrets").resolve(strict=False) in paths


def test_exclusions_empty_when_no_binding_carries_any(plain_root: Path) -> None:
    with wfr.run_workspace(WORKSPACE_ID):
        assert current_folder_binding_exclusions() == ()


# ---------------------------------------------------------------------------
# Family enforcement: read / list / write / glob through the builtin tools.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_tool_refuses_excluded_path(excluded_root: Path) -> None:
    target = excluded_root / "secrets" / "key.pem"
    with wfr.run_workspace(WORKSPACE_ID):
        result = await ReadFileTool().execute(file_path=str(target))
    assert result.get("error"), "excluded key.pem was returned by read_file"
    assert "protected path" in result["error"], result["error"]
    assert KEY_MARKER not in str(result)
    assert target.read_text().count(KEY_MARKER) == 1  # untouched on disk


@pytest.mark.asyncio
async def test_listing_omits_excluded_directory(excluded_root: Path) -> None:
    with wfr.run_workspace(WORKSPACE_ID):
        result = await ListDirectoryTool().execute(directory_path=str(excluded_root))
    assert result.get("error") is None, result
    names = {entry["name"] for entry in result["entries"]}
    assert "notes.md" in names, f"ordinary entry vanished: {sorted(names)}"
    assert "secrets" not in names, f"excluded directory disclosed: {sorted(names)}"


@pytest.mark.asyncio
async def test_write_to_excluded_path_refuses(excluded_root: Path) -> None:
    target = excluded_root / "secrets" / "new.txt"
    with wfr.run_workspace(WORKSPACE_ID):
        result = await WriteFileTool().execute(file_path=str(target), content="x")
    assert result.get("error"), "write_file accepted an excluded path"
    assert "protected path" in result["error"], result["error"]
    assert not target.exists()


@pytest.mark.asyncio
async def test_glob_omits_excluded_paths(excluded_root: Path) -> None:
    with wfr.run_workspace(WORKSPACE_ID):
        result = await GlobFiles().execute(pattern="**/*")
    assert result.get("error") is None, result
    matches = [str(m) for m in result["matches"]]
    assert any(m.endswith("notes.md") for m in matches), matches
    assert not any("secrets" in m for m in matches), matches


# ---------------------------------------------------------------------------
# Parity: with no exclusions bound, behavior is identical to before.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_exclusions_bound_behavior_unchanged(plain_root: Path) -> None:
    with wfr.run_workspace(WORKSPACE_ID):
        read = await ReadFileTool().execute(
            file_path=str(plain_root / "secrets" / "key.pem")
        )
        written = await WriteFileTool().execute(
            file_path=str(plain_root / "out.txt"), content="x"
        )
        listed = await ListDirectoryTool().execute(directory_path=str(plain_root))
    assert read.get("error") is None, read
    assert read["content"].count(KEY_MARKER) == 1
    assert written.get("error") is None, written
    assert (plain_root / "out.txt").read_text() == "x"
    names = {entry["name"] for entry in listed["entries"]}
    assert {"notes.md", "secrets"} <= names, sorted(names)
