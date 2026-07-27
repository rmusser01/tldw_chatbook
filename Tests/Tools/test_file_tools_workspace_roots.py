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
    GlobFiles,
    GrepFiles,
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


# ---------------------------------------------------------------------------
# TASK-850: glob_files/grep_files previously searched the tool sandbox root
# only -- strictly narrower than, and inconsistent with, the three tools
# above, all of which already honour every workspace folder root bound to
# the run. These pin the fix: the search tools now honour the SAME root
# set, merge results across every root, and never widen reach beyond what
# read_file already has.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_finds_file_in_bound_workspace_folder(bound_workspace) -> None:
    (bound_workspace["ro"] / "notes.md").write_text("hello")
    with wfr.run_workspace("ws-a"):
        result = await GlobFiles().execute(pattern="**/*.md")
    assert "notes.md" in {Path(p).name for p in result["matches"]}


@pytest.mark.asyncio
async def test_grep_finds_content_in_bound_workspace_folder(bound_workspace) -> None:
    (bound_workspace["ro"] / "notes.md").write_text("UNIQUE_WORKSPACE_MARKER_71a2\n")
    with wfr.run_workspace("ws-a"):
        result = await GrepFiles().execute(pattern="UNIQUE_WORKSPACE_MARKER_71a2")
    assert any(m["path"].endswith("notes.md") for m in result["matches"])


@pytest.mark.asyncio
async def test_glob_merges_matches_across_sandbox_and_bound_folder(
    bound_workspace,
) -> None:
    """Results are MERGED across every root into one result set -- a match
    reachable only through a bound folder, not the sandbox, is not lost.
    """
    (bound_workspace["sandbox"] / "a.py").write_text("x = 1\n")
    (bound_workspace["ro"] / "b.py").write_text("y = 2\n")
    with wfr.run_workspace("ws-a"):
        result = await GlobFiles().execute(pattern="**/*.py")
    assert {Path(p).name for p in result["matches"]} == {"a.py", "b.py"}


@pytest.mark.asyncio
async def test_grep_candidate_bound_is_shared_across_roots_not_multiplied(
    bound_workspace, monkeypatch
) -> None:
    """`_MAX_CANDIDATES` must bound candidates examined across ALL roots
    COMBINED -- N configured roots must not multiply the worst-case walk
    by N. Each matching file here produces exactly one match, so the
    match count directly pins how many candidates were actually examined.
    """
    monkeypatch.setattr(fot, "_MAX_CANDIDATES", 3)
    monkeypatch.setattr(fot, "_MAX_MATCHES", 1_000)
    for i in range(5):
        (bound_workspace["sandbox"] / f"s{i}.txt").write_text("DEBUG\n")
    for i in range(5):
        (bound_workspace["ro"] / f"r{i}.txt").write_text("DEBUG\n")

    with wfr.run_workspace("ws-a"):
        result = await GrepFiles().execute(pattern="DEBUG")

    assert len(result["matches"]) <= 3


@pytest.mark.asyncio
async def test_dotted_workspace_root_is_skipped_not_fatal_to_other_roots(
    tmp_path, monkeypatch
) -> None:
    """Dotted-root rule, extended to a root SET: with a single root (the
    sandbox alone), a dotted root refuses the WHOLE call (see
    ``test_glob_files_refuses_a_dotted_sandbox_root`` /
    ``test_grep_files_refuses_a_dotted_sandbox_root`` in
    ``test_glob_grep_files.py``). With several roots, this pins the
    documented decision for the multi-root case instead: each root is
    checked independently, a dotted one is excluded from the search, and
    every OTHER, still-valid root's results are returned normally rather
    than the whole call being refused.
    """
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="dotted-root-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")

    dotted_folder = tmp_path / ".hidden-project"
    dotted_folder.mkdir()
    (dotted_folder / "secret.txt").write_text("SHOULD_NOT_BE_FOUND_8b21\n")

    ok_folder = tmp_path / "ok-project"
    ok_folder.mkdir()
    (ok_folder / "notes.txt").write_text("SHOULD_BE_FOUND_8b21\n")

    registry.add_folder_binding("ws-a", dotted_folder)
    registry.add_folder_binding("ws-a", ok_folder)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(sandbox))

    with wfr.run_workspace("ws-a"):
        glob_result = await GlobFiles().execute(pattern="**/*.txt")
        grep_result = await GrepFiles().execute(pattern="8b21")

    assert "error" not in glob_result
    names = {Path(p).name for p in glob_result["matches"]}
    assert "notes.txt" in names
    assert "secret.txt" not in names

    assert grep_result["matches"] != []
    assert all("SHOULD_NOT_BE_FOUND" not in m["line"] for m in grep_result["matches"])


@pytest.mark.asyncio
async def test_glob_grep_and_the_read_family_all_refuse_a_path_outside_every_root(
    bound_workspace, tmp_path
) -> None:
    """Prove TASK-850 does not widen reach beyond what read_file already
    has: a path outside EVERY configured root (sandbox + every bound
    workspace folder) stays refused across all five tools.

    read_file/write_file/list_directory are proven directly, against the
    exact outside path. glob_files/grep_files take no target-path
    argument at all -- only a relative glob PATTERN, which
    ``_rejects_traversal`` refuses outright for any absolute form or
    ``..`` component -- so the meaningful equivalent for them is a
    SYMLINK planted INSIDE an allowed root pointing at the outside
    directory: the one real avenue a search tool could otherwise reach it
    through despite never accepting an absolute/traversal pattern.
    ``is_within``'s resolved-ancestry containment check must refuse the
    symlinked target exactly like it refuses everything else outside
    every root.
    """
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("OUTSIDE_EVERY_ROOT_MARKER_9f3c1a")

    with wfr.run_workspace("ws-a"):
        read_result = await ReadFileTool().execute(file_path=str(outside_file))
        write_result = await WriteFileTool().execute(
            file_path=str(outside_file), content="pwned"
        )
        list_result = await ListDirectoryTool().execute(
            directory_path=str(outside_dir)
        )

        link = bound_workspace["ro"] / "escape"
        os.symlink(outside_dir, link)
        glob_result = await GlobFiles().execute(pattern="**/*")
        grep_result = await GrepFiles().execute(
            pattern="OUTSIDE_EVERY_ROOT_MARKER_9f3c1a"
        )

    assert "error" in read_result
    assert "OUTSIDE_EVERY_ROOT_MARKER_9f3c1a" not in str(read_result)

    assert "error" in write_result
    assert outside_file.read_text() == "OUTSIDE_EVERY_ROOT_MARKER_9f3c1a"  # untouched

    assert "error" in list_result

    assert "secret.txt" not in {Path(p).name for p in glob_result["matches"]}
    assert grep_result["matches"] == []
