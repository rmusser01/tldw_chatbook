"""End-to-end family-1 user-exclusion enforcement (spec 2026-09-20, family 1).

Pins the merged-context wiring from Tasks 1-4 for the local provider family
(``fs_*``, read-only Git, Virtual CLI): one per-binding user exclusion must
keep the excluded subtree invisible to enumeration (fs_glob, fs_grep,
git_status), refuse direct mutation (fs_edit), and refuse a Virtual CLI
``stat`` -- every surface through the same :class:`WorkspaceToolExecutor`
seam. Parent-side refusals surface as ``WorkspaceToolExecutionError`` with
code ``invalid_request`` (the Task-3 executor convention), never as
``LocalToolError``.
"""
from __future__ import annotations

import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

from tldw_chatbook.Tools.virtual_cli_impls import VirtualCliRegistry
from tldw_chatbook.Tools.workspace_tool_executor import (
    WorkspaceToolExecutionError,
    WorkspaceToolExecutor,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("SECRET")
    (root / "public.txt").write_text("hello")
    return root


def _executor(repo: Path) -> WorkspaceToolExecutor:
    return WorkspaceToolExecutor(
        repo, user_exclusion_paths=lambda: (repo / "secrets",)
    )


def test_fs_glob_omits_excluded(repo: Path) -> None:
    result = _executor(repo).execute("fs_glob", {"pattern": "**/*"}, intent="read")
    assert "key.pem" not in result
    assert "public.txt" in result


def test_fs_grep_skips_excluded_content(repo: Path) -> None:
    result = _executor(repo).execute(
        "fs_grep", {"pattern": "SECRET", "mode": "files"}, intent="read"
    )
    assert "key.pem" not in result


def test_fs_edit_refused_on_excluded(repo: Path) -> None:
    with pytest.raises(WorkspaceToolExecutionError) as caught:
        _executor(repo).execute(
            "fs_edit",
            {"path": "secrets/key.pem", "old_string": "S", "new_string": "X"},
            intent="write",
        )

    assert caught.value.code == "invalid_request"
    assert (repo / "secrets" / "key.pem").read_text() == "SECRET"


@pytest.mark.parametrize(
    ("excluded_form", "selected_form"), (("NFC", "NFD"), ("NFD", "NFC"))
)
def test_fs_read_refuses_other_normalization_of_excluded_file(
    tmp_path: Path, excluded_form: str, selected_form: str
) -> None:
    """TASK-37655: APFS opens an NFD spelling as the stored NFC file (and back)."""
    excluded = unicodedata.normalize(excluded_form, "Données/naïve.txt")
    selected = unicodedata.normalize(selected_form, "Données/naïve.txt")
    (tmp_path / excluded).parent.mkdir()
    (tmp_path / excluded).write_text("PRIVATE")
    if not (tmp_path / selected).exists():  # normalization-sensitive filesystem
        (tmp_path / selected).parent.mkdir(exist_ok=True)
        (tmp_path / selected).write_text("PRIVATE")
    executor = WorkspaceToolExecutor(
        tmp_path, user_exclusion_paths=lambda: (tmp_path / excluded,)
    )

    with pytest.raises(WorkspaceToolExecutionError) as caught:
        executor.execute("fs_read", {"path": selected}, intent="read")

    assert "PRIVATE" not in str(caught.value)


def test_vcli_stat_refused_on_excluded(repo: Path) -> None:
    registry = VirtualCliRegistry(repo, workspace_executor=_executor(repo))
    with pytest.raises(WorkspaceToolExecutionError) as caught:
        registry.execute("stat", ["secrets/key.pem"])

    assert caught.value.code == "invalid_request"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_git_status_omits_excluded(tmp_path: Path) -> None:
    root = tmp_path
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "public.txt").write_text("hello")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "baseline")
    (root / "secrets").mkdir()
    (root / "secrets" / "key.pem").write_text("SECRET")
    (root / "public.txt").write_text("changed")

    result = _executor(root).execute("git_status", {}, intent="read")

    assert "key.pem" not in result
    assert "secrets" not in result
    assert "public.txt" in result
