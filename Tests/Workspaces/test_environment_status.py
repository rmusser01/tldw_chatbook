"""Gatherer tests: real temp git repos, no git mocks (gh is mocked at its seam later)."""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.git_workspace import linked_worktree_name


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        capture_output=True, text=True, timeout=30,
    )


@pytest.fixture()
def main_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "mainrepo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-m", "init")
    return repo


def test_linked_worktree_name_is_none_for_a_primary_checkout(main_repo: Path):
    assert linked_worktree_name(main_repo) is None


def test_linked_worktree_name_returns_basename_for_a_linked_worktree(main_repo: Path, tmp_path: Path):
    wt = tmp_path / "feature-wt"
    _git(main_repo, "worktree", "add", str(wt), "-b", "feature-x")
    assert linked_worktree_name(wt) == "feature-wt"


def test_linked_worktree_name_is_none_outside_any_repo(tmp_path: Path):
    bare_dir = tmp_path / "norepo"
    bare_dir.mkdir()
    assert linked_worktree_name(bare_dir) is None
