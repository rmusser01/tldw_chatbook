"""TASK-28238 phase 2: agent worktree lifecycle."""

import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_worktree import (
    AgentWorktree,
    WorktreeRefusal,
    create_agent_worktree,
    discard_agent_worktree,
    prune_stale_agent_worktrees,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    return root


def test_create_yields_isolated_checkout_at_head(repo):
    wt = create_agent_worktree(repo, "run-abc12345")
    assert isinstance(wt, AgentWorktree), getattr(wt, "message", wt)
    assert wt.worktree_path.is_dir()
    assert (wt.worktree_path / "a.txt").read_text() == "base\n"
    assert wt.branch == "run-abc12345" or wt.branch.endswith("run-abc12345")
    # a write in the worktree is invisible in the shared tree
    (wt.worktree_path / "a.txt").write_text("child change\n")
    assert (repo / "a.txt").read_text() == "base\n"
    discard_agent_worktree(repo, wt)


def test_create_refuses_non_git_root(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    refusal = create_agent_worktree(plain, "run-x")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "not_a_git_repo"


def test_discard_removes_worktree_and_branch(repo):
    wt = create_agent_worktree(repo, "run-gone1234")
    assert isinstance(wt, AgentWorktree)
    assert discard_agent_worktree(repo, wt) is None
    assert not wt.worktree_path.exists()
    branches = _git(repo, "branch", "--list", wt.branch)
    assert wt.branch not in branches


def test_uncommitted_shared_changes_do_not_carry(repo):
    (repo / "a.txt").write_text("dirty uncommitted\n")
    wt = create_agent_worktree(repo, "run-clean555")
    assert isinstance(wt, AgentWorktree)
    # clean checkout of HEAD, not the dirty tree (spec decision)
    assert (wt.worktree_path / "a.txt").read_text() == "base\n"
    discard_agent_worktree(repo, wt)


def test_prune_removes_only_dead_runs(repo):
    live = create_agent_worktree(repo, "run-live0001")
    dead = create_agent_worktree(repo, "run-dead0001")
    assert isinstance(live, AgentWorktree) and isinstance(dead, AgentWorktree)
    removed = prune_stale_agent_worktrees(repo, live_run_ids={"run-live0001"})
    assert removed == 1
    assert live.worktree_path.exists()
    assert not dead.worktree_path.exists()
    discard_agent_worktree(repo, live)


def test_create_refuses_when_worktree_base_unwritable(tmp_path, repo, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n")
    monkeypatch.setattr(
        "tldw_chatbook.Agents.agent_worktree._worktrees_base",
        lambda: blocker / "sub",
    )
    result = create_agent_worktree(repo, "run-blocked1")
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "worktree_create_failed"
