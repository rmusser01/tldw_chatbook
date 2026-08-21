"""Commit engine for change-review git modes (TASK-16801 arc B, T3).

Every test drives REAL git in a temp repo -- the engine has no mockable
seam by design (spec
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
AC #5, no mocked git).

`test_prestaged_unrelated_entry_survives` is THE regression pin for the
arc's canonical would-have-shipped bug: a bare `git commit -m` would sweep
in whatever the user had already staged in a terminal. The commit recipe
here is `git add -A -- <selected>` then `git commit -m <msg> -- <selected>`
(a PATHSPEC commit) -- verified empirically (spec §2 probe 1) to commit
EXACTLY the selected paths and leave an unrelated pre-staged index entry
staged and uncommitted.
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.git_workspace import (
    CommitRefusedError,
    CommitResult,
    GitStepOutcome,
    GitWorkspaceError,
    commit_selected,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_prestaged_unrelated_entry_survives(repo):
    # spec §2 probe 1 -- THE regression pin for the index-hijack trap.
    (repo / "keep.txt").write_text("user-staged\n")
    _git(repo, "add", "keep.txt")
    (repo / "a.txt").write_text("agent\n")
    result = commit_selected(
        repo, ["a.txt"], "agent work", None, run_active=lambda: False
    )
    assert isinstance(result, CommitResult)
    assert result.short_sha
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["a.txt"]
    # The pre-staged, unrelated file must survive staged and UNCOMMITTED --
    # a bare `git commit -m` would have swept it in.
    assert _git(repo, "status", "--porcelain") == "A  keep.txt"


def test_run_active_refuses_before_touching(repo):
    (repo / "a.txt").write_text("x\n")
    with pytest.raises(CommitRefusedError):
        commit_selected(repo, ["a.txt"], "m", None, run_active=lambda: True)
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    # Nothing was staged and no commit happened -- the edit is still
    # unstaged (visible in the worktree diff, absent from the index).
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_new_branch_created_then_committed(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "feat/xyz", run_active=lambda: False
    )
    assert result.short_sha
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "feat/xyz"
    # The commit landed on the NEW branch, not main.
    assert _git(repo, "log", "-1", "--format=%s") == "m"


def test_bad_branch_name_refused_preflight(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "-bad", run_active=lambda: False
    )
    assert result.short_sha is None
    assert result.outcomes[0].step == "validate-branch" and not result.outcomes[0].ok
    # nothing staged, no commit happened, still on main
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"


def test_existing_branch_stops_before_commit(repo):
    # `checkout -b main` fails (branch already exists) -- no new commit,
    # nothing staged.
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "main", run_active=lambda: False
    )
    assert result.short_sha is None
    failing = [o for o in result.outcomes if not o.ok]
    assert len(failing) == 1
    assert failing[0].step == "create-branch"
    assert failing[0].detail  # honest stderr excerpt, never blank
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")


def test_merge_in_progress_refused(repo):
    # Build a REAL conflicted merge: two branches diverge from `base`,
    # editing the same line of a.txt, so the merge stops with conflict
    # markers and MERGE_HEAD present -- never hand-write MERGE_HEAD.
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.txt").write_text("feature change\n")
    _git(repo, "commit", "-qam", "feature change")
    _git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("main change\n")
    _git(repo, "commit", "-qam", "main change")
    merge_proc = subprocess.run(
        ["git", "merge", "feature"], cwd=repo, capture_output=True, text=True
    )
    assert merge_proc.returncode != 0  # genuine conflict, not a fast-forward
    assert (repo / ".git" / "MERGE_HEAD").exists()

    result = commit_selected(
        repo, ["a.txt"], "m", None, run_active=lambda: False
    )
    assert result.short_sha is None
    assert result.outcomes[0].step == "in-progress-check" and not result.outcomes[0].ok
    assert "merge" in result.outcomes[0].detail
    # Nothing new was staged/committed by the refused attempt.
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"


def test_dash_leading_message_commits_literally(repo):
    # spec §2 probe 5 -- `-m`'s sticky-arg consumption makes a dash-leading
    # message safe as a plain argv element (never option-injection).
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "--amend", None, run_active=lambda: False
    )
    assert result.short_sha
    assert _git(repo, "log", "-1", "--format=%s") == "--amend"
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"  # a NEW commit, not an amend


def test_deletion_only_selection_commits(repo):
    (repo / "a.txt").unlink()
    result = commit_selected(
        repo, ["a.txt"], "remove a", None, run_active=lambda: False
    )
    assert result.short_sha
    tracked = _git(repo, "ls-tree", "-r", "HEAD", "--name-only").splitlines()
    assert "a.txt" not in tracked
    assert _git(repo, "status", "--porcelain") == ""


def test_unborn_first_commit_works(tmp_path):
    # spec §2 probe 4 -- the pathspec commit works as the first commit on
    # a fresh, unborn-HEAD repo.
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("x\n")
    result = commit_selected(
        root, ["a.txt"], "first commit", None, run_active=lambda: False
    )
    assert result.short_sha
    assert _git(root, "rev-list", "--count", "HEAD") == "1"
    assert _git(root, "show", "--name-only", "--format=", "HEAD").strip() == "a.txt"


def test_empty_files_refused(repo):
    with pytest.raises(GitWorkspaceError):
        commit_selected(repo, [], "m", None, run_active=lambda: False)
    assert _git(repo, "status", "--porcelain") == ""


def test_blank_message_refused(repo):
    (repo / "a.txt").write_text("x\n")
    with pytest.raises(GitWorkspaceError):
        commit_selected(repo, ["a.txt"], "   ", None, run_active=lambda: False)
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_commit_result_and_step_outcome_are_frozen_dataclasses():
    outcome = GitStepOutcome(step="stage", ok=True)
    with pytest.raises(Exception):
        outcome.ok = False  # type: ignore[misc]
    result = CommitResult(outcomes=(outcome,), short_sha="abc1234")
    with pytest.raises(Exception):
        result.short_sha = None  # type: ignore[misc]
