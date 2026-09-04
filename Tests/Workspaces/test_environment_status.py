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


from tldw_chatbook.Chat.console_environment_state import EnvSourceAvailability
from tldw_chatbook.Workspaces.environment_status import gather_git_env


def test_gather_git_env_not_a_repo(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    state = gather_git_env(plain)
    assert state.availability is EnvSourceAvailability.NOT_APPLICABLE


def test_gather_git_env_clean_repo(main_repo: Path):
    state = gather_git_env(main_repo)
    assert state.availability is EnvSourceAvailability.OK
    assert state.branch == "main"
    assert state.adds == 0 and state.dels == 0 and state.files == ()
    assert state.worktree_name is None
    assert not state.dirty


def test_gather_git_env_dirty_repo_counts_lines(main_repo: Path):
    (main_repo / "a.txt").write_text("one\ntwo\nthree\n")
    (main_repo / "new.txt").write_text("hello\n")
    state = gather_git_env(main_repo)
    assert state.dirty
    assert state.adds >= 2  # two lines added to a.txt; untracked adds are 0 by design
    paths = {f.path for f in state.files}
    assert paths == {"a.txt", "new.txt"}


def test_gather_git_env_linked_worktree(main_repo: Path, tmp_path: Path):
    wt = tmp_path / "env-wt"
    _git(main_repo, "worktree", "add", str(wt), "-b", "task-77-branch")
    state = gather_git_env(wt)
    assert state.branch == "task-77-branch"
    assert state.worktree_name == "env-wt"


def test_gather_git_env_detached(main_repo: Path):
    _git(main_repo, "checkout", "--detach")
    state = gather_git_env(main_repo)
    assert state.detached and state.branch is None
    assert state.head_short  # short sha populated


def test_gather_git_env_error_keeps_previous_as_stale(main_repo: Path, monkeypatch):
    import tldw_chatbook.Workspaces.environment_status as mod
    from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError
    good = gather_git_env(main_repo)

    def boom(root, info):
        raise GitWorkspaceError("git status timed out after 30s")

    monkeypatch.setattr(mod, "working_tree_status", boom)
    state = gather_git_env(main_repo, previous=good)
    assert state.stale is True
    assert state.availability is EnvSourceAvailability.OK
    assert state.branch == good.branch


import json

from tldw_chatbook.Chat.console_environment_state import PrEnvState
from tldw_chatbook.Workspaces.environment_status import GhResult, gather_pr_env

_GH_JSON = json.dumps({
    "number": 2281, "title": "Split boot CSS", "state": "OPEN", "isDraft": False,
    "url": "https://github.com/o/r/pull/2281",
    "additions": 36643, "deletions": 2871, "mergedAt": None,
    "statusCheckRollup": [
        {"__typename": "CheckRun", "name": "lint", "status": "COMPLETED",
         "conclusion": "SUCCESS", "detailsUrl": "https://ci/lint"},
        {"__typename": "CheckRun", "name": "tests", "status": "COMPLETED",
         "conclusion": "FAILURE", "detailsUrl": "https://ci/tests"},
        {"__typename": "CheckRun", "name": "build", "status": "IN_PROGRESS",
         "conclusion": None, "detailsUrl": "https://ci/build"},
        {"__typename": "StatusContext", "context": "legacy-ci", "state": "SUCCESS",
         "targetUrl": "https://ci/legacy"},
    ],
})


def test_gather_pr_env_parses_pr_and_both_check_shapes(tmp_path: Path):
    state = gather_pr_env(tmp_path, "feat/x", runner=lambda root, args: GhResult(0, _GH_JSON, ""))
    assert state.availability is EnvSourceAvailability.OK
    assert state.number == 2281 and state.state == "OPEN"
    assert {c.name for c in state.checks} == {"lint", "tests", "build", "legacy-ci"}
    assert [c.name for c in state.failing_checks] == ["tests"]
    assert [c.name for c in state.pending_checks] == ["build"]
    assert state.passing_count == 2


def test_gather_pr_env_no_pr_maps_to_not_applicable(tmp_path: Path):
    result = GhResult(1, "", "no pull requests found for branch \"feat/x\"")
    state = gather_pr_env(tmp_path, "feat/x", runner=lambda root, args: result)
    assert state.availability is EnvSourceAvailability.NOT_APPLICABLE


def test_gather_pr_env_missing_binary(tmp_path: Path):
    state = gather_pr_env(tmp_path, "feat/x", runner=lambda root, args: None)
    assert state.availability is EnvSourceAvailability.MISSING_TOOL


def test_gather_pr_env_detached_branch_skips_entirely(tmp_path: Path):
    def exploding_runner(root, args):  # must not be called
        raise AssertionError("runner must not run for a detached HEAD")
    state = gather_pr_env(tmp_path, None, runner=exploding_runner)
    assert state.availability is EnvSourceAvailability.NOT_APPLICABLE


def test_gather_pr_env_error_keeps_previous_as_stale(tmp_path: Path):
    good = gather_pr_env(tmp_path, "feat/x", runner=lambda root, args: GhResult(0, _GH_JSON, ""))
    state = gather_pr_env(
        tmp_path, "feat/x",
        runner=lambda root, args: GhResult(1, "", "connect: network is unreachable"),
        previous=good,
    )
    assert state.stale is True and state.number == 2281


def test_gather_pr_env_garbage_json_is_error_not_crash(tmp_path: Path):
    state = gather_pr_env(tmp_path, "feat/x", runner=lambda root, args: GhResult(0, "{not json", ""))
    assert state.availability is EnvSourceAvailability.ERROR


def test_run_gh_missing_binary_returns_none(tmp_path: Path):
    from tldw_chatbook.Workspaces.environment_status import run_gh
    import tldw_chatbook.Workspaces.environment_status as mod
    original = mod._GH_EXECUTABLE
    mod._GH_EXECUTABLE = "/nonexistent/gh-binary-for-test"
    try:
        assert run_gh(tmp_path, ["pr", "view"]) is None
    finally:
        mod._GH_EXECUTABLE = original
