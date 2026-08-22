"""Detection groundwork for change-review git modes (TASK-16801).

Every test drives REAL git in a temp repo -- the engine has no mockable
seam by design (spec: AC #5, no mocked git).
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceInfo,
    GitWorkspaceRefusal,
    _run_user_git,
    detect_git_workspace,
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


def test_non_repo_returns_none(tmp_path: Path):
    root = tmp_path / "plain"
    root.mkdir()
    assert detect_git_workspace(root) is None


def test_root_inside_repo_is_refused_with_copy(repo: Path):
    sub = repo / "sub"
    sub.mkdir()
    result = detect_git_workspace(sub)
    assert isinstance(result, GitWorkspaceRefusal)
    assert "repository root" in result.reason


def test_repo_root_detects_branch_and_no_remote(repo: Path):
    info = detect_git_workspace(repo)
    assert isinstance(info, GitWorkspaceInfo)
    assert info.branch == "main"
    assert not info.detached and not info.unborn
    assert info.upstream is None and info.upstream_remote is None
    assert info.remotes == ()
    assert (info.ahead, info.behind) == (0, 0)


def test_unborn_head_detected(tmp_path: Path):
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    info = detect_git_workspace(root)
    assert isinstance(info, GitWorkspaceInfo)
    assert info.unborn and info.branch == "main" and not info.detached


def test_detached_head_detected(repo: Path):
    _git(repo, "checkout", "-q", "--detach")
    info = detect_git_workspace(repo)
    assert info.detached and info.branch is None


def test_ahead_behind_order_left_is_behind(repo: Path, tmp_path: Path):
    # spec §2 probe 7: a swapped parse is the obvious bug.
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", "-u", "origin", "main")
    (repo / "a.txt").write_text("local\n")
    _git(repo, "commit", "-qam", "local-only")
    info = detect_git_workspace(repo)
    assert (info.ahead, info.behind) == (1, 0)
    assert info.upstream == "origin/main" and info.upstream_remote == "origin"


def test_upstream_remote_with_slash_in_name(repo: Path, tmp_path: Path):
    # spec §2 probe 6 regression pin: remote names CAN contain "/".
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _git(repo, "remote", "add", "a/b", str(bare))
    _git(repo, "push", "-q", "-u", "a/b", "main")
    info = detect_git_workspace(repo)
    assert info.upstream_remote == "a/b"


def test_env_posture_preserves_home_scrubs_git_dir(repo: Path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(repo / "nonsense"))
    monkeypatch.setenv("HOME", str(repo.parent))
    # A stray GIT_DIR would break every call; the scrub makes this pass.
    result = _run_user_git(repo, "rev-parse", "--show-toplevel")
    assert Path(result.stdout.strip()).resolve() == repo.resolve()


@pytest.mark.parametrize(
    "var", ["GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"]
)
def test_ambient_pathspec_vars_do_not_break_every_invocation(
    repo: Path, monkeypatch, var
):
    """The C2 fix pins `GIT_LITERAL_PATHSPECS=1`, and git REFUSES the mix.

    `fatal: global 'literal' pathspec setting is incompatible with all
    other global pathspec settings` -- so a user who exports any of these
    in their shell would see every git call in this module die. This
    module deliberately preserves the ambient environment, which makes
    scrubbing these three part of the fix rather than optional tidying.
    """
    monkeypatch.setenv(var, "1")
    # A command that actually PARSES a pathspec -- `rev-parse` never does,
    # so it would pass even with the scrub removed.
    (repo / "a.txt").write_text("edited\n")
    result = _run_user_git(repo, "diff", "HEAD", "--", "a.txt")
    assert "a.txt" in result.stdout, result.stderr
