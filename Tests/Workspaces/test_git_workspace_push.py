"""Push engine for change-review git modes (TASK-16801 arc B, T4).

Every push test drives a REAL local bare remote (`git init --bare`) -- the
engine has no mockable seam by design (spec
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
AC #5, no mocked git).

`test_nonff_push_fails_honestly_no_force` is the regression pin for the
arc's no-force invariant: it wraps `_run_user_git` to capture every argv
issued during the failing push and asserts none of them carry
`--force`/`--force-with-lease` -- a non-fast-forward rejection must
surface git's own stderr excerpt honestly, never be silently retried with
force.
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces import git_workspace
from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceError,
    PushResult,
    _push_failure_detail,
    detect_git_workspace,
    push_current,
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


@pytest.fixture()
def bare(tmp_path: Path) -> Path:
    bare_root = tmp_path / "bare.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare_root)],
        check=True,
        capture_output=True,
    )
    return bare_root


def test_first_push_sets_upstream_and_moves_bare_ref(repo, bare):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    assert info.upstream is None
    assert info.remotes == (("origin", str(bare)),)

    result = push_current(repo, info, None)
    assert isinstance(result, PushResult)
    assert result.state == "pushed"
    assert result.detail == ""

    bare_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert bare_head == _git(repo, "rev-parse", "HEAD")

    info_after = detect_git_workspace(repo)
    assert info_after.upstream == "origin/main"
    assert info_after.upstream_remote == "origin"


def test_second_push_up_to_date(repo, bare):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    first = push_current(repo, info, None)
    assert first.state == "pushed"

    info2 = detect_git_workspace(repo)
    assert info2.upstream == "origin/main"
    second = push_current(repo, info2, None)
    assert second.state == "up_to_date"
    assert second.detail == ""


def test_nonff_push_fails_honestly_no_force(repo, bare, tmp_path, monkeypatch):
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)
    first = push_current(repo, info, None)
    assert first.state == "pushed"

    # A second clone commits and pushes first, so our next push is a
    # genuine non-fast-forward rejection (never a synthetic one).
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(clone)],
        check=True,
        capture_output=True,
    )
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    (clone / "a.txt").write_text("clone change\n")
    _git(clone, "commit", "-qam", "clone change")
    _git(clone, "push", "-q", "origin", "main")

    (repo / "a.txt").write_text("local change\n")
    _git(repo, "commit", "-qam", "local change")

    captured_argv: list[tuple[str, ...]] = []
    real_run = git_workspace._run_user_git

    def _spy(root, *args, **kwargs):
        captured_argv.append(args)
        return real_run(root, *args, **kwargs)

    monkeypatch.setattr(git_workspace, "_run_user_git", _spy)

    info2 = detect_git_workspace(repo)
    assert info2.upstream == "origin/main"
    result = push_current(repo, info2, None)

    assert result.state == "failed"
    assert "rejected" in result.detail

    push_calls = [args for args in captured_argv if args and args[0] == "push"]
    assert push_calls  # sanity: the push was actually attempted
    for args in push_calls:
        assert not any(arg.startswith("--force") for arg in args)

    # The bare remote's ref did NOT move -- the rejection was real.
    bare_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    clone_head = _git(clone, "rev-parse", "HEAD")
    assert bare_head == clone_head


def test_credential_hint_mapping():
    detail = _push_failure_detail(
        "fatal: could not read Username for 'https://host': "
        "terminal prompts disabled"
    )
    assert detail.endswith(
        " — credentials were not available non-interactively; push once "
        "from a terminal or configure a credential helper/ssh agent"
    )


def test_credential_hint_not_appended_for_unrelated_failure():
    detail = _push_failure_detail("fatal: unable to access 'https://host': "
                                   "Could not resolve host")
    assert detail == (
        "fatal: unable to access 'https://host': Could not resolve host"
    )


def test_detached_refused(repo):
    _git(repo, "checkout", "-q", "--detach")
    info = detect_git_workspace(repo)
    assert info.detached and info.branch is None
    with pytest.raises(GitWorkspaceError):
        push_current(repo, info, None)


def test_no_remote_refused(repo):
    info = detect_git_workspace(repo)
    assert info.remotes == ()
    with pytest.raises(GitWorkspaceError):
        push_current(repo, info, None)


def test_explicit_remote_overrides_derivation(repo, bare, tmp_path):
    other_bare = tmp_path / "other.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(other_bare)],
        check=True,
        capture_output=True,
    )
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "remote", "add", "other", str(other_bare))
    info = detect_git_workspace(repo)
    assert len(info.remotes) == 2

    result = push_current(repo, info, "other")
    assert result.state == "pushed"
    other_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=other_bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert other_head == _git(repo, "rev-parse", "HEAD")
    # The untargeted remote was never touched.
    bare_head_proc = subprocess.run(
        ["git", "rev-parse", "main"], cwd=bare, capture_output=True, text=True
    )
    assert bare_head_proc.returncode != 0  # "main" never pushed there


def test_push_result_is_frozen_dataclass():
    result = PushResult(state="pushed", detail="")
    with pytest.raises(Exception):
        result.state = "failed"  # type: ignore[misc]
