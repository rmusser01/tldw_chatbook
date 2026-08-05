"""Tests for the sync ``run_git`` wrapper and ``prepare_repository``.

Port-verification tests for the phase-3b-ii port of tldw_server's
``git_module.py`` runner (see ``tldw_chatbook/Tools/git_tool_impls.py``
header for provenance). Real git is exercised via tmp repos; the whole
module is skipped when git is unavailable (re-plan §2.5).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Tools import git_tool_impls
from tldw_chatbook.Tools.git_tool_impls import (
    GitCommandResult,
    prepare_repository,
    run_git,
)
from tldw_chatbook.Tools.local_tool_impls import LocalToolError

GIT_AVAILABLE = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(not GIT_AVAILABLE, reason="git is not available on this system")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial commit")
    return repo


def test_run_git_version() -> None:
    result = run_git(["git", "--version"])
    assert isinstance(result, GitCommandResult)
    assert result.returncode == 0
    assert "git version" in result.stdout
    assert result.timed_out is False
    assert result.truncated is False


def test_run_git_rejects_disallowed_subcommand() -> None:
    with pytest.raises(LocalToolError, match="not allowlisted"):
        run_git(["git", "push"])
    with pytest.raises(LocalToolError, match="only executes git"):
        run_git(["not-git", "status"])


def test_run_git_rejects_global_option_smuggling() -> None:
    with pytest.raises(LocalToolError, match="not allowlisted"):
        run_git(["git", "--exec-path=/tmp", "status"])
    with pytest.raises(LocalToolError, match="not allowlisted"):
        run_git(["git", "-c", "x=y", "status"])
    # --version must be used alone
    with pytest.raises(LocalToolError):
        run_git(["git", "--version", "status"])
    # -C requires a following path argument
    with pytest.raises(LocalToolError):
        run_git(["git", "-C"])


def test_run_git_timeout() -> None:
    # A zero timeout expires the deadline before any output is read; the
    # process is killed and a model-actionable error is raised.
    with pytest.raises(LocalToolError, match="timed out"):
        run_git(["git", "--version"], timeout=0.0)


def test_run_git_output_cap(tmp_git_repo: Path) -> None:
    result = run_git(["git", "--version"], max_output_bytes=4)
    assert result.truncated is True
    assert result.stdout.startswith("git ")
    assert "truncated" in result.stdout
    # The bounded read killed the process at the cap instead of buffering it.
    assert len(result.stdout) < 100


def test_run_git_env_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    real_popen = subprocess.Popen

    def spy(argv, **kwargs):  # noqa: ANN001 - test spy
        captured.update(kwargs)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(git_tool_impls.subprocess, "Popen", spy)
    run_git(["git", "--version"])
    env = captured["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_PAGER"] == "cat"
    assert "HOME" not in env
    assert "PATH" in env


def test_prepare_repository_finds_root(tmp_git_repo: Path) -> None:
    nested = tmp_git_repo / "sub" / "deep"
    nested.mkdir(parents=True)
    root = prepare_repository(tmp_git_repo, "sub/deep")
    assert root == tmp_git_repo.resolve()


def test_prepare_repository_rejects_non_repo(tmp_path: Path) -> None:
    with pytest.raises(LocalToolError, match="not a git repository"):
        prepare_repository(tmp_path, ".")


def test_prepare_repository_rejects_repo_above_workspace(tmp_git_repo: Path) -> None:
    # Workspace nested INSIDE a repo: the discovered repo root sits above the
    # workspace root, so confinement would leak repo state -> refused.
    ws = tmp_git_repo / "ws"
    ws.mkdir()
    with pytest.raises(LocalToolError, match="workspace"):
        prepare_repository(ws, ".")


def test_git_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(git_tool_impls.shutil, "which", lambda _name: None)
    with pytest.raises(LocalToolError, match="git is not available"):
        prepare_repository(tmp_path, ".")
    with pytest.raises(LocalToolError, match="git is not available"):
        run_git(["git", "--version"])
