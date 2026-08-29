"""Tests for the sync ``run_git`` wrapper and ``prepare_repository``.

Port-verification tests for the phase-3b-ii port of tldw_server's
``git_module.py`` runner (see ``tldw_chatbook/Tools/git_tool_impls.py``
header for provenance). Real git is exercised via tmp repos; the whole
module is skipped when git is unavailable (re-plan §2.5).
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Tools import git_tool_impls
from tldw_chatbook.Tools.git_tool_impls import (
    GitCommandResult,
    git_blame,
    git_branches,
    git_diff,
    git_log,
    git_status,
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
    _git(repo, "config", "commit.gpgsign", "false")
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
    # Cwd is a dedicated runner parameter; -C is never accepted in argv.
    with pytest.raises(LocalToolError):
        run_git(["git", "-C"])
    with pytest.raises(LocalToolError, match="not allowlisted"):
        run_git(["git", "-C", "/tmp", "status"])


def test_run_git_timeout() -> None:
    # A zero timeout expires the deadline before any output is read; the
    # process is killed and a model-actionable error is raised.
    with pytest.raises(LocalToolError, match="timed out"):
        run_git(["git", "--version"], timeout=0.0)


def test_run_git_output_cap(tmp_git_repo: Path) -> None:
    result = run_git(["git", "--version"], max_output_bytes=4)
    combined_output = result.stdout + result.stderr
    assert result.truncated is True
    assert result.stdout.startswith("git ") or result.stderr.startswith("git:")
    assert "truncated" in combined_output
    # The bounded read killed the process at the cap instead of buffering it.
    assert len(combined_output) < 100


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


def test_worker_git_uses_absolute_executable_relative_cwd_without_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker Git stays in the outer helper's retained process tree."""
    seen: dict[str, object] = {}

    class RecordingProcess:
        pid = 1234
        returncode = 0
        stdout = io.BytesIO(b"git version test\n")
        stderr = io.BytesIO()

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def recording_popen(argv: list[str], **kwargs: object) -> RecordingProcess:
        seen["argv"] = argv
        seen.update(kwargs)
        return RecordingProcess()

    monkeypatch.setattr(git_tool_impls.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(git_tool_impls.shutil, "which", lambda _name: "/usr/bin/git")

    result = run_git(
        ["git", "--version"],
        cwd=Path("repo"),
        executable=Path("/usr/bin/git"),
        own_process_group=False,
    )

    assert result.argv == ["git", "--version"]
    assert seen["argv"] == ["/usr/bin/git", "--version"]
    assert "-C" not in seen["argv"]
    assert seen["cwd"] == Path("repo")
    assert seen["start_new_session"] is False


def test_worker_git_timeout_kills_only_direct_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer helper remains the sole descendant-cleanup owner."""
    events: list[str] = []

    class RecordingProcess:
        pid = 1234
        returncode: int | None = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            events.append("direct-kill")
            self.returncode = -9

    monkeypatch.setattr(
        git_tool_impls.subprocess,
        "Popen",
        lambda _argv, **_kwargs: RecordingProcess(),
    )
    monkeypatch.setattr(git_tool_impls.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(
        git_tool_impls.os,
        "killpg",
        lambda *_args: events.append("process-group-kill"),
    )

    with pytest.raises(LocalToolError, match="timed out"):
        run_git(
            ["git", "status"],
            cwd=Path("repo"),
            executable=Path("/usr/bin/git"),
            own_process_group=False,
            timeout=0.0,
        )

    assert events == ["direct-kill"]


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


def _current_branch(repo: Path) -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_file(repo: Path, name: str, text: str, message: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)


# --- git_status ---------------------------------------------------------


def test_git_status_porcelain(tmp_git_repo: Path) -> None:
    clean = git_status(tmp_git_repo)
    assert "clean" in clean

    (tmp_git_repo / "file.txt").write_text("changed\n", encoding="utf-8")
    (tmp_git_repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    out = git_status(tmp_git_repo)
    assert f"branch: {_current_branch(tmp_git_repo)}" in out
    assert "file.txt" in out
    assert "unstaged" in out
    assert "untracked.txt" in out
    assert "untracked" in out


def test_git_status_not_repo(tmp_path: Path) -> None:
    with pytest.raises(LocalToolError, match="not a git repository"):
        git_status(tmp_path)


# --- git_branches -------------------------------------------------------


def test_git_branches(tmp_git_repo: Path) -> None:
    main = _current_branch(tmp_git_repo)
    _git(tmp_git_repo, "checkout", "-b", "feature-x")
    out = git_branches(tmp_git_repo)
    assert main in out
    assert "feature-x" in out
    # The current branch carries the marker.
    assert "* feature-x" in out
    assert f"* {main}" not in out


def test_read_only_git_operations_use_cwd_without_dash_c(
    tmp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every compatibility operation launches Git from its repository cwd."""
    _commit_file(tmp_git_repo, "multi.txt", "line1\n", "add multi")
    captured: list[tuple[list[str], dict[str, object]]] = []
    real_run_git = git_tool_impls.run_git

    def spy(argv: list[str], **kwargs: object) -> GitCommandResult:
        captured.append((list(argv), dict(kwargs)))
        return real_run_git(argv, **kwargs)

    monkeypatch.setattr(git_tool_impls, "run_git", spy)

    git_status(tmp_git_repo)
    git_diff(tmp_git_repo)
    git_log(tmp_git_repo)
    git_blame(tmp_git_repo, "multi.txt")
    git_branches(tmp_git_repo)

    operation_calls = [
        (argv, kwargs)
        for argv, kwargs in captured
        if any(command in argv for command in ("status", "diff", "log", "blame", "branch"))
    ]
    assert len(operation_calls) == 5
    for argv, kwargs in operation_calls:
        assert argv[0] == "git"
        assert "-C" not in argv
        assert Path(kwargs["cwd"]).is_absolute()
        assert kwargs["own_process_group"] is True


# --- git_log ------------------------------------------------------------


def test_git_log(tmp_git_repo: Path) -> None:
    _commit_file(tmp_git_repo, "a.txt", "a\n", "add a")
    _commit_file(tmp_git_repo, "b.txt", "b\n", "add b")

    out = git_log(tmp_git_repo)
    lines = out.strip().splitlines()
    assert len(lines) == 3
    assert "add b" in lines[0]
    assert "add a" in lines[1]
    assert "initial commit" in lines[2]
    assert "Test User" in out

    capped = git_log(tmp_git_repo, count=2)
    assert len(capped.strip().splitlines()) == 2

    filtered = git_log(tmp_git_repo, path="a.txt")
    assert "add a" in filtered
    assert "add b" not in filtered
    assert "initial commit" not in filtered


def test_git_log_count_clamped(
    tmp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    real_run_git = git_tool_impls.run_git

    def spy(argv, **kwargs):  # noqa: ANN001, ANN202 - test spy
        captured.append(list(argv))
        return real_run_git(argv, **kwargs)

    monkeypatch.setattr(git_tool_impls, "run_git", spy)
    git_log(tmp_git_repo, count=0)
    git_log(tmp_git_repo, count=250)
    log_calls = [argv for argv in captured if "log" in argv]
    assert len(log_calls) == 2
    for argv, expected in zip(log_calls, ["1", "100"], strict=True):
        assert argv[argv.index("-n") + 1] == expected


# --- git_diff -----------------------------------------------------------


def test_git_diff_worktree_and_staged(tmp_git_repo: Path) -> None:
    (tmp_git_repo / "file.txt").write_text("hello\nchanged\n", encoding="utf-8")

    unstaged = git_diff(tmp_git_repo)
    assert "+changed" in unstaged
    assert "file.txt" in unstaged
    assert git_diff(tmp_git_repo, staged=True) == "(no changes)"

    _git(tmp_git_repo, "add", "file.txt")
    staged = git_diff(tmp_git_repo, staged=True)
    assert "+changed" in staged
    assert git_diff(tmp_git_repo) == "(no changes)"


def test_git_diff_path_filter_and_stat(tmp_git_repo: Path) -> None:
    _commit_file(tmp_git_repo, "other.txt", "other\n", "add other")
    (tmp_git_repo / "file.txt").write_text("hello\nmore\n", encoding="utf-8")
    (tmp_git_repo / "other.txt").write_text("other\nmore2\n", encoding="utf-8")

    filtered = git_diff(tmp_git_repo, path="file.txt")
    assert "file.txt" in filtered
    assert "other.txt" not in filtered

    stat = git_diff(tmp_git_repo, stat=True)
    assert "files changed" in stat
    assert "+more" not in stat


def test_git_diff_commit_range(tmp_git_repo: Path) -> None:
    (tmp_git_repo / "file.txt").write_text("hello\nsecond\n", encoding="utf-8")
    _git(tmp_git_repo, "add", "file.txt")
    _git(tmp_git_repo, "commit", "-m", "second commit")

    out = git_diff(tmp_git_repo, commit_range="HEAD~1..HEAD")
    assert "+second" in out
    assert "file.txt" in out


def test_git_diff_commit_range_injection_refused(tmp_git_repo: Path) -> None:
    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="HEAD; rm -rf")
    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="HEAD $(whoami)")


# --- git_blame ----------------------------------------------------------


def test_git_blame(tmp_git_repo: Path) -> None:
    _commit_file(tmp_git_repo, "multi.txt", "line1\nline2\nline3\n", "add multi")

    out = git_blame(tmp_git_repo, "multi.txt")
    assert "Test User" in out
    assert "line1" in out
    assert "line3" in out

    ranged = git_blame(tmp_git_repo, "multi.txt", start_line=2, end_line=3)
    assert "line2" in ranged
    assert "line3" in ranged
    assert "line1" not in ranged


def test_git_blame_missing_file(tmp_git_repo: Path) -> None:
    with pytest.raises(LocalToolError, match="not found"):
        git_blame(tmp_git_repo, "nope.txt")


# --- confinement --------------------------------------------------------


def test_path_filter_confined(tmp_git_repo: Path) -> None:
    with pytest.raises(LocalToolError, match="outside"):
        git_diff(tmp_git_repo, path="../x")
    with pytest.raises(LocalToolError, match="outside"):
        git_log(tmp_git_repo, path="../x")
    with pytest.raises(LocalToolError, match="outside"):
        git_blame(tmp_git_repo, "../x")


# --- review fixes: flag smuggling, truncation delivery, process-group kill ---


def test_git_diff_commit_range_flag_smuggling_refused(tmp_git_repo: Path) -> None:
    # Leading-dash values are flags, not refnames: git's last-occurrence-wins
    # would let "--textconv" override the --no-textconv already in argv.
    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="--textconv")
    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="--output=/tmp/x")
    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="-c")


def test_git_diff_textconv_hostile_repo_not_executed(tmp_git_repo: Path) -> None:
    # Hostile repo: .gitattributes routes *.txt through a textconv that
    # writes a marker file. The smuggled --textconv must be refused, and a
    # plain diff must be protected by the --no-textconv already in argv.
    marker = tmp_git_repo / "MARKER"
    (tmp_git_repo / ".gitattributes").write_text("*.txt diff=pwn\n", encoding="utf-8")
    _git(tmp_git_repo, "config", "diff.pwn.textconv", f"touch {marker}; cat")
    (tmp_git_repo / "file.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(LocalToolError, match="commit_range"):
        git_diff(tmp_git_repo, commit_range="--textconv")
    git_diff(tmp_git_repo)
    assert not marker.exists()


def test_git_diff_truncated_returns_partial_output(tmp_git_repo: Path) -> None:
    # A >1 MB diff trips the output cap: git is killed by US, so the bounded
    # partial output (with truncation marker) must be delivered, not a
    # bogus LocalToolError built from the killed process's returncode.
    big = "".join(f"line {i}\n" for i in range(120_000))  # ~1.4 MB
    _commit_file(tmp_git_repo, "big.txt", big, "add big")
    (tmp_git_repo / "big.txt").write_text(big.replace("line", "LINE"), encoding="utf-8")

    out = git_diff(tmp_git_repo)
    assert "diff --git" in out
    assert out.rstrip().endswith("...[output truncated]")


def test_run_git_kills_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake git that spawns a long-lived child: the timeout kill must reap
    # the whole process group, not just the direct child.
    import os
    import time

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "child.pid"
    fake_git = bin_dir / "git"
    fake_git.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pid_file}\nwait\n", encoding="utf-8")
    fake_git.chmod(0o755)

    monkeypatch.setattr(git_tool_impls.shutil, "which", lambda _name: str(fake_git))
    real_env = git_tool_impls._git_environment

    def env_with_fake_path() -> dict:
        env = real_env()
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '/usr/bin:/bin')}"
        return env

    monkeypatch.setattr(git_tool_impls, "_git_environment", env_with_fake_path)

    with pytest.raises(LocalToolError, match="timed out"):
        run_git(["git", "status"], timeout=1.0)

    child_pid = int(pid_file.read_text(encoding="utf-8").strip())
    for _ in range(50):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild process {child_pid} survived the timeout kill")


def test_run_git_bare_argv_requires_subcommand() -> None:
    with pytest.raises(LocalToolError, match="requires a subcommand"):
        run_git(["git"])
