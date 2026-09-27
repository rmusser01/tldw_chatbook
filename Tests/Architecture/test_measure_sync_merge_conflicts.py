"""Integration tests for scripts/measure_sync_merge_conflicts.py.

Builds a real, disposable git repository per test (no mocks) and runs the
script against it as a subprocess, exercising:

- the base sync-merge / conflict-rate report (2 real syncs, 1 conflicting),
- an unrelated nested merge correctly excluded from the sync count (Qodo #4),
- `--merges 0` rejected as an invalid value (Qodo #1),
- `--since` narrowing the window to syncs committed after a cutoff (Qodo #5).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "measure_sync_merge_conflicts.py"

# Fixed commit dates for the two real syncs, so `--since` can deterministically
# split them: the conflicting sync is dated well before the cutoff, the clean
# one well after.
_SYNC_A_DATE = "2020-01-01 00:00:00 +0000"
_SYNC_B_DATE = "2024-01-01 00:00:00 +0000"
_SINCE_CUTOFF = "2022-01-01"


def _env() -> dict[str, str]:
    """A git environment isolated from the real user's global config."""
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "Test"
    env["GIT_AUTHOR_EMAIL"] = "test@example.com"
    env["GIT_COMMITTER_NAME"] = "Test"
    env["GIT_COMMITTER_EMAIL"] = "test@example.com"
    return env


def _git(
    repo: Path, *args: str, env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result


def _head(repo: Path, env: dict[str, str]) -> str:
    return _git(repo, "rev-parse", "HEAD", env=env).stdout.strip()


def _write(repo: Path, name: str, content: str) -> None:
    (repo / name).write_text(content)


def _build_repo(repo: Path) -> None:
    """Build a `dev` history with exactly 2 real sync merges (one conflicting,
    one clean) plus one unrelated nested merge that must NOT be counted.

    Shape:
      C0 -- D1 -- D2 (dev)
       \\           \\
        A1 --------- SA (conflict on f.txt, resolved) --\\
                                                           MA (dev)
        B1 ------------------------------------------ SB (clean) --\\
                                                                      MB (dev)
        side1 (orphan, unrelated) -- MC_inner (branch-c merges side) --\\
                                                                         MC (dev)

    Real syncs: SA (dated 2020, conflicting) and SB (dated 2024, clean).
    MC_inner's second parent (side1) never lands on dev's first-parent chain,
    so it must be excluded from the sync count (Qodo #4).
    """
    env = _env()
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "dev", env=env)
    _git(repo, "config", "commit.gpgsign", "false", env=env)

    for name in ("f.txt", "g.txt", "h.txt"):
        _write(repo, name, "base\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "C0: initial", env=env)
    c0 = _head(repo, env)

    _write(repo, "f.txt", "dev changed f\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "D1: dev changes f.txt", env=env)

    _write(repo, "g.txt", "dev changed g\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "D2: dev changes g.txt", env=env)
    d2 = _head(repo, env)
    _git(repo, "update-ref", "refs/remotes/origin/dev", d2, env=env)

    # Branch A: diverges at C0, changes f.txt -> conflicts with dev's own
    # f.txt change when it syncs.
    _git(repo, "checkout", "-q", "-b", "branch-a", c0, env=env)
    _write(repo, "f.txt", "A changed f\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "A1: branch A changes f.txt", env=env)

    conflict = _git(
        repo, "merge", "--no-ff", "-m", "sync: merge dev into branch A", d2,
        env=env, check=False,
    )
    assert conflict.returncode != 0, "setup expected a real conflict on f.txt"
    _write(repo, "f.txt", "resolved f\n")
    _git(repo, "add", "f.txt", env=env)
    sa_env = dict(env, GIT_AUTHOR_DATE=_SYNC_A_DATE, GIT_COMMITTER_DATE=_SYNC_A_DATE)
    _git(repo, "commit", "-q", "--no-edit", env=sa_env)

    _git(repo, "checkout", "-q", "dev", env=env)
    _git(repo, "merge", "--no-ff", "-m", "Merge branch A into dev", "branch-a", env=env)
    ma = _head(repo, env)
    _git(repo, "update-ref", "refs/remotes/origin/dev", ma, env=env)

    # Branch B: diverges at C0, only ever touches h.txt -> syncs cleanly.
    _git(repo, "checkout", "-q", "-b", "branch-b", c0, env=env)
    _write(repo, "h.txt", "B changed h\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "B1: branch B changes h.txt", env=env)

    sb_env = dict(env, GIT_AUTHOR_DATE=_SYNC_B_DATE, GIT_COMMITTER_DATE=_SYNC_B_DATE)
    _git(repo, "merge", "--no-ff", "-m", "sync: merge dev into branch B", ma, env=sb_env)

    _git(repo, "checkout", "-q", "dev", env=env)
    _git(repo, "merge", "--no-ff", "-m", "Merge branch B into dev", "branch-b", env=env)
    mb = _head(repo, env)
    _git(repo, "update-ref", "refs/remotes/origin/dev", mb, env=env)

    # An unrelated orphan branch, never merged into dev directly.
    _git(repo, "checkout", "-q", "--orphan", "side", env=env)
    _git(repo, "rm", "-rf", "-q", ".", env=env)
    _write(repo, "side.txt", "side content\n")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "side1: unrelated history", env=env)

    # Branch C: merges the unrelated side branch into itself (nested merge
    # whose second parent is NOT on dev's first-parent chain).
    _git(repo, "checkout", "-q", "-b", "branch-c", mb, env=env)
    _git(
        repo, "merge", "--no-ff", "--allow-unrelated-histories",
        "-m", "branch C merges unrelated side", "side", env=env,
    )

    _git(repo, "checkout", "-q", "dev", env=env)
    _git(repo, "merge", "--no-ff", "-m", "Merge branch C into dev", "branch-c", env=env)
    mc = _head(repo, env)
    _git(repo, "update-ref", "refs/remotes/origin/dev", mc, env=env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    _build_repo(repo_dir)
    return repo_dir


def _run_script(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo, env=_env(), capture_output=True, text=True,
    )


def test_counts_two_real_syncs_and_excludes_the_unrelated_merge(repo: Path) -> None:
    result = _run_script(repo, "--merges", "20")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "window: last 20 dev merges"
    assert "sync merges: 2" in result.stdout
    assert "conflicted: 1 (50%)" in result.stdout
    assert re.search(r"^\s*1\s+f\.txt$", result.stdout, re.MULTILINE), result.stdout


def test_merges_zero_is_rejected(repo: Path) -> None:
    result = _run_script(repo, "--merges", "0")
    assert result.returncode != 0


def test_since_narrows_to_the_post_cutoff_sync(repo: Path) -> None:
    result = _run_script(repo, "--merges", "20", "--since", _SINCE_CUTOFF)
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.splitlines()[0]
        == f"window: last 20 dev merges, syncs since {_SINCE_CUTOFF}"
    )
    assert "sync merges: 1" in result.stdout
    assert "conflicted: 0 (0%)" in result.stdout
