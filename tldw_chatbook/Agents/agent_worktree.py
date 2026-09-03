"""TASK-28238 phase 2: git-worktree lifecycle for isolated fleet children.

An isolated child works in `git worktree add <tmp> -b agent/<run_id> HEAD`:
a CLEAN checkout of HEAD (uncommitted shared-tree changes deliberately do not
carry — dirt belongs to the user). All git runs go through
`Workspaces.git_workspace._run_user_git` (user identity, scrubbed redirection
vars); repo detection reuses `detect_git_workspace`, which never raises.
Typed results, no logging (results carry the information).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceError,
    GitWorkspaceInfo,
    _run_user_git,
    detect_git_workspace,
)

_BRANCH_PREFIX = "agent/"


@dataclass(frozen=True)
class AgentWorktree:
    """One isolated child checkout."""

    run_id: str
    worktree_path: Path
    branch: str
    base_sha: str


@dataclass(frozen=True)
class WorktreeRefusal:
    """A reason-coded, never-raised failure."""

    reason_code: str
    message: str


def _worktrees_base() -> Path:
    return Path(tempfile.gettempdir()) / "tldw_agent_worktrees"


def _detect(repo_root: Path) -> WorktreeRefusal | None:
    """None when repo_root is a usable git repo; a refusal otherwise."""
    info = detect_git_workspace(Path(repo_root))
    if not isinstance(info, GitWorkspaceInfo):
        return WorktreeRefusal(
            reason_code="not_a_git_repo",
            message="Worktree isolation requires the workspace root to be a git repository.",
        )
    return None


def _git(repo_root: Path, *args: str) -> tuple[int, str, str]:
    """(returncode, stdout, stderr) via the user-identity git runner; never raises."""
    try:
        result = _run_user_git(Path(repo_root), *args, check=False)
    except GitWorkspaceError as exc:
        # git missing, timed out, or an OS-level launch failure -- check=False
        # only suppresses the nonzero-exit raise, not these.
        return 127, "", str(exc)
    return result.returncode, result.stdout, result.stderr


def create_agent_worktree(repo_root: Path, run_id: str) -> AgentWorktree | WorktreeRefusal:
    """Create an isolated worktree for ``run_id`` at HEAD.

    Args:
        repo_root: The shared workspace root (must be a git repo).
        run_id: The child run's id; names the branch ``agent/<run_id>``.

    Returns:
        The created worktree, or a reason-coded refusal (never raises).
    """
    refusal = _detect(repo_root)
    if refusal is not None:
        return refusal
    code, out, err = _git(repo_root, "rev-parse", "HEAD")
    if code == 127:
        return WorktreeRefusal("git_unavailable", err)
    if code != 0:
        return WorktreeRefusal(
            "worktree_create_failed", f"could not resolve HEAD: {err.strip()[:200]}"
        )
    base_sha = out.strip()
    branch = f"{_BRANCH_PREFIX}{run_id}"
    dest = _worktrees_base() / f"agent-{run_id[:8]}"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return WorktreeRefusal(
            "worktree_create_failed", f"cannot create worktree base: {exc}"
        )
    code, out, err = _git(repo_root, "worktree", "add", str(dest), "-b", branch, "HEAD")
    if code != 0:
        return WorktreeRefusal(
            "worktree_create_failed", f"git worktree add failed: {err.strip()[:200]}"
        )
    return AgentWorktree(run_id=run_id, worktree_path=dest, branch=branch, base_sha=base_sha)


def discard_agent_worktree(repo_root: Path, wt: AgentWorktree) -> WorktreeRefusal | None:
    """Remove the worktree and delete its branch. None on success."""
    code, _out, err = _git(repo_root, "worktree", "remove", "--force", str(wt.worktree_path))
    if code != 0 and wt.worktree_path.exists():
        return WorktreeRefusal(
            "worktree_remove_failed", f"git worktree remove failed: {err.strip()[:200]}"
        )
    _git(repo_root, "branch", "-D", wt.branch)  # best-effort; branch may be merged
    return None


def prune_stale_agent_worktrees(repo_root: Path, live_run_ids: set[str]) -> int:
    """Remove agent worktrees whose run is no longer live. Returns count removed."""
    code, out, _err = _git(repo_root, "worktree", "list", "--porcelain")
    if code != 0:
        return 0
    removed = 0
    current_path: Path | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree "):])
        elif line.startswith("branch ") and current_path is not None:
            branch = line[len("branch "):].removeprefix("refs/heads/")
            if branch.startswith(_BRANCH_PREFIX):
                run_id = branch[len(_BRANCH_PREFIX):]
                if run_id not in live_run_ids:
                    wt = AgentWorktree(
                        run_id=run_id, worktree_path=current_path, branch=branch, base_sha=""
                    )
                    if discard_agent_worktree(repo_root, wt) is None:
                        removed += 1
            current_path = None
    return removed
