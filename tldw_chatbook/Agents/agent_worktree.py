"""TASK-28238 phase 2: git-worktree lifecycle for isolated fleet children.

An isolated child works in `git worktree add <tmp> -b agent/<run_id> HEAD`:
a CLEAN checkout of HEAD (uncommitted shared-tree changes deliberately do not
carry — dirt belongs to the user). All git runs go through
`Workspaces.git_workspace._run_user_git` (user identity, scrubbed redirection
vars); repo detection reuses `detect_git_workspace`, which never raises.
Typed results, no logging (results carry the information).
"""

from __future__ import annotations

import os
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


def _worktree_root_identity(path: Path) -> tuple[tuple[str, int, int, int], ...]:
    """Ancestor-chain identity for an admitted worktree root.

    Mirrors `console_chat_controller._capture_project_root_identity`'s
    `os.lstat` walk, minus its symlink-rejection `None` returns -- the
    admitted-root authority only needs a non-empty tuple, not a veto.
    """
    root = Path(path).resolve()
    identities: list[tuple[str, int, int, int]] = []
    for component in (*reversed(root.parents), root):
        value = os.lstat(component)
        identities.append((str(component), value.st_dev, value.st_ino, value.st_mode))
    return tuple(identities)


def discard_agent_worktree(repo_root: Path, wt: AgentWorktree) -> WorktreeRefusal | None:
    """Remove the worktree and delete its branch. None on success."""
    code, _out, err = _git(repo_root, "worktree", "remove", "--force", str(wt.worktree_path))
    if code != 0 and wt.worktree_path.exists():
        return WorktreeRefusal(
            "worktree_remove_failed", f"git worktree remove failed: {err.strip()[:200]}"
        )
    _git(repo_root, "branch", "-D", wt.branch)  # best-effort; branch may be merged
    return None


@dataclass(frozen=True)
class MergeOutcome:
    """A successful merge-back."""

    mode: str
    diffstat: str
    commit_sha: str | None


def _conflicting_files(stderr: str) -> str:
    """Extract the file name(s) named by a failed `git apply`, in order.

    Real git uses two different shapes depending on the failure:
      - content conflict: "error: patch failed: a.txt:1"
      - new-file collision: "error: a.txt: already exists in working directory"
    Both name the file, but at different positions in the line.
    """
    names: list[str] = []
    for raw in stderr.splitlines():
        line = raw.strip()
        if line.startswith("error: patch failed: "):
            name = line[len("error: patch failed: "):].rsplit(":", 1)[0]
        elif line.startswith("error: ") and "already exists" in line:
            name = line[len("error: "):].split(":", 1)[0]
        else:
            continue
        if name and name not in names:
            names.append(name)
    return ", ".join(names) or stderr.strip()[:200]


def _apply_patch(repo_root: Path, patch: str, *, check_only: bool) -> WorktreeRefusal | None:
    """git-apply the patch text (via a temp file, so this stays on `_git`); a
    refusal naming files on failure, None on success.
    """
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
            fh.write(patch)
            patch_path = fh.name
    except OSError as exc:
        return WorktreeRefusal("apply_conflict", f"cannot stage patch: {exc}")
    try:
        args = ("apply", *(("--check",) if check_only else ()), patch_path)
        code, _out, err = _git(repo_root, *args)
    finally:
        Path(patch_path).unlink(missing_ok=True)
    if code == 0:
        return None
    return WorktreeRefusal(
        "apply_conflict", f"patch does not apply cleanly: {_conflicting_files(err)}"
    )


def preview_agent_worktree_diffstat(repo_root: Path, wt: AgentWorktree) -> str:
    """Cheap, non-mutating diffstat preview for a confirm card.

    A child's work is typically still UNCOMMITTED in its own worktree --
    nothing commits it there; `merge_agent_worktree_changes` is the one
    that does, right before computing its own (post-commit) diffstat. A
    preview must run BEFORE the user consents to anything, so it cannot
    perform that commit -- it diffs `wt.base_sha` against the worktree's
    own WORKING TREE instead (`git diff --stat <base_sha>`, no second
    ref), which reads uncommitted MODIFICATIONS too.

    `git diff` alone never lists untracked files, though, and a new file
    is as common a shape of "child's work" as an edited one (`git add -A`
    is exactly why `merge_agent_worktree_changes` sees them) -- so this
    also lists untracked paths from a plain `git status --porcelain`
    (still read-only; no `git add -N` staging, which would leave the
    child's worktree index touched by a preview alone).

    Args:
        repo_root: Unused by this preview (kept for signature parity with
            this module's other `repo_root`-first functions); everything
            runs inside `wt.worktree_path`.
        wt: The child's worktree record.

    Returns:
        The diffstat text (plus an untracked-files line when any exist),
        or `""` on any git failure (never raises) or when nothing has
        changed yet.

    # ponytail: no rename-detection tuning, no line-count for untracked
    # files (just their names), and filenames with unusual characters
    # print in git's quoted form -- good enough for a confirm card. If a
    # child's diff turns out to need more nuance, revisit then.
    """
    del repo_root
    lines: list[str] = []
    code, out, _err = _git(wt.worktree_path, "diff", "--stat", wt.base_sha)
    if code == 0 and out.strip():
        lines.append(out.strip())
    code, status, _err = _git(wt.worktree_path, "status", "--porcelain")
    if code == 0:
        untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]
        if untracked:
            lines.append(f"new (untracked): {', '.join(untracked)}")
    return "\n".join(lines)


def merge_agent_worktree_changes(
    repo_root: Path, wt: AgentWorktree, mode: str = "apply"
) -> MergeOutcome | WorktreeRefusal:
    """Land the child's changes in the shared tree. Explicit, atomic, typed.

    Args:
        repo_root: The shared workspace root.
        wt: The child's worktree record.
        mode: ``"apply"`` (check-then-apply; lands UNCOMMITTED) or ``"merge"``
            (a real ``--no-ff`` merge commit).

    Returns:
        A MergeOutcome, or a refusal (``nothing_to_merge`` / ``apply_conflict``
        / ``merge_conflict`` naming the conflicting files / ``invalid_mode``
        / ``worktree_commit_failed``).
    """
    if mode not in ("apply", "merge"):
        return WorktreeRefusal("invalid_mode", f"unknown merge mode: {mode!r}")
    # The child only edits files in its worktree -- nothing commits them
    # there, so diff/merge can't see the work until it lands on the branch.
    # A failure here must not fall through to "nothing_to_merge" -- that
    # would silently strand the child's work (lost on discard, no trail).
    code, out, _err = _git(wt.worktree_path, "status", "--porcelain")
    if code == 0 and out.strip():
        add_code, _add_out, add_err = _git(wt.worktree_path, "add", "-A")
        if add_code != 0:
            return WorktreeRefusal(
                "worktree_commit_failed",
                f"could not commit agent work: {add_err.strip()[:200]}",
            )
        # Explicit identity: the worktree may have no resolvable git user
        # (no ~/.gitconfig, no global identity) even though the shared repo
        # does, since `commit` needs one regardless of who is landing it.
        commit_code, _commit_out, commit_err = _git(
            wt.worktree_path,
            "-c", "user.name=tldw-agent",
            "-c", "user.email=agent@tldw.local",
            "commit", "-m", f"agent work ({wt.run_id[:8]})",
        )
        if commit_code != 0:
            return WorktreeRefusal(
                "worktree_commit_failed",
                f"could not commit agent work: {commit_err.strip()[:200]}",
            )
    code, out, _err = _git(repo_root, "diff", "--stat", f"{wt.base_sha}..{wt.branch}")
    diffstat = out.strip()
    if code != 0 or not diffstat:
        return WorktreeRefusal(
            "nothing_to_merge", "the agent worktree has no changes past its base"
        )
    if mode == "apply":
        code, patch, err = _git(repo_root, "diff", "--binary", f"{wt.base_sha}..{wt.branch}")
        if code != 0:
            return WorktreeRefusal("apply_conflict", f"diff failed: {err.strip()[:200]}")
        refusal = _apply_patch(repo_root, patch, check_only=True)
        if refusal is not None:
            return refusal
        refusal = _apply_patch(repo_root, patch, check_only=False)
        if refusal is not None:
            return refusal
        return MergeOutcome(mode="apply", diffstat=diffstat, commit_sha=None)
    # mode == "merge"
    code, _out, err = _git(
        repo_root, "merge", "--no-ff", wt.branch, "-m", f"Merge agent worktree {wt.run_id[:8]}"
    )
    if code != 0:
        _code, files, _err2 = _git(repo_root, "diff", "--name-only", "--diff-filter=U")
        _git(repo_root, "merge", "--abort")
        names = files.strip() or err.strip()[:200]
        return WorktreeRefusal(
            "merge_conflict", f"merge conflicts; resolve manually: {names}"
        )
    code, head, _err3 = _git(repo_root, "rev-parse", "HEAD")
    return MergeOutcome(
        mode="merge", diffstat=diffstat, commit_sha=head.strip() if code == 0 else None
    )


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
