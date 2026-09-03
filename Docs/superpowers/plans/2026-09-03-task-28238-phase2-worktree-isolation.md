# TASK-28238 Phase 2: Git-Worktree Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fleet child can opt into an isolated git worktree; its changes merge back explicitly (apply-as-uncommitted-diff or real merge commit, behind a confirm card), never silently (AC#1).

**Architecture:** A new `Agents/agent_worktree.py` module owns the git lifecycle (create/discard/merge, typed never-raise results, reusing `Workspaces/git_workspace.py`'s `_run_user_git`/`detect_git_workspace` shape). The shared `LocalToolProvider` gains a per-run agent-root map consulted first in `_select_admitted_root`, so an isolated child's fs/git tools auto-route to its worktree via the existing `RunAdmittedWorkspaceRoot` authority machinery (own executor, guard, allow_write). `spawn_subagent` gains `isolation="worktree"`; agent_service creates+admits at the `precreated_run_id` site and retires in `run_child`'s finally (and the thread-start-failure path). Two new runtime tools (`merge_agent_worktree`, `discard_agent_worktree`) follow the wait_agents LoopDeps pattern — NOT in `pure_runtime_tools` — with a blocking confirm card cloned from the `run_skill_script` pattern.

**Tech Stack:** Python ≥3.11, stdlib + existing repo helpers only. No new dependencies.

**Spec:** `backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md` — the "Phase 2 — git-worktree isolation (resolved design, 2026-09-03)" section is binding.

## Global Constraints

- Work on a NEW branch off `origin/dev` in a clean worktree. Branch: `feat/task-28238-phase2-worktree-isolation`.
- Test runner: the worktree venv only — `.venv/bin/python -m pytest` (bare `pytest`/`uv run` fail; if pytest missing: `VIRTUAL_ENV=.venv uv pip install -e . pytest pytest-asyncio pytest-timeout ruff jsonschema hypothesis`).
- ZERO new `logger.*` calls in any production path (a new diagnostic trips both the boot-census ratchet and the derived-artifacts inventory). Typed results carry all information.
- `Agents/agent_worktree.py` must be imported ONLY lazily (inside functions) from boot-resident modules; `tool_catalog.py` schema additions are data-only.
- All git invocations go through `Workspaces/git_workspace.py`'s `_run_user_git(root, *args, timeout=..., check=...)` + `_user_git_env()`; repo detection via `detect_git_workspace(root)` (never-raise, typed-refusal shape). git-missing/not-a-repo → typed refusal, never an exception.
- Single-agent and non-isolated fleet behavior byte-identical: the agent-root map is consulted only for run_ids explicitly admitted; unmapped runs hit today's code paths (phase-1 AC#3 discipline).
- Fail closed: no confirm surface wired (headless) → merge/discard tools refuse honestly; a live (non-terminal) child's worktree is never merged.
- Worktrees live OUTSIDE the repo: `Path(tempfile.gettempdir()) / "tldw_agent_worktrees" / f"agent-{run_id[:8]}"`.
- The stale-write ledger (phase 1) needs no changes; do not touch its code.

---

### Task 1: agent_worktree module — lifecycle core (create/discard/GC)

**Files:**
- Create: `tldw_chatbook/Agents/agent_worktree.py`
- Test: `Tests/Agents/test_agent_worktree.py`

**Interfaces:**
- Consumes: `Workspaces/git_workspace.py`: `_run_user_git(root, *args, timeout=..., check=...) -> GitCmdResult` (`.code`, `.out`, `.err` — VERIFY exact attr names at `Workspaces/git_workspace.py:198-250` and use them), `detect_git_workspace(root)` (returns `GitWorkspaceInfo | GitWorkspaceRefusal | None`).
- Produces (later tasks import these exact names):
  - `@dataclass(frozen=True) AgentWorktree: run_id: str; worktree_path: Path; branch: str; base_sha: str`
  - `@dataclass(frozen=True) WorktreeRefusal: reason_code: str; message: str`
  - `create_agent_worktree(repo_root: Path, run_id: str) -> AgentWorktree | WorktreeRefusal`
  - `discard_agent_worktree(repo_root: Path, wt: AgentWorktree) -> WorktreeRefusal | None` (None = success; removes worktree dir AND deletes the branch)
  - `prune_stale_agent_worktrees(repo_root: Path, live_run_ids: set[str]) -> int` (removes `agent/<run_id>` worktrees whose run_id is not live; returns count)
  - Reason codes: `"not_a_git_repo"`, `"git_unavailable"`, `"worktree_create_failed"`, `"worktree_remove_failed"`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Agents/test_agent_worktree.py
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
```

- [ ] **Step 2: Run to verify RED**

Run: `.venv/bin/python -m pytest Tests/Agents/test_agent_worktree.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Agents.agent_worktree`

- [ ] **Step 3: Implement the module**

```python
# tldw_chatbook/Agents/agent_worktree.py
"""TASK-28238 phase 2: git-worktree lifecycle for isolated fleet children.

An isolated child works in `git worktree add <tmp> -b agent/<run_id> HEAD`:
a CLEAN checkout of HEAD (uncommitted shared-tree changes deliberately do not
carry — dirt belongs to the user). All git runs go through
`Workspaces.git_workspace._run_user_git` (user identity, scrubbed redirection
vars); repo detection mirrors `detect_git_workspace`'s never-raise shape.
Typed results, no logging (results carry the information).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

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


def _detect(repo_root: Path) -> "WorktreeRefusal | None":
    """None when repo_root is a usable git repo; a refusal otherwise."""
    from tldw_chatbook.Workspaces.git_workspace import detect_git_workspace

    try:
        info = detect_git_workspace(Path(repo_root))
    except Exception:  # noqa: BLE001 - detection must never raise outward
        info = None
    # detect_git_workspace returns info | refusal | None; only a positive
    # detection is usable. Refusal/None both mean "not a git workspace here".
    if info is None or type(info).__name__ == "GitWorkspaceRefusal":
        return WorktreeRefusal(
            reason_code="not_a_git_repo",
            message="Worktree isolation requires the workspace root to be a git repository.",
        )
    return None


def _git(repo_root: Path, *args: str) -> "tuple[int, str, str]":
    """(code, stdout, stderr) via the user-identity git runner; never raises."""
    from tldw_chatbook.Workspaces.git_workspace import _run_user_git

    try:
        result = _run_user_git(Path(repo_root), *args, check=False)
    except FileNotFoundError:
        return 127, "", "git is not available on this system"
    except Exception as exc:  # noqa: BLE001 - lifecycle must degrade, not raise
        return 1, "", str(exc)
    code = getattr(result, "code", getattr(result, "returncode", 1))
    out = getattr(result, "out", getattr(result, "stdout", "")) or ""
    err = getattr(result, "err", getattr(result, "stderr", "")) or ""
    return int(code), str(out), str(err)


def create_agent_worktree(
    repo_root: Path, run_id: str
) -> "AgentWorktree | WorktreeRefusal":
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    code, out, err = _git(
        repo_root, "worktree", "add", str(dest), "-b", branch, "HEAD"
    )
    if code != 0:
        return WorktreeRefusal(
            "worktree_create_failed", f"git worktree add failed: {err.strip()[:200]}"
        )
    return AgentWorktree(
        run_id=run_id, worktree_path=dest, branch=branch, base_sha=base_sha
    )


def discard_agent_worktree(
    repo_root: Path, wt: AgentWorktree
) -> "WorktreeRefusal | None":
    """Remove the worktree and delete its branch. None on success."""
    code, _out, err = _git(
        repo_root, "worktree", "remove", "--force", str(wt.worktree_path)
    )
    if code != 0 and wt.worktree_path.exists():
        return WorktreeRefusal(
            "worktree_remove_failed", f"git worktree remove failed: {err.strip()[:200]}"
        )
    _git(repo_root, "branch", "-D", wt.branch)  # best-effort; branch may be merged
    return None


def prune_stale_agent_worktrees(repo_root: Path, live_run_ids: "set[str]") -> int:
    """Remove agent worktrees whose run is no longer live. Returns count removed."""
    code, out, _err = _git(repo_root, "worktree", "list", "--porcelain")
    if code != 0:
        return 0
    removed = 0
    current_path: "Path | None" = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree "):])
        elif line.startswith("branch ") and current_path is not None:
            branch = line[len("branch "):].removeprefix("refs/heads/")
            if branch.startswith(_BRANCH_PREFIX):
                run_id = branch[len(_BRANCH_PREFIX):]
                if run_id not in live_run_ids:
                    wt = AgentWorktree(
                        run_id=run_id,
                        worktree_path=current_path,
                        branch=branch,
                        base_sha="",
                    )
                    if discard_agent_worktree(repo_root, wt) is None:
                        removed += 1
            current_path = None
    return removed
```

Implementer note: FIRST open `Workspaces/git_workspace.py:198-250` and check `_run_user_git`'s real signature and its result's attribute names (`GitCmdResult` — the plan's `getattr` chain covers `.code/.out/.err` vs `.returncode/.stdout/.stderr`; simplify to the real names once read). Also confirm `detect_git_workspace`'s return types by reading its def at `Workspaces/git_workspace.py:309` and replace the `type(...).__name__` check with a real isinstance import if the class is importable without heavy deps.

- [ ] **Step 4: Run to verify GREEN**

Run: `.venv/bin/python -m pytest Tests/Agents/test_agent_worktree.py -q -p no:cacheprovider`
Expected: 5 passed

- [ ] **Step 5: Lint + commit**

Run: `.venv/bin/ruff check tldw_chatbook/Agents/agent_worktree.py Tests/Agents/test_agent_worktree.py --select F`
```bash
git add tldw_chatbook/Agents/agent_worktree.py Tests/Agents/test_agent_worktree.py
git commit -m "feat(agents): agent worktree lifecycle (create/discard/prune) (TASK-28238 P2 T1)"
```

---

### Task 2: agent_worktree module — merge-back operations

**Files:**
- Modify: `tldw_chatbook/Agents/agent_worktree.py`
- Test: `Tests/Agents/test_agent_worktree.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) MergeOutcome: mode: str; diffstat: str; commit_sha: str | None` (commit_sha only for mode="merge")
  - `merge_agent_worktree_changes(repo_root: Path, wt: AgentWorktree, mode: str = "apply") -> MergeOutcome | WorktreeRefusal`
  - New reason codes: `"nothing_to_merge"`, `"merge_conflict"` (message NAMES the conflicting files), `"apply_conflict"` (same), `"invalid_mode"`.
- Semantics (spec-binding):
  - The child must have committed its work? NO — children edit files; nothing commits in the worktree. So merge-back FIRST auto-commits the worktree's dirty state onto the agent branch (`git -C <wt> add -A && commit -m "agent work (<run8>)"` via the worktree path) — a child's uncommitted work is otherwise invisible to diff/merge. If the worktree is clean AND the branch has no commits past base → `nothing_to_merge`.
  - `mode="apply"`: `git diff --binary <base>..<branch>` piped to `git apply --check` in the shared tree; on check failure → `apply_conflict` refusal naming files (parse `error: patch failed: <file>` lines from stderr), NOTHING applied (atomic); on success → `git apply` the same patch (changes land UNCOMMITTED). Branch survives.
  - `mode="merge"`: `git merge --no-ff <branch> -m "Merge agent worktree <run8>"`; on conflict → collect `git diff --name-only --diff-filter=U`, then `git merge --abort`, refuse with the file list; on success → commit_sha = new HEAD. Branch survives (discard deletes it later).
  - Both: diffstat from `git diff --stat <base>..<branch>`.

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/Agents/test_agent_worktree.py
from tldw_chatbook.Agents.agent_worktree import (  # noqa: E402
    MergeOutcome,
    merge_agent_worktree_changes,
)


def test_apply_mode_lands_uncommitted_diff(repo):
    wt = create_agent_worktree(repo, "run-apply001")
    (wt.worktree_path / "a.txt").write_text("child version\n")
    (wt.worktree_path / "new.txt").write_text("brand new\n")
    outcome = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(outcome, MergeOutcome), getattr(outcome, "message", outcome)
    assert outcome.commit_sha is None
    assert (repo / "a.txt").read_text() == "child version\n"
    assert (repo / "new.txt").read_text() == "brand new\n"
    # uncommitted: the shared tree is dirty, no new commit on HEAD
    status = _git(repo, "status", "--porcelain")
    assert " a.txt" in status or "M a.txt" in status.replace("  ", " ")
    assert "a.txt" in outcome.diffstat
    discard_agent_worktree(repo, wt)


def test_merge_mode_creates_merge_commit(repo):
    wt = create_agent_worktree(repo, "run-merge001")
    (wt.worktree_path / "a.txt").write_text("merged version\n")
    before = _git(repo, "rev-parse", "HEAD").strip()
    outcome = merge_agent_worktree_changes(repo, wt, mode="merge")
    assert isinstance(outcome, MergeOutcome), getattr(outcome, "message", outcome)
    after = _git(repo, "rev-parse", "HEAD").strip()
    assert outcome.commit_sha == after != before
    assert (repo / "a.txt").read_text() == "merged version\n"
    parents = _git(repo, "log", "-1", "--format=%P").split()
    assert len(parents) == 2  # a real --no-ff merge commit
    discard_agent_worktree(repo, wt)


def test_apply_conflict_refuses_atomically_naming_file(repo):
    wt = create_agent_worktree(repo, "run-conflict1")
    (wt.worktree_path / "a.txt").write_text("child side\n")
    (repo / "a.txt").write_text("user side\n")  # conflicting shared-tree change
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "apply_conflict"
    assert "a.txt" in refusal.message
    assert (repo / "a.txt").read_text() == "user side\n"  # untouched
    discard_agent_worktree(repo, wt)


def test_merge_conflict_aborts_and_names_file(repo):
    wt = create_agent_worktree(repo, "run-conflict2")
    (wt.worktree_path / "a.txt").write_text("child side\n")
    (repo / "a.txt").write_text("user side committed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "user change")
    before = _git(repo, "rev-parse", "HEAD").strip()
    refusal = merge_agent_worktree_changes(repo, wt, mode="merge")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "merge_conflict"
    assert "a.txt" in refusal.message
    assert _git(repo, "rev-parse", "HEAD").strip() == before  # aborted cleanly
    assert "user side committed" in (repo / "a.txt").read_text()
    discard_agent_worktree(repo, wt)


def test_clean_worktree_is_nothing_to_merge(repo):
    wt = create_agent_worktree(repo, "run-noop0001")
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "nothing_to_merge"
    discard_agent_worktree(repo, wt)
```

- [ ] **Step 2: RED** — `.venv/bin/python -m pytest Tests/Agents/test_agent_worktree.py -q -p no:cacheprovider -k "apply_mode or merge_mode or conflict or nothing_to"` → ImportError (MergeOutcome).

- [ ] **Step 3: Implement**

```python
# append to tldw_chatbook/Agents/agent_worktree.py

@dataclass(frozen=True)
class MergeOutcome:
    """A successful merge-back."""

    mode: str
    diffstat: str
    commit_sha: "str | None"


def _git_in(worktree: Path, repo_root: Path, *args: str) -> "tuple[int, str, str]":
    """Run git inside the WORKTREE (same runner, -C into the worktree)."""
    return _git(repo_root, "-C", str(worktree), *args)


def merge_agent_worktree_changes(
    repo_root: Path, wt: AgentWorktree, mode: str = "apply"
) -> "MergeOutcome | WorktreeRefusal":
    """Land the child's changes in the shared tree. Explicit, atomic, typed.

    Args:
        repo_root: The shared workspace root.
        wt: The child's worktree record.
        mode: ``"apply"`` (3-step check+apply, lands UNCOMMITTED) or
            ``"merge"`` (real ``--no-ff`` merge commit).

    Returns:
        A MergeOutcome, or a refusal (``nothing_to_merge`` /
        ``apply_conflict`` / ``merge_conflict`` naming files / ``invalid_mode``).
    """
    if mode not in ("apply", "merge"):
        return WorktreeRefusal("invalid_mode", f"unknown merge mode: {mode!r}")
    # A child's work is plain edits in its worktree; commit them onto the
    # agent branch first so diff/merge can see them.
    code, out, _err = _git_in(wt.worktree_path, repo_root, "status", "--porcelain")
    if code == 0 and out.strip():
        _git_in(wt.worktree_path, repo_root, "add", "-A")
        _git_in(
            wt.worktree_path,
            repo_root,
            "commit",
            "-m",
            f"agent work ({wt.run_id[:8]})",
        )
    code, out, _err = _git(repo_root, "diff", "--stat", f"{wt.base_sha}..{wt.branch}")
    diffstat = out.strip()
    if code != 0 or not diffstat:
        return WorktreeRefusal(
            "nothing_to_merge", "the agent worktree has no changes past its base"
        )
    if mode == "apply":
        code, patch, err = _git(
            repo_root, "diff", "--binary", f"{wt.base_sha}..{wt.branch}"
        )
        if code != 0:
            return WorktreeRefusal("apply_conflict", f"diff failed: {err.strip()[:200]}")
        check = _apply_patch(repo_root, patch, check_only=True)
        if check is not None:
            return check
        applied = _apply_patch(repo_root, patch, check_only=False)
        if applied is not None:
            return applied
        return MergeOutcome(mode="apply", diffstat=diffstat, commit_sha=None)
    # mode == "merge"
    code, _out, err = _git(
        repo_root,
        "merge",
        "--no-ff",
        wt.branch,
        "-m",
        f"Merge agent worktree {wt.run_id[:8]}",
    )
    if code != 0:
        conflict_code, files, _ = _git(
            repo_root, "diff", "--name-only", "--diff-filter=U"
        )
        _git(repo_root, "merge", "--abort")
        names = files.strip() or err.strip()[:200]
        return WorktreeRefusal(
            "merge_conflict", f"merge conflicts; resolve manually: {names}"
        )
    code, head, _ = _git(repo_root, "rev-parse", "HEAD")
    return MergeOutcome(
        mode="merge", diffstat=diffstat, commit_sha=head.strip() if code == 0 else None
    )


def _apply_patch(
    repo_root: Path, patch: str, *, check_only: bool
) -> "WorktreeRefusal | None":
    """git-apply the patch text via stdin; refusal naming files on failure."""
    from tldw_chatbook.Workspaces.git_workspace import _user_git_env
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        return WorktreeRefusal("git_unavailable", "git is not available")
    argv = [git, "apply"] + (["--check"] if check_only else [])
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            cwd=str(repo_root),
            input=patch,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=_user_git_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return WorktreeRefusal("apply_conflict", f"git apply failed: {exc}")
    if proc.returncode != 0:
        failed = ", ".join(
            line.split(":", 2)[-1].strip()
            for line in proc.stderr.splitlines()
            if "patch failed" in line or "already exists" in line or "error:" in line
        ) or proc.stderr.strip()[:200]
        return WorktreeRefusal(
            "apply_conflict", f"patch does not apply cleanly: {failed}"
        )
    return None
```

Implementer note: check `_user_git_env`'s real name/visibility in `Workspaces/git_workspace.py` (recon says it exists as `_user_git_env()`); if `_run_user_git` accepts stdin input directly, prefer routing `git apply` through it instead of the local subprocess block — keep whichever is smaller after reading the real signature. `git diff --binary` output can be large for big changes — acceptable (merge-back is rare and explicit).

- [ ] **Step 4: GREEN** — the 5 new tests + the Task 1 five: `.venv/bin/python -m pytest Tests/Agents/test_agent_worktree.py -q -p no:cacheprovider` → 10 passed.

- [ ] **Step 5: Lint + commit**

```bash
git add tldw_chatbook/Agents/agent_worktree.py Tests/Agents/test_agent_worktree.py
git commit -m "feat(agents): dual-mode merge-back for agent worktrees (TASK-28238 P2 T2)"
```

---

### Task 3: Provider per-run agent roots (admit/retire + routing)

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (`__init__` ~line 576 region; `_select_admitted_root` ~line 740)
- Test: `Tests/Agents/test_local_tool_provider.py` (append)

**Interfaces:**
- Consumes: existing `RunAdmittedWorkspaceRoot` (:339), `_select_admitted_root` (:740), `current_run_id()` from `Agents/run_context`.
- Produces (Task 5 calls these):
  - `admit_run_workspace_root(self, run_id: str, authority: RunAdmittedWorkspaceRoot) -> None`
  - `retire_run_workspace_root(self, run_id: str) -> None`
- Semantics: a lock-guarded `self._agent_roots: dict[str, RunAdmittedWorkspaceRoot]` (+ `self._agent_roots_lock = threading.Lock()`), SEPARATE from `_admitted_roots`. `_select_admitted_root` consults it FIRST (before the `self._admitted_roots is None` early return): for a path-authority tool name, if `self._agent_roots.get(current_run_id())` is set, return that authority with `root_alias` popped from the args (auto-routing; an explicit alias is ignored for an isolated run — its world IS the worktree). Unmapped runs: behavior byte-identical to today.

- [ ] **Step 1: Write the failing tests**

```python
# append to Tests/Agents/test_local_tool_provider.py

# --- TASK-28238 phase 2: per-run agent worktree roots ---

def _agent_authority(root, alias="agent-x"):
    import os as _os
    import stat as _stat

    root = Path(root).resolve()
    identities = []
    for component in (*reversed(root.parents), root):
        value = _os.lstat(component)
        identities.append((str(component), value.st_dev, value.st_ino, value.st_mode))
    return RunAdmittedWorkspaceRoot(
        workspace_id="agent-worktree",
        binding_id=alias,
        alias=alias,
        root=root,
        locator_fingerprint="f" * 64,
        root_identity=tuple(identities),
        allow_write=True,
        guard=lambda write: root.is_dir(),
        workspace_executor=InProcessWorkspaceExecutor(root),
    )


def test_admitted_run_routes_fs_tools_to_worktree(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "wt"
    shared.mkdir()
    worktree.mkdir()
    provider = _guard_provider(shared)
    provider.admit_run_workspace_root("run-iso", _agent_authority(worktree))
    with use_run_id("run-iso"):
        result = provider.invoke(
            "local:fs_write", {"path": "out.txt", "content": "isolated\n"}
        )
    assert result.ok, str(result.error)
    assert (worktree / "out.txt").read_text() == "isolated\n"
    assert not (shared / "out.txt").exists()


def test_unmapped_run_unchanged_and_retire_restores(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "wt"
    shared.mkdir()
    worktree.mkdir()
    provider = _guard_provider(shared)
    provider.admit_run_workspace_root("run-iso", _agent_authority(worktree))
    # a DIFFERENT run still writes to the shared root
    with use_run_id("run-other"):
        assert provider.invoke(
            "local:fs_write", {"path": "s.txt", "content": "shared\n"}
        ).ok
    assert (shared / "s.txt").read_text() == "shared\n"
    # retire: the isolated run falls back to the shared root
    provider.retire_run_workspace_root("run-iso")
    with use_run_id("run-iso"):
        assert provider.invoke(
            "local:fs_write", {"path": "back.txt", "content": "home\n"}
        ).ok
    assert (shared / "back.txt").read_text() == "home\n"
    assert not (worktree / "back.txt").exists()


def test_agent_root_write_permission_enforced(tmp_path):
    shared = tmp_path / "shared"
    worktree = tmp_path / "wt"
    shared.mkdir()
    worktree.mkdir()
    provider = _guard_provider(shared)
    authority = _agent_authority(worktree)
    object.__setattr__(authority, "allow_write", False)
    provider.admit_run_workspace_root("run-ro", authority)
    with use_run_id("run-ro"):
        result = provider.invoke(
            "local:fs_write", {"path": "no.txt", "content": "x\n"}
        )
    assert not result.ok  # write refused by authority machinery
```

- [ ] **Step 2: RED** — `-k "routes_fs_tools or retire_restores or write_permission"` → AttributeError: no `admit_run_workspace_root`.

- [ ] **Step 3: Implement**

(a) `__init__`, next to `self._read_ledger` (Task-2/phase-1 block):

```python
        # TASK-28238 phase 2: per-run agent worktree authorities. SEPARATE
        # from the constructor's alias map so Console workspace bindings and
        # the legacy single-root path are untouched; consulted FIRST by
        # _select_admitted_root, keyed by current_run_id().
        self._agent_roots: dict[str, RunAdmittedWorkspaceRoot] = {}
        self._agent_roots_lock = threading.Lock()
```

(b) methods next to `_select_admitted_root`:

```python
    def admit_run_workspace_root(
        self, run_id: str, authority: RunAdmittedWorkspaceRoot
    ) -> None:
        """Route ``run_id``'s path tools to ``authority`` (agent worktree).

        Args:
            run_id: The isolated child run's id.
            authority: The worktree authority (own executor, guard, perms).
        """
        with self._agent_roots_lock:
            self._agent_roots[str(run_id)] = authority

    def retire_run_workspace_root(self, run_id: str) -> None:
        """Remove ``run_id``'s agent-root mapping (child finished).

        Args:
            run_id: The run whose mapping to drop; absent is a no-op.
        """
        with self._agent_roots_lock:
            self._agent_roots.pop(str(run_id), None)
```

(c) at the TOP of `_select_admitted_root`, before the existing `if name not in _PATH_AUTHORITY_LOCAL_NAMES or self._admitted_roots is None:` line:

```python
        if name in _PATH_AUTHORITY_LOCAL_NAMES and self._agent_roots:
            from tldw_chatbook.Agents.run_context import current_run_id

            with self._agent_roots_lock:
                agent_root = self._agent_roots.get(current_run_id())
            if agent_root is not None:
                clean_args = dict(args) if type(args) is dict else args
                if isinstance(clean_args, dict):
                    clean_args.pop("root_alias", None)
                return agent_root, clean_args
```

- [ ] **Step 4: GREEN** — the 3 new tests, then the full provider + ledger + worktree suites: `.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py Tests/Agents/test_fs_read_ledger.py Tests/Agents/test_agent_worktree.py -q -p no:cacheprovider` — no NEW failures (phase-1 baseline 235 + 10 worktree + 3 new).

- [ ] **Step 5: Lint + commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat(agents): per-run agent worktree roots in the local provider (TASK-28238 P2 T3)"
```

---

### Task 4: Spawn wiring — isolation="worktree"

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (SPAWN_TOOL_SCHEMA :100-117 and `build_spawn_schema` :120+ — add the property to BOTH)
- Modify: `tldw_chatbook/Agents/agent_service.py` (`_launch_fleet_child` :4460; child_run_id site :4552-4554; `run_child` finally :4675-4694; thread-start-failure teardown :4747-4787; the two `_launch_fleet_child(` call sites :5091 and :5440 — thread `isolation` from the spawn call args)
- Test: `Tests/Agents/test_agent_service.py` OR the fleet test home — IMPLEMENTER: find where `_launch_fleet_child`/spawn is currently tested (grep `spawn` in Tests/Agents/test_agent_service*.py and Tests/Agents/*fleet*) and add there.

**Interfaces:**
- Consumes: Task 1's `create_agent_worktree`/`discard_agent_worktree`/`WorktreeRefusal`; Task 3's `admit_run_workspace_root`/`retire_run_workspace_root`; `registry.resolve_owner_for_name(name) -> (tool_id, provider) | None` (tool_catalog.py:1611) to reach the LocalToolProvider (resolve a known local name, e.g. `"fs_read"`; `isinstance` against the lazily imported `LocalToolProvider`).
- Produces: `self._agent_worktrees: dict[str, AgentWorktree]` on AgentService keyed by handle_id (Task 5 reads it); spawn arg `isolation` (`"worktree"`).
- Semantics:
  - SPAWN schema gains optional `"isolation": {"type": "string", "enum": ["worktree"], "description": "Run this child in an isolated git worktree; its changes stay out of the shared tree until explicitly merged back with merge_agent_worktree."}`.
  - `_launch_fleet_child(spawn_task, agent_name, ..., isolation=None)`; both call sites pass `isolation=call.args.get("isolation")` (or however the spawn args reach them — mirror how `agent_name` flows).
  - At the child_run_id site (right after `child_kwargs["precreated_run_id"] = child_run_id`, :4552): when `isolation == "worktree"`, lazily import agent_worktree; resolve the workspace root — use the LocalToolProvider's `workspace_root` property (the provider found via `resolve_owner_for_name`); `create_agent_worktree(root, child_run_id)`. A `WorktreeRefusal` → release the reserved handle the same way a cap refusal does (mirror the reserve-failure unwind in this function) and return `(None, ToolResult(ok=False, error=f"worktree isolation refused [{r.reason_code}]: {r.message}"))` — the spawn FAILS HONESTLY, never silently shares the tree. On success: build the authority (same shape as the Task-3 test helper `_agent_authority`, but with `WorkspaceToolExecutor(wt.worktree_path)` — lazy import from Tools.workspace_tool_executor — and `guard=lambda write: wt.worktree_path.is_dir()`; compute `root_identity` with the same `os.lstat` loop, inline in a small module-level helper `_worktree_root_identity(path)` in agent_worktree.py — ADD it there with a 2-line docstring); `provider.admit_run_workspace_root(child_run_id, authority)`; `self._agent_worktrees[handle.handle_id] = wt`.
  - Retire in `run_child`'s `finally`, immediately after `fleet.finish(...)` (:4694): lazily re-resolve the provider and `retire_run_workspace_root(child_run_id)` (wrap in `try/except Exception: pass` — teardown must never mask the child's real outcome). The WORKTREE ITSELF SURVIVES (merged or discarded later by Task 5's tools; pruned by GC otherwise).
  - Same retire in the thread-start-failure teardown (:4767-4782), plus `discard_agent_worktree` there (a never-ran child has nothing worth keeping) and drop the `self._agent_worktrees` entry.

- [ ] **Step 1: Write the failing test** (adapt to the fleet-test harness you find; the behavior pins are what matter)

```python
def test_isolated_spawn_writes_are_invisible_until_merge(...existing harness fixtures...):
    """AC#1 core: an isolation='worktree' child's fs_write lands in ITS worktree,
    not the shared tree; a non-isolated sibling still writes the shared tree."""
    # Arrange a real git repo as the workspace root (mirror Task 1's repo fixture),
    # spawn a child with isolation="worktree" whose scripted turn calls
    # fs_write("iso.txt", ...), and a plain child writing plain.txt.
    # Assert: shared/iso.txt does NOT exist; the service's _agent_worktrees entry's
    # worktree_path/iso.txt DOES; shared/plain.txt exists.


def test_isolated_spawn_refuses_on_non_git_workspace(...):
    """Spawn with isolation on a plain directory returns an honest refusal
    ToolResult mentioning the reason, and no child handle is left reserved."""
```

Implementer: these two tests are the acceptance pins; write them against the real harness (scripted model turns / fake gateway — see how existing spawn tests drive a child's tool calls). If the harness cannot drive a child fs_write end-to-end, split the pin: (a) unit-test the launch path by calling `_launch_fleet_child` semantics through the service's public spawn entry with a stub registry provider recording admit/retire calls; (b) keep the non-git refusal test end-to-end (it needs no child tool call). State in your report which shape you built.

- [ ] **Step 2: RED**, **Step 3: implement per the semantics above**, **Step 4: GREEN + full Tests/Agents guard suites no new failures**, **Step 5: lint + commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/agent_worktree.py Tests/Agents/
git commit -m "feat(agents): spawn_subagent isolation=worktree (create/admit/retire) (TASK-28238 P2 T4)"
```

---

### Task 5: merge/discard runtime tools (headless half)

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (schemas beside WAIT_AGENTS_SCHEMA :165)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (LoopDeps fields :513-514 region; dispatch branches beside :2334-2360; do NOT touch pure_runtime_tools :2067-2073)
- Modify: `tldw_chatbook/Agents/agent_service.py` (closures near wait_agents :5106; LoopDeps wiring :6629; new run-entry parameter threading the confirm callable)
- Test: `Tests/Agents/test_agent_service.py` / fleet-test home (same as Task 4)

**Interfaces:**
- Consumes: Task 2's `merge_agent_worktree_changes`/`MergeOutcome`, Task 1's `discard_agent_worktree`, Task 4's `self._agent_worktrees`; the fleet handle's terminal status (`FleetCoordinator` — a handle is mergeable only when TERMINAL; find the status accessor used by check_agents' closure and reuse it).
- Produces:
  - `tool_catalog.py`: `MERGE_AGENT_WORKTREE_TOOL_NAME = "merge_agent_worktree"`, `DISCARD_AGENT_WORKTREE_TOOL_NAME = "discard_agent_worktree"`, `MERGE_AGENT_WORKTREE_SCHEMA` (params: `handle_id` required; `mode` enum ["apply","merge"] default apply — description states apply lands UNCOMMITTED changes for user review and merge creates a real merge commit, both require user confirmation), `DISCARD_AGENT_WORKTREE_SCHEMA` (handle_id).
  - `agent_runtime.py` LoopDeps: `merge_agent_worktree: Callable[[str, str], ToolResult] | None = None`, `discard_agent_worktree: Callable[[str], ToolResult] | None = None`; dispatch branches exactly mirroring wait_agents' `elif call.name == ... and deps.X is not None:` shape (STEP_TOOL_CALL add + result assignment).
  - `agent_service.py`: run-entry kwarg `request_worktree_merge_confirm: "Callable[[dict], dict] | None" = None` threaded to where closures build; closures:

```python
        def merge_agent_worktree_tool(handle_id: str, mode: str = "apply") -> ToolResult:
            wt = self._agent_worktrees.get(str(handle_id))
            if wt is None:
                return ToolResult(ok=False, error=f"no agent worktree for handle {handle_id!r}")
            if not _handle_is_terminal(handle_id):  # reuse check_agents' status source
                return ToolResult(ok=False, error="child is still running; wait for it to finish before merging")
            if request_worktree_merge_confirm is None:
                return ToolResult(ok=False, error="merge requires user confirmation, and no approval surface is available in this session")
            from tldw_chatbook.Agents.agent_worktree import merge_agent_worktree_changes
            # preview diffstat for the card (never mutate before consent)
            decision = request_worktree_merge_confirm({
                "handle_id": handle_id, "mode": mode,
                "branch": wt.branch, "worktree": str(wt.worktree_path),
            })
            if not decision.get("allow", False):
                return ToolResult(ok=False, error="The user declined the worktree merge.")
            outcome = merge_agent_worktree_changes(workspace_root, wt, mode=mode)
            if hasattr(outcome, "reason_code"):
                return ToolResult(ok=False, error=f"[{outcome.reason_code}] {outcome.message}")
            landed = "as UNCOMMITTED changes (review and commit them)" if outcome.commit_sha is None else f"as merge commit {outcome.commit_sha[:9]}"
            return ToolResult(ok=True, content=f"Merged agent worktree {landed}.\n{outcome.diffstat}")
```

  (discard closure analogous, no confirm needed? NO — discard destroys the child's work: same confirm gate, same fail-closed None check.) `workspace_root` = the same root resolved in Task 4's spawn path; resolve once where closures build. Wire `merge_agent_worktree=merge_agent_worktree_tool if fleet_active else None` (and discard) into the LoopDeps construction at :6629, and append both schemas beside WAIT/CHECK in the schema plan at :702 (same `fleet_active`-style condition).
- Fail-closed rulings baked in: no confirm surface → refuse; non-terminal child → refuse; unknown handle → refuse.

- [ ] **Step 1: failing tests** — headless closures are directly testable: build the service the way existing fleet tests do, seed `self._agent_worktrees` with a real Task-1 worktree on a real temp repo, stub `request_worktree_merge_confirm` to allow/deny, assert: deny → refusal + tree untouched; allow+apply → MergeOutcome text + uncommitted change landed; no-confirm-surface → refusal; non-terminal handle → refusal. Also one dispatch test at the runtime layer if the existing runtime-tool tests (`Tests/Agents/test_search_run_log_runtime_tool.py` pattern) make that cheap.
- [ ] **Step 2: RED** → **Step 3: implement** → **Step 4: GREEN + full Tests/Agents guard suites** → **Step 5: lint + commit**

```bash
git commit -m "feat(agents): merge/discard agent-worktree runtime tools, fail-closed confirm (TASK-28238 P2 T5)"
```

---

### Task 6: Console confirm card + bridge wiring

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (clone `request_skill_script_confirm` :13370-13513 → `request_worktree_merge_confirm(payload) -> dict`; reuse the same card widget/round machinery with copy for "Merge agent worktree — mode/branch/diff summary"; keep `use_human_input_wait(run_id)` so the tool-call deadline pauses)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (pass the controller method through to the service run entry, exactly as `run_skill_script`'s confirm flows — find where the bridge forwards run kwargs into AgentService and add `request_worktree_merge_confirm=...`)
- Test: `Tests/Chat/` — mirror however `request_skill_script_confirm` is tested (grep for it in Tests/Chat; if only integration-tested, add a focused test that the bridge threads the kwarg and that a deny propagates as the refusal ToolResult through a scripted turn).

Semantics: the card shows handle_id, mode (apply = "lands uncommitted for your review"; merge = "creates a real merge commit"), branch, and diffstat if cheaply available; Deny is the default action; the decision dict is `{"allow": bool}` (mirror the skill-script decision shape exactly — read what request_skill_script_confirm returns and match it).

- [ ] Steps: RED test → implement → GREEN + no new failures in the touched Tests/Chat files → lint → commit
```bash
git commit -m "feat(console): confirm card for agent-worktree merge/discard (TASK-28238 P2 T6)"
```

---

### Task 7: Guardrail sweeps

Same four sweeps as phase 1's Task 6, same interpretation rules:
- [ ] `.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py Tests/Performance/test_app_import_weight.py -q -p no:cacheprovider` — if red naming `agent_worktree`, convert its importer to lazy (it must already be lazy per Global Constraints).
- [ ] `.venv/bin/python scripts/check_persistent_diagnostic_inventory.py` — clean (zero new logger calls). Drift naming our files = a logging call slipped in; remove it.
- [ ] `.venv/bin/ruff check` `--select F` on every touched file.
- [ ] `Tests/Agents/ + the touched Tests/Chat files`: no NEW failures vs the known env baseline (14 pre-existing Tests/Agents failures reproduce on clean dev — compare NAMES).
- [ ] Commit only if fixes were needed.

---

### Task 8: Docs + task hygiene

**Files:** `backlog/tasks/task-28238 - Worktree-isolation-and-stale-write-guard-for-parallel-sub-agents.md`, `backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md`

- [ ] Tick AC#1 (isolated worktree, explicit merge-back) — with phase 1's AC#2/#3/#4 already ticked, ALL ACs are now done: set frontmatter `status: Done` (edit directly; the backlog CLI is unavailable in the worktree — note the substitution).
- [ ] Append phase-2 implementation notes: the per-run admitted-root routing (and that the sketch's per-child-registry fear was resolved by the existing authority machinery), worktree lifecycle module, dual-mode merge-back + confirm card, fail-closed rulings (no-confirm/non-terminal/unknown-handle), file list, test names per AC.
- [ ] Spec Status line → "Phases 1 and 2 implemented (phase 1 merged in PR #2341; phase 2 on branch feat/task-28238-phase2-worktree-isolation)".
- [ ] Commit: `docs(agents): record TASK-28238 phase-2 completion; task Done`
