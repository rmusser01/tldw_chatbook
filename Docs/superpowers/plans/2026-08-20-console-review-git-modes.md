# Change Review Git Modes Implementation Plan (TASK-16801 arc B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real-git `current` mode on the Change Review screen — working-tree view plus confirmed commit / push / open-PR actions — with active git-workspace detection groundwork.

**Architecture:** One new engine module `Workspaces/git_workspace.py` (detection + runner + status/commit/push/PR-URL, pure of UI, per-step honest outcomes, a THIRD disclosed env posture: ambient env preserved, repo-targeting vars scrubbed); thin wrappers on `AgentRunsChangeReviewProvider`; the screen gains a sentinel `Select` entry whose load runs in an exclusive worker, synthesizes per-root pseudo-rows so the `(row, ChangedFile)` plumbing survives, and gates snapshot-only actions (revert/comments) out of the mode. Commit uses the verified pathspec recipe (`add -A -- sel` + `commit -m msg -- sel`) so pre-staged unrelated work survives.

**Tech Stack:** Python 3.11+, Textual 8.x, subprocess git (argv lists), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md` — the binding authority; §2 lists seven EMPIRICALLY VERIFIED probes this plan's recipes rest on. Conflicts inside this plan resolve against the spec.

## Global Constraints

- NEVER pass `--force`/`--force-with-lease` to push; a non-fast-forward failure surfaces git's stderr excerpt honestly (assert on captured argv in tests).
- NEVER mutate the user's index or working tree from a VIEW operation — no `git diff --no-index`, no `add --intent-to-add`; untracked previews are synthesized in Python.
- Env posture (spec §3.1): start from ambient `os.environ`; scrub case-insensitively `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`, `GIT_COMMON_DIR`, `GIT_CEILING_DIRECTORIES`; set `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0`, `GIT_PAGER=cat`. `HOME`, `SSH_AUTH_SOCK`, `GIT_SSH_COMMAND`, `GIT_ASKPASS`, `SSH_ASKPASS` MUST survive.
- Subprocess discipline: argv lists only (never shell), `cwd=root` (no `-C`), stdin `DEVNULL`, `capture_output=True`, timeouts 30s reads / 120s commit / 300s push, stderr excerpts capped at 400 chars (`.strip()[:400]`), paths after `--`.
- Paths are data: all porcelain/numstat parsing is `-z` NUL-delimited; porcelain rename records are NEW path then OLD path (spec §2 probe 2); `-uall` is mandatory.
- Commit recipe (spec §2 probe 1, pinned): `git add -A -- <selected>` then `git commit -m <msg> -- <selected>`; an unrelated pre-staged index entry MUST survive staged and uncommitted.
- Unborn HEAD (spec §2 probe 4): branch via `symbolic-ref --short -q HEAD` (NEVER `rev-parse --abbrev-ref HEAD`); unborn probe via `rev-parse --verify -q HEAD` exit 1; when unborn, `git diff HEAD` and `--numstat` are never run.
- `upstream_remote` comes from `for-each-ref --format=%(upstream:remotename)` — NEVER from splitting the upstream string on `/` (remote names can contain `/`, spec §2 probe 6).
- Whole-tree status and all mutations run in `run_worker(..., exclusive=True, thread=True)` landing via `call_from_thread`; the single-file diff-on-focus read stays synchronous through `_diff_text_for` (spec §4). Detection/status NEVER run on the Inspector rail's 0.2s tick.
- Kill switch: `[change_review] git_actions` (flat section), default `True`; off ⇒ zero behavior change, byte-compatible snapshot-mode behavior either way (screen still OPENS on the latest turn, never on the pseudo-entry).
- Tests: real git repos + `git init --bare` local remotes — no mocked git for engine/e2e; screen tests use the real CSS stack, the REAL provider, and a FILE-backed AgentRunsDB (never `:memory:`); guard tests proven RED pre-fix where feasible.
- Test command: `VIRTUAL_ENV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:randomly -q <paths>`.
- Test repos MUST set `git config user.email`/`user.name` locally and use `git init -b main` (CI machines may lack global identity/default-branch config); never rely on the runner's global git config.

---

### Task 1: Runner + detection (`Workspaces/git_workspace.py`)

**Files:**
- Create: `tldw_chatbook/Workspaces/git_workspace.py`
- Test: `Tests/Workspaces/test_git_workspace_detection.py`

**Interfaces:**
- Consumes: nothing new (stdlib + loguru).
- Produces (later tasks rely on these EXACT names):
  `GitWorkspaceError(Exception)`; `GitCmdResult(returncode, stdout, stderr)` frozen dataclass;
  `_run_user_git(root: Path, *args: str, timeout: float = READ_TIMEOUT_SECONDS, check: bool = True) -> GitCmdResult`;
  `GitWorkspaceInfo(root, repo_root, branch, detached, unborn, upstream, upstream_remote, remotes, ahead, behind)` frozen dataclass (fields exactly as spec §3);
  `GitWorkspaceRefusal(reason: str)` frozen dataclass;
  `detect_git_workspace(root: Path) -> GitWorkspaceInfo | GitWorkspaceRefusal | None`;
  constants `READ_TIMEOUT_SECONDS = 30.0`, `COMMIT_TIMEOUT_SECONDS = 120.0`, `PUSH_TIMEOUT_SECONDS = 300.0`.

- [ ] **Step 1: Write failing detection tests**

```python
# Tests/Workspaces/test_git_workspace_detection.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest -p no:randomly -q Tests/Workspaces/test_git_workspace_detection.py`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module absent).

- [ ] **Step 3: Implement runner + detection**

Module docstring MUST carry the three-runner env-posture table from spec §3.1 (shadow tracker scrubs ALL `GIT_*`; read-only agent tools strip `HOME`; this module preserves ambient env and scrubs only repo-TARGETING vars — say WHY for each).

```python
_SCRUBBED_VARS = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_COMMON_DIR",
    "GIT_CEILING_DIRECTORIES",
})

def _user_git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() not in _SCRUBBED_VARS}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_PAGER"] = "cat"
    return env

def _run_user_git(root, *args, timeout=READ_TIMEOUT_SECONDS, check=True):
    git = shutil.which("git")
    if git is None:
        raise GitWorkspaceError("git is not installed")
    try:
        proc = subprocess.run(
            [git, *args], cwd=str(root), env=_user_git_env(),
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitWorkspaceError(
            f"git {args[0]} timed out after {timeout:.0f}s"
        ) from exc
    except OSError as exc:
        raise GitWorkspaceError(str(exc)) from exc
    if check and proc.returncode != 0:
        raise GitWorkspaceError(
            f"git {args[0]} failed ({proc.returncode}): "
            f"{(proc.stderr or '').strip()[:400]}"
        )
    return GitCmdResult(proc.returncode, proc.stdout, proc.stderr)
```

`detect_git_workspace(root)` (wrap the whole body so any `GitWorkspaceError` → `None` with a debug log — detection must never raise to the UI):
1. `rev-parse --show-toplevel` check=False → nonzero ⇒ `None`.
2. `Path(toplevel).resolve() != Path(root).resolve()` ⇒ `GitWorkspaceRefusal("workspace is inside a repository — git actions need the workspace root to be the repository root")`.
3. `symbolic-ref --short -q HEAD` check=False → exit 0 ⇒ branch=stdout.strip(), detached=False; exit 1 ⇒ branch=None, detached=True.
4. `rev-parse --verify -q HEAD` check=False → exit != 0 ⇒ `unborn=True`.
5. `remote -v` check=False → parse `(push)` lines into ordered unique `(name, url)` pairs (split on tab, then space).
6. Not detached and not unborn: `rev-parse --abbrev-ref @{upstream}` check=False → exit 0 ⇒ upstream=stdout.strip(); then `for-each-ref --format=%(upstream:remotename) refs/heads/<branch>` check=False ⇒ upstream_remote (empty ⇒ None).
7. Upstream set: `rev-list --left-right --count @{upstream}...HEAD` check=False → parse `behind\tahead` (LEFT is behind).

- [ ] **Step 4: Run tests to verify they pass**

Run: same command. Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/git_workspace.py Tests/Workspaces/test_git_workspace_detection.py
git commit -m "feat(git-modes): user-repo git runner + workspace detection (TASK-16801 T1)"
```

---

### Task 2: Working-tree status, diff, untracked preview

**Files:**
- Modify: `tldw_chatbook/Workspaces/git_workspace.py`
- Test: `Tests/Workspaces/test_git_workspace_status.py`

**Interfaces:**
- Consumes: Task 1's `_run_user_git`, `GitWorkspaceInfo`.
- Produces: `CurrentRootStatus(root: Path, info: GitWorkspaceInfo, files: tuple[ChangedFile, ...], untracked: frozenset[str])` frozen dataclass (`ChangedFile` imported from `tldw_chatbook.Workspaces.change_tracking`);
  `working_tree_status(root: Path, info: GitWorkspaceInfo) -> CurrentRootStatus`;
  `working_tree_diff(root: Path, path: str) -> str`;
  `untracked_preview(root: Path, path: str, max_lines: int) -> str`.

- [ ] **Step 1: Write failing tests**

Test cases (same `repo` fixture pattern as Task 1; every case real git):

```python
def test_status_lists_modified_added_deleted(repo):
    (repo / "a.txt").write_text("edit\n")          # M
    (repo / "new.txt").write_text("x\n")           # ?? -> A + untracked
    (repo / "gone.txt").write_text("y\n")
    _git(repo, "add", "gone.txt"); _git(repo, "commit", "-qm", "add gone")
    (repo / "gone.txt").unlink()                    # D
    info = detect_git_workspace(repo)
    status = working_tree_status(repo, info)
    by_path = {f.path: f for f in status.files}
    assert by_path["a.txt"].status == "M" and by_path["a.txt"].adds == 1
    assert by_path["new.txt"].status == "A" and "new.txt" in status.untracked
    assert by_path["gone.txt"].status == "D"

def test_staged_rename_new_then_old(repo):
    _git(repo, "mv", "a.txt", "b.txt")
    status = working_tree_status(repo, detect_git_workspace(repo))
    row = {f.path: f for f in status.files}["b.txt"]
    assert row.status == "R" and row.old_path == "a.txt"

def test_untracked_directory_lists_per_file(repo):
    (repo / "sub").mkdir(); (repo / "sub" / "x.txt").write_text("x\n")
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert "sub/x.txt" in {f.path for f in status.files}   # -uall pin

def test_added_then_deleted_collapses_to_D(repo):
    (repo / "tmp.txt").write_text("x\n"); _git(repo, "add", "tmp.txt")
    (repo / "tmp.txt").unlink()                     # XY == "AD"
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert {f.path: f for f in status.files}["tmp.txt"].status == "D"

def test_path_with_spaces_and_utf8(repo): ...      # "wei rd ü.txt" round-trips

def test_unborn_repo_all_files_untracked_no_numstat(tmp_path):
    # fresh init + one file; working_tree_status must not run `diff HEAD`
    # (spec §2 probe 4) -- assert adds/dels are 0 and path in untracked.

def test_untracked_preview_text_bounded_and_binary(repo):
    (repo / "t.txt").write_text("\n".join(str(i) for i in range(100)) + "\n")
    text = untracked_preview(repo, "t.txt", max_lines=10)
    assert text.splitlines()[0].startswith("new file: t.txt")
    assert sum(1 for l in text.splitlines() if l.startswith("+")) == 10
    assert "truncated" in text
    (repo / "b.bin").write_bytes(b"\x00\x01\x02")
    assert "binary" in untracked_preview(repo, "b.bin", max_lines=10)

def test_working_tree_diff_returns_unified(repo):
    (repo / "a.txt").write_text("edit\n")
    assert "-base" in working_tree_diff(repo, "a.txt")
```

- [ ] **Step 2: Run to verify FAIL** (ImportError on the new names).

- [ ] **Step 3: Implement**

Porcelain parse: `status --porcelain=v1 -z -uall`, NUL-token walk; record = `XY<space>path`, and when `X in "RC"` consume ONE extra token as `old_path` (NEW first — spec §2 probe 2). XY collapse precedence (with the rationale comments): `??` → `A` + untracked; contains `R` → `R`; contains `C` → `C`; contains `D` → `D` (covers `AD`: the file is GONE from disk — showing `A` would advertise a file that does not exist); contains `A` → `A`; else first non-space char verbatim (`M`, `T`, `U`… pass through, matching `ChangedFile`'s unknown-letters contract). Counts: when not `info.unborn`, ONE `diff HEAD --numstat -z` call merged by path (rename-tolerant token walk copied from `change_tracking.changed_files`); binary (`-`) ⇒ adds=dels=0. `working_tree_diff` = `diff HEAD -- <path>` (callers never invoke it when unborn). `untracked_preview`: `(root / path).read_bytes()` capped read (`max_lines * 400` bytes is plenty), `\x00` in the head 8KB ⇒ `f"new file: {path}\n(binary file, {size} bytes)"`; else header + blank + `+`-prefixed lines up to `max_lines` + `f"… truncated at {max_lines} lines"` when over. OSError ⇒ honest one-line error text, never a raise.

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** (`feat(git-modes): working-tree status/diff/preview engine (T2)`).

---

### Task 3: Commit engine

**Files:**
- Modify: `tldw_chatbook/Workspaces/git_workspace.py`
- Test: `Tests/Workspaces/test_git_workspace_commit.py`

**Interfaces:**
- Consumes: Task 1's runner; `COMMIT_TIMEOUT_SECONDS`.
- Produces: `GitStepOutcome(step: str, ok: bool, detail: str = "")` frozen dataclass;
  `CommitResult(outcomes: tuple[GitStepOutcome, ...], short_sha: str | None)` (sha None on any failure);
  `CommitRefusedError(GitWorkspaceError)`;
  `commit_selected(root: Path, files: Sequence[str], message: str, new_branch: str | None, *, run_active: Callable[[], bool]) -> CommitResult`.

- [ ] **Step 1: Write failing tests**

```python
def test_prestaged_unrelated_entry_survives(repo):
    # spec §2 probe 1 -- THE regression pin for the index-hijack trap.
    (repo / "keep.txt").write_text("user-staged\n"); _git(repo, "add", "keep.txt")
    (repo / "a.txt").write_text("agent\n")
    result = commit_selected(repo, ["a.txt"], "agent work", None,
                             run_active=lambda: False)
    assert result.short_sha
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["a.txt"]
    assert _git(repo, "status", "--porcelain") == "A  keep.txt"

def test_run_active_refuses_before_touching(repo):
    (repo / "a.txt").write_text("x\n")
    with pytest.raises(CommitRefusedError):
        commit_selected(repo, ["a.txt"], "m", None, run_active=lambda: True)
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")

def test_new_branch_created_then_committed(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(repo, ["a.txt"], "m", "feat/xyz",
                             run_active=lambda: False)
    assert result.short_sha
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "feat/xyz"

def test_bad_branch_name_refused_preflight(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(repo, ["a.txt"], "m", "-bad",
                             run_active=lambda: False)
    assert result.short_sha is None
    assert result.outcomes[0].step == "validate-branch" and not result.outcomes[0].ok
    # nothing staged, no commit happened
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"

def test_existing_branch_stops_before_commit(repo): ...   # checkout -b main fails; no new commit

def test_merge_in_progress_refused(repo, tmp_path):
    # build a real conflicted merge, then MERGE_HEAD exists
    ...
    result = commit_selected(repo, ["a.txt"], "m", None, run_active=lambda: False)
    assert result.outcomes[0].step == "in-progress-check" and not result.outcomes[0].ok
    assert "merge" in result.outcomes[0].detail

def test_dash_leading_message_commits_literally(repo):
    # spec §2 probe 5
    (repo / "a.txt").write_text("x\n")
    commit_selected(repo, ["a.txt"], "--amend", None, run_active=lambda: False)
    assert _git(repo, "log", "-1", "--format=%s") == "--amend"

def test_deletion_only_selection_commits(repo): ...       # unlink + commit -> file gone at HEAD

def test_unborn_first_commit_works(tmp_path): ...         # spec §2 probe 4
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

`commit_selected` step order, each step appending a `GitStepOutcome` and stopping at the first failure:
1. `run_active()` ⇒ raise `CommitRefusedError("a run is active on this workspace — finish or stop the run first")` (mirrors `revert_paths`, `change_revert.py:168`).
2. `in-progress-check`: `rev-parse --verify -q` on `MERGE_HEAD`, `REBASE_HEAD`, `CHERRY_PICK_HEAD` (check=False); any exit 0 ⇒ failed outcome `"finish or abort the merge/rebase/cherry-pick first"`.
3. When `new_branch`: `validate-branch` via `check-ref-format --branch <name>` check=False (nonzero ⇒ failed outcome with stderr excerpt); then `create-branch` via `checkout -b <name>`.
4. `stage`: `add -A -- *files`.
5. `commit`: `commit -m <message> -- *files` with `timeout=COMMIT_TIMEOUT_SECONDS` (user hooks run).
6. `rev-parse --short HEAD` ⇒ `short_sha`.
Message and paths are argv elements; paths always after `--`. Empty `files` or blank `message` ⇒ `GitWorkspaceError` (the UI validates first; the engine still refuses).

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** (`feat(git-modes): pathspec commit engine with per-step outcomes (T3)`).

---

### Task 4: Push engine + PR compare URL

**Files:**
- Modify: `tldw_chatbook/Workspaces/git_workspace.py`
- Test: `Tests/Workspaces/test_git_workspace_push.py`, `Tests/Workspaces/test_pr_urls.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `PushResult(state: str, detail: str)` with `state in {"pushed", "up_to_date", "failed"}`;
  `push_current(root: Path, info: GitWorkspaceInfo, remote: str | None) -> PushResult` (remote None ⇒ derive: upstream_remote, else sole remote — caller guarantees the >1-remote case arrives resolved);
  `pr_compare_url(root: Path, info: GitWorkspaceInfo) -> str | GitWorkspaceRefusal`;
  `_parse_remote_url(url: str) -> tuple[str, str, str] | None` (host, owner_path, repo).

- [ ] **Step 1: Write failing tests**

Push e2e (the AC #5 named case — every push against a real `git init --bare` remote):

```python
def test_first_push_sets_upstream_and_moves_bare_ref(repo, bare): ...
    # push_current -> state "pushed"; bare rev-parse main == repo HEAD;
    # detect_git_workspace(repo).upstream == "origin/main"
def test_second_push_up_to_date(repo, bare): ...
def test_nonff_push_fails_honestly_no_force(repo, bare, tmp_path):
    # second clone commits + pushes first; our push -> state "failed",
    # "rejected" in detail; ALSO assert via a wrapper that captured argv
    # never contains any string starting with "--force"
def test_credential_hint_mapping(): ...
    # pure: _push_failure_detail("fatal: could not read Username ...")
    # ends with the credential-helper hint copy
def test_detached_refused(repo): ...   # push_current raises GitWorkspaceError
```

URL builder (pure unit, `test_pr_urls.py`): all three remote shapes (`https://github.com/o/r.git`, `ssh://git@github.com/o/r`, `git@github.com:o/r.git`) → `("github.com", "o", "r")`; gitlab subgroup `https://gitlab.com/g/sub/r.git` → owner_path `g/sub`; `.git` stripped; branch `feat/x` percent-encoding (kept `/` for github path template, encoded for gitlab query param); unicode branch; unsupported host → `GitWorkspaceRefusal` naming the four hosts; codeberg WITH `refs/remotes/origin/HEAD` resolvable → `/compare/main...feat%2Fx`… actually Gitea keeps `/` raw in path — use `quote(branch, safe="/")` there too; codeberg WITHOUT it → refusal "can't determine the default branch". PR precondition: `info.upstream is None` → refusal "push the branch first".

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

`push_current`: refuse detached (`no branch checked out`) and no-remote (`no git remote configured`). Argv: upstream set ⇒ `push <upstream_remote>`; unset ⇒ `push -u <remote> <branch>`; `timeout=PUSH_TIMEOUT_SECONDS`. Classify: rc 0 + `"Everything up-to-date"` in stdout+stderr ⇒ `up_to_date`; rc 0 ⇒ `pushed`; nonzero ⇒ `failed` with excerpt run through `_push_failure_detail` (appends the hint when the excerpt matches any of `could not read Username`, `terminal prompts disabled`, `Permission denied`, `Authentication failed`: `" — credentials were not available non-interactively; push once from a terminal or configure a credential helper/ssh agent"`).

`pr_compare_url`: refuse when upstream unset; find the upstream remote's PUSH url in `info.remotes`; parse; templates exactly as spec §6 (github `?expand=1`; gitlab `merge_request%5Bsource_branch%5D=`; bitbucket `?source=`; codeberg needs base via `symbolic-ref --short -q refs/remotes/<remote>/HEAD` check=False → `<remote>/<base>` → strip the `<remote>/` PREFIX ONLY by length of the known remote name + 1, never by splitting on "/").

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** (`feat(git-modes): push engine + PR compare URLs (T4)`).

---

### Task 5: Provider wrappers + kill switch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (the `AgentRunsChangeReviewProvider` class, near `_configured_cap` at :134)
- Test: `Tests/UI/test_change_review_git_provider.py`

**Interfaces:**
- Consumes: every Task 1-4 name.
- Produces (screen Tasks 6-8 call ONLY these):
  `provider.git_actions_enabled() -> bool` (reads `get_cli_setting("change_review", "git_actions", True)`, bad config ⇒ True, same guard shape as `_configured_cap`);
  `provider.detect_git(roots: Sequence[str]) -> dict[str, GitWorkspaceInfo | GitWorkspaceRefusal | None]` (keyed by str(root); dedupes; empty when the kill switch is off);
  `provider.current_status(root: str) -> CurrentRootStatus`;
  `provider.current_diff_text(root: str, change: ChangedFile) -> str` (branches: `change.path in status.untracked` handled by the SCREEN via preview — this method is tracked-only `working_tree_diff`);
  `provider.untracked_preview(root: str, path: str) -> str` (max_lines = `self.diff_display_max_lines`);
  `provider.commit_selected(root, files, message, new_branch)` (passes `run_active=self.run_active` — exactly how revert threads it);
  `provider.push_current(root, info, remote)`; `provider.pr_url(root, info)`.

- [ ] **Step 1: Write failing tests** — REAL provider over a file-backed `AgentRunsDB(tmp_path / "runs.db")` + real `ShadowRepoService` + real temp repo (the fixture-invented-shapes rule): kill switch off (`monkeypatch` `get_cli_setting`) ⇒ `detect_git` returns `{}`; on ⇒ info for a real repo root; `commit_selected` refuses when `provider.run_active` returns True (wire `provider.run_active = lambda: True` — proves the thread-through, not a mock of git).
- [ ] **Step 2: FAIL** (AttributeError). **Step 3: Implement** (thin delegation only — no logic in the provider beyond the kill-switch read and dedupe). **Step 4: PASS.** **Step 5: Commit** (`feat(git-modes): provider seam + [change_review] git_actions kill switch (T5)`).

---

### Task 6: Screen `current` mode (view only)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py`
- Test: `Tests/UI/test_change_review_current_mode.py`

**Interfaces:**
- Consumes: Task 5 provider methods; existing `_load_turn` / `_leaves` / `_focus_leaf` / `_render_diff` / `_diff_text_for` machinery.
- Produces: module constant `CURRENT_MODE_SENTINEL = "__git_current__"`; `ChangeReviewScreen(..., workspace_roots: Sequence[str] | None = None)` ctor kwarg; `self._current_mode_active() -> bool`; pseudo-row shape `{"root": str(root), "kind": "git_current", "id": -1}`; `self._current_untracked: dict[str, frozenset[str]]` (root → untracked paths); `self._current_infos: dict[str, GitWorkspaceInfo]`.

- [ ] **Step 1: Write failing tests** (real CSS stack, real provider, file-backed DB, real repos; `run_test()` harness as the existing screen tests):
  - pseudo-entry PRESENT (first Select option) when a candidate root is a repo and the switch is on; screen still OPENS on the latest turn (assert the Select's value is the newest run_id — the byte-compatible default);
  - pseudo-entry ABSENT: non-repo root (AC #4), kill switch off (guard proven RED by flipping only the switch), no candidate roots;
  - root-inside-repo Refusal → entry absent + banner carries the refusal copy;
  - selecting the entry → loading state, then tree lists the real repo's changed files; focusing a tracked file renders the real `git diff HEAD` text through `diff_pane_text()`; an untracked file renders the synthesized preview ("new file:" header);
  - unborn-HEAD repo: all files render via the preview path — assert no `diff HEAD` ran (wrap `working_tree_diff` with a counting shim at the PROVIDER boundary, not a git mock);
  - mode gating (row-consumers table, spec §4.1): `action_revert_file`, `action_undo_all`, `action_comment_file`, and the `c` key each no-op with their notify copy in current mode (assert via `notify` capture AND that the notes DB row count is unchanged);
  - stale-land guard: switch back to a turn while the current-mode worker is in flight → landing is discarded (drive by delaying the worker body with an event).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement**
  - `_populate_turn_select`: prepend `(label, CURRENT_MODE_SENTINEL)` when `provider.git_actions_enabled()` and a fresh `provider.detect_git(candidates)` (run inside `on_mount`'s existing load path — worker-side) yields ≥1 `GitWorkspaceInfo`; candidates = distinct row roots ∪ `workspace_roots` kwarg. Label: `Working tree (current) — {branch}` / `detached HEAD` / `{branch} (no commits yet)`.
  - Select-changed handler: sentinel ⇒ `_load_current_mode()`; else existing `_load_turn` path untouched.
  - `_load_current_mode()`: bump `_diff_cache_generation`, capture `token = self._current_load_token = object()`; `run_worker(exclusive=True, thread=True)` → per detected root: fresh `detect_git` + `current_status`; `call_from_thread(self._land_current_mode, token, results)`; land no-ops when `token is not self._current_load_token` or the Select moved off the sentinel (arc A's dispatch-scope guard shape, `chat_screen.py` `_land_console_changed_files` precedent).
  - Landing: synthesize pseudo-rows, populate `_leaves` with `(pseudo_row, ChangedFile)`, store `_current_untracked`/`_current_infos`, build the tree per root (reuse the existing per-root grouping), header/banner: `branch ↑ahead ↓behind → upstream` per root + totals; empty ⇒ "working tree clean".
  - `_diff_text_for`: branch on `row.get("kind") == "git_current"` → untracked ⇒ `provider.untracked_preview`, else `provider.current_diff_text` (memo key unchanged — pseudo-row identity + generation already isolate it).
  - `_current_mode_active()`: the Select's value is the sentinel. Gate the four snapshot-only actions at the TOP with the notify copy from spec §4.1; `_marked_diff_lines`/notes-strip computation short-circuits to empty in current mode (pseudo `id=-1` must never reach the DB — assert in the gating test).
- [ ] **Step 4: PASS.** **Step 5: Commit** (`feat(git-modes): current working-tree mode on change review (T6)`).

---

### Task 7: Commit UI

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py`
- Test: `Tests/UI/test_change_review_commit_ui.py`

**Interfaces:**
- Consumes: Task 5/6.
- Produces: `ChangeGitCommitModal(SafeModalDismissMixin, ModalScreen[dict | None])` (result `{"message": str, "new_branch": str | None, "files": list[str], "root": str}` or None); `action_git_commit` + `Commit…` Button (id `change-review-git-commit-btn`); busy flag `self._git_busy`.

- [ ] **Step 1: Write failing tests**
  - Commit button visible/enabled only in current mode with ≥1 file; disabled (with reason tooltip) on clean tree;
  - `run_active` True ⇒ notify refusal, no modal;
  - FRESH preflight: create a file on disk AFTER the view loaded, open the modal ⇒ the new file IS in the checklist (spec §5 step 2 — this is the test that proves the modal never trusts the stale view);
  - modal: all files pre-checked; unchecking one excludes it from the commit (assert via `git show --name-only`); blank message blocks submit; detached/main warnings rendered (query the warning Static's text against a repo on `main`);
  - merge-in-progress ⇒ refusal copy shown, no commit;
  - e2e: message + one unchecked file ⇒ real commit lands, notify carries short sha, view reloads (tree drops the committed file);
  - busy: buttons disabled while the commit worker runs.
  - CSS: any new styles go in the `_change_review` bundle module + `python -m tldw_chatbook.css.build_css` — NEVER hand-edit the bundle; modal follows `ChangeRevertConfirmModal`'s classes.
- [ ] **Step 2: FAIL.** **Step 3: Implement** (modal compose mirrors `ChangeRevertConfirmModal` :1977; preflight worker → modal push with fresh `CurrentRootStatus`; on result: exclusive worker → `provider.commit_selected` → notify outcome (failure names the STEP + excerpt) → `_load_current_mode()`; multi-root: modal carries the focused leaf's root, root `Select` only when >1 detected root and no focused leaf). Labels through `rich.text.Text` (Button.label markup-parse trap). **Step 4: PASS.** **Step 5: Commit** (`feat(git-modes): confirmed file-picked commit UI (T7)`).

---

### Task 8: Push + PR UI

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py`
- Test: `Tests/UI/test_change_review_push_ui.py`

**Interfaces:**
- Consumes: Tasks 4-6; Task 7's `_git_busy`.
- Produces: `ChangeGitPushModal` (names branch, remote — `Select` when >1 remote and no upstream — and upstream state incl. ahead count); `action_git_push`; `Push` / `Open PR` buttons (ids `change-review-git-push-btn`, `change-review-git-pr-btn`).

- [ ] **Step 1: Write failing tests**
  - Push enabled only when detection found ≥1 remote; disabled reason "no git remote configured" (AC #2); detached ⇒ disabled "no branch checked out";
  - push NOT refused while `run_active` is True, commit IS — both asserted side by side (spec §6's explicit contract);
  - e2e to a real bare remote: confirm ⇒ pushed, upstream set, header ahead-count refreshes to ↑0; second push ⇒ "up to date" notify;
  - non-FF ⇒ failure notify carries "rejected" + no `--force` in any captured argv;
  - PR disabled with "push the branch first" before upstream exists; after push, PR press calls `app.open_url` with the exact expected compare URL (stub `open_url` on the test app — a URL-open stub, not a git mock); unsupported host ⇒ disabled reason naming the four hosts.
- [ ] **Step 2: FAIL.** **Step 3: Implement** (push worker with `PUSH_TIMEOUT_SECONDS`; `self.app.open_url(url)` — NEVER `webbrowser`). **Step 4: PASS.** **Step 5: Commit** (`feat(git-modes): confirmed push + PR compare-URL UI (T8)`).

---

### Task 9: Opener wiring + User Guide

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_open_change_review` :18673, `_console_change_review_provider` :18630)
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Test: `Tests/UI/test_change_review_opener_roots.py` (or extend the existing opener wiring test file)

**Interfaces:**
- Consumes: Task 6's `workspace_roots` ctor kwarg.
- Produces: `_open_change_review` passes `workspace_roots=` resolved via `controller.resolve_turn_execution_context(active_session_id).workspace_roots` wrapped in try/except (`None` on any failure — the opener must degrade, matching its existing posture); every other `ChangeReviewScreen(...)` caller unchanged (`None` default is byte-compatible).

- [ ] **Step 1: Write failing test** — opener passes the controller's roots through to the screen ctor (drive the real opener seam the way the existing rail click-through test does); a controller raise ⇒ screen still opens with `workspace_roots=None`.
- [ ] **Step 2: FAIL.** **Step 3: Implement wiring.**
- [ ] **Step 4: Guide** — new "Git actions in change review" subsection under the TASK-1972/18060 material: what the `current` entry is, when it appears (real repo at the workspace root + `[change_review] git_actions`), the commit/push/PR flows, each "why unavailable" reason, the no-force-push guarantee; update the page's "Verified against" stamp.
- [ ] **Step 5: Run the branch sweep** — every test file this plan created plus `Tests/UI/test_change_review_screen.py`, `Tests/Chat/test_change_notes_db.py`, `Tests/UI/test_console_changed_files_wiring.py` (rail regression), and a `--collect-only` sweep of `Tests/`. Expected: all green with READ counts.
- [ ] **Step 6: Commit** (`feat(git-modes): opener workspace roots + user guide (T9)`).

---

## Self-review notes (writing-plans checklist applied)

- Spec coverage: §3→T1, §4→T2/T6, §4.1→T6, §5→T3/T7, §6→T4/T8, §7→T5, §8→T5/T6, §9 distributed into each task's tests, §10→T9 (the API-PR follow-up task is FILED AT CLOSE-OUT, deliberately not a plan task — backlog rule against referencing future tasks).
- Type consistency: `CurrentRootStatus`/`GitWorkspaceInfo`/`GitStepOutcome` names identical across T1-T8; provider method names in T5 match every T6-T8 call site; `CURRENT_MODE_SENTINEL` defined once (T6).
- Deliberate scope cuts (do NOT "fix" these): no force-push, no API PR creation, no `RuntimeBindingKind` registry writes, no per-hunk staging, no commit from snapshot modes.
