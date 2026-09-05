# Console Inspect Rail — Environment Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Console Inspect rail around a Codex-style Environment panel — git changes/branch/worktree, PR + CI via `gh`, a backlog Tasks card, and the agent fleet moved over from the left rail — keeping all current rail content.

**Architecture:** One pure state module (`Chat/console_environment_state.py`) projects an `EnvironmentSnapshot` into `ConsoleInspectorSectionState`s; one impure gatherer module (`Workspaces/environment_status.py`) assembles the snapshot from git/gh/backlog; a small controller (`UI/Console_Modules/environment.py`) owns dispatch cadence, TTL, and backoff with injected seams; ChatScreen wires workers, landing, and row-activation handlers. The rail mounts three `ConsoleInspectorSection` widgets (the existing primitive — clickable rows, `sync_state`, collapse) and expansions are just extra rows emitted by the projection when a row id is in the expanded set.

**Tech Stack:** Python ≥3.11, Textual 8.x, PyYAML (core dep), `gh` CLI (optional at runtime), real-SQLite/real-git test fixtures, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md` (read it first; it carries the owner decisions and degradation matrix).

## Global Constraints

- **Base branch:** `origin/dev` (spec verified against `f6896176c8`/`3668571d7b`; re-fetch and re-verify cited line numbers at execution time). Branch name: `feat/console-inspector-environment`. PR targets `dev`. Never commit to `main`.
- **Worktree:** create at `<repo>/.worktrees/console-inspector-environment` (NEVER under `/tmp` — macOS cleaner has destroyed work there; standing owner rule). Inside it: `uv venv && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` (the venv is uv-managed; plain `pip` is absent).
- **First commit** on the branch adds the spec + this plan (they were drafted in a different checkout and are not yet committed anywhere).
- **Backlog hygiene:** create one umbrella backlog task via `backlog task create` before coding; assign the ID only after sweeping ALL remotes + worktrees for collisions (standing rule: ten-plus collisions have happened). Set it In Progress with this plan as the Implementation Plan; mark Done only per the repo's Definition of Done.
- **pytest:** run targeted files, never the full suite mid-task. Redirect stdout to a file and grep the summary (`pytest ... > .pytest-out.txt 2>.pytest-warn.txt; grep -E "passed|failed" .pytest-out.txt`) — tmpdir-cleanup warnings bury the summary when merged into one stream. `.pytest-out.txt` stays untracked; never commit it. "No tests ran" is a FAILED gate; a gate passes only on a read nonzero passed-count.
- **CSS:** edit `tldw_chatbook/css/components/_agentic_terminal.tcss` only; regenerate bundles with `python -m tldw_chatbook.css.build_css`; never hand-edit `tldw_cli_modular.tcss`. Do not raise any ADR-097 ratchet constant.
- **New git subprocess calls** go through `git_workspace._run_user_git` (inherits `GIT_OPTIONAL_LOCKS=0`, scrubbed env, timeouts). Branch names and paths are argv data, never shell.
- **Strict ownership:** do NOT add rows to `ConsoleInspectorState` — the new sections are `ConsoleInspectorSection` widgets (like the fleet section), which the ownership classifier does not govern. Adding an inspector *row label* without a `ROW_IDS`/`ROW_GROUPS` entry crashes the rail under STRICT policy.
- **Run `./scripts/preflight.sh`** before opening the PR.
- Decisions resolved during planning (spec ambiguities picked): row-expansion state is **in-memory per screen** (a `set[str]`); only section collapse (`environment_open`, `tasks_open`) persists. The git tier keeps `git_workspace`'s internal 30s `READ_TIMEOUT_SECONDS` (not the spec's 5s — the function doesn't expose a timeout and the worker thread never blocks the UI); the 5s timeout applies to `gh`. The "refresh" affordance is the section's built-in `view_all_label="Refresh"` slot (posts `ViewAllRequested`), not a header glyph.

---

### Task 1: Pure foundations — enums, dataclasses, formatting helpers

**Files:**
- Create: `tldw_chatbook/Chat/console_environment_state.py`
- Test: `Tests/Chat/test_console_environment_state.py`

**Interfaces:**
- Consumes: `ChangedFile` from `tldw_chatbook.Workspaces.change_tracking` (frozen dataclass: `path: str`, `status: str`, `adds: int = 0`, `dels: int = 0`, `old_path: str | None = None`, `binary: bool = False`).
- Produces (later tasks rely on these exact names):
  - `EnvSourceAvailability` (str-Enum: `OK`, `NOT_APPLICABLE`, `MISSING_TOOL`, `ERROR`)
  - `ExecTargetKind` (str-Enum: `LOCAL`, `REMOTE_TLDW_SERVER`)
  - `GitEnvState`, `PrEnvState`, `PrCheck`, `TasksEnvState`, `BacklogTaskEntry`, `BranchTaskState`, `ExecTargetState`, `EnvironmentSnapshot` (all frozen dataclasses, fields below)
  - `compact_count(n: int) -> str`, `signed_change_counts(adds: int, dels: int) -> str`
  - `branch_task_id(branch: str | None) -> str | None`
  - `relative_age(then: datetime | None, now: datetime) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_environment_state.py
"""Pure-state tests for the Console Environment panel (no I/O, no Textual app)."""
from datetime import datetime, timedelta, timezone

from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    ExecTargetKind,
    GitEnvState,
    PrEnvState,
    TasksEnvState,
    ExecTargetState,
    EnvironmentSnapshot,
    branch_task_id,
    compact_count,
    relative_age,
    signed_change_counts,
)


def test_compact_count_small_numbers_keep_thousands_separators():
    assert compact_count(0) == "0"
    assert compact_count(1204) == "1,204"
    assert compact_count(99_999) == "99,999"


def test_compact_count_large_numbers_compress():
    assert compact_count(277_870) == "278k"
    assert compact_count(1_679_102) == "1.7M"


def test_signed_change_counts_pairs_plus_and_minus():
    assert signed_change_counts(1204, 86) == "+1,204 −86"
    assert signed_change_counts(1_679_102, 277_870) == "+1.7M −278k"


def test_branch_task_id_matches_plain_and_subtask_ids():
    assert branch_task_id("feat/task-3401-video-generation-foundation") == "3401"
    assert branch_task_id("fix/task-3401.6-comfyui-adapter") == "3401.6"
    assert branch_task_id("chore/no-task-reference-here") is None
    assert branch_task_id(None) is None


def test_relative_age_buckets():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    assert relative_age(None, now) == ""
    assert relative_age(now - timedelta(minutes=5), now) == "5m ago"
    assert relative_age(now - timedelta(hours=3), now) == "3h ago"
    assert relative_age(now - timedelta(days=6), now) == "6d ago"


def test_environment_snapshot_defaults_are_not_applicable():
    snapshot = EnvironmentSnapshot()
    assert snapshot.git.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.pr.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.tasks.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.target.kind is ExecTargetKind.LOCAL
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_environment_state.py -v > .pytest-out.txt 2>&1; grep -E "passed|failed|error" .pytest-out.txt`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_environment_state'`

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Chat/console_environment_state.py
"""Pure display state for the Console Environment panel (Inspect rail).

No I/O here: gatherers live in ``Workspaces/environment_status.py`` and
projections consume only these frozen dataclasses. Follows the
``console_display_state.py`` convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from tldw_chatbook.Workspaces.change_tracking import ChangedFile


class EnvSourceAvailability(str, Enum):
    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    MISSING_TOOL = "missing_tool"
    ERROR = "error"


class ExecTargetKind(str, Enum):
    LOCAL = "local"
    REMOTE_TLDW_SERVER = "remote_tldw_server"


@dataclass(frozen=True)
class GitEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.NOT_APPLICABLE
    root: str = ""
    branch: str | None = None
    detached: bool = False
    unborn: bool = False
    head_short: str = ""
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    adds: int = 0
    dels: int = 0
    files: tuple[ChangedFile, ...] = ()
    worktree_name: str | None = None
    stale: bool = False

    @property
    def dirty(self) -> bool:
        return bool(self.files)


@dataclass(frozen=True)
class PrCheck:
    name: str
    conclusion: str  # "success" | "failure" | "pending" (normalized)
    details_url: str = ""


@dataclass(frozen=True)
class PrEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.NOT_APPLICABLE
    number: int = 0
    title: str = ""
    state: str = ""  # "OPEN" | "MERGED" | "CLOSED"
    is_draft: bool = False
    url: str = ""
    adds: int = 0
    dels: int = 0
    merged_at: datetime | None = None
    checks: tuple[PrCheck, ...] = ()
    stale: bool = False

    @property
    def failing_checks(self) -> tuple[PrCheck, ...]:
        return tuple(c for c in self.checks if c.conclusion == "failure")

    @property
    def pending_checks(self) -> tuple[PrCheck, ...]:
        return tuple(c for c in self.checks if c.conclusion == "pending")

    @property
    def passing_count(self) -> int:
        return sum(1 for c in self.checks if c.conclusion == "success")


@dataclass(frozen=True)
class BacklogTaskEntry:
    task_id: str
    title: str
    status: str  # "To Do" | "In Progress" | "Done" | other verbatim


@dataclass(frozen=True)
class BranchTaskState:
    task_id: str
    title: str
    status: str
    ac_done: int = 0
    ac_total: int = 0
    path: str = ""


@dataclass(frozen=True)
class TasksEnvState:
    availability: EnvSourceAvailability = EnvSourceAvailability.NOT_APPLICABLE
    branch_task: BranchTaskState | None = None
    in_progress: int = 0
    todo: int = 0
    entries: tuple[BacklogTaskEntry, ...] = ()
    scanning: bool = False


@dataclass(frozen=True)
class ExecTargetState:
    kind: ExecTargetKind = ExecTargetKind.LOCAL


@dataclass(frozen=True)
class EnvironmentSnapshot:
    git: GitEnvState = field(default_factory=GitEnvState)
    target: ExecTargetState = field(default_factory=ExecTargetState)
    pr: PrEnvState = field(default_factory=PrEnvState)
    tasks: TasksEnvState = field(default_factory=TasksEnvState)


_BRANCH_TASK_RE = re.compile(r"task-(\d+(?:\.\d+)*)")


def branch_task_id(branch: str | None) -> str | None:
    """Extract a backlog task id (subtasks included) from a branch name."""
    if not branch:
        return None
    match = _BRANCH_TASK_RE.search(branch)
    return match.group(1) if match else None


def compact_count(n: int) -> str:
    """Humanize a line count: exact with separators below 100k, compact above."""
    if n < 100_000:
        return f"{n:,}"
    if n < 1_000_000:
        return f"{round(n / 1_000)}k"
    return f"{n / 1_000_000:.1f}M"


def signed_change_counts(adds: int, dels: int) -> str:
    return f"+{compact_count(adds)} −{compact_count(dels)}"


def relative_age(then: datetime | None, now: datetime) -> str:
    """Coarse '5m ago' / '3h ago' / '6d ago' bucket; '' for None."""
    if then is None:
        return ""
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
```

Note the minus sign in `signed_change_counts` is U+2212, matching the test literal `−86`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_environment_state.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ passed" .pytest-out.txt`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_environment_state.py Tests/Chat/test_console_environment_state.py
git commit -m "feat(console): pure state foundations for Environment panel"
```

---

### Task 2: `linked_worktree_name` helper in git_workspace

**Files:**
- Modify: `tldw_chatbook/Workspaces/git_workspace.py` (add one function near `detect_git_workspace`, ~L340)
- Test: `Tests/Workspaces/test_environment_status.py` (new file; later tasks append to it)

**Interfaces:**
- Consumes: `_run_user_git(root, *args, timeout=..., check=True) -> GitCmdResult` (module-private, same module — direct call is fine from inside the module), `GitWorkspaceError`.
- Produces: `linked_worktree_name(root: Path) -> str | None` — the directory basename when `root` is a **linked** git worktree, else `None`. Never raises.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Workspaces/test_environment_status.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Workspaces/test_environment_status.py -v > .pytest-out.txt 2>&1; grep -E "passed|failed|error" .pytest-out.txt`
Expected: FAIL — `ImportError: cannot import name 'linked_worktree_name'`

- [ ] **Step 3: Implement in `git_workspace.py`**

Place directly after `detect_git_workspace` (docstring style matches neighbors):

```python
def linked_worktree_name(root: Path) -> str | None:
    """Return the directory basename when ``root`` is a linked git worktree.

    A linked worktree has ``--git-dir`` != ``--git-common-dir``. Returns
    ``None`` for a primary checkout, a non-repo, or any git failure —
    this is a display probe and must never raise.
    """
    try:
        result = _run_user_git(root, "rev-parse", "--git-dir", "--git-common-dir")
    except GitWorkspaceError:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        return None
    git_dir = (root / lines[0]).resolve() if not Path(lines[0]).is_absolute() else Path(lines[0]).resolve()
    common_dir = (root / lines[1]).resolve() if not Path(lines[1]).is_absolute() else Path(lines[1]).resolve()
    if git_dir == common_dir:
        return None
    return root.resolve().name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Workspaces/test_environment_status.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ passed" .pytest-out.txt`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/git_workspace.py Tests/Workspaces/test_environment_status.py
git commit -m "feat(workspaces): linked_worktree_name probe for Environment panel"
```

---

### Task 3: Environment section projection

**Files:**
- Modify: `tldw_chatbook/Chat/console_environment_state.py` (append)
- Test: `Tests/Chat/test_console_environment_state.py` (append)

**Interfaces:**
- Consumes: Task 1 dataclasses; `InspectorSectionRow` and `ConsoleInspectorSectionState` from `tldw_chatbook.Widgets.Console.console_inspector_section` (frozen dataclasses — `InspectorSectionRow(row_id, primary_text, secondary_text="", status="", clickable=False, cancellable=False)`; `ConsoleInspectorSectionState(rows, summary)` with BOTH fields required).
- Produces (later tasks rely on these):
  - Constants: `ENVIRONMENT_SECTION_ID = "environment"`, `TASKS_SECTION_ID = "tasks"`, and row ids `ENV_ROW_CHANGES = "env-changes"`, `ENV_ROW_LOCAL = "env-local"`, `ENV_ROW_BRANCH = "env-branch"`, `ENV_ROW_COMMIT_PUSH = "env-commit-push"`, `ENV_ROW_PR = "env-pr"`, `ENV_ROW_CHECKS = "env-checks"`, `ENV_ROW_PR_OPEN = "env-pr-open"`, `ENV_ROW_PR_ADD = "env-pr-add"`, `ENV_ROW_CHECKS_FIX = "env-checks-fix"`, `ENV_FILE_ROW_PREFIX = "env-file-"`
  - `EXPANDABLE_ENV_ROWS = frozenset({ENV_ROW_CHANGES, ENV_ROW_LOCAL, ENV_ROW_BRANCH, ENV_ROW_PR, ENV_ROW_CHECKS})`
  - `project_environment_section(snapshot: EnvironmentSnapshot, expanded: frozenset[str], *, now: datetime) -> ConsoleInspectorSectionState`
  - `pr_summary_text(pr: PrEnvState) -> str` (the composer-insert payload for "Add to chat")
  - `failing_checks_text(pr: PrEnvState) -> str` (the composer-insert payload for "Fix")

- [ ] **Step 1: Write the failing tests** (append to the Task 1 test file)

```python
from tldw_chatbook.Workspaces.change_tracking import ChangedFile
from tldw_chatbook.Chat.console_environment_state import (
    ENV_ROW_BRANCH,
    ENV_ROW_CHANGES,
    ENV_ROW_CHECKS,
    ENV_ROW_CHECKS_FIX,
    ENV_ROW_COMMIT_PUSH,
    ENV_ROW_LOCAL,
    ENV_ROW_PR,
    ENV_ROW_PR_ADD,
    ENV_ROW_PR_OPEN,
    PrCheck,
    failing_checks_text,
    pr_summary_text,
    project_environment_section,
)

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _git_state(**kw) -> GitEnvState:
    base = dict(
        availability=EnvSourceAvailability.OK,
        root="/w/repo",
        branch="feat/task-3401-video-generation",
        adds=1204,
        dels=86,
        files=(ChangedFile(path="a.py", status="M", adds=1200, dels=80),
               ChangedFile(path="b.py", status="A", adds=4, dels=6)),
    )
    base.update(kw)
    return GitEnvState(**base)


def test_no_git_workspace_projects_a_single_quiet_row():
    state = project_environment_section(EnvironmentSnapshot(), frozenset(), now=_NOW)
    assert [r.row_id for r in state.rows] == ["env-empty"]
    assert state.rows[0].primary_text == "No git workspace"
    assert not state.rows[0].clickable


def test_changes_row_shows_signed_totals_and_branch_row_shows_divergence():
    snapshot = EnvironmentSnapshot(git=_git_state(ahead=2, behind=1, upstream="origin/feat/x"))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    by_id = {r.row_id: r for r in state.rows}
    assert by_id[ENV_ROW_CHANGES].secondary_text == "+1,204 −86"
    assert "↑2" in by_id[ENV_ROW_BRANCH].secondary_text
    assert "↓1" in by_id[ENV_ROW_BRANCH].secondary_text
    assert by_id[ENV_ROW_CHANGES].clickable and by_id[ENV_ROW_BRANCH].clickable


def test_commit_or_push_row_hidden_when_clean_and_synced_shown_when_dirty():
    clean = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=()))
    dirty = EnvironmentSnapshot(git=_git_state())
    clean_ids = [r.row_id for r in project_environment_section(clean, frozenset(), now=_NOW).rows]
    dirty_ids = [r.row_id for r in project_environment_section(dirty, frozenset(), now=_NOW).rows]
    assert ENV_ROW_COMMIT_PUSH not in clean_ids
    assert ENV_ROW_COMMIT_PUSH in dirty_ids


def test_push_only_variant_when_tree_clean_but_ahead():
    snapshot = EnvironmentSnapshot(git=_git_state(adds=0, dels=0, files=(), ahead=2))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_COMMIT_PUSH].primary_text == "Push ↑2"


def test_changes_expansion_lists_files_with_per_file_counts():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_CHANGES}), now=_NOW)
    ids = [r.row_id for r in state.rows]
    assert "env-file-0" in ids and "env-file-1" in ids
    file_row = next(r for r in state.rows if r.row_id == "env-file-0")
    assert file_row.primary_text == "M a.py"
    assert file_row.secondary_text == "+1,200 −80"


def test_pr_rows_absent_without_pr_and_present_with_actions_when_expanded():
    no_pr = EnvironmentSnapshot(git=_git_state())
    assert ENV_ROW_PR not in [r.row_id for r in project_environment_section(no_pr, frozenset(), now=_NOW).rows]
    pr = PrEnvState(
        availability=EnvSourceAvailability.OK, number=2281, title="Split boot CSS",
        state="OPEN", url="https://github.com/o/r/pull/2281", adds=36643, dels=2871,
        checks=(PrCheck("lint", "success"), PrCheck("ci", "failure", "https://ci/1"),
                PrCheck("docs", "pending")),
    )
    snapshot = EnvironmentSnapshot(git=_git_state(), pr=pr)
    collapsed = project_environment_section(snapshot, frozenset(), now=_NOW)
    by_id = {r.row_id: r for r in collapsed.rows}
    assert by_id[ENV_ROW_PR].primary_text == "PR #2281 · Open"
    assert by_id[ENV_ROW_CHECKS].primary_text == "1 failing check"
    expanded = project_environment_section(
        snapshot, frozenset({ENV_ROW_PR, ENV_ROW_CHECKS}), now=_NOW)
    expanded_ids = [r.row_id for r in expanded.rows]
    assert ENV_ROW_PR_OPEN in expanded_ids and ENV_ROW_PR_ADD in expanded_ids
    assert ENV_ROW_CHECKS_FIX in expanded_ids


def test_detached_head_labels_and_skipped_pr():
    snapshot = EnvironmentSnapshot(git=_git_state(branch=None, detached=True, head_short="abc1234"))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_BRANCH].primary_text == "detached @ abc1234"


def test_stale_marker_survives_on_error_with_prior_data():
    snapshot = EnvironmentSnapshot(git=_git_state(stale=True))
    by_id = {r.row_id: r for r in project_environment_section(snapshot, frozenset(), now=_NOW).rows}
    assert by_id[ENV_ROW_CHANGES].status == "blocked"


def test_local_row_expansion_shows_remote_placeholder():
    snapshot = EnvironmentSnapshot(git=_git_state())
    state = project_environment_section(snapshot, frozenset({ENV_ROW_LOCAL}), now=_NOW)
    texts = [r.primary_text for r in state.rows]
    assert any("Remote tldw_server" in t for t in texts)


def test_composer_payload_builders():
    pr = PrEnvState(availability=EnvSourceAvailability.OK, number=7, title="T",
                    state="OPEN", url="https://x/pull/7",
                    checks=(PrCheck("ci", "failure", "https://ci/1"),))
    assert "PR #7" in pr_summary_text(pr) and "https://x/pull/7" in pr_summary_text(pr)
    fix = failing_checks_text(pr)
    assert "ci" in fix and "https://ci/1" in fix
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_environment_state.py -v > .pytest-out.txt 2>&1; grep -E "passed|failed|error" .pytest-out.txt`
Expected: FAIL — `ImportError` on the new names.

- [ ] **Step 3: Implement the projection** (append to `console_environment_state.py`)

```python
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)

ENVIRONMENT_SECTION_ID = "environment"
TASKS_SECTION_ID = "tasks"

ENV_ROW_CHANGES = "env-changes"
ENV_ROW_LOCAL = "env-local"
ENV_ROW_BRANCH = "env-branch"
ENV_ROW_COMMIT_PUSH = "env-commit-push"
ENV_ROW_PR = "env-pr"
ENV_ROW_CHECKS = "env-checks"
ENV_ROW_PR_OPEN = "env-pr-open"
ENV_ROW_PR_ADD = "env-pr-add"
ENV_ROW_CHECKS_FIX = "env-checks-fix"
ENV_FILE_ROW_PREFIX = "env-file-"

EXPANDABLE_ENV_ROWS = frozenset(
    {ENV_ROW_CHANGES, ENV_ROW_LOCAL, ENV_ROW_BRANCH, ENV_ROW_PR, ENV_ROW_CHECKS}
)

_MAX_FILE_ROWS = 12


def _git_status_class(stale: bool) -> str:
    return "blocked" if stale else ""


def _branch_primary(git: GitEnvState) -> str:
    if git.detached:
        return f"detached @ {git.head_short or 'HEAD'}"
    if git.unborn:
        return f"{git.branch or '?'} (no commits yet)"
    return git.branch or "?"


def _branch_secondary(git: GitEnvState) -> str:
    parts: list[str] = []
    if git.ahead:
        parts.append(f"↑{git.ahead}")
    if git.behind:
        parts.append(f"↓{git.behind}")
    if git.worktree_name:
        parts.append(f"wt:{git.worktree_name}")
    return " ".join(parts)


def project_environment_section(
    snapshot: EnvironmentSnapshot,
    expanded: frozenset[str],
    *,
    now: datetime,
) -> ConsoleInspectorSectionState:
    git = snapshot.git
    if git.availability is not EnvSourceAvailability.OK:
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(row_id="env-empty", primary_text="No git workspace"),),
            summary="",
        )
    status = _git_status_class(git.stale)
    rows: list[InspectorSectionRow] = []

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_CHANGES, primary_text="Changes",
        secondary_text=signed_change_counts(git.adds, git.dels),
        status=status, clickable=True,
    ))
    if ENV_ROW_CHANGES in expanded:
        for index, change in enumerate(git.files[:_MAX_FILE_ROWS]):
            rows.append(InspectorSectionRow(
                row_id=f"{ENV_FILE_ROW_PREFIX}{index}",
                primary_text=f"{change.status} {change.path}",
                secondary_text=signed_change_counts(change.adds, change.dels),
            ))
        if len(git.files) > _MAX_FILE_ROWS:
            rows.append(InspectorSectionRow(
                row_id="env-file-more",
                primary_text=f"… {len(git.files) - _MAX_FILE_ROWS} more — Review opens all",
            ))
        rows.append(InspectorSectionRow(
            row_id="env-changes-review", primary_text="Review in Change Review",
            clickable=True,
        ))

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_LOCAL, primary_text="Local", clickable=True,
    ))
    if ENV_ROW_LOCAL in expanded:
        rows.append(InspectorSectionRow(
            row_id="env-local-current", primary_text="Local instance ✓",
        ))
        rows.append(InspectorSectionRow(
            row_id="env-local-remote",
            primary_text="Remote tldw_server — not configured",
        ))

    rows.append(InspectorSectionRow(
        row_id=ENV_ROW_BRANCH, primary_text=_branch_primary(git),
        secondary_text=_branch_secondary(git), status=status, clickable=True,
    ))
    if ENV_ROW_BRANCH in expanded:
        rows.append(InspectorSectionRow(
            row_id="env-branch-detail",
            primary_text=git.branch or _branch_primary(git),
            secondary_text=(
                f"upstream {git.upstream} (↑↓ vs last fetch)"
                if git.upstream else "no upstream"
            ),
        ))
        if git.worktree_name:
            rows.append(InspectorSectionRow(
                row_id="env-branch-worktree",
                primary_text=f"worktree {git.worktree_name}",
                secondary_text=git.root,
            ))

    if git.dirty or git.ahead:
        if git.dirty:
            label = f"Commit or push · {len(git.files)} files"
        else:
            label = f"Push ↑{git.ahead}"
        rows.append(InspectorSectionRow(
            row_id=ENV_ROW_COMMIT_PUSH, primary_text=label, clickable=True,
        ))

    pr = snapshot.pr
    if pr.availability is EnvSourceAvailability.OK and pr.number:
        state_label = "Draft" if (pr.is_draft and pr.state == "OPEN") else pr.state.capitalize()
        secondary = ""
        if pr.state == "MERGED" and pr.merged_at is not None:
            secondary = f"Merged {relative_age(pr.merged_at, now)}"
        rows.append(InspectorSectionRow(
            row_id=ENV_ROW_PR,
            primary_text=f"PR #{pr.number} · {state_label}",
            secondary_text=secondary,
            status="blocked" if pr.stale else "",
            clickable=True,
        ))
        if ENV_ROW_PR in expanded:
            rows.append(InspectorSectionRow(
                row_id="env-pr-title", primary_text=pr.title,
                secondary_text=signed_change_counts(pr.adds, pr.dels),
            ))
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_PR_OPEN, primary_text="Open in browser", clickable=True,
            ))
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_PR_ADD, primary_text="Add to chat", clickable=True,
            ))
        if pr.checks:
            failing = len(pr.failing_checks)
            pending = len(pr.pending_checks)
            if failing:
                checks_primary = f"{failing} failing check" + ("s" if failing != 1 else "")
                checks_status = "error"
            elif pending:
                checks_primary = f"{pending} pending check" + ("s" if pending != 1 else "")
                checks_status = "running"
            else:
                checks_primary = f"{pr.passing_count} checks passed"
                checks_status = "done"
            rows.append(InspectorSectionRow(
                row_id=ENV_ROW_CHECKS, primary_text=checks_primary,
                secondary_text=(
                    f"{pr.passing_count} passed · {pending} pending" if failing else ""
                ),
                status=checks_status, clickable=True,
            ))
            if ENV_ROW_CHECKS in expanded:
                for index, check in enumerate(pr.failing_checks):
                    rows.append(InspectorSectionRow(
                        row_id=f"env-check-{index}", primary_text=check.name,
                        status="error",
                    ))
                if failing:
                    rows.append(InspectorSectionRow(
                        row_id=ENV_ROW_CHECKS_FIX,
                        primary_text="Fix — add failure summary to chat",
                        clickable=True,
                    ))

    summary = f"{_branch_primary(git)} {signed_change_counts(git.adds, git.dels)}"
    return ConsoleInspectorSectionState(rows=tuple(rows), summary=summary)


def pr_summary_text(pr: PrEnvState) -> str:
    """Composer-insert payload for the PR 'Add to chat' action."""
    lines = [f"PR #{pr.number}: {pr.title} [{pr.state}]", pr.url]
    if pr.failing_checks:
        lines.append("Failing checks: " + ", ".join(c.name for c in pr.failing_checks))
    return "\n".join(line for line in lines if line)


def failing_checks_text(pr: PrEnvState) -> str:
    """Composer-insert payload for the failing-checks 'Fix' action."""
    lines = [f"CI is failing on PR #{pr.number} — please investigate and fix:"]
    for check in pr.failing_checks:
        suffix = f" — {check.details_url}" if check.details_url else ""
        lines.append(f"- {check.name}{suffix}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_environment_state.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ passed" .pytest-out.txt`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_environment_state.py Tests/Chat/test_console_environment_state.py
git commit -m "feat(console): Environment section projection with state-driven expansions"
```

---

### Task 4: Tasks section projection

**Files:**
- Modify: `tldw_chatbook/Chat/console_environment_state.py` (append)
- Test: `Tests/Chat/test_console_environment_state.py` (append)

**Interfaces:**
- Consumes: Task 1/3 types.
- Produces: `TASKS_ROW_HEAD = "task-head"`, `TASKS_ROW_ADD = "task-add"`, `TASKS_ENTRY_ROW_PREFIX = "task-entry-"`, `MAX_TASK_LIST_ROWS = 30`, and `project_tasks_section(snapshot: EnvironmentSnapshot, expanded: frozenset[str]) -> ConsoleInspectorSectionState`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from tldw_chatbook.Chat.console_environment_state import (
    BacklogTaskEntry,
    BranchTaskState,
    TASKS_ROW_ADD,
    TASKS_ROW_HEAD,
    project_tasks_section,
)


def _tasks_state(**kw) -> TasksEnvState:
    base = dict(
        availability=EnvSourceAvailability.OK,
        branch_task=BranchTaskState(task_id="3401", title="Video gen foundation",
                                    status="In Progress", ac_done=3, ac_total=6,
                                    path="backlog/tasks/task-3401 - Video.md"),
        in_progress=3, todo=12,
        entries=(BacklogTaskEntry("3401", "Video gen foundation", "In Progress"),
                 BacklogTaskEntry("25704", "Render-path sweep", "To Do")),
    )
    base.update(kw)
    return TasksEnvState(**base)


def test_tasks_card_absent_without_backlog_dir():
    state = project_tasks_section(EnvironmentSnapshot(), frozenset())
    assert state.rows == ()


def test_branch_task_headline_with_ac_progress():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    head = project_tasks_section(snapshot, frozenset()).rows[0]
    assert head.row_id == TASKS_ROW_HEAD
    assert head.primary_text == "task-3401 · In Progress"
    assert head.secondary_text == "3/6 ACs · Video gen foundation"
    assert head.clickable


def test_counts_headline_when_no_branch_task():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state(branch_task=None))
    head = project_tasks_section(snapshot, frozenset()).rows[0]
    assert head.primary_text == "3 in progress · 12 to do"


def test_expansion_lists_entries_in_progress_first_and_add_action():
    snapshot = EnvironmentSnapshot(tasks=_tasks_state())
    rows = project_tasks_section(snapshot, frozenset({TASKS_ROW_HEAD})).rows
    ids = [r.row_id for r in rows]
    assert TASKS_ROW_ADD in ids
    entry_rows = [r for r in rows if r.row_id.startswith("task-entry-")]
    assert entry_rows[0].primary_text.startswith("task-3401")
    assert entry_rows[0].status == "running"


def test_scanning_placeholder():
    snapshot = EnvironmentSnapshot(
        tasks=TasksEnvState(availability=EnvSourceAvailability.OK, scanning=True))
    rows = project_tasks_section(snapshot, frozenset()).rows
    assert rows[0].primary_text == "Scanning backlog…"
```

- [ ] **Step 2: Run to verify FAIL** — same pytest command, expect `ImportError` on new names.

- [ ] **Step 3: Implement** (append to `console_environment_state.py`)

```python
TASKS_ROW_HEAD = "task-head"
TASKS_ROW_ADD = "task-add"
TASKS_ENTRY_ROW_PREFIX = "task-entry-"
MAX_TASK_LIST_ROWS = 30

_STATUS_ROW_CLASS = {"In Progress": "running", "Done": "done"}


def project_tasks_section(
    snapshot: EnvironmentSnapshot,
    expanded: frozenset[str],
) -> ConsoleInspectorSectionState:
    tasks = snapshot.tasks
    if tasks.availability is not EnvSourceAvailability.OK:
        return ConsoleInspectorSectionState(rows=(), summary="")
    if tasks.scanning and not tasks.entries and tasks.branch_task is None:
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(row_id="task-scanning",
                                      primary_text="Scanning backlog…"),),
            summary="",
        )
    rows: list[InspectorSectionRow] = []
    if tasks.branch_task is not None:
        bt = tasks.branch_task
        ac = f"{bt.ac_done}/{bt.ac_total} ACs · " if bt.ac_total else ""
        rows.append(InspectorSectionRow(
            row_id=TASKS_ROW_HEAD,
            primary_text=f"task-{bt.task_id} · {bt.status}",
            secondary_text=f"{ac}{bt.title}",
            status=_STATUS_ROW_CLASS.get(bt.status, ""),
            clickable=True,
        ))
    else:
        rows.append(InspectorSectionRow(
            row_id=TASKS_ROW_HEAD,
            primary_text=f"{tasks.in_progress} in progress · {tasks.todo} to do",
            clickable=True,
        ))
    if TASKS_ROW_HEAD in expanded:
        ordered = sorted(
            tasks.entries,
            key=lambda e: (0 if e.status == "In Progress" else 1, e.task_id),
        )
        for index, entry in enumerate(ordered[:MAX_TASK_LIST_ROWS]):
            rows.append(InspectorSectionRow(
                row_id=f"{TASKS_ENTRY_ROW_PREFIX}{index}",
                primary_text=f"task-{entry.task_id} · {entry.title}",
                secondary_text=entry.status,
                status=_STATUS_ROW_CLASS.get(entry.status, ""),
            ))
        if len(tasks.entries) > MAX_TASK_LIST_ROWS:
            rows.append(InspectorSectionRow(
                row_id="task-entry-more",
                primary_text=f"… {len(tasks.entries) - MAX_TASK_LIST_ROWS} more",
            ))
        if tasks.branch_task is not None:
            rows.append(InspectorSectionRow(
                row_id=TASKS_ROW_ADD, primary_text="Add task to chat", clickable=True,
            ))
    summary = (
        f"task-{tasks.branch_task.task_id} · {tasks.branch_task.status}"
        if tasks.branch_task else f"{tasks.in_progress} doing · {tasks.todo} todo"
    )
    return ConsoleInspectorSectionState(rows=tuple(rows), summary=summary)
```

- [ ] **Step 4: Run to verify PASS** — expect 21 passed in the file.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_environment_state.py Tests/Chat/test_console_environment_state.py
git commit -m "feat(console): Tasks section projection (branch task + list)"
```

---

### Task 5: Git gatherer

**Files:**
- Create: `tldw_chatbook/Workspaces/environment_status.py`
- Test: `Tests/Workspaces/test_environment_status.py` (append)

**Interfaces:**
- Consumes: `detect_git_workspace(root: Path) -> GitWorkspaceInfo | GitWorkspaceRefusal | None` (never raises); `working_tree_status(root: Path, info: GitWorkspaceInfo) -> CurrentRootStatus` (raises `GitWorkspaceError`); `linked_worktree_name(root)` (Task 2); Task 1 dataclasses.
- Produces: `gather_git_env(root: Path, *, previous: GitEnvState | None = None) -> GitEnvState`. Never raises. On `GitWorkspaceError` with a `previous` OK state: returns `previous` with `stale=True` (via `dataclasses.replace`); without one: `availability=ERROR`.

- [ ] **Step 1: Write the failing tests** (append to the Task 2 test file; reuse the `main_repo`/`_git` fixtures)

```python
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
```

- [ ] **Step 2: Run to verify FAIL** — `ModuleNotFoundError: tldw_chatbook.Workspaces.environment_status`.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Workspaces/environment_status.py
"""Impure gatherers for the Console Environment panel.

Everything here runs on a worker thread. Functions never raise: failures
map to availability enums (spec: absence is silent, errors keep last
good data with a stale marker).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loguru import logger

from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    GitEnvState,
)
from tldw_chatbook.Workspaces.git_workspace import (
    GitWorkspaceError,
    GitWorkspaceInfo,
    detect_git_workspace,
    linked_worktree_name,
    working_tree_status,
    _run_user_git,
)


def _head_short(root: Path) -> str:
    try:
        result = _run_user_git(root, "rev-parse", "--short", "HEAD")
    except GitWorkspaceError:
        return ""
    return result.stdout.strip()


def gather_git_env(root: Path, *, previous: GitEnvState | None = None) -> GitEnvState:
    """Assemble the git tier of the Environment snapshot. Never raises."""
    info = detect_git_workspace(root)
    if not isinstance(info, GitWorkspaceInfo):
        return GitEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
    try:
        status = working_tree_status(root, info)
    except GitWorkspaceError as exc:
        logger.debug("environment_status: working_tree_status failed: {}", exc)
        if previous is not None and previous.availability is EnvSourceAvailability.OK:
            return replace(previous, stale=True)
        return GitEnvState(availability=EnvSourceAvailability.ERROR)
    files = status.files
    return GitEnvState(
        availability=EnvSourceAvailability.OK,
        root=str(status.root),
        branch=info.branch,
        detached=info.detached,
        unborn=info.unborn,
        head_short=_head_short(root) if info.detached else "",
        upstream=info.upstream,
        ahead=info.ahead,
        behind=info.behind,
        adds=sum(f.adds for f in files),
        dels=sum(f.dels for f in files),
        files=files,
        worktree_name=linked_worktree_name(root),
        stale=False,
    )
```

Note: `_run_user_git` is module-private to `git_workspace` — importing it across sibling modules in the same package is accepted here for `_head_short` (same subprocess hardening); if the reviewer objects, add a public `head_short_sha(root)` next to `linked_worktree_name` instead.

- [ ] **Step 4: Run to verify PASS** — expect 9 passed in `Tests/Workspaces/test_environment_status.py`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/environment_status.py Tests/Workspaces/test_environment_status.py
git commit -m "feat(workspaces): git tier gatherer for Environment panel"
```

---

### Task 6: gh gatherer (PR + CI)

**Files:**
- Modify: `tldw_chatbook/Workspaces/environment_status.py` (append)
- Test: `Tests/Workspaces/test_environment_status.py` (append)

**Interfaces:**
- Produces:
  - `GhRunner = Callable[[Path, list[str]], "GhResult | None"]` — seam for tests
  - `GhResult` frozen dataclass: `returncode: int`, `stdout: str`, `stderr: str`
  - `run_gh(root: Path, args: list[str], *, timeout: float = 5.0) -> GhResult | None` — `None` means the binary is missing
  - `gather_pr_env(root: Path, branch: str | None, *, runner: GhRunner = run_gh, previous: PrEnvState | None = None) -> PrEnvState`. Never raises.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

- [ ] **Step 2: Run to verify FAIL** — `ImportError` on `GhResult`/`gather_pr_env`.

- [ ] **Step 3: Implement** (append to `environment_status.py`)

```python
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from tldw_chatbook.Chat.console_environment_state import PrCheck, PrEnvState

_GH_EXECUTABLE = "gh"
_GH_TIMEOUT_SECONDS = 5.0
_PR_JSON_FIELDS = (
    "number,title,state,isDraft,url,additions,deletions,mergedAt,statusCheckRollup"
)


@dataclass(frozen=True)
class GhResult:
    returncode: int
    stdout: str
    stderr: str


def run_gh(root: Path, args: list[str], *, timeout: float = _GH_TIMEOUT_SECONDS) -> GhResult | None:
    """Run gh non-interactively in ``root``. ``None`` == binary missing."""
    env = dict(os.environ)
    env.update({"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "NO_COLOR": "1"})
    try:
        completed = subprocess.run(
            [_GH_EXECUTABLE, *args],
            cwd=str(root), env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return None
    except (subprocess.TimeoutExpired, OSError) as exc:
        return GhResult(returncode=124, stdout="", stderr=str(exc))
    return GhResult(completed.returncode, completed.stdout, completed.stderr)


GhRunner = Callable[[Path, list[str]], GhResult | None]


def _parse_check(entry: dict) -> PrCheck | None:
    typename = entry.get("__typename", "")
    if typename == "CheckRun":
        status = (entry.get("status") or "").upper()
        conclusion_raw = (entry.get("conclusion") or "").upper()
        if status != "COMPLETED":
            conclusion = "pending"
        elif conclusion_raw in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            conclusion = "success"
        else:
            conclusion = "failure"
        return PrCheck(name=str(entry.get("name") or "?"), conclusion=conclusion,
                       details_url=str(entry.get("detailsUrl") or ""))
    if typename == "StatusContext":
        state = (entry.get("state") or "").upper()
        conclusion = ("success" if state == "SUCCESS"
                      else "pending" if state in {"PENDING", "EXPECTED"} else "failure")
        return PrCheck(name=str(entry.get("context") or "?"), conclusion=conclusion,
                       details_url=str(entry.get("targetUrl") or ""))
    return None


def _parse_merged_at(raw: object) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def gather_pr_env(
    root: Path,
    branch: str | None,
    *,
    runner: GhRunner = run_gh,
    previous: PrEnvState | None = None,
) -> PrEnvState:
    """Fetch PR + checks for ``branch`` via gh. Never raises."""
    if not branch:
        return PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
    result = runner(root, ["pr", "view", branch, "--json", _PR_JSON_FIELDS])
    if result is None:
        return PrEnvState(availability=EnvSourceAvailability.MISSING_TOOL)
    if result.returncode != 0:
        if "no pull requests found" in result.stderr.lower():
            return PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
        logger.debug("environment_status: gh failed rc={} err={}",
                     result.returncode, result.stderr[:200])
        if previous is not None and previous.availability is EnvSourceAvailability.OK:
            return replace(previous, stale=True)
        return PrEnvState(availability=EnvSourceAvailability.ERROR)
    try:
        payload = json.loads(result.stdout)
        checks = tuple(
            check for entry in (payload.get("statusCheckRollup") or [])
            if isinstance(entry, dict) and (check := _parse_check(entry)) is not None
        )
        return PrEnvState(
            availability=EnvSourceAvailability.OK,
            number=int(payload.get("number") or 0),
            title=str(payload.get("title") or ""),
            state=str(payload.get("state") or ""),
            is_draft=bool(payload.get("isDraft")),
            url=str(payload.get("url") or ""),
            adds=int(payload.get("additions") or 0),
            dels=int(payload.get("deletions") or 0),
            merged_at=_parse_merged_at(payload.get("mergedAt")),
            checks=checks,
        )
    except (ValueError, TypeError) as exc:
        logger.debug("environment_status: gh JSON parse failed: {}", exc)
        return PrEnvState(availability=EnvSourceAvailability.ERROR)
```

- [ ] **Step 4: Run to verify PASS** — expect 16 passed in the Workspaces test file.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/environment_status.py Tests/Workspaces/test_environment_status.py
git commit -m "feat(workspaces): gh PR+checks gatherer with graceful degradation"
```

---

### Task 7: Backlog task scanner

**Files:**
- Modify: `tldw_chatbook/Workspaces/environment_status.py` (append)
- Test: `Tests/Workspaces/test_environment_status.py` (append)

**Interfaces:**
- Produces: `class BacklogTaskScanner` with `scan(self, workspace_root: Path, branch: str | None) -> TasksEnvState`. Instance-scoped `(mtime, size) -> parsed` cache (NEVER module-global — a module-global cache leaked between tests in a prior review). Filenames give id + title (`task-<id> - <title>.md`); frontmatter is parsed only for changed files; AC checkboxes only for the branch task's file.

- [ ] **Step 1: Write the failing tests** (append)

```python
from tldw_chatbook.Workspaces.environment_status import BacklogTaskScanner


def _write_task(tasks_dir: Path, task_id: str, title: str, status: str,
                body: str = "") -> Path:
    path = tasks_dir / f"task-{task_id} - {title}.md"
    path.write_text(f"---\nid: task-{task_id}\ntitle: {title}\nstatus: {status}\n---\n\n{body}")
    return path


@pytest.fixture()
def backlog_ws(tmp_path: Path) -> Path:
    tasks_dir = tmp_path / "backlog" / "tasks"
    tasks_dir.mkdir(parents=True)
    _write_task(tasks_dir, "101", "Fix frobnicator", "In Progress")
    _write_task(tasks_dir, "102", "Polish widget", "To Do")
    _write_task(tasks_dir, "103", "Old thing", "Done")
    _write_task(
        tasks_dir, "3401", "Video foundation", "In Progress",
        body=("## Acceptance Criteria\n\n- [x] adapter builds\n- [x] tests green\n"
              "- [ ] docs updated\n"),
    )
    return tmp_path


def test_scan_counts_statuses_and_excludes_done_from_entries(backlog_ws: Path):
    state = BacklogTaskScanner().scan(backlog_ws, branch=None)
    assert state.availability is EnvSourceAvailability.OK
    assert state.in_progress == 2 and state.todo == 1
    assert {e.task_id for e in state.entries} == {"101", "102", "3401"}


def test_scan_no_backlog_dir_is_not_applicable(tmp_path: Path):
    state = BacklogTaskScanner().scan(tmp_path, branch=None)
    assert state.availability is EnvSourceAvailability.NOT_APPLICABLE


def test_branch_task_gets_ac_progress(backlog_ws: Path):
    state = BacklogTaskScanner().scan(backlog_ws, branch="feat/task-3401-video")
    assert state.branch_task is not None
    assert state.branch_task.task_id == "3401"
    assert state.branch_task.ac_done == 2 and state.branch_task.ac_total == 3


def test_malformed_frontmatter_is_skipped_not_fatal(backlog_ws: Path):
    bad = backlog_ws / "backlog" / "tasks" / "task-999 - Broken.md"
    bad.write_text("---\nstatus: [unclosed\n---\n")
    state = BacklogTaskScanner().scan(backlog_ws, branch=None)
    assert "999" not in {e.task_id for e in state.entries}
    assert state.in_progress == 2  # rest of the scan unaffected


def test_mtime_cache_avoids_reparsing_unchanged_files(backlog_ws: Path, monkeypatch):
    scanner = BacklogTaskScanner()
    scanner.scan(backlog_ws, branch=None)
    calls = {"n": 0}
    original = BacklogTaskScanner._parse_status

    def counting(self, path):
        calls["n"] += 1
        return original(self, path)

    monkeypatch.setattr(BacklogTaskScanner, "_parse_status", counting)
    scanner.scan(backlog_ws, branch=None)
    assert calls["n"] == 0  # nothing changed on disk -> zero re-parses
```

- [ ] **Step 2: Run to verify FAIL** — `ImportError: BacklogTaskScanner`.

- [ ] **Step 3: Implement** (append)

```python
import re as _re

import yaml

from tldw_chatbook.Chat.console_environment_state import (
    BacklogTaskEntry,
    BranchTaskState,
    TasksEnvState,
    branch_task_id,
)

_TASK_FILENAME_RE = _re.compile(r"^task-(\d+(?:\.\d+)*) - (.+)\.md$")
_FRONT_MATTER_RE = _re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", _re.DOTALL)
_AC_DONE_RE = _re.compile(r"^- \[x\]", _re.MULTILINE | _re.IGNORECASE)
_AC_OPEN_RE = _re.compile(r"^- \[ \]", _re.MULTILINE)


class BacklogTaskScanner:
    """Scans <workspace>/backlog/tasks/ with an instance-scoped (mtime, size) cache."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple[float, int], str]] = {}

    def _parse_status(self, path: Path) -> str:
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError:
            return ""
        match = _FRONT_MATTER_RE.match(head)
        if not match:
            return ""
        try:
            meta = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return ""
        if not isinstance(meta, dict):
            return ""
        return str(meta.get("status") or "").strip()

    def _status_for(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError:
            return ""
        signature = (stat.st_mtime, stat.st_size)
        cached = self._cache.get(str(path))
        if cached is not None and cached[0] == signature:
            return cached[1]
        status = self._parse_status(path)
        self._cache[str(path)] = (signature, status)
        return status

    @staticmethod
    def _ac_progress(path: Path) -> tuple[int, int]:
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return (0, 0)
        done = len(_AC_DONE_RE.findall(body))
        open_count = len(_AC_OPEN_RE.findall(body))
        return (done, done + open_count)

    def scan(self, workspace_root: Path, branch: str | None) -> TasksEnvState:
        tasks_dir = workspace_root / "backlog" / "tasks"
        if not tasks_dir.is_dir():
            return TasksEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)
        wanted_id = branch_task_id(branch)
        entries: list[BacklogTaskEntry] = []
        branch_task: BranchTaskState | None = None
        in_progress = todo = 0
        try:
            listing = sorted(tasks_dir.iterdir())
        except OSError:
            return TasksEnvState(availability=EnvSourceAvailability.ERROR)
        for path in listing:
            match = _TASK_FILENAME_RE.match(path.name)
            if not match:
                continue
            task_id, title = match.group(1), match.group(2)
            status = self._status_for(path)
            if not status:
                continue
            if status == "In Progress":
                in_progress += 1
            elif status == "To Do":
                todo += 1
            if status != "Done" or task_id == wanted_id:
                entries.append(BacklogTaskEntry(task_id=task_id, title=title, status=status))
            if wanted_id is not None and task_id == wanted_id:
                done, total = self._ac_progress(path)
                branch_task = BranchTaskState(
                    task_id=task_id, title=title, status=status,
                    ac_done=done, ac_total=total, path=str(path),
                )
        return TasksEnvState(
            availability=EnvSourceAvailability.OK,
            branch_task=branch_task,
            in_progress=in_progress,
            todo=todo,
            entries=tuple(entries),
        )
```

- [ ] **Step 4: Run to verify PASS** — expect 21 passed in the Workspaces test file. Also re-run `Tests/Chat/test_console_environment_state.py` (still 21 passed).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Workspaces/environment_status.py Tests/Workspaces/test_environment_status.py
git commit -m "feat(workspaces): backlog task scanner with instance-scoped mtime cache"
```

---

### Task 8: Persisted section-collapse booleans

**Files:**
- Modify: `tldw_chatbook/Chat/console_rail_state.py` (six touch points, pattern of `inspector_more_open`)
- Test: `Tests/Chat/test_console_rail_state.py` if it exists on dev (check first: `ls Tests/Chat | grep rail`); otherwise create `Tests/Chat/test_console_rail_environment_prefs.py`

**Interfaces:**
- Produces: `ConsoleRailPreferences.environment_open: bool = True`, `.tasks_open: bool = True`; same fields on `ConsoleRailState`; both round-trip through `coerce_console_rail_preferences` / `serialize_console_rail_preferences` and propagate through `build_console_rail_state`; `"environment"` and `"tasks"` accepted by the screen's `section_updates` writer via `CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Chat/test_console_rail_environment_prefs.py
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS,
    ConsoleRailPreferences,
    coerce_console_rail_preferences,
    serialize_console_rail_preferences,
)


def test_environment_and_tasks_open_default_true_and_round_trip():
    defaults = ConsoleRailPreferences()
    assert defaults.environment_open is True
    assert defaults.tasks_open is True
    serialized = serialize_console_rail_preferences(defaults)
    assert serialized["environment_open"] is True and serialized["tasks_open"] is True
    coerced = coerce_console_rail_preferences({"environment_open": False, "tasks_open": False})
    assert coerced.environment_open is False and coerced.tasks_open is False


def test_disclosure_ids_accept_environment_and_tasks():
    assert "environment" in CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS
    assert "tasks" in CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS


def test_coerce_garbage_falls_back_to_defaults():
    coerced = coerce_console_rail_preferences({"environment_open": "banana"})
    assert coerced.environment_open is True
```

- [ ] **Step 2: Run to verify FAIL** — `AttributeError`/`TypeError` on the new fields.

- [ ] **Step 3: Implement** — the six edits in `console_rail_state.py` (line refs from origin/dev, re-verify):
  1. `ConsoleRailPreferences` (~L138): add `environment_open: bool = True` and `tasks_open: bool = True`.
  2. `ConsoleRailState` (~L182): add the same two fields with `= True` defaults.
  3. `coerce_console_rail_preferences` (~L388): add `environment_open=_coerce_bool(raw.get("environment_open"), defaults.environment_open), tasks_open=_coerce_bool(raw.get("tasks_open"), defaults.tasks_open),`.
  4. `serialize_console_rail_preferences` (~L419): add `"environment_open": bool(preferences.environment_open), "tasks_open": bool(preferences.tasks_open),`.
  5. `build_console_rail_state` return (~L882): pass `environment_open=preferences.environment_open, tasks_open=preferences.tasks_open` (mirror how `inspector_more_open` flows).
  6. Disclosure registration (~L29-33): add `CONSOLE_ENVIRONMENT_DISCLOSURE_ID = "environment"` and `CONSOLE_TASKS_DISCLOSURE_ID = "tasks"`, and append both to `CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS`. Do NOT add them to `CONSOLE_RAIL_SECTION_IDS` (that tuple is left-rail sections and other code iterates it).

  Also check `serialize_console_rail_stored_preferences` (~L423) — if it enumerates fields explicitly, add the two names there too.

- [ ] **Step 4: Run to verify PASS**, then run the existing rail-state suite: `pytest Tests/Chat/ -k "rail_state or rail_preference" -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ passed" .pytest-out.txt` — pre-existing tests may pin the serialized-dict key set; update any such pin in the same commit.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_rail_state.py Tests/Chat/
git commit -m "feat(console): persist Environment/Tasks section collapse in rail preferences"
```

---

### Task 9: Mount Environment + Tasks sections in the right rail

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py` (ctor kwargs + compose)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (rail construction site, ~L14509 region — pass initial states)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (only if a new rule is genuinely needed — reuse `.console-inspector-section*` classes first)
- Test: `Tests/UI/test_console_environment_section.py` (new)

**Interfaces:**
- Consumes: `ConsoleInspectorSection(title=..., section_id=..., rows=..., summary=..., collapsible=True, open=..., view_all_label=..., id=...)`; `ENVIRONMENT_SECTION_ID`/`TASKS_SECTION_ID` and projections from Tasks 3-4.
- Produces: rail ctor gains `environment_section_state: ConsoleInspectorSectionState | None = None`, `tasks_section_state: ConsoleInspectorSectionState | None = None`, `environment_open: bool = True`, `tasks_open: bool = True`. DOM: `#console-environment-section` and `#console-tasks-section` mounted at the TOP of `_InspectorOuterBody` (before the staged-context tray). Sections with empty `rows` get `styles.display = "none"` (fleet pattern).

- [ ] **Step 1: Write the failing widget test**

```python
# Tests/UI/test_console_environment_section.py
"""Widget-level tests for the Environment/Tasks sections (ConsolidatedCSSApp harness)."""
import pytest
from datetime import datetime, timezone

from tldw_chatbook.Chat.console_environment_state import (
    EnvironmentSnapshot,
    EnvSourceAvailability,
    GitEnvState,
    project_environment_section,
)
from tldw_chatbook.Widgets.Console.console_inspector_section import ConsoleInspectorSection

from Tests.UI.consolidated_css import ConsolidatedCSSApp

_NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def _section() -> ConsoleInspectorSection:
    snapshot = EnvironmentSnapshot(git=GitEnvState(
        availability=EnvSourceAvailability.OK, root="/w", branch="feat/task-1-x",
        adds=10, dels=2,
        files=(),
    ))
    state = project_environment_section(snapshot, frozenset(), now=_NOW)
    return ConsoleInspectorSection(
        title="Environment", section_id="environment",
        rows=state.rows, summary=state.summary,
        collapsible=True, open=True, view_all_label="Refresh",
        id="console-environment-section",
    )


class _Harness(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.view_all_events = []

    def compose(self):
        yield _section()

    def on_console_inspector_section_view_all_requested(self, event):
        self.view_all_events.append(event.section_id)


@pytest.mark.asyncio
async def test_environment_section_renders_rows_and_refresh_slot():
    app = _Harness()
    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        section = app.query_one("#console-environment-section", ConsoleInspectorSection)
        primaries = [s.renderable for s in section.query(".console-inspector-section-row-primary")]
        assert any("Changes" in str(p) for p in primaries)
        view_all = app.query_one("#console-inspector-section-environment-view-all")
        view_all.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.view_all_events == ["environment"]
```

(Adapt the message-handler name to the harness convention in `Tests/UI/test_console_inspector_section.py` — that file registers `@on(ConsoleInspectorSection.ViewAllRequested)`-style collectors; copy its exact pattern, including the pilot settle loops it uses.)

- [ ] **Step 2: Run to verify FAIL** (section renders but the file/imports don't exist yet → collect error first, then assertion refinement).

Run: `pytest Tests/UI/test_console_environment_section.py -v > .pytest-out.txt 2>&1; grep -E "passed|failed|error" .pytest-out.txt`

- [ ] **Step 3: Implement rail changes**

In `right_rail.py`:
- Add ctor kwargs (after `inspector_more_open`): `environment_section_state`, `tasks_section_state`, `environment_open: bool = True`, `tasks_open: bool = True`; store on `self`. Default `None` → `ConsoleInspectorSectionState(rows=(), summary="")`.
- In `compose()`, as the FIRST children inside `_InspectorOuterBody` (before the staged-context tray yield):

```python
environment_section = ConsoleInspectorSection(
    title="Environment",
    section_id="environment",
    rows=self._environment_section_state.rows,
    summary=self._environment_section_state.summary,
    collapsible=True,
    open=self._environment_open,
    view_all_label="Refresh",
    id="console-environment-section",
)
environment_section.styles.display = (
    "block" if self._environment_section_state.rows else "none"
)
yield environment_section
tasks_section = ConsoleInspectorSection(
    title="Tasks",
    section_id="tasks",
    rows=self._tasks_section_state.rows,
    summary=self._tasks_section_state.summary,
    collapsible=True,
    open=self._tasks_open,
    id="console-tasks-section",
)
tasks_section.styles.display = "block" if self._tasks_section_state.rows else "none"
yield tasks_section
```

In `chat_screen.py` at the rail construction call (search `ConsoleInspectorRail(`): pass `environment_section_state=self._console_environment_section_state()`, `tasks_section_state=self._console_tasks_section_state()`, `environment_open=rail_state.environment_open`, `tasks_open=rail_state.tasks_open`. Add the two small builder methods next to the call site returning the projections from the controller's current snapshot (Task 11 provides it; until then they can return the empty-state projection of a default `EnvironmentSnapshot()` — hidden section, which is correct pre-wiring).

CSS: reuse `.console-inspector-section*` classes — add new rules ONLY if the live check in Task 14 shows a gap; if added, regenerate: `python -m tldw_chatbook.css.build_css`.

- [ ] **Step 4: Run to verify PASS**, plus the rail order census: `pytest Tests/UI/test_console_right_rail.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ (passed|failed)" .pytest-out.txt`. `test_mounted_inspector_semantic_census_matches_actual_right_rail_order` WILL fail — update its expected order to include the two new (hidden-when-empty) sections in the same commit. Re-check the pre-existing dev-red list in the spec's memory before blaming your change for an unrelated failure: verify any suspicious failure against a pristine `origin/dev` worktree first.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/right_rail.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(console): mount Environment and Tasks sections in the Inspect rail"
```

---

### Task 10: Move the agent fleet section from the left rail to the right rail (atomic)

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/left_rail.py` (remove ctor kwarg `agent_fleet_section_state` ~L337/L458 and the compose block ~L2239-2250)
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py` (add the kwarg + compose the section after `#console-tasks-section`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (move the `agent_fleet_section_state=` kwarg from the left-rail construction ~L14098 to the right-rail construction; `_sync_console_agent_section` ~L7788 needs NO query change — it queries by id `#console-agent-section-subagents`, which moves with the widget)
- Test: `Tests/UI/` — sweep and update

**Interfaces:**
- Consumes: `_console_agent_fleet_section_state(self) -> ConsoleInspectorSectionState` (unchanged), `CONSOLE_AGENT_FLEET_SECTION_ID = "agent-fleet"` (unchanged), widget id `console-agent-section-subagents` (unchanged — keep it so the sync path and its tests keep working).
- Produces: fleet section rendered inside `#console-inspector-rail-body`, title "Agents" (renamed from "Sub-agents" — see sweep note), `open=False` as today.

- [ ] **Step 1: Sweep BEFORE touching anything** (the `Sources`→`Retrieval` blast-radius lesson): run all of

```bash
git grep -n "Sub-agents" -- 'tldw_chatbook/' 'Tests/'
git grep -n "console-agent-section-subagents" -- 'tldw_chatbook/' 'Tests/'
git grep -n "agent_fleet_section_state" -- 'tldw_chatbook/' 'Tests/'
git grep -n "agent-fleet" -- 'tldw_chatbook/' 'Tests/'
```

List every hit and decide its fate before editing. If the "Sub-agents" title has more than ~3 test consumers, KEEP the title "Sub-agents" and drop the rename (the move is the deliverable; the rename is cosmetic). Record the decision in the commit message.

- [ ] **Step 2: Write the failing test** — in `Tests/UI/test_console_right_rail.py` style (use its `make_console_pilot`):

```python
async def test_agent_fleet_section_lives_in_the_right_rail():
    async with make_console_pilot() as (pilot, screen):
        section = screen.query_one("#console-agent-section-subagents")
        body = screen.query_one("#console-inspector-rail-body")
        node = section
        while node is not None and node is not body:
            node = node.parent
        assert node is body, "fleet section must be a descendant of the Inspect rail body"
```

- [ ] **Step 3: Run to verify FAIL** (it's in the left rail today).

- [ ] **Step 4: Implement the move** — left rail: delete kwarg, stored attr, and compose block; right rail: add kwarg `agent_fleet_section_state: ConsoleInspectorSectionState | None = None` and compose after the tasks section (copy the exact block from left_rail L2239-2250, unchanged id, unchanged `open=False`, display toggle on rows); chat_screen: move the kwarg between the two construction sites.

- [ ] **Step 5: Run to verify PASS + update the sweep's casualties**

Run the fleet + rail suites: `pytest Tests/UI/test_console_right_rail.py Tests/UI/test_console_agent_fleet_sync_coalescing.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ (passed|failed)" .pytest-out.txt`, plus every test file the Step 1 sweep flagged (left-rail order/census tests will need their expected inventories updated). Verify any ambiguous failure against pristine `origin/dev` before treating it as yours.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/ tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(console): move agent fleet section from left rail to Inspect rail"
```

---

### Task 11: Environment controller (cadence, TTL, backoff — no app)

**Files:**
- Create: `tldw_chatbook/UI/Console_Modules/environment.py`
- Test: `Tests/UI/test_console_environment_controller.py` (new; pure unit tests, no Textual app)

**Interfaces:**
- Consumes: gatherers from Task 5-7, snapshot types from Task 1.
- Produces `ConsoleEnvironmentController` with this exact surface (Task 12 wires it):

```python
class ConsoleEnvironmentController:
    LOCAL_WORKER_GROUP = "console-environment-local"
    NET_WORKER_GROUP = "console-environment-net"

    def __init__(
        self, *,
        run_worker: Callable[..., Any],          # screen.run_worker passthrough
        marshal_to_ui: Callable[..., None],      # app.call_from_thread passthrough
        workspace_root_accessor: Callable[[], str | None],
        rail_open_accessor: Callable[[], bool],
        on_snapshot: Callable[[EnvironmentSnapshot], None],  # runs on UI thread
        now: Callable[[], datetime] = ...,       # injectable clock
    ) -> None: ...
    snapshot: EnvironmentSnapshot                # current, UI-thread reads
    def request_refresh(self, *, include_net: bool = False, force_net: bool = False) -> None
    def poll_tick(self) -> None                  # called every 10s by the screen timer
    def notify_rail_opened(self) -> None         # local + net (TTL-respecting)
```

Behavior (each a test): no dispatch while rail closed; no dispatch when root is None; local dispatch uses `LOCAL_WORKER_GROUP` with `thread=True, exclusive=True` and net uses `NET_WORKER_GROUP` (never the same group — the 10s poll must not cancel an in-flight gh fetch); net respects a 60s TTL keyed by `(root, branch)` unless `force_net`; 3 consecutive failures of a tier pause it until `force_net`/root change; a snapshot landed for a root that is no longer current is dropped (stale-scope guard); `on_snapshot` is invoked via `marshal_to_ui`.

- [ ] **Step 1: Write the failing tests** — the controller must import the gatherers as module attributes (`from ... import gather_git_env` at module top) so tests can patch `tldw_chatbook.UI.Console_Modules.environment.gather_git_env`. Shared fake harness for the whole file:

```python
# Tests/UI/test_console_environment_controller.py
"""Controller cadence/TTL/backoff tests — no Textual app, synchronous fakes."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import tldw_chatbook.UI.Console_Modules.environment as env_mod
from tldw_chatbook.Chat.console_environment_state import (
    EnvironmentSnapshot, EnvSourceAvailability, GitEnvState, PrEnvState, TasksEnvState,
)
from tldw_chatbook.UI.Console_Modules.environment import ConsoleEnvironmentController


class Fixture:
    def __init__(self, monkeypatch, *, root="/w/repo", rail_open=True):
        self.dispatched: list[dict] = []
        self.snapshots: list[EnvironmentSnapshot] = []
        self.root: str | None = root
        self.rail_open = rail_open
        self.clock = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        self.git_calls = 0
        self.pr_calls = 0

        def fake_git(path, previous=None):
            self.git_calls += 1
            return GitEnvState(availability=EnvSourceAvailability.OK,
                               root=str(path), branch="feat/task-1-x")

        def fake_pr(path, branch, runner=None, previous=None):
            self.pr_calls += 1
            return PrEnvState(availability=EnvSourceAvailability.OK, number=7,
                              title="T", state="OPEN", url="https://x/pull/7")

        monkeypatch.setattr(env_mod, "gather_git_env", fake_git)
        monkeypatch.setattr(env_mod, "gather_pr_env", fake_pr)
        monkeypatch.setattr(
            env_mod.BacklogTaskScanner, "scan",
            lambda scanner, ws, branch: TasksEnvState(
                availability=EnvSourceAvailability.NOT_APPLICABLE),
        )

        def run_worker(fn, **kwargs):
            self.dispatched.append(kwargs)
            fn()  # synchronous: the "worker" runs inline

        self.controller = ConsoleEnvironmentController(
            run_worker=run_worker,
            marshal_to_ui=lambda fn, *a: fn(*a),
            workspace_root_accessor=lambda: self.root,
            rail_open_accessor=lambda: self.rail_open,
            on_snapshot=self.snapshots.append,
            now=lambda: self.clock,
        )


def test_no_dispatch_while_rail_closed(monkeypatch):
    fx = Fixture(monkeypatch, rail_open=False)
    fx.controller.request_refresh(include_net=True)
    assert fx.dispatched == []


def test_local_and_net_use_distinct_worker_groups(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    groups = {d["group"] for d in fx.dispatched}
    assert groups == {ConsoleEnvironmentController.LOCAL_WORKER_GROUP,
                      ConsoleEnvironmentController.NET_WORKER_GROUP}
    assert all(d["thread"] is True and d["exclusive"] is True for d in fx.dispatched)


def test_net_ttl_suppresses_refetch_within_60s_and_force_busts_it(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.request_refresh(include_net=True)
    fx.clock += timedelta(seconds=30)
    fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 1  # TTL held
    fx.controller.request_refresh(include_net=True, force_net=True)
    assert fx.pr_calls == 2
    fx.clock += timedelta(seconds=61)
    fx.controller.request_refresh(include_net=True)
    assert fx.pr_calls == 3


def test_three_failures_pause_the_local_tier_until_forced(monkeypatch):
    fx = Fixture(monkeypatch)
    monkeypatch.setattr(
        env_mod, "gather_git_env",
        lambda path, previous=None: GitEnvState(availability=EnvSourceAvailability.ERROR),
    )
    for _ in range(3):
        fx.controller.poll_tick()
    dispatched_before = len(fx.dispatched)
    fx.controller.poll_tick()  # paused: no new dispatch
    assert len(fx.dispatched) == dispatched_before


def test_stale_scope_snapshot_is_dropped_when_root_changes_mid_flight(monkeypatch):
    fx = Fixture(monkeypatch)

    def run_worker_scope_shift(fn, **kwargs):
        fx.root = "/other/repo"  # root changes while the "worker" runs
        fn()

    fx.controller._run_worker = run_worker_scope_shift
    fx.controller.request_refresh()
    assert fx.snapshots == []  # landed result discarded by the stale-scope guard


def test_poll_tick_dispatches_local_only(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.poll_tick()
    assert fx.git_calls == 1 and fx.pr_calls == 0


def test_rail_open_dispatches_both_tiers(monkeypatch):
    fx = Fixture(monkeypatch)
    fx.controller.notify_rail_opened()
    assert fx.git_calls == 1 and fx.pr_calls == 1
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement the controller** (~150 lines). Structure:

```python
class ConsoleEnvironmentController:
    def __init__(self, *, run_worker, marshal_to_ui, workspace_root_accessor,
                 rail_open_accessor, on_snapshot, now=None) -> None:
        self._run_worker = run_worker
        self._marshal_to_ui = marshal_to_ui
        self._workspace_root_accessor = workspace_root_accessor
        self._rail_open_accessor = rail_open_accessor
        self._on_snapshot = on_snapshot
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.snapshot = EnvironmentSnapshot()
        self._scanner = BacklogTaskScanner()
        self._net_fetched_at: tuple[str, str, datetime] | None = None  # (root, branch, at)
        self._failures = {"local": 0, "net": 0}
        self._NET_TTL = timedelta(seconds=60)
        self._MAX_FAILURES = 3
```

`request_refresh`: read root; bail on None/closed rail; capture `scope_root = root`; dispatch a named closure per tier via `self._run_worker(job, thread=True, exclusive=True, group=GROUP)` (named `def` closures, never `partial` — `run_worker(partial(...))` silently anonymises the worker); each job gathers, then `self._marshal_to_ui(self._land, scope_root, tier, result)`. `_land` drops the result if `scope_root != self._workspace_root_accessor()` (stale-scope) or the tier failed (`availability is ERROR` increments `self._failures[tier]`, success resets it), rebuilds `self.snapshot` via `dataclasses.replace`, and calls `self._on_snapshot(self.snapshot)`. `poll_tick`: local tier only, skipping when paused; also detects root changes (compare with the root of the last landed snapshot) and then refreshes both tiers. `notify_rail_opened`: `request_refresh(include_net=True)`.

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/environment.py Tests/UI/test_console_environment_controller.py
git commit -m "feat(console): environment controller with tiered workers, TTL, backoff"
```

---

### Task 12: `initial_current_mode` on ChangeReviewScreen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (`__init__` ~L1369; `_land_git_detection` tail ~L1755-1775)
- Test: `Tests/Workspaces/` — find the existing change-review screen test file (`ls Tests/Workspaces | grep change_review`) and append there, following its harness

**Interfaces:**
- Consumes: `CURRENT_MODE_SENTINEL = "__git_current__"` (L75), `turn_select_options` property (L1636), `git_detection_settled` property (L1646), `_load_current_mode()` (L2306).
- Produces: `ChangeReviewScreen.__init__(..., initial_current_mode: bool = False)`. When True and git detection lands with the current-mode option present, the screen selects `CURRENT_MODE_SENTINEL` and loads current mode once; when the option never appears (no git), the flag is a no-op.

- [ ] **Step 1: Write the failing test** (in the existing change-review screen test file's harness style — it drives the screen with a fake provider):

```python
async def test_initial_current_mode_selects_working_tree_after_detection():
    # build the screen exactly as the neighboring tests do, adding initial_current_mode=True
    # settle until screen.git_detection_settled is True (pilot.pause loop)
    select = screen.query_one("#change-review-turn-select", Select)
    assert select.value == CURRENT_MODE_SENTINEL


async def test_initial_current_mode_is_noop_without_git():
    # same harness with a provider whose detect_git returns {} — flag must not crash
    assert screen.git_detection_settled
```

- [ ] **Step 2: Run to verify FAIL** — `TypeError: unexpected keyword argument 'initial_current_mode'`.

- [ ] **Step 3: Implement** — store `self._initial_current_mode = bool(initial_current_mode)` in `__init__`; at the end of `_land_git_detection`, after the current-mode option is prepended:

```python
if self._initial_current_mode:
    self._initial_current_mode = False  # one-shot
    select = self.query_one("#change-review-turn-select", Select)
    select.value = CURRENT_MODE_SENTINEL
    self._load_current_mode()
```

(Check whether setting `select.value` fires `_on_turn_changed` and double-loads; if it does, drop the explicit `_load_current_mode()` call — assert single-load in the test via a counter monkeypatch on `_load_current_mode`.)

- [ ] **Step 4: Run to verify PASS** plus the file's existing tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/change_review_screen.py Tests/Workspaces/
git commit -m "feat(change-review): initial_current_mode opens on the working tree"
```

---

### Task 13: Screen wiring — workers, timer, landing, row actions

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_environment_wiring.py` (new; `make_console_pilot` harness from `Tests/UI/test_console_right_rail.py` L71)

**Interfaces:**
- Consumes: everything above, plus dev seams — `self._review_selection._console_change_review_workspace_roots() -> tuple[str, ...] | None` (first root = primary), `self._open_change_review()` (L18161), `self._insert_prompt_text_into_composer(text, replace=False)` (L17008), `self.app.open_url(url)` (never `webbrowser.open` — corrupts the TUI), `self._set_console_rail_preference(section_updates=...)` (L11813), `action_toggle_console_inspector_rail` (L2385), the fleet coalescer `_run_coalesced_console_agent_fleet_sync` (L7740), `@on(ConsoleInspectorSection.RowActivated)` handler (L2257) and `@on(ConsoleInspectorSection.CollapseToggled)` / `ViewAllRequested`.
- Produces on ChatScreen:
  - `self._console_environment = ConsoleEnvironmentController(run_worker=self.run_worker, marshal_to_ui=self.app.call_from_thread, workspace_root_accessor=self._console_environment_root, rail_open_accessor=lambda: self._is_console_widget_displayed("console-right-rail"), on_snapshot=self._land_console_environment)` — built in `__init__`/mount alongside the other controllers
  - `_console_environment_root(self) -> str | None` — first entry of `_console_change_review_workspace_roots()`, else None
  - `_console_environment_expanded: set[str]` — in-memory expansion state
  - `_land_console_environment(self, snapshot)` — re-projects both sections (`project_environment_section(snapshot, frozenset(self._console_environment_expanded), now=...)`, `project_tasks_section(...)`), `sync_state`s `#console-environment-section` / `#console-tasks-section`, toggles their `styles.display` by `bool(rows)` — all inside `try/except NoMatches: return` (rail may be un-mounted)
  - `self.set_interval(10.0, self._console_environment.poll_tick)` in `on_mount` near the existing timers
  - `notify_rail_opened()` call added inside `action_toggle_console_inspector_rail` on the `opening` branch
  - a local-tier nudge (`request_refresh()`) at the end of `_run_coalesced_console_agent_fleet_sync` (agent work just changed the tree) — guarded so it no-ops while the rail is closed
  - an app-focus nudge (spec: "app focus regained schedules a local-tier refresh"): `@on(events.AppFocus)` handler on the screen (`from textual import events`) calling `self._console_environment.request_refresh()` — the controller's rail-open guard makes it free when the rail is closed
  - row-activation routing: extend the existing `@on(ConsoleInspectorSection.RowActivated)` handler — when `event.section_id in {"environment", "tasks"}` call `self._handle_console_environment_row(event.section_id, event.row_id)`:

```python
def _handle_console_environment_row(self, section_id: str, row_id: str) -> None:
    from tldw_chatbook.Chat.console_environment_state import (
        ENV_ROW_CHECKS_FIX, ENV_ROW_COMMIT_PUSH, ENV_ROW_PR_ADD, ENV_ROW_PR_OPEN,
        EXPANDABLE_ENV_ROWS, TASKS_ROW_ADD, TASKS_ROW_HEAD,
        failing_checks_text, pr_summary_text,
    )
    snapshot = self._console_environment.snapshot
    if row_id in EXPANDABLE_ENV_ROWS or row_id == TASKS_ROW_HEAD:
        self._console_environment_expanded.symmetric_difference_update({row_id})
        self._land_console_environment(snapshot)
        return
    if row_id in ("env-changes-review",):
        self._open_change_review()
        return
    if row_id == ENV_ROW_COMMIT_PUSH:
        self._open_change_review_current_mode()  # thin wrapper: _open_change_review + initial_current_mode=True
        return
    if row_id == ENV_ROW_PR_OPEN and snapshot.pr.url:
        try:
            self.app.open_url(snapshot.pr.url)
        except Exception:  # noqa: BLE001 -- never raise out of a handler
            logger.warning("environment: open_url failed")
        return
    if row_id == ENV_ROW_PR_ADD:
        self._insert_prompt_text_into_composer(pr_summary_text(snapshot.pr), replace=False)
        return
    if row_id == ENV_ROW_CHECKS_FIX:
        self._insert_prompt_text_into_composer(failing_checks_text(snapshot.pr), replace=False)
        return
    if row_id == TASKS_ROW_ADD and snapshot.tasks.branch_task is not None:
        bt = snapshot.tasks.branch_task
        self._insert_prompt_text_into_composer(
            f"Working on task-{bt.task_id}: {bt.title}\n{bt.path}", replace=False)
        return
```

  - `_open_change_review_current_mode()` — copy of `_open_change_review`'s body passing `initial_current_mode=True` (or refactor `_open_change_review` to accept and forward the kwarg — preferred, one seam)
  - `ViewAllRequested` with `section_id == "environment"` → `self._console_environment.request_refresh(include_net=True, force_net=True)`
  - `CollapseToggled` for the two section ids → `self._set_console_rail_preference(section_updates={event.section_id: event.open}, notify_on_failure=False)`

- [ ] **Step 1: Write the failing wiring tests** — this is the both-seams gate (a projection-only test passes with the screen unwired; that is how a prior fix shipped broken). Minimum set, all via `make_console_pilot` and monkeypatching the controller's gatherers to return canned states:

One fully worked example — the others follow the identical shape (canned snapshot → activate row → assert the seam):

```python
async def test_fix_row_inserts_failure_text_into_composer():
    async with make_console_pilot() as (pilot, screen):
        from tldw_chatbook.Chat.console_environment_state import (
            ENV_ROW_CHECKS_FIX, EnvironmentSnapshot, EnvSourceAvailability,
            GitEnvState, PrCheck, PrEnvState,
        )
        snapshot = EnvironmentSnapshot(
            git=GitEnvState(availability=EnvSourceAvailability.OK,
                            root="/w", branch="feat/task-1-x"),
            pr=PrEnvState(availability=EnvSourceAvailability.OK, number=7, title="T",
                          state="OPEN", url="https://x/pull/7",
                          checks=(PrCheck("ci-tests", "failure", "https://ci/1"),)),
        )
        screen._console_environment.snapshot = snapshot
        screen._land_console_environment(snapshot)
        await pilot.pause()
        screen._handle_console_environment_row("environment", ENV_ROW_CHECKS_FIX)
        await pilot.pause()
        from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.draft_text()
        assert "ci-tests" in draft and "https://ci/1" in draft
```

Remaining tests in the same file, same harness and canned-snapshot pattern:

```python
async def test_row_activation_toggles_expansion_and_rerenders():
    # _handle_console_environment_row("environment", ENV_ROW_CHANGES) on a snapshot with
    # 2 files -> after pilot.pause(), section DOM contains a row id ending "-row-1";
    # activating again removes the file rows.
async def test_commit_push_row_opens_change_review_in_current_mode():
    # monkeypatch screen._open_change_review to capture kwargs;
    # _handle_console_environment_row("environment", ENV_ROW_COMMIT_PUSH);
    # assert captured kwargs == {"initial_current_mode": True}.
async def test_collapse_toggle_persists_via_rail_preferences():
    # monkeypatch screen._set_console_rail_preference to capture; post
    # ConsoleInspectorSection.CollapseToggled(section_id="environment", open=False)
    # via the section's set_open(False); assert section_updates == {"environment": False}.
async def test_refresh_view_all_forces_net_tier():
    # monkeypatch screen._console_environment.request_refresh to capture; post
    # ViewAllRequested from #console-environment-section; assert force_net=True.
async def test_landing_for_stale_root_does_not_touch_sections():
    # point workspace_root_accessor at None, call _land_console_environment(snapshot)
    # directly; assert it returns without raising and the section keeps its prior rows.
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement** per the Produces block above. Keep every handler body synchronous and focus-neutral: do NOT move focus in any of these paths (burn-down lesson: focus decisions happen synchronously at the call site, and never onto the control that undoes the action).

- [ ] **Step 4: Run to verify PASS**, then the neighboring suites: `pytest Tests/UI/test_console_environment_wiring.py Tests/UI/test_console_right_rail.py Tests/UI/test_console_environment_section.py -v > .pytest-out.txt 2>&1; grep -E "[0-9]+ (passed|failed)" .pytest-out.txt`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(console): wire Environment panel workers, actions, and persistence"
```

---

### Task 14: Docs, preflight, live verification, close-out

**Files:**
- Modify: the `Docs/User_Guide/console/` page that documents the Inspect rail (find it: `git grep -ln "Inspect" Docs/User_Guide/console/`) — describe the Environment/Tasks/Agents sections, the row actions, the degradation behavior (gh optional), and refresh the page's "Verified against" stamp
- Modify: backlog umbrella task file (Implementation Notes + AC ticks)

- [ ] **Step 1: Update the User Guide page** (repo rule: UI PRs update the matching guide page). Cover: what each Environment row shows, that PR/CI rows require `gh` and hide without it, "vs last fetch" divergence honesty, the Refresh slot, and that the fleet section moved from the left rail.

- [ ] **Step 2: Regenerate + verify derived artifacts**

```bash
python -m tldw_chatbook.css.build_css
./scripts/preflight.sh
```
Expected: preflight all green. If the CSS bundle diff is only the `Generated:` timestamp line, revert it (noise; the sync checker ignores that line).

- [ ] **Step 3: Live 80×24 verification (REQUIRED — the harness cannot reproduce small-terminal clipping).** Use the project's `verify` skill recipe (tmux). Script: launch the app in a `tmux` session sized 80×24 in a workspace pointing at a real dirty git repo with a `backlog/` dir; open Console; press Alt+I; confirm (a) Environment section renders with real counts, (b) expanding Changes shows file rows and nothing clips the rail's LAST child, (c) collapse both new sections, restart the app, confirm collapse persisted, (d) with `gh` authenticated on a branch with an open PR: PR + checks rows appear; with `PATH` stripped of gh: rows silently absent. Capture a screenshot/paste of each state into the backlog task's Implementation Notes.

- [ ] **Step 4: Targeted regression pass** — every test file this branch touched plus a `--collect-only` sweep of `Tests/UI Tests/Chat Tests/Workspaces` to catch import breakage: `pytest Tests/UI Tests/Chat Tests/Workspaces --collect-only -q > .pytest-out.txt 2>&1; tail -3 .pytest-out.txt` (expect a clean collection count, zero errors).

- [ ] **Step 5: Close out** — tick the umbrella task's ACs, write Implementation Notes (approach, deviations, evidence incl. the live-run captures), commit docs + task file, push, open the PR against `dev` with the standard footer. If review or live verification surfaced a generalisable trap, add it to the relevant `backlog/docs/lessons-*.md` with the incident.

```bash
git add Docs/ backlog/
git commit -m "docs(console): Environment panel guide + close-out"
git push -u origin feat/console-inspector-environment
```
