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


from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSectionState,
    InspectorSectionRow,
)

ENVIRONMENT_SECTION_ID = "environment"
TASKS_SECTION_ID = "tasks"

ENV_ROW_CHANGES = "env-changes"
ENV_ROW_ERROR = "env-error"
ENV_ROW_EMPTY = "env-empty"
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

# Column budget for the Environment section's header summary.
#
# The section header is `title (1fr) + summary (auto) + toggle (3)` on ONE
# line (`console_inspector_section.py::compose`). The summary Static is
# `width: auto`, so an unbudgeted summary takes whatever it wants and the
# 1fr title -- "Environment" -- is starved to nothing. Live-verified at both
# 80x24 and 200x50: the Inspect rail's body is ~34 columns wide regardless
# of terminal size, so any branch name over ~33 characters (routine here:
# `feat/console-inspector-environment`) squeezed the title AND the collapse
# chevron off the header entirely.
#
# 18 leaves 34 - 18 - 3 = 13 columns for the title, which fits
# "Environment" (11) with slack. Truncation lives HERE, in the pure
# projection, not in the widget -- this arc's rule, so it is testable
# without a running app.
ENV_SUMMARY_BUDGET = 18


def _git_status_class(stale: bool) -> str:
    return "blocked" if stale else ""


def _ellipsize(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` columns, marking the cut with a trailing "…".

    Head-anchored (keeps the start, drops the tail) because this repo's
    branch names lead with the identifying fragment -- ``feat/task-31450-…``
    -- so the head is what tells the branches apart.

    Args:
        text: Text to fit.
        limit: Maximum column count; ``<= 0`` yields ``""``.

    Returns:
        ``text`` unchanged when it already fits, else its head plus "…".
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1] + "…"


def environment_summary(git: GitEnvState, *, budget: int = ENV_SUMMARY_BUDGET) -> str:
    """Build the Environment header summary, fitted to ``budget`` columns.

    The signed ± counts are the priority half -- they are the number the
    user is scanning for and they are already compacted
    (``compact_count``), so they are never truncated. Whatever the counts
    leave over is the branch fragment's budget; when that is too small to
    say anything (under two columns, i.e. not even one character plus the
    ellipsis) the branch is dropped and the counts stand alone.

    Args:
        git: The git tier state to describe.
        budget: Column budget for the whole summary.

    Returns:
        A summary string of at most ``budget`` columns (or exactly the
        counts, when the counts alone already exceed it).
    """
    counts = signed_change_counts(git.adds, git.dels)
    room = budget - len(counts) - 1  # -1 for the separating space
    if room < 2:
        return counts
    return f"{_ellipsize(_branch_primary(git), room)} {counts}"


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
    if git.availability is EnvSourceAvailability.ERROR:
        # ERROR is NOT "there is nothing here" -- it is "we could not look".
        # Rendering it as the NOT_APPLICABLE empty state told a user whose
        # git call timed out (or whose tier had backed off after 3 failures)
        # that their repository was not a git workspace, with no hint that
        # the Refresh slot would revive it.
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(
                row_id=ENV_ROW_ERROR,
                primary_text="Environment unavailable — Refresh to retry",
                status="blocked",
            ),),
            summary="",
        )
    if git.availability is not EnvSourceAvailability.OK:
        return ConsoleInspectorSectionState(
            rows=(InspectorSectionRow(row_id=ENV_ROW_EMPTY, primary_text="No git workspace"),),
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
            count = len(git.files)
            label = f"Commit or push · {count} file" + ("s" if count != 1 else "")
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

    return ConsoleInspectorSectionState(
        rows=tuple(rows), summary=environment_summary(git)
    )


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
