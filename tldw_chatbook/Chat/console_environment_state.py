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
