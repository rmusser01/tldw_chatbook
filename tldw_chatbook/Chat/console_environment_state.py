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
