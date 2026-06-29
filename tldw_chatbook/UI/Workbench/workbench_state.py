"""Immutable state snapshots for shared Workbench widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Density = Literal["normal", "compact"]
WorkbenchStatus = Literal[
    "ready",
    "running",
    "blocked",
    "error",
    "paused",
    "empty",
    "loading",
]


def _classes(*names: str) -> str:
    """Return a stable class string without empty parts."""
    return " ".join(name for name in names if name)


@dataclass(frozen=True)
class WorkbenchAction:
    """A visible Workbench command."""

    id: str
    label: str
    tooltip: str = ""
    disabled: bool = False
    primary: bool = False

    @property
    def css_classes(self) -> str:
        """Return TCSS classes for the action state."""
        return _classes(
            "workbench-action",
            "is-primary" if self.primary else "",
            "is-disabled" if self.disabled else "",
        )


@dataclass(frozen=True)
class WorkbenchMode:
    """A top-level Workbench mode chip."""

    id: str
    label: str
    active: bool = False
    status: WorkbenchStatus = "ready"

    @property
    def css_classes(self) -> str:
        """Return TCSS classes for the mode state."""
        return _classes(
            "workbench-mode",
            "is-active" if self.active else "",
            f"status-{self.status}",
        )


@dataclass(frozen=True)
class WorkbenchHeaderState:
    """Header copy and readiness state for a Workbench destination."""

    title: str
    subtitle: str = ""
    status: WorkbenchStatus = "ready"
    density: Density = "normal"


@dataclass(frozen=True)
class WorkbenchPaneState:
    """Visible state for a Workbench pane frame."""

    id: str
    title: str
    status: WorkbenchStatus = "ready"
    collapsed: bool = False


@dataclass(frozen=True)
class RecoveryState:
    """Blocked or empty-state recovery copy and optional action."""

    title: str
    body: str
    action: WorkbenchAction | None = None
    visible: bool = True


@dataclass(frozen=True)
class WorkbenchState:
    """Complete visible state snapshot for a Workbench frame."""

    header: WorkbenchHeaderState
    modes: tuple[WorkbenchMode, ...] = ()
    actions: tuple[WorkbenchAction, ...] = ()
    panes: tuple[WorkbenchPaneState, ...] = ()
    recovery: RecoveryState | None = None
    density: Density = "normal"
    route_id: str = ""

    def __post_init__(self) -> None:
        """Validate action identity before widgets render the state."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for action in self.actions:
            if action.id in seen:
                duplicates.append(action.id)
                continue
            seen.add(action.id)
        if duplicates:
            duplicate_ids = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"duplicate action id: {duplicate_ids}")
