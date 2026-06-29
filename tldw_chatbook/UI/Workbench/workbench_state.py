"""Immutable state snapshots for shared Workbench widgets."""

from __future__ import annotations

from dataclasses import dataclass
import re
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


def normalize_workbench_id(value: str) -> str:
    """Return the Textual-safe identity segment used by Workbench widgets."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    normalized = normalized.strip("-")
    return normalized or "item"


def _raise_on_duplicate_ids(
    values: tuple[str, ...],
    *,
    label: str,
    normalized: bool = False,
) -> None:
    """Raise if values contain duplicate exact or normalized IDs."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = normalize_workbench_id(value) if normalized else value
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
    if duplicates:
        duplicate_ids = ", ".join(sorted(set(duplicates)))
        normalized_label = "normalized " if normalized else ""
        raise ValueError(f"duplicate {normalized_label}{label} id: {duplicate_ids}")


def _raise_on_non_canonical_ids(values: tuple[str, ...], *, label: str) -> None:
    """Raise if values are not already safe for Textual widget IDs."""
    invalid = sorted(
        value for value in values if value != normalize_workbench_id(value)
    )
    if invalid:
        raise ValueError(f"non-canonical {label} id: {', '.join(invalid)}")


def _raise_on_non_canonical_id(value: str, *, label: str) -> None:
    """Raise if one value is not already safe for Textual widget IDs."""
    _raise_on_non_canonical_ids((value,), label=label)


@dataclass(frozen=True)
class WorkbenchAction:
    """A visible Workbench command."""

    id: str
    label: str
    tooltip: str = ""
    disabled: bool = False
    primary: bool = False

    def __post_init__(self) -> None:
        """Validate public action identity before widgets render it."""
        _raise_on_non_canonical_id(self.id, label="action")

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

    def __post_init__(self) -> None:
        """Validate public mode identity before widgets render it."""
        _raise_on_non_canonical_id(self.id, label="mode")

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

    def __post_init__(self) -> None:
        """Validate public pane identity before widgets render it."""
        _raise_on_non_canonical_id(self.id, label="pane")


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
        action_ids = tuple(action.id for action in self.actions)
        mode_ids = tuple(mode.id for mode in self.modes)
        pane_ids = tuple(pane.id for pane in self.panes)
        route_ids = (self.route_id,) if self.route_id else ()

        _raise_on_duplicate_ids(action_ids, label="action")
        _raise_on_duplicate_ids(mode_ids, label="mode")
        _raise_on_duplicate_ids(pane_ids, label="pane")

        _raise_on_duplicate_ids(action_ids, label="action", normalized=True)
        _raise_on_duplicate_ids(mode_ids, label="mode", normalized=True)
        _raise_on_duplicate_ids(pane_ids, label="pane", normalized=True)

        _raise_on_non_canonical_ids(action_ids, label="action")
        _raise_on_non_canonical_ids(mode_ids, label="mode")
        _raise_on_non_canonical_ids(pane_ids, label="pane")
        _raise_on_non_canonical_ids(route_ids, label="route")
