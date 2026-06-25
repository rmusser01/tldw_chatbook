"""Shared keyboard focus helpers for destination workbench panes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from textual.css.query import QueryError
from textual.widget import Widget


@dataclass(frozen=True)
class WorkbenchPaneTarget:
    """A major workbench pane and the preferred child to focus inside it."""

    pane_id: str
    preferred_focus_ids: tuple[str, ...]


def focus_relative_workbench_pane(
    screen: Widget,
    targets: Iterable[WorkbenchPaneTarget],
    *,
    direction: int,
) -> Widget | None:
    """Focus the next available workbench pane target.

    Args:
        screen: Mounted screen that owns the workbench panes.
        targets: Ordered pane targets for the screen.
        direction: Positive for next, negative for previous.

    Returns:
        The focused widget, or ``None`` when no available target exists.
    """

    available_targets = _available_targets(screen, targets)
    if not available_targets:
        return None

    focused = getattr(getattr(screen, "app", None), "focused", None)
    current_index = _focused_pane_index(focused, available_targets)
    if current_index is None:
        next_index = 0 if direction >= 0 else len(available_targets) - 1
    else:
        next_index = (current_index + (1 if direction >= 0 else -1)) % len(
            available_targets
        )

    _pane, focus_target = available_targets[next_index]
    focus_target.focus()
    return focus_target


def _available_targets(
    screen: Widget,
    targets: Iterable[WorkbenchPaneTarget],
) -> list[tuple[Widget, Widget]]:
    available: list[tuple[Widget, Widget]] = []
    for target in targets:
        pane = _query_by_id(screen, target.pane_id)
        if pane is None or not _is_available(pane):
            continue
        focus_target = _resolve_focus_target(pane, target.preferred_focus_ids)
        if focus_target is None or not _is_available(focus_target):
            continue
        available.append((pane, focus_target))
    return available


def _resolve_focus_target(pane: Widget, preferred_focus_ids: tuple[str, ...]) -> Widget | None:
    for focus_id in preferred_focus_ids:
        if pane.id == focus_id and _is_available(pane):
            return pane
        widget = _query_by_id(pane, focus_id)
        if (
            widget is not None
            and not getattr(widget, "disabled", False)
            and _is_available(widget)
        ):
            return widget
    if getattr(pane, "can_focus", False):
        return pane
    return None


def _focused_pane_index(
    focused: Widget | None,
    available_targets: list[tuple[Widget, Widget]],
) -> int | None:
    if focused is None:
        return None
    for index, (pane, focus_target) in enumerate(available_targets):
        if focused is pane or focused is focus_target:
            return index
        if _is_descendant_of(focused, pane):
            return index
    return None


def _is_descendant_of(widget: Widget, ancestor: Widget) -> bool:
    parent = widget.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def _is_available(widget: Widget) -> bool:
    current: Widget | None = widget
    while current is not None:
        if getattr(current, "display", True) is False:
            return False
        if getattr(getattr(current, "styles", None), "display", None) == "none":
            return False
        current = current.parent
    return True


def _query_by_id(root: Widget, widget_id: str) -> Widget | None:
    selector = f"#{widget_id.lstrip('#')}"
    try:
        return root.query_one(selector, Widget)
    except QueryError:
        return None
