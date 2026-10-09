"""Shared keyboard focus helpers for destination workbench panes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from textual.css.query import NoMatches, QueryError
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


def _resolve_focus_target(
    pane: Widget, preferred_focus_ids: tuple[str, ...]
) -> Widget | None:
    # TASK-34000.8 AC#3: prefer the first control that is ON SCREEN over
    # one painted past the terminal's edge -- F6 used to focus the note
    # editor's Save while its region sat beyond the right edge of a
    # 120-column terminal, with the title field in plain view under it.
    # The off-screen control is only PASSED OVER, never ruled out: when no
    # preferred control is on screen the first focusable one is still the
    # landing, because focusing it may be what reveals its pane (the narrow
    # Artifacts stage keeps its reader collapsed until F6 focuses its body:
    # ``test_narrow_f6_reveals_reader_and_returns_through_items_grip``).
    fallback: Widget | None = None
    for focus_id in preferred_focus_ids:
        if pane.id == focus_id and _is_focusable(pane):
            return pane
        widget = _query_by_id(pane, focus_id)
        if widget is None or not _is_focusable(widget):
            continue
        if widget_is_painted_off_screen(widget):
            if fallback is None:
                fallback = widget
            continue
        return widget
    if fallback is not None:
        return fallback
    if _is_focusable(pane):
        return pane
    return None


def _has_region(widget: Widget) -> bool:
    region = getattr(widget, "region", None)
    return region is not None and region.width > 0 and region.height > 0


def widget_is_painted_off_screen(widget: Widget) -> bool:
    """Whether ``widget`` has a region that does not fit inside its screen.

    The laid-out-but-unseeable case: a displayed, visible, focusable
    control whose region lies partly or wholly past the terminal's edge
    (the Library note editor's Save at 120x36 before TASK-34000.8). A
    widget with no region at all answers False -- it is not laid out (or
    its pane is collapsed), which focus may legitimately reveal.

    Args:
        widget: A mounted widget.

    Returns:
        True only when the region is non-empty and the screen's region does
        not contain it.
    """
    if not _has_region(widget):
        return False
    try:
        screen = widget.screen
    except Exception:  # NoScreen -- not attached to any screen
        return False
    return not screen.region.contains_region(widget.region)


def widget_has_visible_region(widget: Widget) -> bool:
    """Whether ``widget`` occupies a non-empty region inside its screen.

    The geometry half of "can the user see this control" for a caller that
    ADVERTISES it (the footer's Enter chip): no region, an off-screen
    region and no screen all answer False.

    Args:
        widget: A mounted widget.

    Returns:
        True when the widget's region is non-empty and lies inside its
        screen's region.
    """
    return _has_region(widget) and not widget_is_painted_off_screen(widget)


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
        # ``visibility: hidden`` keeps a widget's cells but paints nothing
        # and drops it from the focus chain (Textual 8.2.8); it is not a
        # landing either (TASK-34000.8: Discard new note is hidden this way
        # so the task row keeps its width).
        if getattr(current, "visible", True) is False:
            return False
        current = current.parent
    return True


def _is_focusable(widget: Widget) -> bool:
    return (
        bool(getattr(widget, "can_focus", False))
        and not getattr(widget, "disabled", False)
        and _is_available(widget)
    )


def _query_by_id(root: Widget, widget_id: str) -> Widget | None:
    selector = f"#{widget_id.lstrip('#')}"
    try:
        return root.query_one(selector, Widget)
    except (NoMatches, QueryError):
        return None
