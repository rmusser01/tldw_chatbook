"""Focus-order helpers for Workbench destinations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkbenchFocusRegistry:
    """Visible-pane focus order for a Workbench destination."""

    pane_order: tuple[str, ...]

    def next_after(
        self,
        current: str | None,
        *,
        hidden: set[str] | frozenset[str] = frozenset(),
    ) -> str | None:
        """Return the next visible pane after ``current``."""
        visible = tuple(pane for pane in self.pane_order if pane not in hidden)
        if not visible:
            return None
        if current not in visible:
            return visible[0]
        index = visible.index(current)
        return visible[(index + 1) % len(visible)]

    def previous_before(
        self,
        current: str | None,
        *,
        hidden: set[str] | frozenset[str] = frozenset(),
    ) -> str | None:
        """Return the previous visible pane before ``current``."""
        visible = tuple(pane for pane in self.pane_order if pane not in hidden)
        if not visible:
            return None
        if current not in visible:
            return visible[-1]
        index = visible.index(current)
        return visible[(index - 1) % len(visible)]
