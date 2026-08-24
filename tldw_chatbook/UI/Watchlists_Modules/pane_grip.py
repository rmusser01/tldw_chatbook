"""Clickable ASCII grips for collapsible Watchlists panes."""

from __future__ import annotations

from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button

from .region_layout import PANE_GRIP_WIDTH, Region


_PANE_NAMES: dict[Region, str] = {
    Region.LEFT_RAIL: "Navigation",
    Region.ITEMS: "Feed Items",
    Region.RIGHT_RAIL: "Inspector",
}


class RegionToggled(Message):
    """Request to toggle one Watchlists pane."""

    def __init__(self, region: Region) -> None:
        super().__init__()
        self.region = region


class WatchlistsPaneGrip(Button):
    """A focused five-column control for expanding or collapsing a pane."""

    expanded: reactive[bool] = reactive(False)

    def __init__(
        self,
        region: Region,
        expanded: bool,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            id=id,
            classes="watchlists-pane-grip",
            compact=True,
        )
        self.pane_region = region
        self.styles.width = PANE_GRIP_WIDTH
        self.styles.min_width = PANE_GRIP_WIDTH
        self.styles.max_width = PANE_GRIP_WIDTH
        self.styles.line_pad = 0
        self.set_reactive(WatchlistsPaneGrip.expanded, expanded)
        self._relabel()

    def watch_expanded(self, _expanded: bool) -> None:
        """Refresh the arrow and action copy without replacing the widget."""
        self._relabel()

    def _relabel(self) -> None:
        pane_name = _PANE_NAMES[self.pane_region]
        action = "Collapse" if self.expanded else "Expand"
        points_right = self.pane_region is Region.RIGHT_RAIL
        if not self.expanded:
            points_right = not points_right

        self.label = "--->" if points_right else "<---"
        copy = f"{action} {pane_name}"
        self.tooltip = copy
        self._name = copy

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate native mouse and keyboard activation into one message."""
        event.stop()
        self.post_message(RegionToggled(self.pane_region))
