"""Home focus canvas: the selected work item and its actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from tldw_chatbook.Home.dashboard_state import HomeCanvasState


class HomeCanvas(Vertical):
    """Render the selected item (or next best action) with its controls.

    Attributes:
        canvas: Current canvas display state.
        action_button_factory: Callable building an action Button so the
            screen can keep its fallback-press wiring (HomeActionButton).
    """

    def __init__(
        self,
        canvas: HomeCanvasState,
        *,
        action_button_factory: Callable[[str, str, bool], Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        self.action_button_factory = action_button_factory
        self.styles.width = "13fr"
        self.styles.min_width = 40

    def sync_state(self, canvas: HomeCanvasState) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        """Render the canvas title, detail lines, actions, and callout.

        Returns:
            ComposeResult for the focus canvas.
        """
        yield Static(
            self.canvas.title,
            id="home-canvas-title",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            "\n".join(self.canvas.lines),
            id="home-canvas-lines",
            markup=False,
        )
        action_row = Vertical(id="home-canvas-actions")
        action_row.styles.height = "auto"
        with action_row:
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                for control in self.canvas.actions:
                    yield self.action_button_factory(
                        control.label,
                        control.control_id,
                        control.control_id == self.canvas.primary_control_id,
                    )
                if self.canvas.next_action_is_canvas:
                    yield self.action_button_factory(
                        self.canvas.next_action.label, "home-primary-action", False
                    )
        if not self.canvas.next_action_is_canvas:
            yield Static(
                f"Next: {self.canvas.next_action.label} — {self.canvas.next_action.reason}",
                id="home-next-action-callout",
                classes="home-rail-empty-copy",
                markup=False,
            )
            yield self.action_button_factory(
                self.canvas.next_action.label, "home-primary-action", False
            )
