"""Home focus canvas: the selected work item and its actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from tldw_chatbook.Home.dashboard_state import HOME_PRIMARY_ACTION_ID, HomeCanvasState


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

        B3 (task-282): when the item/toolbar shape is unchanged (same
        title, actions, next-action, and primary control -- i.e. the same
        item is still shown) but the detail ``lines`` differ (e.g. a
        background count refresh updating the idle canvas's content-counts
        line), patch the lines Static in place instead of tearing down and
        remounting the whole canvas. A different selected item, or any
        change to the action toolbar, still gets a full recompose.

        Args:
            canvas: Latest canvas display state.

        Returns:
            None.
        """
        previous = self.canvas
        self.canvas = canvas
        if canvas == previous:
            # Nothing actually changed (e.g. a background refresh whose
            # data didn't move the needle) -- avoid a no-op layout pass.
            return
        if (
            canvas.title == previous.title
            and canvas.actions == previous.actions
            and canvas.next_action == previous.next_action
            and canvas.next_action_is_canvas == previous.next_action_is_canvas
            and canvas.primary_control_id == previous.primary_control_id
        ):
            try:
                self.query_one("#home-canvas-lines", Static).update(
                    "\n".join(canvas.lines)
                )
                return
            except Exception:
                # Defensive: fall through to a full recompose rather than
                # leave the canvas showing stale lines.
                pass
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
                        self.canvas.next_action.label,
                        HOME_PRIMARY_ACTION_ID,
                        # T190: the ready-idle canvas makes the next-action
                        # button itself the primary "Start a conversation"
                        # control, so it must honor primary emphasis too.
                        self.canvas.primary_control_id == HOME_PRIMARY_ACTION_ID,
                    )
        if not self.canvas.next_action_is_canvas:
            yield Static(
                f"Next: {self.canvas.next_action.label} — {self.canvas.next_action.reason}",
                id="home-next-action-callout",
                classes="home-rail-empty-copy",
                markup=False,
            )
            yield self.action_button_factory(
                self.canvas.next_action.label,
                HOME_PRIMARY_ACTION_ID,
                self.canvas.primary_control_id == HOME_PRIMARY_ACTION_ID,
            )
