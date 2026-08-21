"""Disposable terminal-native floating view for the app-owned Persona Buddy."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.geometry import Offset
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Persona_Buddy.preferences import (
    PERSONA_BUDDY_UNPOSITIONED_COORDINATE,
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
)

_DEFAULT_WIDTH = 28
_DEFAULT_HEIGHT = 12
_MIN_WIDTH = 18
_MIN_HEIGHT = 6
_COMPACT_HEIGHT = 3
_POLL_SECONDS = 0.10
_FRAME_SECONDS = 0.02


class PersonaBuddyWidget(Widget, can_focus=True):
    """Paint one controller snapshot and own only disposable view mechanics."""

    BINDINGS = [
        Binding("h", "move_left", "Move left", show=False),
        Binding("j", "move_down", "Move down", show=False),
        Binding("k", "move_up", "Move up", show=False),
        Binding("l", "move_right", "Move right", show=False),
        Binding("H", "shrink_width", "Narrower", show=False),
        Binding("L", "grow_width", "Wider", show=False),
        Binding("J", "grow_height", "Taller", show=False),
        Binding("K", "shrink_height", "Shorter", show=False),
        Binding("0", "reset_geometry", "Reset", show=False),
        Binding("c", "toggle_collapse", "Collapse", show=False),
        Binding("x", "close", "Close", show=False),
    ]

    BUNDLED_CSS = """
    PersonaBuddyWidget {
        position: absolute;
        overlay: screen;
        layer: overlay;
        width: 28;
        height: 12;
        min-width: 1;
        min-height: 1;
        padding: 0 1;
        background: $panel;
        color: $text;
        border: round $accent;
        overflow: hidden hidden;
    }

    PersonaBuddyWidget:focus {
        outline: heavy $accent;
    }

    PersonaBuddyWidget #persona-buddy-header {
        width: 100%;
        height: 3;
        layout: horizontal;
        background: $surface;
    }

    PersonaBuddyWidget #persona-buddy-title {
        width: 1fr;
        height: 3;
        padding: 1 0 0 1;
        color: $text;
        text-style: bold;
    }

    PersonaBuddyWidget .persona-buddy-control {
        width: auto;
        min-width: 7;
        height: 3;
        padding: 0 1;
        margin: 0;
        border: none;
        background: $surface-darken-1;
        color: $text-muted;
    }

    PersonaBuddyWidget .persona-buddy-control:focus {
        outline: heavy $accent;
        color: $text;
    }

    PersonaBuddyWidget #persona-buddy-frame {
        width: 100%;
        height: 1fr;
        content-align: center middle;
        color: $text;
    }

    PersonaBuddyWidget #persona-buddy-status,
    PersonaBuddyWidget #persona-buddy-hints {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-align: center;
    }

    PersonaBuddyWidget.persona-buddy-collapsed,
    PersonaBuddyWidget.persona-buddy-compact {
        height: 3;
        padding: 0;
    }

    PersonaBuddyWidget.persona-buddy-collapsed #persona-buddy-frame,
    PersonaBuddyWidget.persona-buddy-collapsed #persona-buddy-status,
    PersonaBuddyWidget.persona-buddy-collapsed #persona-buddy-hints,
    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-frame,
    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-status,
    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-hints,
    PersonaBuddyWidget.persona-buddy-collapsed .persona-buddy-control,
    PersonaBuddyWidget.persona-buddy-compact .persona-buddy-control {
        display: none;
    }

    PersonaBuddyWidget.persona-buddy-collapsed #persona-buddy-header,
    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-header,
    PersonaBuddyWidget.persona-buddy-collapsed #persona-buddy-title,
    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-title {
        height: 1;
        width: 100%;
        padding: 0;
        content-align: center middle;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        controller: Any,
        view_generation: int,
        reconcile: Callable[[], Awaitable[None] | None],
    ) -> None:
        super().__init__(id="persona-buddy-widget")
        self._controller = controller
        self.view_generation = view_generation
        self._reconcile = reconcile
        self._snapshot: Any = None
        self._visual_identity: object | None = None
        self._working_preferences: PersonaBuddyPreferences = (
            controller.current_preferences()
        )
        self.frame_index = 0
        self.snapshot_polling_active = False
        self._poll_timer: Timer | None = None
        self._frame_timer: Timer | None = None
        self._interaction: tuple[str, int, int, PersonaBuddyGeometry] | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="persona-buddy-header"):
            yield Static("Buddy", id="persona-buddy-title")
            yield Button(
                "Fold", id="persona-buddy-collapse", classes="persona-buddy-control"
            )
            yield Button(
                "Close", id="persona-buddy-close", classes="persona-buddy-control"
            )
        yield Static("Visual pending", id="persona-buddy-frame")
        yield Static("State: idle", id="persona-buddy-status")
        yield Static("h/j/k/l move · Shift resize · 0 reset", id="persona-buddy-hints")

    def on_mount(self) -> None:
        """Start bounded snapshot and frame polling without changing focus."""

        self.snapshot_polling_active = True
        self.border_title = "Persona Buddy"
        self._apply_geometry(self._working_preferences.geometry)
        self.refresh_from_controller()
        self._poll_timer = self.set_interval(
            _POLL_SECONDS, self.refresh_from_controller
        )
        self._frame_timer = self.set_interval(_FRAME_SECONDS, self.advance_frame)

    def on_unmount(self) -> None:
        """Release capture and stop view-owned timers during any teardown path."""

        self.snapshot_polling_active = False
        self.release_interaction_capture()
        if self._poll_timer is not None:
            self._poll_timer.stop()
        if self._frame_timer is not None:
            self._frame_timer.stop()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-clamp geometry whenever the terminal viewport changes."""

        setter = getattr(self._controller, "set_viewport_generation", None)
        if callable(setter):
            snapshot = self._controller.snapshot()
            setter(snapshot.viewport_generation + 1)
        self._apply_geometry(self._working_preferences.geometry)

    def refresh_from_controller(self) -> None:
        """Refresh paint/state from an immutable snapshot without touching focus."""

        if not self.is_attached:
            return
        snapshot = self._controller.snapshot()
        self._snapshot = snapshot
        preferences = self._controller.current_preferences()
        if self._interaction is not None:
            preferences = replace(
                preferences,
                geometry=self._working_preferences.geometry,
            )
        self._working_preferences = preferences
        self._sync_compact_state(snapshot.collapsed)

        visual = snapshot.visual
        visual_identity = (
            id(visual),
            getattr(visual, "requested_state", None),
            getattr(visual, "resolved_state", None),
        )
        if visual_identity != self._visual_identity:
            self._visual_identity = visual_identity
            self.frame_index = 0

        title = self.query_one("#persona-buddy-title", Static)
        title.update("Buddy")
        self.query_one("#persona-buddy-status", Static).update(
            f"State: {snapshot.state.replace('_', ' ')}"
        )
        self._paint_frame()

    def advance_frame(self) -> None:
        """Advance animation only while a full visible animated view is painted."""

        snapshot = self._snapshot
        if (
            snapshot is None
            or not self.display
            or snapshot.collapsed
            or self.has_class("persona-buddy-compact")
        ):
            return
        visual = snapshot.visual
        frames = getattr(visual, "frames", ()) if visual is not None else ()
        if not getattr(visual, "animate", False) or len(frames) < 2:
            return
        self.frame_index = (self.frame_index + 1) % len(frames)
        self._paint_frame()

    def _paint_frame(self) -> None:
        if not self.is_attached:
            return
        target = self.query_one("#persona-buddy-frame", Static)
        visual = getattr(self._snapshot, "visual", None)
        frames = getattr(visual, "frames", ()) if visual is not None else ()
        if frames:
            self.frame_index = min(self.frame_index, len(frames) - 1)
            target.update(frames[self.frame_index].renderable)
            return
        reason = getattr(visual, "reason", None)
        target.update("Visual unavailable" if reason else "Visual pending")

    def _sync_compact_state(self, collapsed: bool) -> None:
        size = self.screen.size
        tiny = size.width < _MIN_WIDTH or size.height < _MIN_HEIGHT
        self.set_class(tiny, "persona-buddy-compact")
        self.set_class(collapsed and not tiny, "persona-buddy-collapsed")
        collapse = self.query_one("#persona-buddy-collapse", Button)
        collapse.label = "Open" if collapsed else "Fold"
        self._apply_geometry(self._working_preferences.geometry)

    def _clamped_geometry(self, geometry: PersonaBuddyGeometry) -> PersonaBuddyGeometry:
        viewport = self.screen.size
        tiny = viewport.width < _MIN_WIDTH or viewport.height < _MIN_HEIGHT
        if tiny:
            width = max(1, viewport.width)
            height = min(_COMPACT_HEIGHT, max(1, viewport.height))
        else:
            width = min(max(_MIN_WIDTH, geometry.width), viewport.width)
            if self._snapshot and self._snapshot.collapsed:
                height = min(_COMPACT_HEIGHT, viewport.height)
            else:
                height = min(max(_MIN_HEIGHT, geometry.height), viewport.height)
        max_x = max(0, viewport.width - width)
        max_y = max(0, viewport.height - height)
        x = (
            max_x
            if geometry.x == PERSONA_BUDDY_UNPOSITIONED_COORDINATE
            else min(geometry.x, max_x)
        )
        y = (
            max_y
            if geometry.y == PERSONA_BUDDY_UNPOSITIONED_COORDINATE
            else min(geometry.y, max_y)
        )
        return PersonaBuddyGeometry(x=x, y=y, width=width, height=height)

    def _apply_geometry(self, geometry: PersonaBuddyGeometry) -> None:
        if not self.is_attached:
            return
        clamped = self._clamped_geometry(geometry)
        self.styles.width = clamped.width
        self.styles.height = clamped.height
        self.absolute_offset = Offset(clamped.x, clamped.y)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Arm header drag or lower-right resize for the real terminal event shape."""

        if event.button != 1 or not self.is_attached:
            return
        screen_x = int(event.screen_x if event.screen_x is not None else event.x)
        screen_y = int(event.screen_y if event.screen_y is not None else event.y)
        region = self.region
        resize = screen_x >= region.right - 2 and screen_y >= region.bottom - 1
        if not resize and screen_y >= region.y + 3:
            return
        mode = "resize" if resize else "drag"
        geometry = PersonaBuddyGeometry(
            x=self.absolute_offset.x,
            y=self.absolute_offset.y,
            width=region.width,
            height=region.height,
        )
        self._interaction = (mode, screen_x, screen_y, geometry)
        self.focus(scroll_visible=False)
        self.capture_mouse(True)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Apply bounded working geometry without writing config per mouse move."""

        if self._interaction is None:
            return
        mode, origin_x, origin_y, original = self._interaction
        screen_x = int(event.screen_x if event.screen_x is not None else event.x)
        screen_y = int(event.screen_y if event.screen_y is not None else event.y)
        dx = screen_x - origin_x
        dy = screen_y - origin_y
        if mode == "drag":
            candidate = replace(
                original, x=max(0, original.x + dx), y=max(0, original.y + dy)
            )
        else:
            candidate = replace(
                original,
                width=max(1, original.width + dx),
                height=max(1, original.height + dy),
            )
        clamped = self._clamped_geometry(candidate)
        self._working_preferences = replace(self._working_preferences, geometry=clamped)
        self._apply_geometry(clamped)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Finish one interaction, release capture, and persist exactly once."""

        if self._interaction is None:
            self.release_interaction_capture()
            return
        self._interaction = None
        self.release_interaction_capture()
        self._schedule_preferences(self._working_preferences, reconcile=False)
        event.stop()

    def release_interaction_capture(self) -> None:
        """Release mouse capture safely for removal, recompose, and mouse-up."""

        self._interaction = None
        try:
            self.release_mouse()
        except Exception:
            if self.is_attached:
                self.app.capture_mouse(None)

    def _geometry_action(
        self, *, dx: int = 0, dy: int = 0, dw: int = 0, dh: int = 0
    ) -> None:
        geometry = self._clamped_geometry(self._working_preferences.geometry)
        candidate = replace(
            geometry,
            x=max(0, geometry.x + dx),
            y=max(0, geometry.y + dy),
            width=max(1, geometry.width + dw),
            height=max(1, geometry.height + dh),
        )
        clamped = self._clamped_geometry(candidate)
        self._working_preferences = replace(self._working_preferences, geometry=clamped)
        self._apply_geometry(clamped)
        self._schedule_preferences(self._working_preferences, reconcile=False)

    def action_move_left(self) -> None:
        self._geometry_action(dx=-1)

    def action_move_down(self) -> None:
        self._geometry_action(dy=1)

    def action_move_up(self) -> None:
        self._geometry_action(dy=-1)

    def action_move_right(self) -> None:
        self._geometry_action(dx=1)

    def action_shrink_width(self) -> None:
        self._geometry_action(dw=-1)

    def action_grow_width(self) -> None:
        self._geometry_action(dw=1)

    def action_grow_height(self) -> None:
        self._geometry_action(dh=1)

    def action_shrink_height(self) -> None:
        self._geometry_action(dh=-1)

    def action_reset_geometry(self) -> None:
        geometry = PersonaBuddyGeometry(width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT)
        clamped = self._clamped_geometry(geometry)
        self._working_preferences = replace(self._working_preferences, geometry=clamped)
        self._apply_geometry(clamped)
        self._schedule_preferences(self._working_preferences, reconcile=False)

    def action_toggle_collapse(self) -> None:
        preferences = replace(
            self._working_preferences,
            collapsed=not self._working_preferences.collapsed,
        )
        self._working_preferences = preferences
        self._schedule_preferences(preferences, reconcile=False)

    def action_close(self) -> None:
        self._schedule_awaitable(self.close_and_persist())

    async def close_and_persist(self) -> None:
        """Persist the explicit close and reconcile only this app's active screen."""

        preferences = replace(self._working_preferences, open=False)
        self._working_preferences = preferences
        await self._controller.update_preferences(preferences)
        pending = self._reconcile()
        if inspect.isawaitable(pending):
            await pending

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "persona-buddy-collapse":
            self.action_toggle_collapse()
        elif event.button.id == "persona-buddy-close":
            self.action_close()
        event.stop()

    def _schedule_preferences(
        self,
        preferences: PersonaBuddyPreferences,
        *,
        reconcile: bool,
    ) -> None:
        async def update() -> None:
            await self._controller.update_preferences(preferences)
            self.refresh_from_controller()
            if reconcile:
                pending = self._reconcile()
                if inspect.isawaitable(pending):
                    await pending

        self._schedule_awaitable(update())

    def _schedule_awaitable(self, awaitable: Awaitable[None]) -> None:
        self.run_worker(awaitable, group="persona-buddy-preferences")


__all__ = ["PersonaBuddyWidget"]
