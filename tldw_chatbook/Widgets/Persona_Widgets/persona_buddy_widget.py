"""Disposable terminal-native floating view for the app-owned Persona Buddy."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.geometry import Offset
from textual.timer import Timer
from textual.worker import Worker
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyVisualSnapshot
from tldw_chatbook.Persona_Buddy.preferences import (
    PERSONA_BUDDY_UNPOSITIONED_COORDINATE,
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
)

_DEFAULT_WIDTH = 28
_DEFAULT_HEIGHT = 12
_MIN_WIDTH = 18
_MIN_HEIGHT = 6
_OPERABLE_WIDTH = 10
_OPERABLE_HEIGHT = 4
_BOUNDARY_SIZE = 2
_COMPACT_HEIGHT = 1
_RESIZE_GRIP_WIDTH = 2
_POLL_SECONDS = 0.10
_ACTIONABLE_ALERTS = {
    "approval_needed": "Approval needed",
    "error": "Error",
    "offline": "Offline",
}


def _visual_identity(visual: Any) -> tuple[object, ...]:
    frames = getattr(visual, "frames", ())
    return (
        getattr(visual, "source", None),
        getattr(visual, "persona_id", None),
        getattr(visual, "persona_revision", None),
        getattr(visual, "requested_state", None),
        getattr(visual, "resolved_state", None),
        getattr(visual, "animation_id", None),
        getattr(visual, "graph_identity", None),
        getattr(visual, "cache_identity", None),
        getattr(visual, "frame_rate", None),
        getattr(visual, "loop", None),
        getattr(visual, "animate", None),
        tuple(
            (
                getattr(frame, "cache_identity", None),
                getattr(frame, "asset_id", None),
                getattr(frame, "asset_sha256", None),
                getattr(frame, "manifest_frame_index", None),
                getattr(frame, "selected_frame", None),
                getattr(frame, "duration_ms", None),
                getattr(frame, "paint_digest", None),
            )
            for frame in frames
        ),
    )


def _stable_content_box(visual: Any) -> tuple[int, int] | None:
    frames = getattr(visual, "frames", ())
    dimensions = tuple(
        (getattr(frame, "width", None), getattr(frame, "height", None))
        for frame in frames
    )
    if any(
        type(width) is not int or width < 1 or type(height) is not int or height < 1
        for width, height in dimensions
    ):
        return None
    return (
        max((_OPERABLE_WIDTH, *(width for width, _height in dimensions))),
        max(
            (
                _OPERABLE_HEIGHT,
                *((height + 1) // 2 for _width, height in dimensions),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _AcceptedRender:
    authority: tuple[object, ...]
    visual_identity: tuple[object, ...]
    visual: PersonaBuddyVisualSnapshot = field(repr=False, compare=False)
    collapsed: bool
    content_width: int
    content_height: int


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
        padding: 0;
        background: $panel;
        color: $text;
        border: round $accent;
        overflow: hidden hidden;
    }

    PersonaBuddyWidget:focus {
        outline: heavy $accent;
    }

    PersonaBuddyWidget .persona-buddy-control {
        position: absolute;
        layer: controls;
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        padding: 0;
        margin: 0;
        border: none;
        background: $surface-darken-1;
        color: $text-muted;
        content-align: center middle;
        text-align: center;
    }

    PersonaBuddyWidget .persona-buddy-control:focus {
        background: $accent;
        color: $text;
        text-style: bold underline;
        outline: none;
    }

    PersonaBuddyWidget #persona-buddy-frame {
        width: 100%;
        height: 100%;
        padding: 0;
        layer: pet;
        content-align: center middle;
        text-align: center;
        color: $text;
    }

    PersonaBuddyWidget #persona-buddy-frame.persona-buddy-alert {
        color: $warning;
        text-style: bold;
    }

    PersonaBuddyWidget #persona-buddy-collapse {
        offset: 0 0;
    }

    PersonaBuddyWidget.persona-buddy-compact {
        height: 1;
        padding: 0;
        border: none;
    }

    PersonaBuddyWidget.persona-buddy-compact #persona-buddy-frame {
        display: none;
    }

    PersonaBuddyWidget.persona-buddy-compact .persona-buddy-control {
        height: 1;
        width: 3;
        min-width: 3;
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
        is_current: Callable[["PersonaBuddyWidget"], bool] | None = None,
    ) -> None:
        super().__init__(id="persona-buddy-widget")
        self._controller = controller
        self.view_generation = view_generation
        self._reconcile = reconcile
        self._current_view = is_current
        self._snapshot: Any = None
        self._painted_visual_identity: object | None = None
        self._accepted_render: _AcceptedRender | None = None
        self._display_content_boxes: dict[bool, tuple[int, int]] = {}
        self._alert_entry_authority: tuple[str, bool, tuple[int, int] | None] | None = (
            None
        )
        self._working_preferences: PersonaBuddyPreferences = (
            controller.current_preferences()
        )
        self.frame_index = 0
        self.snapshot_polling_active = False
        self._poll_timer: Timer | None = None
        self._frame_timer: Timer | None = None
        self._resolution_worker: Worker[None] | None = None
        self._resolved_authority: tuple[object, ...] | None = None
        self._requested_authority: tuple[object, ...] | None = None
        self._interaction: tuple[str, int, int, PersonaBuddyGeometry] | None = None
        self._last_screen_size: tuple[int, int] | None = None
        self._next_frame_at = 0.0
        self._frame_was_frozen = True

    def compose(self) -> ComposeResult:
        yield Static("", id="persona-buddy-frame")
        collapse = Button(
            "▾", id="persona-buddy-collapse", classes="persona-buddy-control"
        )
        collapse.tooltip = "Fold"
        yield collapse
        close = Button("×", id="persona-buddy-close", classes="persona-buddy-control")
        close.tooltip = "Close"
        yield close

    def on_mount(self) -> None:
        """Start bounded snapshot and frame polling without changing focus."""

        self.snapshot_polling_active = True
        self._apply_geometry(self._working_preferences.geometry)
        self.refresh_from_controller(schedule_resolution=False)
        self._poll_timer = self.set_interval(
            _POLL_SECONDS, self.refresh_from_controller
        )

    def on_unmount(self) -> None:
        """Release capture and stop view-owned timers during any teardown path."""

        self.snapshot_polling_active = False
        self.release_interaction_capture()
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._stop_frame_timer()
        if self._resolution_worker is not None:
            self._resolution_worker.cancel()

    def resume_resolution(self) -> None:
        """Restart resolution when a covering modal returns to this view."""

        if not self._is_current_view():
            return
        worker = self._resolution_worker
        if worker is not None and not worker.is_finished:
            return
        self.refresh_from_controller()

    def _is_current_view(self) -> bool:
        if not self.is_attached:
            return False
        return self._current_view(self) if self._current_view is not None else True

    def _requested_render_budget(self, snapshot: Any) -> tuple[int, int] | None:
        """Return a deterministic preferred content budget for one resolve."""

        if (
            not self.is_attached
            or not self.display
            or self._compact_for_geometry(self._working_preferences.geometry)
        ):
            return None
        if snapshot.collapsed:
            return _OPERABLE_WIDTH, _OPERABLE_HEIGHT
        viewport = self.screen.size
        geometry = self._working_preferences.geometry
        cols = min(geometry.width, viewport.width) - _BOUNDARY_SIZE
        lines = min(geometry.height, viewport.height) - _BOUNDARY_SIZE
        if cols < 1 or lines < 1:
            return None
        return cols, lines

    async def _resolution_loop(self) -> None:
        """Resolve only changed semantic authority while this view is current."""

        while self._is_current_view():
            snapshot = self._controller.snapshot()
            budget = self._requested_render_budget(snapshot)
            authority = self._resolution_authority(snapshot)
            if (
                authority is None
                or budget is None
                or authority == self._resolved_authority
            ):
                return
            visual = await self._controller.resolve_current_visual(
                cols=budget[0],
                lines=budget[1],
            )
            if not self._is_current_view():
                return
            current_snapshot = self._controller.snapshot()
            if self._resolution_authority(current_snapshot) != authority:
                continue
            if visual is not None:
                content_box = _stable_content_box(visual)
                if content_box is not None:
                    self._accepted_render = _AcceptedRender(
                        authority=authority,
                        visual_identity=_visual_identity(visual),
                        visual=visual,
                        collapsed=bool(current_snapshot.collapsed),
                        content_width=content_box[0],
                        content_height=content_box[1],
                    )
                    collapsed = bool(current_snapshot.collapsed)
                    alert_entry = self._alert_entry_authority
                    if current_snapshot.state not in _ACTIONABLE_ALERTS:
                        self._display_content_boxes[collapsed] = content_box
                    elif (
                        alert_entry is not None
                        and alert_entry[:2] == (current_snapshot.state, collapsed)
                        and alert_entry[2] != budget
                    ):
                        self._display_content_boxes[collapsed] = content_box
                        self._alert_entry_authority = (
                            current_snapshot.state,
                            collapsed,
                            budget,
                        )
            self._resolved_authority = authority
            self.refresh_from_controller(schedule_resolution=False)
            current_visual = current_snapshot.visual
            if visual is not None and not visual.available and current_visual is visual:
                confirm = getattr(
                    self.screen, "confirm_persona_buddy_unavailable", None
                )
                if not callable(confirm) or not confirm(
                    view=self,
                    controller=self._controller,
                    snapshot=current_snapshot,
                    visual=visual,
                ):
                    self._resolved_authority = None
                    await asyncio.sleep(_POLL_SECONDS)
                    continue
                pending = self._reconcile()
                removed = False
                if inspect.isawaitable(pending):
                    reconcile_worker = self.app.run_worker(
                        pending,
                        group="persona-buddy-unavailable-reconcile",
                        exclusive=True,
                    )
                    removed = bool(await asyncio.shield(reconcile_worker.wait()))
                else:
                    removed = bool(pending)
                if removed or not self._is_current_view():
                    return
                current_snapshot = self._controller.snapshot()
                confirmed = getattr(
                    self.screen, "is_persona_buddy_confirmed_unavailable", None
                )
                if callable(confirmed) and confirmed(
                    self._controller, current_snapshot
                ):
                    return
                self._resolved_authority = None
                continue
            return

    def _resolution_authority(self, snapshot: Any) -> tuple[object, ...] | None:
        """Return the full semantic/cache/viewport key for one costly resolve."""

        budget = self._requested_render_budget(snapshot)
        if (
            not snapshot.enabled
            or not snapshot.open
            or snapshot.selection is None
            or budget is None
        ):
            return None
        return (
            id(self._controller),
            self.view_generation,
            snapshot.generation,
            snapshot.selection,
            snapshot.state,
            snapshot.state_source,
            snapshot.state_owner,
            snapshot.preferences_generation,
            snapshot.profile_generation,
            snapshot.viewport_generation,
            snapshot.collapsed,
            budget,
        )

    def _ensure_resolution(self, snapshot: Any) -> None:
        authority = self._resolution_authority(snapshot)
        if authority is None or authority == self._resolved_authority:
            return
        worker = self._resolution_worker
        if (
            worker is not None
            and not worker.is_finished
            and authority == self._requested_authority
        ):
            return
        self._requested_authority = authority
        self._resolution_worker = self.run_worker(
            partial(self._resolution_loop),
            group="persona-buddy-visual-resolution",
            exclusive=True,
        )

    def on_resize(self, _event: events.Resize) -> None:
        """Re-clamp geometry whenever the terminal viewport changes."""

        if not self._is_current_view():
            return
        self._sync_viewport_generation()
        self._apply_geometry(self._working_preferences.geometry)

    def _sync_viewport_generation(self) -> None:
        """Publish terminal-size authority even when fitted bounds stay unchanged."""

        screen_size = (self.screen.size.width, self.screen.size.height)
        if screen_size != self._last_screen_size:
            self._last_screen_size = screen_size
            setter = getattr(self._controller, "set_viewport_generation", None)
            if callable(setter):
                snapshot = self._controller.snapshot()
                setter(snapshot.viewport_generation + 1)

    def refresh_from_controller(self, *, schedule_resolution: bool = True) -> None:
        """Refresh paint/state from an immutable snapshot without touching focus."""

        if not self._is_current_view():
            return
        self._sync_viewport_generation()
        snapshot = self._controller.snapshot()
        self._snapshot = snapshot
        preferences = self._controller.current_preferences()
        if self._interaction is not None:
            preferences = replace(
                preferences,
                geometry=self._working_preferences.geometry,
            )
        self._working_preferences = preferences
        alert_state = snapshot.state if snapshot.state in _ACTIONABLE_ALERTS else None
        alert_key = (alert_state, bool(snapshot.collapsed))
        if alert_state is None:
            self._alert_entry_authority = None
        elif (
            self._alert_entry_authority is None
            or self._alert_entry_authority[:2] != alert_key
        ):
            self._alert_entry_authority = (
                alert_state,
                alert_key[1],
                self._requested_render_budget(snapshot),
            )
        self._sync_compact_state(snapshot.collapsed)

        accepted = self._accepted_render
        visual_identity = accepted.visual_identity if accepted is not None else None
        if visual_identity != self._painted_visual_identity:
            self._stop_frame_timer()
            self._painted_visual_identity = visual_identity
            self.frame_index = 0
            self._next_frame_at = time.monotonic() + self._frame_duration_seconds()
            self._frame_was_frozen = False

        self._paint_frame()
        self._sync_frame_timer()
        if schedule_resolution:
            self._ensure_resolution(snapshot)

    def advance_frame(self) -> None:
        """Advance animation only while a full visible animated view is painted."""

        self._stop_frame_timer()

        snapshot = self._snapshot
        accepted = self._accepted_render_for_paint()
        if accepted is None or snapshot is None or snapshot.collapsed:
            self._frame_was_frozen = True
            return
        visual = accepted.visual
        frames = getattr(visual, "frames", ())
        if not getattr(visual, "animate", False) or len(frames) < 2:
            return
        now = time.monotonic()
        if now < self._next_frame_at:
            self._sync_frame_timer()
            return
        if self.frame_index == len(frames) - 1 and not getattr(visual, "loop", False):
            return
        self.frame_index = (self.frame_index + 1) % len(frames)
        self._next_frame_at = now + self._frame_duration_seconds()
        self._paint_frame()
        self._sync_frame_timer()

    def _stop_frame_timer(self) -> None:
        timer = self._frame_timer
        self._frame_timer = None
        if timer is not None:
            timer.stop()

    def _sync_frame_timer(self) -> None:
        snapshot = self._snapshot
        accepted = self._accepted_render_for_paint()
        visual = accepted.visual if accepted is not None else None
        frames = getattr(visual, "frames", ()) if visual is not None else ()
        active = bool(
            accepted is not None
            and snapshot is not None
            and not snapshot.collapsed
            and getattr(visual, "animate", False)
            and len(frames) > 1
            and (self.frame_index < len(frames) - 1 or getattr(visual, "loop", False))
        )
        if not active:
            self._stop_frame_timer()
            self._frame_was_frozen = True
            return
        if self._frame_timer is not None:
            return
        now = time.monotonic()
        if self._frame_was_frozen:
            self._next_frame_at = now + self._frame_duration_seconds()
            self._frame_was_frozen = False
        delay = max(0.001, self._next_frame_at - now)
        self._frame_timer = self.set_timer(delay, self.advance_frame)

    def _frame_duration_seconds(self) -> float:
        accepted = self._accepted_render
        visual = accepted.visual if accepted is not None else None
        frames = getattr(visual, "frames", ()) if visual is not None else ()
        if frames:
            duration_ms = getattr(frames[self.frame_index], "duration_ms", None)
            if type(duration_ms) is int and duration_ms > 0:
                return duration_ms / 1000.0
        frame_rate = getattr(visual, "frame_rate", None)
        if isinstance(frame_rate, (int, float)) and frame_rate > 0:
            return 1.0 / float(frame_rate)
        return 0.1

    def _accepted_render_for_paint(self) -> _AcceptedRender | None:
        """Return the accepted pet render only when it may supply visible pixels."""

        snapshot = self._snapshot
        accepted = self._accepted_render
        if (
            not self._is_current_view()
            or not self.display
            or self.has_class("persona-buddy-compact")
            or snapshot is None
            or accepted is None
            or snapshot.state in _ACTIONABLE_ALERTS
            or accepted.collapsed != bool(snapshot.collapsed)
        ):
            return None
        return accepted

    def _paint_frame(self) -> None:
        if not self.is_attached:
            return
        target = self.query_one("#persona-buddy-frame", Static)
        if self.has_class("persona-buddy-compact"):
            target.set_class(False, "persona-buddy-alert")
            target.update("")
            return
        alert = _ACTIONABLE_ALERTS.get(getattr(self._snapshot, "state", None))
        target.set_class(alert is not None, "persona-buddy-alert")
        if alert is not None:
            target.update(alert)
            return
        accepted = self._accepted_render_for_paint()
        visual = accepted.visual if accepted is not None else None
        frames = getattr(visual, "frames", ()) if visual is not None else ()
        if frames:
            self.frame_index = min(self.frame_index, len(frames) - 1)
            target.update(frames[self.frame_index].renderable)
            return
        target.update("")

    def _sync_compact_state(self, collapsed: bool) -> None:
        compact = self._compact_presentation_for_geometry(
            self._working_preferences.geometry
        )
        self.set_class(compact, "persona-buddy-compact")
        self.set_class(collapsed and not compact, "persona-buddy-collapsed")
        self._sync_control_presentation(collapsed)
        self._apply_geometry(self._working_preferences.geometry)

    def _sync_control_presentation(self, collapsed: bool) -> None:
        """Expose a focused action label without adding resting layout chrome."""

        collapse = self.query_one("#persona-buddy-collapse", Button)
        close = self.query_one("#persona-buddy-close", Button)
        collapse_action = "Open" if collapsed else "Fold"
        if collapse.tooltip != collapse_action:
            collapse.tooltip = collapse_action
        collapse.label = (
            collapse_action
            if self.app.focused is collapse
            else ("▴" if collapsed else "▾")
        )
        close.label = "Close" if self.app.focused is close else "×"

    def on_descendant_focus(self, _event: events.DescendantFocus) -> None:
        snapshot = self._snapshot
        self._sync_control_presentation(bool(snapshot and snapshot.collapsed))
        self._apply_geometry(self._working_preferences.geometry)

    def on_descendant_blur(self, _event: events.DescendantBlur) -> None:
        self.call_after_refresh(self._sync_controls_after_blur)

    def _sync_controls_after_blur(self) -> None:
        if not self.is_attached:
            return
        snapshot = self._snapshot
        self._sync_control_presentation(bool(snapshot and snapshot.collapsed))
        self._apply_geometry(self._working_preferences.geometry)

    def _compact_for_geometry(self, geometry: PersonaBuddyGeometry) -> bool:
        viewport = self.screen.size
        available_width = min(geometry.width, max(1, viewport.width)) - _BOUNDARY_SIZE
        available_height = (
            min(geometry.height, max(1, viewport.height)) - _BOUNDARY_SIZE
        )
        return available_width < _OPERABLE_WIDTH or available_height < _OPERABLE_HEIGHT

    def _compact_presentation_for_geometry(
        self, geometry: PersonaBuddyGeometry
    ) -> bool:
        """Hide an accepted pet that cannot fit the current render budget."""

        if self._compact_for_geometry(geometry):
            return True
        snapshot = self._snapshot
        accepted = self._accepted_render
        if (
            snapshot is None
            or accepted is None
            or snapshot.state in _ACTIONABLE_ALERTS
            or accepted.collapsed != bool(snapshot.collapsed)
        ):
            return False
        viewport = self.screen.size
        available_width = min(geometry.width, viewport.width) - _BOUNDARY_SIZE
        available_height = min(geometry.height, viewport.height) - _BOUNDARY_SIZE
        return (
            accepted.content_width > available_width
            or accepted.content_height > available_height
        )

    def _display_geometry(self, geometry: PersonaBuddyGeometry) -> PersonaBuddyGeometry:
        """Fit accepted content without changing the saved render budget."""

        viewport = self.screen.size
        compact = self._compact_presentation_for_geometry(geometry)
        if compact:
            available_width = min(geometry.width, max(1, viewport.width))
            width = min(max(6, available_width), max(1, viewport.width))
            height = min(_COMPACT_HEIGHT, max(1, viewport.height))
        else:
            width = min(max(_MIN_WIDTH, geometry.width), viewport.width)
            collapsed = bool(self._snapshot and self._snapshot.collapsed)
            content_box = self._display_content_boxes.get(collapsed)
            if (
                content_box is None
                and self._snapshot is not None
                and self._snapshot.state in _ACTIONABLE_ALERTS
            ):
                content_box = (_OPERABLE_WIDTH, _OPERABLE_HEIGHT)
            if content_box is not None:
                width = min(
                    content_box[0] + _BOUNDARY_SIZE,
                    viewport.width,
                )
                height = min(
                    content_box[1] + _BOUNDARY_SIZE,
                    viewport.height,
                )
            elif self._snapshot and self._snapshot.collapsed:
                width = min(_OPERABLE_WIDTH + _BOUNDARY_SIZE, viewport.width)
                height = min(_OPERABLE_HEIGHT + _BOUNDARY_SIZE, viewport.height)
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

    def _clamped_geometry(self, geometry: PersonaBuddyGeometry) -> PersonaBuddyGeometry:
        return self._display_geometry(geometry)

    def _apply_geometry(self, geometry: PersonaBuddyGeometry) -> None:
        if not self.is_attached:
            return
        clamped = self._display_geometry(geometry)
        self.styles.width = clamped.width
        self.styles.height = clamped.height
        self.absolute_offset = Offset(clamped.x, clamped.y)
        frame = self.query_one("#persona-buddy-frame", Static)
        frame.styles.width = "100%"
        frame.styles.height = "100%"
        collapse = self.query_one("#persona-buddy-collapse", Button)
        close = self.query_one("#persona-buddy-close", Button)
        collapse_width = len(str(collapse.label)) + 2
        close_width = len(str(close.label)) + 2
        collapse.styles.width = collapse_width
        close.styles.width = close_width
        content_width = clamped.width - (
            0 if self.has_class("persona-buddy-compact") else _BOUNDARY_SIZE
        )
        resize_grip_width = (
            _RESIZE_GRIP_WIDTH if self.has_class("persona-buddy-compact") else 0
        )
        close.styles.offset = Offset(
            max(0, content_width - close_width - resize_grip_width), 0
        )

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Arm pet-surface drag or lower-right resize for a terminal event."""

        if event.button != 1 or not self._is_current_view():
            return
        screen_x = int(event.screen_x if event.screen_x is not None else event.x)
        screen_y = int(event.screen_y if event.screen_y is not None else event.y)
        region = self.region
        resize = (
            screen_x >= region.right - _RESIZE_GRIP_WIDTH
            and screen_y >= region.bottom - 1
        )
        for button in self.query(Button):
            if (
                button.display
                and button.region.x <= screen_x < button.region.right
                and button.region.y <= screen_y < button.region.bottom
            ):
                return
        content = self.content_region
        in_content = (
            content.x <= screen_x < content.right
            and content.y <= screen_y < content.bottom
        )
        if not resize and not in_content:
            return
        mode = "resize" if resize else "drag"
        preferred = self._working_preferences.geometry
        geometry = replace(
            preferred,
            x=self.absolute_offset.x,
            y=self.absolute_offset.y,
        )
        self._interaction = (mode, screen_x, screen_y, geometry)
        self.focus(scroll_visible=False)
        self.capture_mouse(True)
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Apply bounded working geometry without writing config per mouse move."""

        if self._interaction is None:
            return
        if not self._is_current_view():
            self.release_interaction_capture()
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
        preferred = self._preferred_geometry_at_clamped_position(candidate)
        self._working_preferences = replace(
            self._working_preferences, geometry=preferred
        )
        self._apply_geometry(preferred)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Finish one interaction, release capture, and persist exactly once."""

        if self._interaction is None:
            self.release_interaction_capture()
            return
        if not self._is_current_view():
            self.release_interaction_capture()
            return
        self._interaction = None
        self.release_interaction_capture()
        self._apply_and_schedule_preferences(
            geometry=self._working_preferences.geometry,
            reconcile=False,
        )
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
        if not self._is_current_view():
            return
        geometry = self._working_preferences.geometry
        displayed = self._clamped_geometry(geometry)
        candidate = replace(
            geometry,
            x=max(0, displayed.x + dx),
            y=max(0, displayed.y + dy),
            width=max(1, geometry.width + dw),
            height=max(1, geometry.height + dh),
        )
        preferred = self._preferred_geometry_at_clamped_position(candidate)
        self._working_preferences = replace(
            self._working_preferences, geometry=preferred
        )
        self._apply_geometry(preferred)
        self._apply_and_schedule_preferences(geometry=preferred, reconcile=False)

    def _preferred_geometry_at_clamped_position(
        self, geometry: PersonaBuddyGeometry
    ) -> PersonaBuddyGeometry:
        """Keep preferred dimensions while constraining the displayed position."""

        displayed = self._clamped_geometry(geometry)
        return replace(geometry, x=displayed.x, y=displayed.y)

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
        if not self._is_current_view():
            return
        geometry = PersonaBuddyGeometry(width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT)
        preferred = self._preferred_geometry_at_clamped_position(geometry)
        self._working_preferences = replace(
            self._working_preferences, geometry=preferred
        )
        self._apply_geometry(preferred)
        self._apply_and_schedule_preferences(geometry=preferred, reconcile=False)

    def action_toggle_collapse(self) -> None:
        if not self._is_current_view():
            return
        preferences = replace(
            self._working_preferences,
            collapsed=not self._working_preferences.collapsed,
        )
        self._working_preferences = preferences
        self._apply_and_schedule_preferences(
            collapsed=preferences.collapsed,
            reconcile=False,
        )

    def action_close(self) -> None:
        if not self._is_current_view():
            return
        preferences = replace(self._working_preferences, open=False)
        self._working_preferences = preferences
        self._apply_and_schedule_preferences(open=False, reconcile=True)

    async def close_and_persist(self) -> None:
        """Persist the explicit close and reconcile only this app's active screen."""

        if not self._is_current_view():
            return
        preferences = replace(self._working_preferences, open=False)
        self._working_preferences = preferences
        revision = self._controller.apply_preferences_patch(open=False)
        await self._controller.persist_preferences_revision(revision)
        if not self._is_current_view():
            return
        pending = self._reconcile()
        if inspect.isawaitable(pending):
            await pending

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self._is_current_view():
            return
        if event.button.id == "persona-buddy-collapse":
            self.action_toggle_collapse()
        elif event.button.id == "persona-buddy-close":
            self.action_close()
        event.stop()

    def _apply_and_schedule_preferences(
        self, *, reconcile: bool, **changes: object
    ) -> None:
        """Publish an exact field patch now and persist its revision on the app."""

        revision = self._controller.apply_preferences_patch(**changes)

        async def update() -> None:
            await self._controller.persist_preferences_revision(revision)
            if not self._is_current_view():
                return
            self.refresh_from_controller()
            if reconcile:
                pending = self._reconcile()
                if inspect.isawaitable(pending):
                    await pending

        self._schedule_awaitable(update())

    def _schedule_awaitable(self, awaitable: Awaitable[None]) -> None:
        self.app.run_worker(awaitable, group="persona-buddy-preferences")


__all__ = ["PersonaBuddyWidget"]
