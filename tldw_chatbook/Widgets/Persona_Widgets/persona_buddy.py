"""Native floating Persona Buddy; controller-owned snapshots enter, requests leave."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from time import monotonic
from typing import Any, ClassVar

from PIL import Image
from rich.text import Text
from textual import errors, events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Persona_Visual.runtime import PersonaVisualResolution
from tldw_chatbook.Utils.mosaic_render import mosaic_from_image


@dataclass(frozen=True, slots=True)
class PreparedBuddyFrame:
    """UI-free rendered frame and its display duration."""

    renderable: Text
    duration: float


@dataclass(frozen=True, slots=True)
class PreparedBuddyFrames:
    """Frame preparation result safe to return from a worker."""

    frames: tuple[PreparedBuddyFrame, ...]
    reason: str | None = None


def prepare_buddy_frames(
    resolution: PersonaVisualResolution,
    width: int,
    height: int,
    *,
    monochrome: bool = False,
) -> PreparedBuddyFrames:
    """Decode contained mosaics off the event loop, preserving a static fallback."""
    prepared = []
    reason = None
    sources = resolution.frames or (
        (resolution.portrait,) if resolution.portrait else ()
    )
    for frame in sources:
        try:
            with Image.open(BytesIO(frame.data)) as image:
                image.seek(frame.selected_frame)
                image.load()
                region = getattr(frame, "region", None)
                source = image
                if region is not None:
                    if (
                        region.x < 0
                        or region.y < 0
                        or region.width < 1
                        or region.height < 1
                        or region.x + region.width > image.width
                        or region.y + region.height > image.height
                    ):
                        raise ValueError("invalid frame region")
                    source = image.crop(
                        (
                            region.x,
                            region.y,
                            region.x + region.width,
                            region.y + region.height,
                        )
                    )
                renderable = mosaic_from_image(
                    source, width, height, fit="contain", monochrome=monochrome
                )
                milliseconds = getattr(frame, "duration_ms", None)
                duration = (
                    milliseconds / 1000
                    if milliseconds
                    else 1 / (resolution.frame_rate or 10)
                )
                prepared.append(PreparedBuddyFrame(renderable, max(0.02, duration)))
        except Exception:  # noqa: BLE001 - public frame failures must remain path-free
            # Exceptions may contain private storage details. Retain already
            # prepared static art, and publish only the stable public category.
            reason = "persona_buddy_frame_unavailable"
            break
    if not prepared and resolution.frames and resolution.portrait:
        try:
            with Image.open(BytesIO(resolution.portrait.data)) as image:
                image.seek(resolution.portrait.selected_frame)
                prepared.append(
                    PreparedBuddyFrame(
                        mosaic_from_image(
                            image, width, height, fit="contain", monochrome=monochrome
                        ),
                        1.0,
                    )
                )
        except Exception:  # noqa: BLE001 - public frame failures must remain path-free
            reason = "persona_buddy_frame_unavailable"
    return PreparedBuddyFrames(tuple(prepared), reason)


class BuddyPreferencesRequested(Message):
    """Request an explicit local preference change from the app owner."""

    def __init__(self, view: PersonaBuddyView, changes: dict[str, object]) -> None:
        super().__init__()
        self._view = view
        self.changes = changes

    @property
    def control(self) -> PersonaBuddyView:
        """Origin used by the host to reject queued events from replaced views."""
        return self._view


class PersonaBuddyView(Vertical):
    """Focusable screen overlay with no flow budget or controller ownership."""

    can_focus = True
    DEFAULT_CSS = """
    PersonaBuddyView {
        position: absolute;
        overlay: screen;
        width: 28;
        height: 16;
        padding: 0;
        border: round $primary;
        background: $surface;
        overflow: hidden hidden;
    }
    PersonaBuddyView:focus { border: round $accent; outline: none; }
    PersonaBuddyView #persona-buddy-header { height: 1; width: 100%; }
    PersonaBuddyView #persona-buddy-name { width: 1fr; height: 1; }
    PersonaBuddyView #persona-buddy-collapse,
    PersonaBuddyView #persona-buddy-close {
        width: auto !important;
        min-width: 0 !important;
        height: 1 !important;
        min-height: 1 !important;
        padding: 0 1 !important;
        margin: 0 !important;
        border: none !important;
    }
    PersonaBuddyView #persona-buddy-image {
        height: 1fr; width: 100%; content-align: center middle;
        overflow: hidden hidden;
    }
    PersonaBuddyView #persona-buddy-state { height: 1; color: $text-muted; }
    PersonaBuddyView #persona-buddy-grip {
        height: 1; text-align: right; color: $text-muted;
    }
    PersonaBuddyView.-compact { border: none; }
    PersonaBuddyView.-compact #persona-buddy-collapse { display: none; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "move(-1, 0)", "Move left", show=False),
        Binding("right", "move(1, 0)", "Move right", show=False),
        Binding("up", "move(0, -1)", "Move up", show=False),
        Binding("down", "move(0, 1)", "Move down", show=False),
        Binding("shift+left", "resize(-1, 0)", "Narrower", show=False),
        Binding("shift+right", "resize(1, 0)", "Wider", show=False),
        Binding("shift+up", "resize(0, -1)", "Shorter", show=False),
        Binding("shift+down", "resize(0, 1)", "Taller", show=False),
        Binding("r", "reset", "Reset Buddy", show=False),
        Binding("c", "collapse", "Collapse Buddy", show=False),
        Binding("x", "close", "Close Buddy", show=False),
    ]

    def __init__(self, snapshot: Any, preferences: Any) -> None:
        super().__init__(id="persona-buddy")
        self._snapshot = snapshot
        self._prepared = snapshot.prepared
        self._preferences = preferences
        self._frame_index = 0
        self._last_frame = monotonic()
        self._compact = False
        self._gesture: tuple[str, int, int, int, int] | None = None
        self._timer = None
        self.tooltip = (
            "Arrows move · Shift+arrows resize · R reset · C collapse · X close"
        )

    def compose(self) -> ComposeResult:
        with Horizontal(id="persona-buddy-header"):
            yield Static("", id="persona-buddy-name", markup=False)
            yield Button("Collapse", id="persona-buddy-collapse")
            yield Button("Close", id="persona-buddy-close")
        yield Static("", id="persona-buddy-image", markup=False)
        yield Static("", id="persona-buddy-state", markup=False)
        yield Static("Resize ◢", id="persona-buddy-grip", markup=False)

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.02, self._advance_frame)
        self.apply_preferences(self._preferences)
        self._paint()

    def apply_snapshot(self, snapshot: Any) -> None:
        """Accept only current-persona snapshots that do not roll back generation."""
        if snapshot.persona_id != self._preferences.local_persona_id:
            return
        if snapshot.generation < self._snapshot.generation:
            return
        identity = getattr(snapshot, "identity", None)
        same_authority = (
            identity is not None
            and identity == getattr(self._snapshot, "identity", None)
            and snapshot.persona_id == self._snapshot.persona_id
        )
        availability_transition = (
            same_authority
            and snapshot.requested_state == self._snapshot.requested_state
            and "unavailable"
            in (snapshot.resolution.source, self._snapshot.resolution.source)
        )
        if (
            snapshot.generation == self._snapshot.generation
            and snapshot.resolution.cache_identity
            != self._snapshot.resolution.cache_identity
            and not availability_transition
        ):
            return
        prepared = snapshot.prepared
        if (
            same_authority
            and isinstance(prepared, PreparedBuddyFrames)
            and not prepared.frames
            and isinstance(self._prepared, PreparedBuddyFrames)
            and self._prepared.frames
        ):
            prepared = PreparedBuddyFrames(
                (self._prepared.frames[self._frame_index],),
                prepared.reason or "persona_buddy_frame_unavailable",
            )
        same_animation = (
            snapshot.resolution.cache_identity
            == self._snapshot.resolution.cache_identity
            and snapshot.resolution.animate == self._snapshot.resolution.animate
            and prepared == self._prepared
        )
        self._snapshot = snapshot
        self._prepared = prepared
        if not same_animation:
            self._frame_index = 0
            self._last_frame = monotonic()
        if self.is_mounted:
            self._paint()

    def apply_preferences(self, preferences: Any) -> None:
        """Apply canonical preferences without overwriting desired geometry."""
        self._preferences = preferences
        self.display = bool(
            preferences.enabled
            and preferences.open
            and preferences.local_persona_id == self._snapshot.persona_id
        )
        if not self.display:
            self._release_gesture()
        if self.is_mounted:
            self.sync_geometry()
            self._paint()

    def sync_geometry(self) -> None:
        """Clamp rendered geometry while retaining the stored desired dimensions."""
        if not self.is_mounted:
            return
        width, height = self.screen.size
        available_height = max(1, height - 2)
        self._compact = width < 24 or available_height < 8
        collapsed = self._preferences.collapsed or self._compact
        render_width = min(width, max(24, self._preferences.width))
        render_height = min(
            available_height,
            1
            if self._compact
            else 3
            if collapsed
            else max(8, self._preferences.height),
        )
        x = self._preferences.x
        y = self._preferences.y
        top = 1 if height >= 3 else 0
        self.styles.width = max(1, render_width)
        self.styles.height = max(1, render_height)
        self.styles.offset = (
            min(
                max(0, width - render_width if x is None else x),
                max(0, width - render_width),
            ),
            min(
                max(top, height - 1 - render_height if y is None else y),
                max(top, height - 1 - render_height),
            ),
        )
        self.set_class(self._compact, "-compact")
        for selector in (
            "#persona-buddy-image",
            "#persona-buddy-state",
            "#persona-buddy-grip",
        ):
            self.query_one(selector).display = not collapsed
        self.query_one("#persona-buddy-collapse", Button).label = (
            "Expand" if collapsed else "Collapse"
        )
        self._paint()

    def on_resize(self) -> None:
        self.sync_geometry()

    def _paint(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#persona-buddy-name", Static).update(
            "Buddy" if self._compact else self._snapshot.name
        )
        prepared = self._prepared
        frames = prepared.frames if isinstance(prepared, PreparedBuddyFrames) else ()
        if frames:
            self._frame_index %= len(frames)
            art = frames[self._frame_index].renderable
        else:
            art = Text("Portrait unavailable")
        self.query_one("#persona-buddy-image", Static).update(art)
        self.query_one("#persona-buddy-state", Static).update(
            self._snapshot.requested_state.replace("_", " ")
            + (" · Frame unavailable" if getattr(prepared, "reason", None) else "")
        )

    def _active(self) -> bool:
        return (
            self.is_mounted
            and self.display
            and self.visible
            and self.app.screen is self.screen
        )

    def _advance_frame(self) -> None:
        if (
            not self._active()
            or self._compact
            or self._preferences.collapsed
            or not self._snapshot.resolution.animate
        ):
            self._last_frame = monotonic()
            return
        prepared = self._prepared
        if not isinstance(prepared, PreparedBuddyFrames) or len(prepared.frames) < 2:
            return
        if monotonic() - self._last_frame < prepared.frames[self._frame_index].duration:
            return
        if (
            not self._snapshot.resolution.loop
            and self._frame_index == len(prepared.frames) - 1
        ):
            return
        self._frame_index = (self._frame_index + 1) % len(prepared.frames)
        self._last_frame = monotonic()
        self._paint()

    def _request(self, **changes: object) -> None:
        if self._active():
            self.post_message(BuddyPreferencesRequested(self, changes))

    def _request_geometry(self, **changes: int) -> None:
        """Persist actual gesture geometry within viewport and storage limits."""
        if not self._active():
            return
        screen_width, screen_height = self.screen.size
        # BuddyController intentionally rejects invalid stored coordinates.
        # User gestures reach this clamp before crossing that strict boundary.
        storage_limit = 10_000
        width_limit = max(1, min(storage_limit, screen_width))
        height_limit = max(1, min(storage_limit, screen_height - 2))
        top = 1 if screen_height >= 3 else 0
        bounds = {
            "x": (0, min(storage_limit, max(0, screen_width - self.region.width))),
            "y": (
                top,
                min(storage_limit, max(top, screen_height - 1 - self.region.height)),
            ),
            "width": (min(24, width_limit), width_limit),
            "height": (min(8, height_limit), height_limit),
        }
        self._request(
            **{
                key: min(high, max(low, int(value)))
                for key, value in changes.items()
                for low, high in (bounds[key],)
            }
        )

    def action_move(self, dx: int, dy: int) -> None:
        self._request_geometry(x=self.region.x + dx, y=self.region.y + dy)

    def action_resize(self, dw: int, dh: int) -> None:
        self._request_geometry(
            width=max(24, self._preferences.width + dw),
            height=max(8, self._preferences.height + dh),
        )

    def action_reset(self) -> None:
        self._request(x=None, y=None, width=28, height=16)

    def action_collapse(self) -> None:
        self._release_gesture()
        self._request(collapsed=not self._preferences.collapsed)

    def action_close(self) -> None:
        self._release_gesture()
        self._request(open=False)

    @on(Button.Pressed, "#persona-buddy-collapse")
    def _collapse_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_collapse()

    @on(Button.Pressed, "#persona-buddy-close")
    def _close_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1 or not self._active():
            return
        target = event.widget
        if target is None:
            # Native XTerm events have no widget; Pilot events already do.
            # Resolve against the current compositor, after the screen guard.
            try:
                target, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
            except errors.NoWidget:
                return
        if target is not None and target.id in (
            "persona-buddy-name",
            "persona-buddy-header",
        ):
            mode, a, b = "move", self.region.x, self.region.y
        elif target is not None and target.id == "persona-buddy-grip":
            mode, a, b = "resize", self.region.width, self.region.height
        else:
            return
        self.focus()
        self._gesture = (mode, event.screen_x, event.screen_y, a, b)
        self.capture_mouse()
        self.screen.clear_selection()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._gesture is None:
            return
        self._apply_gesture_position(event.screen_x, event.screen_y)
        event.stop()

    def _apply_gesture_position(self, screen_x: int, screen_y: int) -> None:
        if self._gesture is None:
            return
        if not self._active():
            self._release_gesture()
            return
        mode, x, y, a, b = self._gesture
        dx, dy = screen_x - x, screen_y - y
        if mode == "move":
            self._request_geometry(x=a + dx, y=b + dy)
        else:
            self._request_geometry(width=a + dx, height=b + dy)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._gesture is not None:
            # A terminal may coalesce moves; release carries the final point.
            self._apply_gesture_position(event.screen_x, event.screen_y)
            self._release_gesture()
            event.stop()

    def _release_gesture(self) -> None:
        if self._gesture is not None:
            self._gesture = None
            self.release_mouse()

    def on_hide(self) -> None:
        self._release_gesture()

    def on_unmount(self) -> None:
        self._release_gesture()
        if self._timer is not None:
            self._timer.stop()
