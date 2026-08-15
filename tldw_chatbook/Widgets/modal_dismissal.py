"""Shared primitives for safe modal cancellation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
import time
from typing import TYPE_CHECKING, Any, Protocol, cast
from weakref import ReferenceType, ref

from textual import events
from textual.app import App
from textual.screen import Screen
from textual.widget import Widget

if TYPE_CHECKING:

    class _SafeModalHost(Protocol):
        """Textual surface required by ``SafeModalDismissMixin``."""

        @property
        def app(self) -> App[Any]: ...

        @property
        def is_mounted(self) -> bool: ...

        def query_one(self, selector: str, expect_type: type[Widget]) -> Widget: ...

        def dismiss(self, result: object) -> object: ...


_safe_request_generation: ContextVar[tuple[int, int] | None] = ContextVar(
    "safe_modal_request_generation", default=None
)


def is_modal_backdrop_click(
    *,
    button: int,
    provenance_known: bool,
    target_is_content_or_descendant: bool,
    point_is_in_content_region: bool,
) -> bool:
    """Return whether a classified click is on a modal's backdrop.

    Args:
        button: Textual mouse-button number; primary is ``1``.
        provenance_known: Whether both the event target and coordinates are known.
        target_is_content_or_descendant: Whether the target belongs to the modal.
        point_is_in_content_region: Whether the screen point falls inside the modal.

    Returns:
        ``True`` only for a known primary click outside modal content.
    """
    return (
        button == 1
        and provenance_known
        and not target_is_content_or_descendant
        and not point_is_in_content_region
    )


def _is_safe_focus_target(widget: Widget | None) -> bool:
    return bool(
        widget is not None
        and widget.is_mounted
        and widget.is_attached
        and widget.display
        and widget.visible
        and not widget.disabled
        and widget.can_focus
        and widget.focusable
        and all(node.display for node in widget.ancestors_with_self)
    )


def _restore_focus_after_dismissal(
    app: App[Any],
    revealed_screen: Screen[Any],
    opener_ref: ReferenceType[Widget] | None,
    opener_id: str | None,
) -> None:
    if app.screen is not revealed_screen:
        return

    opener = opener_ref() if opener_ref is not None else None
    if _is_safe_focus_target(opener):
        assert opener is not None
        if opener.screen is revealed_screen:
            opener.focus()
            return

    if opener_id is not None:
        eligible_matches = [
            widget
            for widget in revealed_screen.query(f"#{opener_id}")
            if _is_safe_focus_target(widget)
        ]
        if len(eligible_matches) == 1:
            eligible_matches[0].focus()
            return

    focus_console_composer = getattr(
        revealed_screen, "_focus_console_composer_if_needed", None
    )
    if callable(focus_console_composer):
        focus_console_composer(force=True)


class _BackdropClickShield(Widget):
    """One-cell overlay that consumes the remainder of a backdrop click chain."""

    can_focus = False

    DEFAULT_CSS = """
    _BackdropClickShield {
        width: 1;
        height: 1;
        position: absolute;
        overlay: screen;
        background: transparent;
    }
    """

    @staticmethod
    def _consume(event: events.MouseEvent) -> None:
        event.stop()
        event.prevent_default()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._consume(event)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._consume(event)

    def on_click(self, event: events.Click) -> None:
        self._consume(event)


async def _remove_backdrop_click_shield(shield: _BackdropClickShield) -> None:
    if shield.is_attached:
        await shield.remove()


async def _mount_backdrop_click_shield(
    app: App[Any],
    revealed_screen: Screen[Any],
    event_time: float,
    screen_x: int,
    screen_y: int,
) -> None:
    if app.screen is not revealed_screen:
        return
    remaining = app.CLICK_CHAIN_TIME_THRESHOLD - (time.monotonic() - event_time)
    if remaining <= 0:
        return

    shield = _BackdropClickShield()
    shield.styles.offset = (screen_x, screen_y)
    await revealed_screen.mount(shield)

    remaining = app.CLICK_CHAIN_TIME_THRESHOLD - (time.monotonic() - event_time)
    if remaining <= 0:
        await shield.remove()
        return
    revealed_screen.set_timer(
        remaining,
        lambda: _remove_backdrop_click_shield(shield),
    )


def _shield_revealed_screen_from_click_chain(
    app: App[Any], event_time: float, screen_x: int, screen_y: int
) -> None:
    revealed_screen = app.screen
    revealed_screen.call_after_refresh(
        _mount_backdrop_click_shield,
        app,
        revealed_screen,
        event_time,
        screen_x,
        screen_y,
    )


class SafeModalDismissMixin:
    """Provide one-shot cancellation and backdrop handling for a modal screen."""

    SAFE_MODAL_CONTENT: str | None = None
    SAFE_MODAL_RESTORE_FOCUS = True

    _safe_cancel_pending = False
    _safe_cancel_effect_committed = False
    _safe_dismiss_committed = False
    _safe_opener_focus_ref: ReferenceType[Widget] | None = None
    _safe_opener_focus_id: str | None = None
    _safe_backdrop_event_in_attempt: tuple[float, int, int] | None = None
    _safe_mount_generation = 0

    def on_mount(self) -> None:
        """Remember the opener's focused widget for post-dismiss restoration."""
        self._safe_cancel_pending = False
        self._safe_cancel_effect_committed = False
        self._safe_dismiss_committed = False
        self._safe_opener_focus_ref = None
        self._safe_opener_focus_id = None
        self._safe_backdrop_event_in_attempt = None
        self._safe_mount_generation += 1

        host = cast("_SafeModalHost", self)
        screen_stack = host.app.screen_stack
        if len(screen_stack) < 2:
            return
        opener = screen_stack[-2].focused
        if opener is not None:
            self._safe_opener_focus_ref = ref(opener)
            self._safe_opener_focus_id = opener.id or None

    def on_unmount(self) -> None:
        """Release the opener reference when the modal leaves the DOM."""
        self._safe_opener_focus_ref = None
        self._safe_opener_focus_id = None

    async def action_request_safe_cancel(self) -> None:
        """Route Escape to the modal's safe cancellation request."""
        await self.request_safe_cancel(source="escape")

    async def request_safe_cancel(self, *, source: str) -> None:
        """Run one cancellation request while consuming concurrent requests."""
        if self._safe_cancel_pending:
            return
        request_generation = self._safe_mount_generation
        request_identity = (id(self), request_generation)
        self._safe_cancel_pending = True
        generation_token = _safe_request_generation.set(request_identity)
        try:
            await self._perform_safe_cancel(source=source)
        finally:
            if self._safe_mount_generation == request_generation:
                self._safe_backdrop_event_in_attempt = None
                if cast("_SafeModalHost", self).is_mounted:
                    self._safe_cancel_pending = False
            _safe_request_generation.reset(generation_token)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Perform the default terminal cancellation."""
        del source
        self.dismiss_safe_once(None)

    async def run_cancel_effect_once(
        self, effect: Callable[[], Awaitable[None]]
    ) -> None:
        """Commit and invoke a cancellation side effect at most once."""
        if self._safe_cancel_effect_committed:
            return
        self._safe_cancel_effect_committed = True
        await effect()

    def dismiss_safe_once(self, result: object) -> bool:
        """Dismiss only this mounted, topmost modal and restore opener focus."""
        request_identity = _safe_request_generation.get()
        current_identity = (id(self), self._safe_mount_generation)
        if request_identity is not None and request_identity != current_identity:
            return False
        if self._safe_dismiss_committed:
            return False
        host = cast("_SafeModalHost", self)
        if not host.is_mounted or host.app.screen is not self:
            return False

        self._safe_dismiss_committed = True
        app = host.app
        opener_ref = self._safe_opener_focus_ref
        opener_id = self._safe_opener_focus_id
        backdrop_event = self._safe_backdrop_event_in_attempt
        host.dismiss(result)
        revealed_screen = app.screen
        if backdrop_event is not None:
            _shield_revealed_screen_from_click_chain(app, *backdrop_event)
        if self.SAFE_MODAL_RESTORE_FOCUS:
            revealed_screen.call_after_refresh(
                _restore_focus_after_dismissal,
                app,
                revealed_screen,
                opener_ref,
                opener_id,
            )
        return True

    async def on_click(self, event: events.Click) -> None:
        """Request cancellation for a known primary click on the backdrop."""
        if self.SAFE_MODAL_CONTENT is None:
            return

        host = cast("_SafeModalHost", self)
        content = host.query_one(self.SAFE_MODAL_CONTENT, Widget)
        target = event.widget
        coordinates_known = event.screen_x is not None and event.screen_y is not None
        provenance_known = isinstance(target, Widget) and coordinates_known
        target_is_content_or_descendant = bool(
            isinstance(target, Widget)
            and (target is content or content in target.ancestors)
        )
        point_is_in_content_region = bool(
            coordinates_known
            and content.region.contains(event.screen_x, event.screen_y)
        )

        if not is_modal_backdrop_click(
            button=event.button,
            provenance_known=provenance_known,
            target_is_content_or_descendant=target_is_content_or_descendant,
            point_is_in_content_region=point_is_in_content_region,
        ):
            return

        event.stop()
        event.prevent_default()
        self._safe_backdrop_event_in_attempt = (
            time.monotonic(),
            int(event.screen_x),
            int(event.screen_y),
        )
        await self.request_safe_cancel(source="backdrop")
