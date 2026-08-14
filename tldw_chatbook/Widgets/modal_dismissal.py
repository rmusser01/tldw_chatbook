"""Shared primitives for safe modal cancellation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
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


def is_modal_backdrop_click(
    *,
    button: int,
    provenance_known: bool,
    target_is_content_or_descendant: bool,
    point_is_in_content_region: bool,
) -> bool:
    """Return whether a classified click is on a modal's backdrop."""
    return (
        button == 1
        and provenance_known
        and not target_is_content_or_descendant
        and not point_is_in_content_region
    )


def _restore_focus_after_dismissal(
    app: App[Any], opener_ref: ReferenceType[Widget] | None
) -> None:
    opener = opener_ref() if opener_ref is not None else None
    if opener is not None and opener.is_mounted and opener.is_attached:
        opener.focus()
        return

    focus_console_composer = getattr(
        app.screen, "_focus_console_composer_if_needed", None
    )
    if callable(focus_console_composer):
        focus_console_composer(force=True)


def _release_backdrop_click_shield(app: App[Any], revealed_screen: Screen[Any]) -> None:
    if app.mouse_captured is revealed_screen:
        revealed_screen.release_mouse()


def _capture_revealed_screen_for_click_chain(
    app: App[Any], revealed_screen: Screen[Any]
) -> None:
    if app.screen is not revealed_screen:
        return
    revealed_screen.capture_mouse()
    revealed_screen.set_timer(
        app.CLICK_CHAIN_TIME_THRESHOLD,
        partial(_release_backdrop_click_shield, app, revealed_screen),
    )


def _shield_revealed_screen_from_click_chain(app: App[Any]) -> None:
    revealed_screen = app.screen
    revealed_screen.call_after_refresh(
        _capture_revealed_screen_for_click_chain, app, revealed_screen
    )


class SafeModalDismissMixin:
    """Provide one-shot cancellation and backdrop handling for a modal screen."""

    SAFE_MODAL_CONTENT: str | None = None

    _safe_cancel_pending = False
    _safe_cancel_effect_committed = False
    _safe_dismiss_committed = False
    _safe_opener_focus_ref: ReferenceType[Widget] | None = None
    _safe_backdrop_seen_committed = False

    def on_mount(self) -> None:
        """Remember the opener's focused widget for post-dismiss restoration."""
        self._safe_cancel_pending = False
        self._safe_cancel_effect_committed = False
        self._safe_dismiss_committed = False
        self._safe_opener_focus_ref = None
        self._safe_backdrop_seen_committed = False

        host = cast("_SafeModalHost", self)
        screen_stack = host.app.screen_stack
        if len(screen_stack) < 2:
            return
        opener = screen_stack[-2].focused
        self._safe_opener_focus_ref = ref(opener) if opener is not None else None

    def on_unmount(self) -> None:
        """Release the opener reference when the modal leaves the DOM."""
        self._safe_opener_focus_ref = None

    async def action_request_safe_cancel(self) -> None:
        """Route Escape to the modal's safe cancellation request."""
        await self.request_safe_cancel(source="escape")

    async def request_safe_cancel(self, *, source: str) -> None:
        """Run one cancellation request while consuming concurrent requests."""
        if source == "backdrop":
            self._safe_backdrop_seen_committed = True
        if self._safe_cancel_pending:
            return
        self._safe_cancel_pending = True
        try:
            await self._perform_safe_cancel(source=source)
        finally:
            if cast("_SafeModalHost", self).is_mounted:
                self._safe_cancel_pending = False

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
        if self._safe_dismiss_committed:
            return False
        host = cast("_SafeModalHost", self)
        if not host.is_mounted or host.app.screen is not self:
            return False

        self._safe_dismiss_committed = True
        app = host.app
        opener_ref = self._safe_opener_focus_ref
        host.dismiss(result)
        if self._safe_backdrop_seen_committed:
            _shield_revealed_screen_from_click_chain(app)
        app.screen.call_after_refresh(_restore_focus_after_dismissal, app, opener_ref)
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
        await self.request_safe_cancel(source="backdrop")
