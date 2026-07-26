"""Mouse-capture guard mixin for non-screen widgets that recompose themselves.

See ``tldw_chatbook.UI.Navigation.base_app_screen.BaseAppScreen`` for the
original bug and its full root-cause writeup (task-627): ``Widget.recompose()``
unconditionally removes and remounts every child. If ``App.mouse_captured``
currently points at one of those children -- e.g. an ``Input`` mid click/
selection whose ``MouseUp`` hasn't arrived yet -- capture is left referencing
a removed widget forever, because ``Input`` (unlike ``TextArea``/
``ScrollBar``) has no ``_on_hide`` handler to release it on removal. From
then on, every mouse event anywhere in the app is silently swallowed
(``Screen._forward_event``/``_handle_mouse_move`` both special-case
``if self.app.mouse_captured: ... self.find_widget(widget)``, which raises
``NoWidget`` for a detached target). Only a real screen switch self-heals
this (``App.push_screen``/``switch_screen``/``_replace_screen`` already call
``capture_mouse(None)`` defensively before swapping screens).

``BaseAppScreen`` closes that gap for the SCREEN's own recompose. It does
nothing for a *descendant* widget that recomposes itself independently --
either via an explicit ``self.refresh(recompose=True)`` call or a
``reactive(..., recompose=True)`` field -- while the enclosing screen is
never itself recomposed (task-637): a capture held by one of THAT widget's
own descendants at recompose time leaks exactly the same way.

``RecomposeCaptureGuard`` packages the same fix as a reusable mixin instead
of copy-pasting ``BaseAppScreen``'s override into every affected widget.
Add it to a widget's base list BEFORE its concrete Textual base, e.g.::

    class MyRail(RecomposeCaptureGuard, Vertical):
        ...

One deliberate difference from ``BaseAppScreen``: a ``Screen`` recompose
tears down its ENTIRE content, so any capture whatsoever must belong to
what's about to be removed (only one screen is ever interactively active),
and ``BaseAppScreen`` releases unconditionally. A non-screen widget is
typically only ONE part of a larger, otherwise-untouched screen -- releasing
unconditionally here could drop a legitimate, still-attached capture that
belongs to a sibling widget having nothing to do with this recompose. So
this mixin's pre-teardown release (in both ``refresh()`` and ``recompose()``)
only fires when the current capture is ``self`` or one of ``self``'s own
descendants (via ``ancestors_with_self``); the post-recompose sweep stays
unconditional on attachment, same as ``BaseAppScreen``, since a detached
widget can never be a legitimate capture for anyone.
"""

from typing import TYPE_CHECKING, Optional
from loguru import logger

from textual.geometry import Region
from textual.widget import Widget

if TYPE_CHECKING:
    from typing_extensions import Self


class RecomposeCaptureGuard:
    """Mixin: release stale mouse capture around this widget's own recompose.

    Semantics mirror ``BaseAppScreen.refresh``/``BaseAppScreen.recompose``
    (task-627), narrowed to this widget's own subtree (see module docstring
    for why): a capture on ``self`` or a descendant of ``self`` is released
    before the teardown that would otherwise orphan it; a capture left
    pointing at an already-detached widget after the recompose completes is
    swept regardless of ownership. A capture belonging to an unrelated,
    still-attached widget elsewhere on the screen is never touched.
    """

    def _capture_is_within_self(self, captured: Optional[Widget]) -> bool:
        """True if ``captured`` is ``self`` or lives inside ``self``'s subtree."""
        if captured is None:
            return False
        if captured is self:
            return True
        try:
            return self in captured.ancestors_with_self
        except Exception:
            return False

    def _release_own_capture_if_any(self, *, context: str) -> None:
        try:
            app = self.app
        except Exception:
            return
        captured = getattr(app, "mouse_captured", None)
        if not self._capture_is_within_self(captured):
            return
        try:
            app.capture_mouse(None)
        except Exception:
            logger.debug(
                f"{type(self).__name__}: mouse-capture release {context} skipped.",
                exc_info=True,
            )

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> "Self":
        """Recompose, releasing this widget's own stale mouse capture first.

        Mirrors ``BaseAppScreen.refresh`` -- released here (at the moment
        ``refresh(recompose=True)`` is CALLED) so the common, already-idle
        case never depends on the deferred-teardown guard below at all.
        """
        if recompose and self.is_running:
            self._release_own_capture_if_any(context="before recompose")
        return super().refresh(  # type: ignore[misc]
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    async def recompose(self) -> None:
        """Release capture again immediately before the actual teardown, then
        sweep any capture left dangling on a now-detached widget afterward.

        Mirrors ``BaseAppScreen.recompose`` exactly (see its docstring for
        the full reasoning on why a single call-time release in ``refresh()``
        is not enough: ``refresh(recompose=True)`` only *schedules* the real
        teardown via ``call_next``, and Textual lets each child's own message
        pump drain during ``super().recompose()``'s own removal, so a
        capture can land AFTER this method's own pre-teardown release but
        DURING the drain it triggers).
        """
        self._release_own_capture_if_any(context="before recompose teardown")
        await super().recompose()  # type: ignore[misc]
        if self.is_running:
            captured = self.app.mouse_captured
            if captured is not None and not captured.is_attached:
                try:
                    self.app.capture_mouse(None)
                except Exception:
                    logger.debug(
                        f"{type(self).__name__}: stale post-recompose "
                        "mouse-capture sweep skipped.",
                        exc_info=True,
                    )
