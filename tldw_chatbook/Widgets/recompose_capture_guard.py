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
from textual.css.query import QueryError
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

    #: True from the start of this widget's own ``recompose()`` teardown until
    #: its replacement children have mounted (TASK-33621.12). In that window a
    #: focused child has already been removed and its same-id replacement does
    #: not exist yet, so a focus-recovery pass that resolves "the control with
    #: the id that had focus" must wait rather than settle for a neighbour --
    #: see ``ConsoleLeftRail._recover_pending_focus``. Class-level default so
    #: it reads False on an instance that has never recomposed.
    recompose_in_flight: bool = False

    #: Opt-in (TASK-33621.12): when one of this widget's own descendants had
    #: focus as its rebuild began, put focus back on the unique same-id
    #: replacement once the rebuild has mounted. The rebuild removes the
    #: focused control, and Textual's ``_reset_focus`` then moves focus to
    #: whatever precedes it in the focus chain -- for the Console's
    #: Conversations tray that was the section toggle, "New conversation", or
    #: the Console header's Settings control. Only focus still sitting on
    #: that automatic reset target is taken back: a move made on purpose
    #: while the rebuild ran (selecting a chat focuses the composer mid-
    #: rebuild) is left alone. Off by default so no other guarded widget
    #: changes behaviour.
    RECOMPOSE_KEEPS_FOCUS: bool = False

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
            # Loguru does not honor the stdlib `exc_info=True` kwarg -- it is
            # bound as an opaque "extra" field instead, and the traceback is
            # silently dropped. `logger.opt(exception=True)` is loguru's own
            # mechanism for attaching the current exception to a log record.
            logger.opt(exception=True).debug(
                f"{type(self).__name__}: mouse-capture release {context} skipped."
            )

    def _focus_to_keep(self) -> Optional[tuple[str, Optional[Widget]]]:
        """The focused descendant's id and where Textual's reset will send it.

        Returns ``None`` unless a descendant of ``self`` (not ``self``) with a
        DOM id holds this screen's focus. Read from the screen that owns
        ``self`` -- the one whose ``_reset_focus`` the teardown triggers --
        so a modal on top does not hide a focused row underneath it.
        """
        try:
            focused = self.screen.focused
        except Exception:
            return None
        if focused is None or focused is self or self not in focused.ancestors:
            return None
        if not focused.id:
            return None
        return focused.id, self._predicted_reset_target(focused)

    def _predicted_reset_target(self, focused: Widget) -> Optional[Widget]:
        """Mirror ``Screen._reset_focus`` for a teardown of all of ``self``'s children.

        Textual walks the focus chain backwards from the removed widget,
        wrapping around, and takes the first control not being removed; a
        focused widget outside the chain falls back to a focusable sibling.
        ``Tests/Widgets/test_recompose_capture_guard.py`` pins the wrap-around.
        """

        def removed(widget: Widget) -> bool:
            return self in widget.ancestors

        try:
            chain = self.screen.focus_chain
        except Exception:
            return None
        if focused not in chain:
            for sibling in focused.visible_siblings:
                if not removed(sibling) and sibling.focusable:
                    return sibling
            return None
        index = chain.index(focused)
        for candidate in reversed(chain[index + 1 :] + chain[:index]):
            if not removed(candidate):
                return candidate
        return None

    def _refocus_rebuilt_descendant(
        self, widget_id: str, reset_target: Optional[Widget]
    ) -> None:
        """Return focus to the same-id replacement a rebuild just mounted.

        Only while focus is still exactly where Textual's reset put it
        (``reset_target``, predicted before the teardown). Anything else means
        focus moved during the rebuild -- the user clicked, or code such as
        a chat selection focused the composer -- and that newer move wins.
        The focus is set synchronously, so a move that was only queued
        (``Widget.focus`` defers through ``call_later``) still lands after
        this one. Exactly one focusable match is required, so an ambiguous
        or vanished id changes nothing.
        """
        if not self.is_attached:
            return
        try:
            screen = self.screen
        except Exception:
            return
        if screen.focused is not reset_target:
            return
        try:
            candidates = list(self.query(f"#{widget_id}"))
        except QueryError:
            return
        matches = [
            widget for widget in candidates if widget.is_mounted and widget.focusable
        ]
        if len(matches) == 1:
            screen.set_focus(matches[0])

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

        Args:
            *regions: Regions to repaint; forwarded to ``Widget.refresh``
                unchanged.
            repaint: Whether to repaint the widget; forwarded unchanged.
            layout: Whether to trigger a layout pass; forwarded unchanged.
            recompose: If ``True``, schedules a full recompose. Only this
                case triggers the pre-teardown mouse-capture release.

        Returns:
            ``self``, per ``Widget.refresh``'s own return contract.
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
        # Await nothing between this and the teardown: the prediction must see
        # the focus chain the teardown's reset will see. (Only a contended
        # widget lock inside super().recompose() can still delay it; a stale
        # prediction then just means no restore -- the pre-opt-in behaviour.)
        keep_focus = self._focus_to_keep() if self.RECOMPOSE_KEEPS_FOCUS else None
        self.recompose_in_flight = True
        try:
            await super().recompose()  # type: ignore[misc]
        finally:
            self.recompose_in_flight = False
        if keep_focus is not None:
            self._refocus_rebuilt_descendant(*keep_focus)
        if self.is_running:
            captured = self.app.mouse_captured
            if captured is not None and not captured.is_attached:
                try:
                    self.app.capture_mouse(None)
                except Exception:
                    # See the matching comment in
                    # `_release_own_capture_if_any`: loguru needs
                    # `logger.opt(exception=True)`, not stdlib's
                    # `exc_info=True`, to attach a traceback.
                    logger.opt(exception=True).debug(
                        f"{type(self).__name__}: stale post-recompose "
                        "mouse-capture sweep skipped."
                    )
