"""Post-recompose callback plumbing for canvas-scoped Library updates.

task-15457. A Library screen handler that recomposed the WHOLE screen could
schedule its follow-up work -- almost always "focus the control the user
should now be on" -- with ``screen.call_after_refresh(...)``: both the
recompose and the callback are driven by the screen's own message pump, and
``Screen._on_timer_update`` runs the recompose via ``call_next`` BEFORE it
runs ``_invoke_and_clear_callbacks``, so the ordering is guaranteed.

A canvas-scoped update breaks that guarantee. ``canvas.refresh(recompose=
True)`` is serviced by the CANVAS's pump (its ``_on_idle`` ->
``_check_recompose``), while ``screen.call_after_refresh`` still queues onto
the screen's callback list -- two independent pumps with no ordering between
them. Reproduced while converting the notes sort/filter strips: every
converted site's focus follow-up ran against the OLD children (or after they
were removed), leaving DOM focus stranded on an unrelated widget outside the
canvas instead of the strip's opener.

This mixin restores the ordering by hanging the follow-up on the widget that
actually does the work: the callback is stored on the canvas instance and
fired from its own ``recompose()``, immediately after the new children are
mounted. Storing it on the INSTANCE (not a screen-side dict keyed by widget
id) is deliberate -- the recompose-lifecycle rule from the July programme:
fresh widgets must carry current state, and per-widget bookkeeping that lives
anywhere else goes stale the moment a widget is replaced.
"""

from __future__ import annotations

from typing import Callable, Optional

from loguru import logger
from textual.css.query import NoMatches


class PostRecomposeCallback:
    """Mixin: run one queued callback right after this widget recomposes.

    Cooperative with ``RecomposeCaptureGuard`` -- both override
    ``recompose()`` and chain through ``super()``, so a canvas can use both
    (list them in whatever order; each does its own work around the shared
    ``super().recompose()``).
    """

    #: Class-level default so the attribute is readable before any queue call
    #: and on any subclass that never queues one.
    _post_recompose_callback: Optional[Callable[[], None]] = None

    def queue_after_recompose(self, callback: Optional[Callable[[], None]]) -> None:
        """Queue ``callback`` to run once, after the next recompose.

        A second call before that recompose replaces the first: the pending
        callback is always the most recent caller's intent, matching how a
        second ``refresh(recompose=True)`` supersedes the first.

        Args:
            callback: Zero-argument callable, or ``None`` to clear.

        Returns:
            None.
        """
        self._post_recompose_callback = callback

    def preserve_same_id_focus_after_recompose(self) -> None:
        """Replace a focused child with its newly composed same-ID owner."""
        focused = self.app.focused
        focused_id = getattr(focused, "id", None)
        if not self.is_attached or focused is None or not focused_id:
            return
        belongs_to_canvas = self in focused.ancestors
        if focused.parent is None and not belongs_to_canvas:
            try:
                self.query_one(f"#{focused_id}")
            except NoMatches:
                return
            belongs_to_canvas = True
        if not belongs_to_canvas:
            return

        pending = self._post_recompose_callback
        self.screen.set_focus(None)

        def restore_focused_child_after_recompose() -> None:
            live_focus = self.app.focused
            if not (
                live_focus is not None
                and live_focus is not focused
                and live_focus.parent is not None
            ):
                try:
                    self.query_one(f"#{focused_id}").focus()
                except NoMatches:
                    pass
            if pending is not None:
                pending()

        self.queue_after_recompose(restore_focused_child_after_recompose)

    def _after_recompose(self) -> None:
        """Subclass hook: post-compose wiring, run BEFORE the queued callback.

        ``on_mount`` fires once, when the widget itself mounts -- a
        ``refresh(recompose=True)`` remounts only its CHILDREN, so any
        wiring that hook performs has to be re-run here. Ordered ahead of
        the callback deliberately: a queued follow-up focuses or measures a
        child, and must see it in its final, post-wiring form (for the notes
        canvas that means after ``apply_compact_presentation`` has rewritten
        the compact labels).
        """
        return None

    async def recompose(self) -> None:
        """Recompose, then re-wire and fire the queued callback."""
        await super().recompose()  # type: ignore[misc]
        callback = self._post_recompose_callback
        # Cleared unconditionally: a callback that itself queues another one
        # (or raises) must not be re-run by the next recompose, and a
        # callback whose recompose never happened must not fire against some
        # unrelated LATER one.
        self._post_recompose_callback = None
        # ``Widget.recompose`` early-returns without touching the children
        # when the widget is detached or being pruned. Nothing was rebuilt,
        # so neither the re-wiring nor the follow-up has a tree to act on --
        # and a follow-up that focuses a child of a detached widget is
        # actively wrong, not merely useless.
        if not self.is_attached or getattr(self, "_pruning", False):
            return
        self._after_recompose()
        if callback is None:
            return
        # A newer recompose is already queued: this widget's children are
        # about to be replaced again, so firing the follow-up now would run it
        # against a tree that is already stale. Textual re-arms
        # ``_recompose_required`` whenever ``refresh(recompose=True)`` is
        # called, which is exactly what a second ``sync_state`` landing while
        # this recompose was awaiting ``mount_all`` does -- the shape the
        # notes canvas's own ``_apply_post_compose_state`` guard was added for
        # (list -> loading -> editor row press). Re-queue instead of dropping,
        # so the follow-up runs once, against the final children.
        if getattr(self, "_recompose_required", False):
            self._post_recompose_callback = callback
            return
        try:
            callback()
        except Exception:
            logger.debug("Library post-recompose callback failed")
