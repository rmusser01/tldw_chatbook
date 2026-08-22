"""Presentation-only bounded content body for direct Console rail sections."""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantBlur, DescendantFocus
from textual.widget import Widget
from textual.widgets import Static


MAX_SECTION_CONTENT_LINES = 20
LOCAL_SCROLL_HINT = "▼ more — scroll"


class _BoundedSectionViewport(VerticalScroll):
    """Native Textual viewport with a scroll-position observation seam."""

    def __init__(
        self,
        *children: Widget,
        on_scroll_changed: Callable[[], None],
        id: str,
    ) -> None:
        super().__init__(
            *children,
            id=id,
            classes="console-bounded-section-viewport",
            can_focus=False,
        )
        self._on_scroll_changed = on_scroll_changed

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Retain native scroll behavior and refresh only the fold copy."""

        super().watch_scroll_y(old_value, new_value)
        if old_value != new_value:
            self._on_scroll_changed()


class ConsoleBoundedSection(Vertical):
    """Lay out owner-supplied content within a bounded local viewport.

    The widget owns only transient presentation state. Owners retain domain data,
    headers, mutation timing, allocation coordination, and focus recovery policy.

    Args:
        *content: Already-built widgets to mount inside the content viewport.
        section_id: Stable suffix used to derive the root, viewport, and hint IDs.
        allocation: Optional owner allocation below the 20-line ceiling. ``None``
            uses the full ceiling.
        max_content_lines: Rail ceiling. The shared Console contract fixes this at
            20; the argument makes that contract explicit at construction.
        on_focus_recovery: Called when the last focused content descendant is
            removed before reconciliation.
        classes: Additional classes for the root widget.
    """

    def __init__(
        self,
        *content: Widget,
        section_id: str,
        allocation: int | None = None,
        max_content_lines: int = MAX_SECTION_CONTENT_LINES,
        on_focus_recovery: Callable[[], None] | None = None,
        classes: str | None = None,
    ) -> None:
        if max_content_lines != MAX_SECTION_CONTENT_LINES:
            raise ValueError(
                "Console bounded sections use a fixed 20-line content ceiling"
            )
        self.section_id = section_id
        self.root_id = f"console-bounded-section-{section_id}"
        self.viewport_id = f"{self.root_id}-viewport"
        self.hint_id = f"{self.root_id}-hint"
        root_classes = "console-bounded-section"
        if classes:
            root_classes = f"{root_classes} {classes}"
        super().__init__(id=self.root_id, classes=root_classes)

        self.max_content_lines = max_content_lines
        self._allocation = self._normalize_allocation(allocation)
        self._on_focus_recovery = on_focus_recovery
        self._desired_content_lines = 0
        self._has_overflow = False
        self._hint_text = ""
        self._reconcile_scheduled = False
        self._focused_descendant: Widget | None = None
        self._focus_recovery_notified = False

        self._viewport = _BoundedSectionViewport(
            *content,
            on_scroll_changed=self._update_hint,
            id=self.viewport_id,
        )
        self._hint = Static(
            "",
            id=self.hint_id,
            classes="console-bounded-section-hint",
            markup=False,
        )
        self._hint.can_focus = False
        self._hint.display = False

    @staticmethod
    def _normalize_allocation(allocation: int | None) -> int | None:
        if allocation is None:
            return None
        if isinstance(allocation, bool) or not isinstance(allocation, int):
            raise TypeError("allocation must be an integer or None")
        if allocation < 0:
            raise ValueError("allocation must be non-negative")
        return min(allocation, MAX_SECTION_CONTENT_LINES)

    @property
    def allocation(self) -> int | None:
        """Current owner-supplied content-row allocation."""

        return self._allocation

    @allocation.setter
    def allocation(self, value: int | None) -> None:
        normalized = self._normalize_allocation(value)
        if normalized == self._allocation:
            return
        self._allocation = normalized
        self.request_reconcile()

    def set_allocation(self, allocation: int | None) -> None:
        """Apply an owner allocation and schedule one post-refresh reconcile."""

        self.allocation = allocation

    @property
    def desired_content_lines(self) -> int:
        """Uncapped physical content demand from the latest laid-out snapshot."""

        return self._desired_content_lines

    @property
    def viewport(self) -> VerticalScroll:
        """The local native Textual scroll viewport."""

        return self._viewport

    @property
    def hint(self) -> Static:
        """The always-mounted, non-focusable local fold hint."""

        return self._hint

    def compose(self) -> ComposeResult:
        yield self._viewport
        yield self._hint

    async def recompose(self) -> None:
        """Reconcile the stable owner-supplied subtree without pruning it.

        Textual's default recompose removes direct children before invoking
        ``compose`` again. Here those children contain widgets built and owned by
        the caller, so pruning would destroy the only content instances available
        to compose. The scaffold has no reactive composition branches; retaining it
        and scheduling the normal post-refresh geometry pass is the safe equivalent.
        """

        if self.is_mounted:
            # Queue the request itself after refresh so a same-tick content update
            # has completed layout before the ordinary reconciler snapshots it.
            self.call_after_refresh(self.request_reconcile)

    def on_mount(self) -> None:
        self._focus_recovery_notified = False
        self.request_reconcile()

    def on_show(self) -> None:
        self.request_reconcile()

    def on_resize(self) -> None:
        self.request_reconcile()

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember content focus and fully reveal it in the local viewport."""

        target = event.widget
        if target is self._viewport:
            return
        if self._viewport in target.ancestors:
            self._focus_recovery_notified = False
            self._focused_descendant = target
            self._viewport.scroll_to_widget(
                target,
                animate=False,
                immediate=True,
            )
            self._update_hint()

    def on_descendant_blur(self, event: DescendantBlur) -> None:
        """Close a recovery incident once focus reaches a valid outside owner."""

        if self._is_valid_outside_focus(self.app.focused):
            self._focus_recovery_notified = False

    def request_reconcile(self) -> None:
        """Coalesce geometry reconciliation into one post-refresh callback."""

        if not self.is_mounted or self._reconcile_scheduled:
            return
        self._reconcile_scheduled = True
        self.call_after_refresh(self._run_scheduled_reconcile)

    def _run_scheduled_reconcile(self) -> None:
        self._reconcile_scheduled = False
        self._reconcile()

    def _reconcile(self) -> None:
        """Apply one equality-guarded snapshot of laid-out content geometry."""

        if not self.display or any(
            isinstance(ancestor, Widget) and not ancestor.display
            for ancestor in self.ancestors
        ):
            # A hidden same-instance section has no honest geometry to measure.
            # Preserve its transient offset; ``Show`` schedules the real snapshot.
            return

        try:
            viewport = self.query_one(f"#{self.viewport_id}", VerticalScroll)
            hint = self.query_one(f"#{self.hint_id}", Static)
        except NoMatches:
            # During a recompose there may be a refresh where one part is absent.
            # Disable the retained parts without inventing geometry; the next owner,
            # mount, show, or resize request retries from a complete layout.
            self._fail_closed()
            return

        desired = self._measure_content_lines(viewport)
        limit = self.max_content_lines
        if self._allocation is not None:
            limit = min(limit, self._allocation)
        target_height = min(desired, limit)

        if self._desired_content_lines != desired:
            self._desired_content_lines = desired

        current_height = viewport.content_region.height
        if current_height != target_height:
            viewport.styles.height = target_height
            viewport.scroll_y = min(
                viewport.scroll_y,
                max(0, desired - target_height),
            )
            self._set_viewport_focusable(
                viewport,
                focusable=False,
                recover_owned_focus=not (desired > target_height > 0),
            )
            self._has_overflow = False
            self._set_hint_layout(hint, visible=False)
            self.request_reconcile()
            return

        max_scroll_y = max(0, viewport.max_scroll_y)
        if viewport.scroll_y > max_scroll_y:
            viewport.scroll_y = max_scroll_y

        has_overflow = desired > current_height > 0 and max_scroll_y > 0
        self._set_viewport_focusable(
            viewport,
            focusable=has_overflow,
            recover_owned_focus=not has_overflow,
        )
        self._has_overflow = has_overflow
        self._set_hint_layout(hint, visible=has_overflow)
        self._recover_removed_focus_target()
        self._update_hint()

    @staticmethod
    def _measure_content_lines(viewport: VerticalScroll) -> int:
        """Return the uncapped bottom edge of laid-out visible children."""

        return max(
            (
                child.virtual_region_with_margin.bottom
                for child in viewport.children
                if child.display
            ),
            default=0,
        )

    def _recover_removed_focus_target(self) -> None:
        target = self._focused_descendant
        if target is None:
            return
        if target.is_mounted and self._viewport in target.ancestors:
            focused = self.app.focused
            if focused is not target and (
                focused is None or self._viewport not in focused.ancestors
            ):
                self._focused_descendant = None
            return

        self._focused_descendant = None
        focused = self.app.focused
        if self._is_valid_outside_focus(focused):
            self._focus_recovery_notified = False
            return
        self._notify_focus_recovery()

    def _set_viewport_focusable(
        self,
        viewport: VerticalScroll,
        *,
        focusable: bool,
        recover_owned_focus: bool,
    ) -> None:
        was_focusable = viewport.can_focus
        if was_focusable is not focusable:
            viewport.can_focus = focusable
        if focusable and not was_focusable:
            # Becoming a scroll owner again starts a fresh focus lifecycle.
            self._focus_recovery_notified = False
        if (
            recover_owned_focus
            and was_focusable
            and not focusable
            and self.app.focused is viewport
        ):
            # One owner callback covers the whole invalidated focus incident.
            # Textual may move a removed descendant to this viewport before the
            # geometry pass; clearing the stale target prevents a second callback.
            self._focused_descendant = None
            self._notify_focus_recovery()

    def _owns_widget(self, widget: Widget) -> bool:
        return widget is self._viewport or self._viewport in widget.ancestors

    def _is_valid_outside_focus(self, widget: Widget | None) -> bool:
        return bool(
            widget is not None
            and not self._owns_widget(widget)
            and widget.is_mounted
            and widget.focusable
            and widget.display
            and all(
                not isinstance(ancestor, Widget) or ancestor.display
                for ancestor in widget.ancestors
            )
        )

    def _notify_focus_recovery(self) -> None:
        callback = self._on_focus_recovery
        if callback is None or self._focus_recovery_notified:
            return
        self._focus_recovery_notified = True
        callback()

    def _fail_closed(self) -> None:
        self._has_overflow = False
        self._viewport.can_focus = False
        self._set_hint_layout(self._hint, visible=False)

    def _set_hint_layout(self, hint: Static, *, visible: bool) -> None:
        if hint.display is not visible:
            hint.display = visible
        if not visible:
            self._set_hint_text("")

    def _update_hint(self) -> None:
        if not self._has_overflow or not self._hint.display:
            self._set_hint_text("")
            return
        has_more_below = (
            self._viewport.max_scroll_y > 0
            and self._viewport.scroll_y < self._viewport.max_scroll_y
        )
        self._set_hint_text(LOCAL_SCROLL_HINT if has_more_below else "")

    def _set_hint_text(self, text: str) -> None:
        if text == self._hint_text:
            return
        self._hint_text = text
        self._hint.update(text)


__all__ = [
    "ConsoleBoundedSection",
    "LOCAL_SCROLL_HINT",
    "MAX_SECTION_CONTENT_LINES",
]
