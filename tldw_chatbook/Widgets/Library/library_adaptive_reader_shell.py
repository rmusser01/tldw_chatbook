"""Retained three-role structure for Library adaptive readers."""

from __future__ import annotations

from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Horizontal
from textual.events import DescendantFocus
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    PaneName,
)


LIBRARY_ARROW_UPPER_POSITION_RATIO = 0.35

#: Shared class every adaptive reader shell puts on BOTH of its pane grips.
#: Named here so focus code can recognise a grip without a magic string
#: (task-31567: the grips are the shell's first focusable widgets, so a
#: recompose hands them focus unless someone puts it back).
LIBRARY_ADAPTIVE_READER_GRIP_CLASS = "library-adaptive-reader-pane-grip"


class PaneToggleRequested(Message):
    """Request a manual toggle of one optional pane."""

    def __init__(self, pane: Literal["library", "items"]) -> None:
        super().__init__()
        self.pane = pane


class AdaptiveReaderShellResized(Message):
    """Report that the settled shell allocation may need resolving."""


class LibraryPaneVisibilityChanged(Message):
    """Report that an optional pane's APPLIED visibility changed.

    task-32225: distinct from ``AdaptiveReaderShellResized``, which the shell
    posts from ``on_resize`` -- its OWN size. A pane toggle only changes child
    widths, so nothing announced "the Library pane is now closed" and the
    footer's narrow-stage return chip went stale in both directions. Posted
    from ``sync_layout``, the one place an applied layout is installed, so no
    destination can forget to announce it.

    Attributes:
        pane: Which optional pane changed.
        open: Its applied visibility after the change.
    """

    def __init__(self, pane: PaneName, open: bool) -> None:
        super().__init__()
        self.pane = pane
        self.open = open


class LibraryAdaptiveReaderPaneGrip(Button):
    """Narrow keyboard and pointer control for one optional pane.

    ``width`` is the destination profile's ``grip_width``: the grip paints
    exactly the columns the resolver held back for it (task-31633 AC#2).
    """

    BINDINGS = [Binding("enter,space", "press", "Press button", show=False)]

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        extra_classes: str = "",
        width: int = PANE_GRIP_WIDTH,
        **kwargs: Any,
    ) -> None:
        """Build one pane grip sized to its destination's profile.

        Args:
            pane: Which optional pane this grip toggles -- ``"library"`` or
                ``"items"``. Carried on the ``PaneToggleRequested`` message
                the press posts.
            open: Whether that pane is open right now. Decides the arrow and
                the action copy only; geometry never changes with it.
            pane_label: Human name of the pane, used verbatim in the tooltip
                and accessible name ("Collapse Items pane").
            extra_classes: Space-separated CSS classes appended to
                ``LIBRARY_ADAPTIVE_READER_GRIP_CLASS`` for per-destination
                styling. The shared class is always present -- the focus
                restore seam reads it to know it must never land here.
            width: The destination profile's ``grip_width``, in cells. The
                grip paints exactly the columns the resolver held back for it
                (task-31633 AC#2); below four cells the arrow becomes a
                one-cell guillemet.
            **kwargs: Forwarded to ``Button`` (``id``, ``disabled``, ...).
        """
        self.pane = pane
        self.pane_label = pane_label
        self.grip_width = width
        classes = LIBRARY_ADAPTIVE_READER_GRIP_CLASS
        if extra_classes:
            classes = f"{classes} {extra_classes}"
        super().__init__(compact=True, flat=True, classes=classes, **kwargs)
        self.sync_width(width)
        self.styles.height = "100%"
        self.styles.padding = 0
        self.styles.line_pad = 0
        self.styles.border = ("none", "transparent")
        self.styles.content_align = ("center", "middle")
        self.sync_open(open)

    def sync_width(self, width: int) -> None:
        """Size the grip to the layout's reservation, in place.

        Args:
            width: The resolved layout's ``grip_width`` in cells.

        Returns:
            None.
        """
        self.grip_width = width
        self.styles.width = width
        self.styles.min_width = width
        self.styles.max_width = width

    def sync_open(self, open: bool) -> None:
        """Patch arrow and action copy without changing geometry.

        The in-place alternative to recomposing the grip: label, accessible
        name and tooltip are assigned only when they actually differ, so a
        shell re-sync that changes nothing costs no refresh -- and the grip
        cannot be the widget a recompose detaches while it holds focus.

        Args:
            open: Whether the pane this grip toggles is now open.

        Returns:
            None.
        """
        action = "Collapse" if open else "Expand"
        copy = f"{action} {self.pane_label} pane"
        # task-31633 AC#2: the arrow is as wide as the grip. The
        # "<---"/"--->" run is four cells, so a grip narrower than that would
        # paint a truncated "<" -- it takes the one-cell guillemet instead.
        if self.grip_width < len("<---"):
            label = "‹" if open else "›"
        else:
            label = "<---" if open else "--->"
        if self.label != label:
            self.label = label
        if self._name != copy:
            self._name = copy
        if self.tooltip != copy:
            self.tooltip = copy

    def render(self) -> Content:
        """Paint the Library pair around the single centered Items arrow.

        Returns:
            Content: Full-height grip content with arrows at the approved rows.
        """
        height = max(self.content_region.height, 1)
        last_row = height - 1
        if self.pane == "library" and height > 1:
            upper_row = round(last_row * LIBRARY_ARROW_UPPER_POSITION_RATIO)
            arrow_rows = {upper_row, last_row - upper_row}
        else:
            arrow_rows = {last_row // 2}
        arrow = self.label.plain
        return Content.from_text(
            "\n".join(arrow if row in arrow_rows else " " for row in range(height))
        )

    @on(Button.Pressed)
    def request_toggle(self, event: Button.Pressed) -> None:
        """Translate native Button activation into the shell message."""
        if event.button is not self:
            return
        event.stop()
        self.post_message(PaneToggleRequested(self.pane))


class LibraryAdaptiveReaderShell(Horizontal):
    """Own adaptive reader structure while callers own state and behavior."""

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        """Assemble the three-pane shell around caller-owned pane widgets.

        Args:
            library: Widget for the leftmost (Library rail) pane.
            items: Widget for the middle (list) pane.
            work: Widget for the work pane -- the Reader or its equivalent.
            layout: The resolved layout to mount with: which optional panes
                are open and how wide each is.
            id_prefix: Per-destination id stem (``"library-media"``) for the
                composed grips, giving each destination its own stable
                selectors.
            library_label: Human name of the Library pane, for grip copy.
            items_label: Human name of the items pane, for grip copy.
            grip_classes: Extra CSS classes for both grips, for
                per-destination styling.
            **kwargs: Forwarded to ``Horizontal`` (``id``, ``classes``, ...).

        Both grips are sized from ``layout.grip_width`` -- the width the
        resolver held back for them (task-31952 AC#3), so a caller cannot
        paint a grip the resolver never reserved.
        """
        super().__init__(**kwargs)
        self.add_class("library-adaptive-reader-shell")
        self.library = library
        self.items = items
        self.work = work
        self.library.add_class("library-adaptive-reader-library")
        self.items.add_class("library-adaptive-reader-items")
        self.work.add_class("library-adaptive-reader-work")
        self.library_grip = LibraryAdaptiveReaderPaneGrip(
            "library",
            open=layout.library_open,
            pane_label=library_label,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-library-grip",
        )
        self.items_grip = LibraryAdaptiveReaderPaneGrip(
            "items",
            open=layout.items_open,
            pane_label=items_label,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-items-grip",
        )
        self._last_focused_descendant: dict[PaneName, Widget | None] = {
            "library": None,
            "items": None,
        }
        self.effective_layout = layout
        self._applied_layout: AdaptiveReaderEffectiveLayout | None = None

    def compose(self) -> ComposeResult:
        """Compose retained Library, Items, grips, and Work widgets."""
        yield self.library
        yield self.library_grip
        yield self.items
        yield self.items_grip
        yield self.work

    def on_mount(self) -> None:
        """Apply initial geometry and request a settled resize projection."""
        self.sync_layout(self.effective_layout)
        self.call_after_refresh(self.post_message, AdaptiveReaderShellResized())

    def on_resize(self) -> None:
        """Request layout resolution after the shell allocation changes."""
        self.post_message(AdaptiveReaderShellResized())

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Remember optional-pane focus before a grip activation moves it."""
        target = event.widget
        for pane_name, pane in (
            ("library", self.library),
            ("items", self.items),
        ):
            if self._is_valid_focus_target(pane, target):
                self._last_focused_descendant[pane_name] = target
                return

    def _pane_focus_chain(self, pane: Widget) -> list[Widget]:
        """Return currently reachable pane targets in Textual focus order."""
        if not self.is_mounted:
            return []
        return [
            target
            for target in self.app.screen.focus_chain
            if target is pane or pane in target.ancestors
        ]

    def _is_valid_focus_target(self, pane: Widget, target: Widget | None) -> bool:
        """Return whether ``target`` is currently reachable within ``pane``."""
        return target is not None and target in self._pane_focus_chain(pane)

    def sync_layout(
        self,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        manual_reopen: PaneName | None = None,
    ) -> None:
        """Patch pane display and exact cell widths in place."""
        previous_layout = self._applied_layout
        self.effective_layout = layout
        focused = self.app.focused if self.is_mounted else None
        evacuation_target: Widget | None = None
        manual_reopen_pane: Widget | None = None
        manual_reopen_name: PaneName | None = None
        automatic_reopen_target: Widget | None = None
        for pane_name, pane, grip, open, width, was_open in (
            (
                "library",
                self.library,
                self.library_grip,
                layout.library_open,
                layout.library_width,
                (
                    previous_layout.library_open
                    if previous_layout is not None
                    else layout.library_open
                ),
            ),
            (
                "items",
                self.items,
                self.items_grip,
                layout.items_open,
                layout.items_width,
                (
                    previous_layout.items_open
                    if previous_layout is not None
                    else layout.items_open
                ),
            ),
        ):
            if (
                not open
                and focused is not None
                and (focused is pane or pane in focused.ancestors)
            ):
                if focused is not pane and self._is_valid_focus_target(pane, focused):
                    self._last_focused_descendant[pane_name] = focused
                evacuation_target = grip
            if pane.display != open:
                pane.display = open
            if pane.disabled != (not open):
                pane.disabled = not open
            if pane.styles.width is None or pane.styles.width.value != width:
                pane.styles.width = width
            if pane.styles.min_width is None or pane.styles.min_width.value != width:
                pane.styles.min_width = width
            if pane.styles.max_width is None or pane.styles.max_width.value != width:
                pane.styles.max_width = width
            if previous_layout is None:
                pane.styles.height = "100%"
            if open and not was_open and focused is grip:
                automatic_reopen_target = next(
                    (
                        candidate
                        for candidate in self.screen.focus_chain
                        if (candidate is pane or pane in candidate.ancestors)
                        and candidate.display
                        and not candidate.disabled
                    ),
                    grip,
                )
            if grip.grip_width != layout.grip_width:
                # Reserve-and-paint holds for every layout the shell is given,
                # not only the one it was built with (task-31952 AC#3).
                grip.sync_width(layout.grip_width)
            grip.sync_open(open)
            if open and not was_open and manual_reopen == pane_name:
                manual_reopen_pane = pane
                manual_reopen_name = pane_name
        if previous_layout is None:
            self.work.display = True
            self.work.styles.width = "1fr"
            self.work.styles.min_width = 0
            self.work.styles.height = "100%"
        for pane_name, was_open, now_open in (
            (
                "library",
                None if previous_layout is None else previous_layout.library_open,
                layout.library_open,
            ),
            (
                "items",
                None if previous_layout is None else previous_layout.items_open,
                layout.items_open,
            ),
        ):
            if was_open != now_open:
                self.post_message(LibraryPaneVisibilityChanged(pane_name, now_open))
        self._applied_layout = layout
        if evacuation_target is not None:
            self.screen.set_focus(evacuation_target, scroll_visible=False)
        elif manual_reopen_pane is not None and manual_reopen_name is not None:
            focus_chain = self._pane_focus_chain(manual_reopen_pane)
            target = self._last_focused_descendant[manual_reopen_name]
            if target not in focus_chain:
                target = next(iter(focus_chain), None)
            if target is not None:
                self.screen.set_focus(target, scroll_visible=False)
        elif automatic_reopen_target is not None:
            # Keep focus recovery synchronous so it cannot overwrite a newer
            # explicit focus change queued before the next refresh.
            self.screen.set_focus(automatic_reopen_target, scroll_visible=False)
