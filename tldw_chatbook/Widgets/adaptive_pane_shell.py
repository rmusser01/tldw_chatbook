"""Shared adaptive pane shell for destination frames (Roleplay frame B0).

One three-pane structure -- a navigation rail, an items list and a work pane,
with a full-height grip after each optional pane -- shared by the Library's
adaptive readers and (from Roleplay frame B1) the Roleplay destination. The
grip, shell and messages moved here from the Library's
``Widgets/Library/library_adaptive_reader_shell.py`` unchanged in behaviour;
that module keeps the Library's names as thin subclasses and same-object
aliases.

Placement rules (the shared adaptive-pane-shell ADR, ``backlog/decisions/``):

- Never import this module from a UI-ready-resident module (for example
  ``Widgets/destination_rail.py``): the UI-ready census has no headroom.
  Destination modules import it, and they load with their own route.
- Import nothing from ``Widgets/Library/``, ``Library/`` or
  ``UI/Library_Modules/``: Roleplay imports this module and must never pull
  the Library package in with it.
- Declare no ``DEFAULT_CSS``, ``CSS`` or ``BUNDLED_CSS``. Every shell, grip
  and row rule is keyed to a destination's own classes
  (``AdaptivePaneClasses``) and lives in that destination's lazy split
  sheet, so this module costs zero boot CSS bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

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
    AdaptivePaneLayout,
    PaneName,
)

#: Where the navigation pane's upper arrow sits, as a fraction of the grip's
#: height; the lower arrow mirrors it (task-32355). The items grip paints one
#: centred arrow.
ARROW_UPPER_POSITION_RATIO = 0.35


@dataclass(frozen=True)
class AdaptivePaneClasses:
    """The destination classes one adaptive pane shell puts on its parts.

    Every rule that styles a shell is keyed to these classes and lives in the
    destination's lazy split sheet (the shared pane-shell ADR). The CSS build moves a rule to a
    lazy sheet only when every class token in its selector carries that
    sheet's owner prefix; a neutral token has no owner, so it would pin the
    rule to the boot bundle. Each value therefore starts with the
    destination's split prefix.

    Attributes:
        shell: Class on the shell container itself.
        nav: Class on the navigation rail (the resolver's ``"library"`` pane).
        items: Class on the items list (the ``"items"`` pane).
        work: Class on the work pane.
        grip: Class on both pane grips. Focus code also reads it to recognise
            a grip without a magic string.
    """

    shell: str
    nav: str
    items: str
    work: str
    grip: str


class PaneToggleRequested(Message):
    """Request a manual toggle of one optional pane.

    Attributes:
        pane: ``"library"`` (the navigation pane) or ``"items"``: the
            resolver's ``PaneName`` values, the same for every destination.
    """

    def __init__(self, pane: PaneName) -> None:
        super().__init__()
        self.pane = pane


class AdaptivePaneShellResized(Message):
    """Report that the settled shell allocation may need resolving."""


class PaneVisibilityChanged(Message):
    """Report that an optional pane's APPLIED visibility changed.

    task-32225: distinct from ``AdaptivePaneShellResized``, which the shell
    posts from ``on_resize`` -- its OWN size. A pane toggle only changes child
    widths, so nothing announced "the navigation pane is now closed" and a
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


class AdaptivePaneGrip(Button):
    """Narrow keyboard and pointer control for one optional pane.

    ``width`` is the destination profile's ``grip_width``: the grip paints
    exactly the columns the resolver held back for it (task-31633 AC#2). The
    grip carries its destination's grip class and paints its pane's name down
    its own column, through ``painted_names`` where the painted noun differs
    from the spoken one.
    """

    BINDINGS = [Binding("enter,space", "press", "Press button", show=False)]

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        destination_class: str,
        painted_names: Mapping[str, str] | None = None,
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
            destination_class: The destination's grip class
                (``AdaptivePaneClasses.grip``). Always present: focus code
                reads it to know it must never restore focus onto a grip.
            painted_names: ``pane_label`` -> the noun painted down the column
                where it differs from the spoken label. ``None`` paints the
                label itself.
            extra_classes: Space-separated CSS classes appended to the
                destination class.
            width: The destination profile's ``grip_width``, in cells. Below
                four cells the arrow becomes a one-cell guillemet.
            **kwargs: Forwarded to ``Button`` (``id``, ``disabled``, ...).
        """
        self.pane = pane
        self.pane_label = pane_label
        self.painted_names: dict[str, str] = dict(painted_names or {})
        self.grip_width = width
        self.pane_open = open
        classes = destination_class
        if extra_classes:
            classes = f"{classes} {extra_classes}"
        super().__init__(compact=True, flat=True, classes=classes, **kwargs)
        self.sync_width(width)
        self.add_class("h-full")
        self.add_class("p-0")
        self.styles.line_pad = 0
        self.add_class("border-none")
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
        # ds-runtime: profile-supplied grip columns reserved by the layout
        self.set_styles(width=width)
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
        self.pane_open = open
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

    def sync_label(self, pane_label: str) -> None:
        """Rename the pane this grip controls, repainting only on a change.

        ``sync_open`` patches only the reactive ``label`` (the arrow), so a new
        ``pane_label`` with an unchanged open state would leave the old painted
        name on screen: nothing reactive changed. This re-derives the
        accessible copy and repaints once, and does nothing at all when the
        name is unchanged (Roleplay's list grip renames with the kind).

        Args:
            pane_label: The pane's new human name.

        Returns:
            None.
        """
        if pane_label == self.pane_label:
            return
        self.pane_label = pane_label
        self.sync_open(self.pane_open)
        self.refresh()

    def painted_name(self) -> str:
        """Return the name this grip paints down its own column.

        (task-32355) The handle used to carry its name only in ``_name`` and
        ``tooltip`` -- neither of which a terminal paints -- so every collapsed
        pane was an unexplained ``--->``. The column is a few cells wide and
        twenty to forty-five rows tall, so the name goes DOWN it.

        Returns:
            The letters to paint, one per row, already trimmed to the rows
            above the first arrow. Empty when there is no room at all.
        """
        name = self.painted_names.get(self.pane_label, self.pane_label)
        return "".join(name.split())[: max(self._first_arrow_row(), 0)]

    def _first_arrow_row(self) -> int:
        """Return the topmost row ``render`` paints an arrow on."""
        return min(self._arrow_rows())

    def _arrow_rows(self) -> set[int]:
        """Return the rows the collapse arrow is painted on."""
        height = max(self.content_region.height, 1)
        last_row = height - 1
        if self.pane == "library" and height > 1:
            upper_row = round(last_row * ARROW_UPPER_POSITION_RATIO)
            return {upper_row, last_row - upper_row}
        return {last_row // 2}

    def render(self) -> Content:
        """Paint the pane's name above the arrows it already carries.

        Returns:
            Content: Full-height grip content -- the name one letter per row
            from the top, then the arrows at the approved rows.
        """
        height = max(self.content_region.height, 1)
        arrow_rows = self._arrow_rows()
        arrow = self.label.plain
        name = self.painted_name()
        lines = [
            name[row]
            if row < len(name)
            else arrow
            if row in arrow_rows
            else " "
            for row in range(height)
        ]
        return Content.from_text("\n".join(lines))

    @on(Button.Pressed)
    def request_toggle(self, event: Button.Pressed) -> None:
        """Translate native Button activation into the shell message."""
        if event.button is not self:
            return
        event.stop()
        self.post_message(PaneToggleRequested(self.pane))


class AdaptivePaneShell(Horizontal):
    """Own adaptive pane structure while callers own state and behavior.

    A destination supplies its own ``AdaptivePaneClasses``; a destination that
    needs its own grip TYPE (the Library keeps ``LibraryAdaptiveReaderPaneGrip``
    so type queries by that name still match) overrides ``grip_type``. A
    subclass must not define ``on_mount``, ``on_resize`` or
    ``on_descendant_focus``: Textual dispatches each once per class in the MRO
    that defines one, so a redefinition would run the shared body twice.
    """

    #: The grip class this shell builds.
    grip_type: ClassVar[type[AdaptivePaneGrip]] = AdaptivePaneGrip

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptivePaneLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        destination: AdaptivePaneClasses,
        painted_names: Mapping[str, str] | None = None,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        """Assemble the three-pane shell around caller-owned pane widgets.

        Args:
            library: Widget for the leftmost (navigation rail) pane.
            items: Widget for the middle (list) pane.
            work: Widget for the work pane.
            layout: The resolved layout to mount with: which optional panes
                are open and how wide each is.
            id_prefix: Per-destination id stem for the composed grips, giving
                each destination its own stable selectors.
            library_label: Human name of the navigation pane, for grip copy.
            items_label: Human name of the items pane, for grip copy.
            destination: The destination's classes for every part.
            painted_names: Passed to both grips (see ``AdaptivePaneGrip``).
            grip_classes: Extra CSS classes for both grips.
            **kwargs: Forwarded to ``Horizontal`` (``id``, ``classes``, ...).

        Both grips are sized from ``layout.grip_width`` -- the width the
        resolver held back for them (task-31952 AC#3), so a caller cannot
        paint a grip the resolver never reserved.
        """
        super().__init__(**kwargs)
        self.destination = destination
        self.add_class(destination.shell)
        self.library = library
        self.items = items
        self.work = work
        self.library.add_class(destination.nav)
        self.items.add_class(destination.items)
        self.work.add_class(destination.work)
        self.library_grip = self.grip_type(
            "library",
            open=layout.library_open,
            pane_label=library_label,
            destination_class=destination.grip,
            painted_names=painted_names,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-library-grip",
        )
        self.items_grip = self.grip_type(
            "items",
            open=layout.items_open,
            pane_label=items_label,
            destination_class=destination.grip,
            painted_names=painted_names,
            extra_classes=grip_classes,
            width=layout.grip_width,
            id=f"{id_prefix}-items-grip",
        )
        self._last_focused_descendant: dict[PaneName, Widget | None] = {
            "library": None,
            "items": None,
        }
        self.effective_layout = layout
        self._applied_layout: AdaptivePaneLayout | None = None

    def compose(self) -> ComposeResult:
        """Compose retained navigation, items, grips, and work widgets."""
        yield self.library
        yield self.library_grip
        yield self.items
        yield self.items_grip
        yield self.work

    def on_mount(self) -> None:
        """Apply initial geometry and request a settled resize projection."""
        self.sync_layout(self.effective_layout)
        self.call_after_refresh(self.post_message, AdaptivePaneShellResized())

    def on_resize(self) -> None:
        """Request layout resolution after the shell allocation changes."""
        self.post_message(AdaptivePaneShellResized())

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
        layout: AdaptivePaneLayout,
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
                pane.remove_class("w-fill", "w-3fr")
                # ds-runtime: pane columns resolved from the measured shell width
                pane.set_styles(width=width)
            if pane.styles.min_width is None or pane.styles.min_width.value != width:
                pane.styles.min_width = width
            if pane.styles.max_width is None or pane.styles.max_width.value != width:
                pane.styles.max_width = width
            if previous_layout is None:
                pane.add_class("h-full")
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
            self.work.add_class("w-fill")
            self.work.styles.min_width = 0
            self.work.add_class("h-full")
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
                self.post_message(PaneVisibilityChanged(pane_name, now_open))
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
