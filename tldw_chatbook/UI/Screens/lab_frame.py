"""The shared frame behind the Lab destination's three screens.

Renders a destination header, an optional status row, the mode strip, and a
three-region workbench. Modes supply content through the hooks below; the
frame owns collapse state, the deferred body mount, and status refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Button, Static

from ..Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)
from ..Lab_Modules.lab_rail_store import load_rail_layout, save_rail_layout
from ..Lab_Modules.lab_workbench import LabWorkbench
from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LAB_MODE_CHIP_IDS, LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


@dataclass(frozen=True)
class LabStatusChip:
    """One chip in the Lab status row.

    Attributes:
        chip_id: Stable id suffix identifying this chip across refreshes.
        text: Rendered copy, e.g. ``"Servers: 2 running"``.
    """

    chip_id: str
    text: str


@dataclass(frozen=True)
class LabInspectorRow:
    """One row in the Lab inspector that refreshes in place.

    Mirrors :class:`LabStatusChip`: the id must match a widget the mode
    already composed via ``compose_lab_inspector`` -- ``refresh_lab_status``
    mutates it rather than recomposing the inspector.

    Attributes:
        row_id: The widget's full DOM id (not just a suffix, unlike
            ``LabStatusChip.chip_id``: inspector rows do not share a common
            id prefix across modes).
        text: Rendered copy, e.g. ``"● llama.cpp — running"``.
    """

    row_id: str
    text: str


class LabScreen(BaseAppScreen):
    """Base for the Lab destination's screens.

    Subclasses override the ``lab_*`` hooks to supply content. The frame owns
    everything else: rail collapse and its persistence, the deferred body
    mount, and status-row refresh.
    """

    #: `[` / `]` move focus along the mode strip; they never navigate. Enter is
    #: then ordinary Button activation on the focused chip, which posts
    #: NavigateToScreen -- so cycling builds zero intermediate screens.
    #:
    #: Both are printable keys, so text inputs consume them first and these act
    #: only from button or list focus. Escape is deliberately unbound:
    #: EvalsScreen already binds it to its own back action.
    BINDINGS = [
        Binding("left_square_bracket", "lab_mode_focus(-1)", "Prev mode", show=False),
        Binding("right_square_bracket", "lab_mode_focus(1)", "Next mode", show=False),
    ]

    #: Footer hints registered for every Lab mode.
    LAB_FOOTER_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("[ / ]", "Switch mode"),
        ("Enter", "Go"),
    )

    def __init__(self, app_instance: "TldwCli", screen_name: str, **kwargs: Any) -> None:
        """Create a Lab screen.

        Args:
            app_instance: The running application.
            screen_name: This screen's shell route (``"llm"``, ``"stts"``, or
                ``"evals"``). Doubles as the mode strip's active route.
            kwargs: Forwarded to ``BaseAppScreen``.
        """
        super().__init__(app_instance, screen_name, **kwargs)
        self.rail_layout: LabRailLayout = load_rail_layout()
        #: Ids this screen has already warned about missing, so a 2-second
        #: refresh timer logs a stale/unknown chip or row once rather than
        #: forever.
        self._warned_ids: set[str] = set()

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        """Log a warning the first time ``key`` is seen this screen instance.

        Args:
            key: Stable identifier for the condition being warned about,
                e.g. ``"chip:servers"``. Namespaced by caller so chip and
                inspector-row warnings never collide on a shared id.
            message: Loguru-style message template.
            args: Values to interpolate into ``message``.
        """
        if key in self._warned_ids:
            return
        self._warned_ids.add(key)
        logger.warning(message, *args)

    # -- hooks -----------------------------------------------------------

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return this mode's destination header copy.

        Returns:
            The header state. Subclasses must override.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("Lab modes must supply lab_header_state()")

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        """Return this mode's status chips.

        Called on compose and on every refresh, so it must be cheap and safe
        to call repeatedly.

        Returns:
            The chips, or an empty tuple to render no status row at all.
        """
        return ()

    def lab_inspector_rows(self) -> tuple[LabInspectorRow, ...]:
        """Return this mode's inspector rows to refresh in place.

        Called on every refresh alongside the status chips, so it must be
        cheap and safe to call repeatedly. Rows are declared here purely for
        refresh; the widgets themselves are composed by
        ``compose_lab_inspector`` with matching ids -- the two must stay in
        sync, since ``refresh_lab_status`` mutates the composed widget by id
        rather than recomposing the inspector.

        Returns:
            The rows, or an empty tuple for a mode with no refreshable
            inspector content.
        """
        return ()

    def compose_lab_rail(self) -> ComposeResult:
        """Yield this mode's catalog rail contents.

        Returns:
            A ``ComposeResult``; empty by default.
        """
        return iter(())

    def build_lab_body(self) -> Widget | None:
        """Build this mode's body widget.

        A factory rather than a generator: the body is mounted after first
        paint, and widget instances do not survive ``recompose=True`` while
        factories do.

        Returns:
            The body widget, or None for a mode with no body.
        """
        return None

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield this mode's inspector contents.

        Returns:
            A ``ComposeResult``; empty by default.
        """
        return iter(())

    def on_lab_body_ready(self) -> None:
        """Called once, after the deferred body has mounted.

        Modes that need to touch their body -- registering watchers, reading
        widgets -- must do it here, never in ``on_mount``: the body does not
        exist yet at mount time.
        """

    # -- composition -----------------------------------------------------

    def compose_content(self) -> ComposeResult:
        """Compose the frame: header, optional status row, mode strip, workbench."""
        yield DestinationHeader(self.lab_header_state(), id="lab-destination-header")

        chips = self.lab_status_chips()
        if chips:
            with Horizontal(id="lab-status-row"):
                for chip in chips:
                    yield Static(
                        chip.text,
                        id=f"lab-status-chip-{chip.chip_id}",
                        classes="lab-status-chip",
                        markup=False,
                    )

        yield LabModeStrip(active_route=self.screen_name, id="lab-mode-strip")

        workbench = LabWorkbench(rail_layout=self.rail_layout, id="lab-workbench")
        yield workbench

    def on_mount(self) -> None:
        """Populate the rail and inspector, then defer the body mount.

        The body is mounted from ``call_after_refresh`` so first paint is not
        blocked by composing it -- Models' body costs 488-787 ms.
        """
        super().on_mount()
        self._populate_regions()
        self.call_after_refresh(self._mount_lab_body)
        self.register_footer_shortcuts(
            source="lab", shortcuts=self.LAB_FOOTER_SHORTCUTS
        )

    def action_lab_mode_focus(self, delta: int) -> None:
        """Move focus to an adjacent mode chip, wrapping at both ends.

        Does not navigate: Enter on the focused chip commits, which is what
        keeps cycling free of intermediate screen mounts.

        Args:
            delta: ``-1`` for the previous chip, ``1`` for the next.
        """
        focused = self.focused
        focused_id = getattr(focused, "id", None)
        if focused_id in LAB_MODE_CHIP_IDS:
            index = LAB_MODE_CHIP_IDS.index(focused_id)
        else:
            # Focus is elsewhere: start from the chip for this screen's own
            # mode so the first press lands beside it, not at a strip end.
            index = self._active_mode_chip_index()
        target = LAB_MODE_CHIP_IDS[(index + delta) % len(LAB_MODE_CHIP_IDS)]
        try:
            self.query_one(f"#{target}", Button).focus()
        except QueryError:
            logger.warning("Lab mode chip {} missing; focus not moved.", target)

    def _active_mode_chip_index(self) -> int:
        """Return the strip index of this screen's own mode.

        Returns:
            The index of the chip carrying ``is-active``, or 0 when the strip
            has not composed one.
        """
        for index, chip_id in enumerate(LAB_MODE_CHIP_IDS):
            try:
                if "is-active" in self.query_one(f"#{chip_id}", Button).classes:
                    return index
            except QueryError:
                continue
        return 0

    def _populate_regions(self) -> None:
        """Mount rail and inspector contents into their regions."""
        for region_id, content in (
            ("#lab-rail", list(self.compose_lab_rail())),
            ("#lab-inspector", list(self.compose_lab_inspector())),
        ):
            if not content:
                continue
            try:
                self.query_one(region_id).mount_all(content)
            except QueryError:
                logger.warning("Lab region {} missing; skipped.", region_id)

    def _mount_lab_body(self) -> None:
        """Mount the deferred body and notify the mode.

        Two very different situations both used to reach the same
        `except QueryError: logger.warning(...)`, and only one of them is
        harmless:

        - The screen was torn down before this ``call_after_refresh``
          callback ran (e.g. the user navigated away during the deferral
          window). Normal; return silently.
        - `#lab-body` is missing while the screen is still mounted. That
          means the frame's own composition is broken -- ``compose_content``
          never yielded the region ``LabWorkbench`` provides -- and yields a
          permanently blank screen. A warning here is exactly how that shipped
          undetected through 78 unit tests; it must surface loudly instead.

        Raises:
            QueryError: If ``#lab-body`` is missing while the screen is
                still mounted.
        """
        if not self.is_mounted:
            return
        body = self.build_lab_body()
        if body is not None:
            try:
                self.query_one("#lab-body").mount(body)
            except QueryError:
                logger.error(
                    "Lab body region #lab-body missing on a mounted screen; "
                    "this is a composition bug, not a normal teardown race."
                )
                raise
        self.on_lab_body_ready()

    # -- status ----------------------------------------------------------

    def refresh_lab_status(self) -> None:
        """Re-read this mode's chips and inspector rows and update in place.

        Mutates the existing ``Static`` for each chip and inspector row id
        rather than recomposing: recomposing on a timer churns widgets and
        can steal focus. The inspector is refreshed on the same cadence as
        the status row -- a mode with live chips (e.g. Models' server count)
        needs its per-server inspector rows to agree, not lag a poll behind.
        A chip or row whose id was not composed is logged once and ignored,
        since mounting new widgets from a timer is never intended.
        """
        for chip in self.lab_status_chips():
            try:
                self.query_one(f"#lab-status-chip-{chip.chip_id}", Static).update(
                    chip.text
                )
            except QueryError:
                self._warn_once(
                    f"chip:{chip.chip_id}",
                    "Unknown Lab status chip id {!r}; ignoring.",
                    chip.chip_id,
                )
        for row in self.lab_inspector_rows():
            try:
                self.query_one(f"#{row.row_id}", Static).update(row.text)
            except QueryError:
                self._warn_once(
                    f"row:{row.row_id}",
                    "Unknown Lab inspector row id {!r}; ignoring.",
                    row.row_id,
                )

    # -- collapse --------------------------------------------------------

    @on(Button.Pressed, "#lab-rail-open")
    def _handle_lab_rail_open(self, event: Button.Pressed) -> None:
        """Expand the catalog rail from its collapsed handle."""
        event.stop()
        self.toggle_lab_rail(LAB_RAIL_LEFT)

    @on(Button.Pressed, "#lab-rail-collapse")
    def _handle_lab_rail_collapse(self, event: Button.Pressed) -> None:
        """Collapse the catalog rail from its header button."""
        event.stop()
        self.toggle_lab_rail(LAB_RAIL_LEFT)

    @on(Button.Pressed, "#lab-inspector-open")
    def _handle_lab_inspector_open(self, event: Button.Pressed) -> None:
        """Expand the inspector from its collapsed handle."""
        event.stop()
        self.toggle_lab_rail(LAB_RAIL_INSPECTOR)

    @on(Button.Pressed, "#lab-inspector-collapse")
    def _handle_lab_inspector_collapse(self, event: Button.Pressed) -> None:
        """Collapse the inspector from its header button."""
        event.stop()
        self.toggle_lab_rail(LAB_RAIL_INSPECTOR)

    def toggle_lab_rail(self, rail: str) -> None:
        """Collapse or expand one rail and persist the new state.

        Applies the new layout to the existing workbench in place via
        ``apply_rail_layout`` rather than ``self.refresh(recompose=True)``:
        a screen-level recompose would rebuild ``compose_content()`` --
        including a brand-new ``LabWorkbench`` with empty regions -- without
        re-firing ``on_mount()``, so ``_populate_regions()`` and the
        deferred ``_mount_lab_body()`` would never run again and the mode's
        entire content would vanish for the life of the screen.

        Args:
            rail: ``LAB_RAIL_LEFT`` or ``LAB_RAIL_INSPECTOR``.
        """
        self.rail_layout = self.rail_layout.toggle(rail)
        save_rail_layout(self.rail_layout)
        try:
            self.query_one(LabWorkbench).apply_rail_layout(self.rail_layout)
        except QueryError:
            logger.warning("Lab workbench missing; rail toggle not applied.")
