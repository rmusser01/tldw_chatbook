"""The shared frame behind the Lab destination's three screens.

Renders a destination header, an optional status row, the mode strip, and a
three-region workbench. Modes supply content through the hooks below; the
frame owns collapse state, the deferred body mount, and status refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Static

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
from .lab_mode_strip import LabModeStrip

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


class LabScreen(BaseAppScreen):
    """Base for the Lab destination's screens.

    Subclasses override the ``lab_*`` hooks to supply content. The frame owns
    everything else: rail collapse and its persistence, the deferred body
    mount, and status-row refresh.
    """

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
        """Mount the deferred body and notify the mode."""
        body = self.build_lab_body()
        if body is not None:
            try:
                self.query_one("#lab-body").mount(body)
            except QueryError:
                logger.warning("Lab body region missing; body not mounted.")
                return
        self.on_lab_body_ready()

    # -- status ----------------------------------------------------------

    def refresh_lab_status(self) -> None:
        """Re-read this mode's chips and update the row in place.

        Mutates the existing ``Static`` for each ``chip_id`` rather than
        recomposing: recomposing on a timer churns widgets and can steal
        focus. A chip whose id was not composed is logged and ignored, since
        mounting new widgets from a timer is never intended.
        """
        for chip in self.lab_status_chips():
            try:
                self.query_one(f"#lab-status-chip-{chip.chip_id}", Static).update(
                    chip.text
                )
            except QueryError:
                logger.warning(
                    "Unknown Lab status chip id {!r}; ignoring.", chip.chip_id
                )

    # -- collapse --------------------------------------------------------

    def toggle_lab_rail(self, rail: str) -> None:
        """Collapse or expand one rail and persist the new state.

        Args:
            rail: ``LAB_RAIL_LEFT`` or ``LAB_RAIL_INSPECTOR``.
        """
        self.rail_layout = self.rail_layout.toggle(rail)
        save_rail_layout(self.rail_layout)
        self.refresh(recompose=True)
