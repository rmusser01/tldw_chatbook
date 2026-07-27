"""The Lab frame's three-region workbench: rail | body | inspector.

Renders a :class:`LabRailLayout` as two collapsible rails around a body,
with a compact handle standing in for each collapsed rail. The container
holds no mode knowledge -- the frame mounts mode content into the regions.

Deliberately not `DestinationWorkbench`: that is a fixed Horizontal of
equal-width panes with no collapse. Deliberately not `WatchlistsWorkbench`
either: that is bound to a five-member Region enum with a stacked centre and
solo semantics, none of which Lab needs.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button

from ...Widgets.destination_rail import DestinationRailHandle
from .lab_rail_layout import LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT, LabRailLayout

#: Width of the expanded catalog rail. Sized to the longest rail label
#: ("Speech Recognition", 18 characters) plus padding and frame border, and
#: chosen against Console's observed ~34 -- Console's rail holds conversation
#: titles, Lab's holds fixed short labels.
LAB_RAIL_WIDTH = 26
#: Width of the expanded inspector.
LAB_INSPECTOR_WIDTH = 30
#: Width of a collapsed rail's handle, matching Console's.
LAB_HANDLE_WIDTH = 11
#: Class every rail row carries; styled app-tier in features/_lab.tcss.
LAB_RAIL_ROW_CLASS = "lab-rail-row"


class LabWorkbench(Horizontal):
    """Two collapsible rails around a body, rendered from a rail layout."""

    def __init__(self, *, rail_layout: LabRailLayout, **kwargs: Any) -> None:
        """Create the workbench.

        Args:
            rail_layout: Which rails are collapsed. The attribute is named
                ``rail_layout``, never ``layout``: ``Widget.layout`` is an
                existing unsettable Textual property that the compositor
                calls ``.arrange()`` on, and shadowing it crashes rendering.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"lab-workbench {classes}".strip(), **kwargs)
        self.rail_layout = rail_layout

    def compose(self) -> ComposeResult:
        """Render handles and regions according to the rail layout.

        Returns:
            A ``ComposeResult`` yielding, left to right: the rail handle, the
            rail, the body, the inspector, and the inspector handle. A
            collapsed region and its handle swap visibility.
        """
        rail_collapsed = self.rail_layout.is_collapsed(LAB_RAIL_LEFT)
        inspector_collapsed = self.rail_layout.is_collapsed(LAB_RAIL_INSPECTOR)

        if rail_collapsed:
            yield DestinationRailHandle(
                label="Catalog",
                button_id="lab-rail-open",
                badge_id="lab-rail-badge",
                side="left",
                id="lab-rail-handle",
            )

        rail = VerticalScroll(id="lab-rail", classes="lab-region lab-rail")
        rail.display = not rail_collapsed
        yield rail

        body = Vertical(id="lab-body", classes="lab-region lab-body")
        yield body

        inspector = VerticalScroll(
            id="lab-inspector", classes="lab-region lab-inspector"
        )
        inspector.display = not inspector_collapsed
        yield inspector

        if inspector_collapsed:
            yield DestinationRailHandle(
                label="Inspector",
                button_id="lab-inspector-open",
                badge_id="lab-inspector-badge",
                side="right",
                id="lab-inspector-handle",
            )
