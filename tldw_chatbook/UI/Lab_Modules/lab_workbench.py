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
from textual.widgets import Button, Static

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
        """Render both handles and both regions, left to right.

        Both handles are always composed alongside their regions -- never
        conditionally -- so a later collapse toggle can flip ``display``
        in place via :meth:`apply_rail_layout` instead of remounting
        anything. Remounting would drop whatever the frame already mounted
        into ``#lab-rail``/``#lab-body``/``#lab-inspector``, since a fresh
        ``LabWorkbench`` instance starts with empty regions and nothing
        re-populates them outside ``LabScreen.on_mount``.

        Each region also carries a small header with a collapse button --
        the ``DestinationRailHandle`` is only the *expand* affordance, so
        without this there would be no way back to collapsed once opened.
        The header is composed here, by the workbench itself, rather than
        by mode content (``compose_lab_rail``/``compose_lab_inspector``):
        collapse is frame-owned, not a per-mode concern. It reuses Console's
        and Personas' ``console-rail-header``/``console-rail-title``/
        ``console-rail-collapse-button`` classes rather than inventing a
        new look.

        Returns:
            A ``ComposeResult`` yielding, in order: the rail handle, the
            rail (with its collapse header first), the body, the inspector
            (with its collapse header first), and the inspector handle.
            Visibility is applied afterwards, in ``on_mount``.
        """
        yield DestinationRailHandle(
            label="Catalog",
            button_id="lab-rail-open",
            badge_id="lab-rail-badge",
            side="left",
            id="lab-rail-handle",
        )

        with VerticalScroll(id="lab-rail", classes="lab-region lab-rail"):
            yield from self._region_collapse_header(
                title="Catalog",
                button_id="lab-rail-collapse",
                glyph="<",
                tooltip="Collapse Catalog rail",
            )

        yield Vertical(id="lab-body", classes="lab-region lab-body")

        with VerticalScroll(id="lab-inspector", classes="lab-region lab-inspector"):
            yield from self._region_collapse_header(
                title="Inspector",
                button_id="lab-inspector-collapse",
                glyph=">",
                tooltip="Collapse Inspector rail",
            )

        yield DestinationRailHandle(
            label="Inspector",
            button_id="lab-inspector-open",
            badge_id="lab-inspector-badge",
            side="right",
            id="lab-inspector-handle",
        )

    @staticmethod
    def _region_collapse_header(
        *, title: str, button_id: str, glyph: str, tooltip: str
    ) -> ComposeResult:
        """Yield one region's title row and collapse button.

        Mounted as the first child of the region's ``VerticalScroll``, ahead
        of whatever mode content ``LabScreen._populate_regions`` mounts
        afterwards via ``mount_all`` (which appends).

        Args:
            title: Region name shown beside the collapse button.
            button_id: DOM id for the collapse button, e.g. ``lab-rail-collapse``.
            glyph: Button glyph -- ``"<"`` for the left rail, ``">"`` for the
                inspector, matching Personas' literal-glyph convention.
            tooltip: Tooltip text for the collapse button.

        Returns:
            A ``ComposeResult`` yielding the header ``Horizontal``.
        """
        with Horizontal(classes="console-rail-header"):
            title_widget = Static(title, classes="console-rail-title")
            yield title_widget
            collapse_button = Button(
                glyph,
                id=button_id,
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = tooltip
            yield collapse_button

    def on_mount(self) -> None:
        """Apply the rail layout's initial visibility once children exist."""
        self.apply_rail_layout(self.rail_layout)

    def apply_rail_layout(self, rail_layout: LabRailLayout) -> None:
        """Show or hide each region and its handle, with no widget churn.

        A region is visible exactly when its handle is not, and vice versa.
        Setting ``display`` never removes or remounts anything, so this is
        safe to call after the initial mount -- e.g. from a rail toggle --
        without losing whatever the frame has already mounted into the
        regions.

        Args:
            rail_layout: Which rails are collapsed.
        """
        self.rail_layout = rail_layout
        rail_collapsed = rail_layout.is_collapsed(LAB_RAIL_LEFT)
        inspector_collapsed = rail_layout.is_collapsed(LAB_RAIL_INSPECTOR)

        self.query_one("#lab-rail").display = not rail_collapsed
        self.query_one("#lab-rail-handle").display = rail_collapsed
        self.query_one("#lab-inspector").display = not inspector_collapsed
        self.query_one("#lab-inspector-handle").display = inspector_collapsed
