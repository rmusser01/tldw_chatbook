"""One resident adaptive shell shared by the Media and Notes browse routes.

Phase C (``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-
design.md``, "Design record — phase C, media") measured that a Library
rail-mode switch rebuilt the whole screen: 177 mounts for a media switch-in,
of which **55** existed only because the rail lives INSIDE the route's reader
shell (``#library-shell-grid > <route shell> > LibraryRail``) and the two
routes used two different shell ids. Keeping the rail resident therefore
requires the SHELL to be resident, which requires the two ids to become one.

This class is that one shell. It carries a route-neutral id
(``#library-browse-reader-shell``) and wears a per-route **marker class**
--- ``.library-media-route`` / ``.library-notes-route`` --- set exactly while
that route owns it. The marker is what preserves the meaning of the ~35
production sites that used to read "is ``#library-media-reader-shell``
mounted?" as "is the media route active?": under residency shell PRESENCE is
permanently true, but the marker is not, so those sites become
``.library-media-route`` and keep their original semantics with one seam
(``apply_route``) as the single writer.
"""

from __future__ import annotations

from typing import Any

from textual.widget import Widget

from tldw_chatbook.Library.library_media_reader_state import (
    MediaReaderEffectiveLayout,
)

from .library_adaptive_reader_shell import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    PaneToggleRequested as SharedPaneToggleRequested,
)

MediaShellResized = AdaptiveReaderShellResized
PaneToggleRequested = SharedPaneToggleRequested

#: The one resident browse shell's id. Route-neutral on purpose: it is the
#: same widget on both routes, so naming it after either one would lie.
LIBRARY_BROWSE_READER_SHELL_ID = "library-browse-reader-shell"

#: Marker classes. Exactly one is present at a time, and ``apply_route`` is
#: their only writer.
LIBRARY_MEDIA_ROUTE_CLASS = "library-media-route"
LIBRARY_NOTES_ROUTE_CLASS = "library-notes-route"

#: Selector equivalents, so callers spell the marker once.
LIBRARY_MEDIA_ROUTE_SELECTOR = f".{LIBRARY_MEDIA_ROUTE_CLASS}"
LIBRARY_NOTES_ROUTE_SELECTOR = f".{LIBRARY_NOTES_ROUTE_CLASS}"

#: The visual grip class the Media route used to bake into its grips at
#: construction. Toggled per route now (the grips are shared).
LIBRARY_MEDIA_PANE_GRIP_CLASS = "library-media-pane-grip"

#: Every class ``apply_route`` flips. They are QUERY markers, not style hooks:
#: no rule in the app's stylesheet references any of them, which is what lets
#: ``apply_route`` flip them without a restyle (see its body). Pinned by
#: ``Tests/UI/test_library_phase_c_switch_storm.py::
#: test_route_marker_classes_have_no_stylesheet_rules``, which reads the
#: PARSED stylesheet -- widget ``DEFAULT_CSS`` counts, so grepping ``css/``
#: would not be enough.
LIBRARY_ROUTE_MARKER_CLASSES = (
    LIBRARY_MEDIA_ROUTE_CLASS,
    LIBRARY_NOTES_ROUTE_CLASS,
    LIBRARY_MEDIA_PANE_GRIP_CLASS,
)

#: The two routes this shell hosts.
LIBRARY_BROWSE_ROUTE_MEDIA = "media"
LIBRARY_BROWSE_ROUTE_NOTES = "notes"


class LibraryBrowseReaderShell(LibraryAdaptiveReaderShell):
    """Adaptive reader shell that outlives a Media<->Notes route switch."""

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: MediaReaderEffectiveLayout,
        *,
        route: str = LIBRARY_BROWSE_ROUTE_MEDIA,
        **kwargs: Any,
    ) -> None:
        """Assemble the shared shell around the active route's panes.

        Args:
            library: Widget for the Library rail pane -- resident across a
                Media<->Notes switch, which is the whole point of this class.
            items: Widget for the list pane (``#library-canvas``), which hosts
                both routes' canvases once each has been visited.
            work: Widget for the work pane -- Media's Reader or Notes' work
                pane. Swapped per route by ``swap_work``; it is small (2-3
                widgets) and every route entry already rebuilt it, so it is
                deliberately NOT made resident.
            layout: The resolved layout to mount with.
            route: Which route owns the shell right now.
            **kwargs: Forwarded to ``LibraryAdaptiveReaderShell`` (``id``,
                ``classes``, ...). The shared identity arguments (id prefix,
                pane labels) are fixed here.
        """
        super().__init__(
            library=library,
            items=items,
            work=work,
            layout=layout,
            id_prefix="library-browse",
            library_label="Library",
            items_label="Items",
            **kwargs,
        )
        self.route = ""
        self.apply_route(route)

    @property
    def reader(self) -> Widget:
        """The work pane under Media's own vocabulary.

        Media callers reach for ``shell.reader``; the shared shell calls the
        same pane ``work``. Kept as a read-only alias rather than a second
        attribute so the two cannot drift after ``swap_work``.
        """
        return self.work

    def apply_route(self, route: str) -> None:
        """Project which route owns this shell onto its marker classes.

        The single writer of ``.library-media-route`` / ``.library-notes-
        route``. Every "is this route active?" DOM probe reads those markers,
        so this method is the seam that keeps them honest -- it is called both
        at construction (whole-screen recompose) and on the resident switch
        path.

        **``update=False`` is load-bearing, not a micro-optimisation.** A
        class flip with Textual's default ``update=True`` calls
        ``DOMNode.update_node_styles`` -> ``App.update_styles(self)`` ->
        ``stylesheet.update_nodes(self.walk_children(with_self=True))``: one
        ``Stylesheet.apply`` for EVERY node in this shell's subtree, which is
        the whole Library. Phase-C task 2.5 measured the two SHELL flips below
        at **238 of the 423 apply calls on a media switch-back (43 ms of its
        86 ms of restyle)**, and this method as a whole -- four classes across
        the three ``set_class`` calls, the grips included -- at **240**;
        194-208 on the notes arms. That is the single largest
        restyle originator on all three, larger than every mount on the switch
        put together. The work is entirely wasted because these classes are
        QUERY markers: no rule in the app's stylesheet references any of them,
        so no node's computed styles can change when they flip. That premise is
        not assumed, it is pinned -- ``Tests/UI/test_library_phase_c_switch_
        storm.py::test_route_marker_classes_have_no_stylesheet_rules`` reads
        the parsed stylesheet and fails the moment a rule starts depending on
        one, which is the signal to restore a (single, coalesced) restyle here.

        Args:
            route: ``"media"`` or ``"notes"``.

        Returns:
            None.
        """
        self.route = route
        self.set_class(
            route == LIBRARY_BROWSE_ROUTE_MEDIA,
            LIBRARY_MEDIA_ROUTE_CLASS,
            update=False,
        )
        self.set_class(
            route == LIBRARY_BROWSE_ROUTE_NOTES,
            LIBRARY_NOTES_ROUTE_CLASS,
            update=False,
        )
        for grip in (self.library_grip, self.items_grip):
            grip.set_class(
                route == LIBRARY_BROWSE_ROUTE_MEDIA,
                LIBRARY_MEDIA_PANE_GRIP_CLASS,
                update=False,
            )
        # The items pane is called "Notes" on one route and "Items" on the
        # other, and that label is the grip's tooltip AND accessible name --
        # user-visible copy, so it travels with the route rather than being
        # frozen at construction the way a per-route shell could afford.
        items_label = "Notes" if route == LIBRARY_BROWSE_ROUTE_NOTES else "Items"
        if self.items_grip.pane_label != items_label:
            self.items_grip.pane_label = items_label
            self.items_grip.sync_open(self.effective_layout.items_open)

    def adopt_route_layout(self, layout: MediaReaderEffectiveLayout) -> None:
        """Apply the destination route's layout as if freshly composed.

        The two routes keep SEPARATE resolved layouts, and each route's
        resolver only re-applies its own when it differs from that route's
        stored value -- so after a switch the shell would otherwise keep the
        outgoing route's pane geometry until something else moved. Dropping
        ``_applied_layout`` first takes ``sync_layout``'s initial-mount branch
        (no focus evacuation, no automatic pane-reopen focus), which is
        exactly what a composed shell gets.

        **A route that has never resolved a layout has nothing to adopt.** Its
        stored value is the all-zero default, and applying it closes both
        panes -- which phase-C task 2.5 measured on the FIRST Notes switch as
        two ``disabled`` flips at 23 ms and two more when the real resolver
        landed at 76 ms: **205 of that switch's 463 ``Stylesheet.apply`` calls
        and 41 ms of restyle** (``Widget.disabled`` is a pseudo-class, so each
        flip restyles the pane's whole subtree), on top of a ~50 ms flash of a
        collapsed shell nobody asked for. The reset above still happens, so
        the resolver that follows -- the swap schedules it, and the canvas
        sync calls it -- still gets the initial-mount branch and its
        application is the FIRST one, instead of the third.

        The all-zero test is this codebase's existing spelling of "never
        resolved": the six ``_sync_library_*_reader_layout_from_shell``
        resolvers each drop ``previous`` on exactly this condition.

        Args:
            layout: The destination route's resolved layout, or its all-zero
                default when that route has not been resolved yet.

        Returns:
            None.
        """
        self._applied_layout = None
        if not (layout.reader_width or layout.library_width or layout.items_width):
            return
        self.sync_layout(layout)

    async def swap_work(self, work: Widget) -> None:
        """Replace the work pane in place, keeping the shell mounted.

        Mounted before the outgoing pane is removed so the shell is never
        childless mid-switch (Textual would otherwise reflow to a two-pane
        allocation for a frame). The new pane inherits exactly the geometry
        ``sync_layout`` gives a freshly composed work pane.

        Args:
            work: The destination route's work pane.

        Returns:
            None.
        """
        previous = self.work
        if previous is work:
            return
        work.add_class("library-adaptive-reader-work")
        await self.mount(work, after=self.items_grip)
        self.work = work
        work.display = True
        work.styles.width = "1fr"
        work.styles.min_width = 0
        work.styles.height = "100%"
        if previous is not None and previous.parent is self:
            await previous.remove()

    def on_mount(self) -> None:
        """Hide the rail's legacy collapse control beside the grips.

        No ``super().on_mount()``: Textual's dispatcher already invokes
        ``LibraryAdaptiveReaderShell.on_mount`` separately for this Mount
        event (TASK-31822).
        """
        collapse = self.query("#library-rail-collapse")
        if collapse:
            collapse.first().display = False
