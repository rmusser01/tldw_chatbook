"""Media<->Notes route switching without a whole-screen recompose.

Phase C's motivating change (``Docs/superpowers/specs/2026-09-01-library-
screen-decomposition-design.md``, "Design record — phase C, media"). Measured
before: a media rail switch ran ONE whole-screen ``Widget.recompose()`` and
177 mounts / 162 unmounts, 124 ms of main-thread block. The recompose is the
reason no canvas could be resident -- ``Widget.recompose()`` removes every
child of the screen -- so residency and the recompose could only land
together.

Two mechanisms in one file:

* **The route swap** (``swap_library_browse_route``): the targeted path that
  ``_select_library_rail_row_after_source_admission`` takes instead of
  ``await self.recompose()``. It mounts or removes exactly the structural
  delta between the two routes (the Notes source strip), re-points the one
  resident shell at the destination route, and re-applies the per-switch
  work a real switch owes (rail selection, header, layout, stage classes,
  footer context, focus).
* **Canvas residency, mechanism C** (mount-once-then-toggle): the Media and
  Notes list canvases both live in ``#library-canvas`` once visited and are
  toggled by ``display`` rather than rebuilt. Residency is deliberately
  *stateless* -- "resident" means "already a child of the canvas host", read
  off the DOM every time. Any other path that replaces the canvas host's
  children (the ordinary-route projection, a recompose) therefore evicts
  residency for free, and the next switch pays one lazy re-mount instead of
  carrying a stale widget reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Library import (
    LIBRARY_BROWSE_ROUTE_MEDIA,
    LIBRARY_BROWSE_ROUTE_NOTES,
    LibraryBrowseReaderShell,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
    LibraryRail,
)

from .canvas_sync import _sync_library_canvas
from .screen_constants import (
    LIBRARY_CANVAS_KIND_NOTES,
    LIBRARY_NOTES_SOURCE_DATABASE,
    LIBRARY_NOTES_SOURCE_FILES,
    LIBRARY_NOTES_SOURCE_STRIP_CANVAS_KINDS,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Library.library_shell_state import LibraryShellState
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

#: The two canvases residency owns. A child of ``#library-canvas`` with one
#: of these ids is kept (hidden) across a switch; anything else there is
#: transient chrome (the loading row, the lookup-error card, the Trash
#: canvas) and is removed on the way out, exactly as before phase C.
LIBRARY_RESIDENT_CANVAS_IDS = frozenset(
    {"library-media-canvas", "library-notes-canvas"}
)

#: Route -> the canvas id that route shows in its list view.
_ROUTE_LIST_CANVAS_ID = {
    LIBRARY_BROWSE_ROUTE_MEDIA: "library-media-canvas",
    LIBRARY_BROWSE_ROUTE_NOTES: "library-notes-canvas",
}

#: Route -> the ``_sync_library_canvas`` kind that repaints it in place.
LIBRARY_BROWSE_ROUTE_SYNC_KIND = {
    LIBRARY_BROWSE_ROUTE_MEDIA: "media",
    LIBRARY_BROWSE_ROUTE_NOTES: "notes",
}


def library_browse_route_for_canvas_kind(screen: "LibraryScreen", kind: str) -> str | None:
    """Return which browse route owns a canvas kind, if either does.

    Args:
        screen: The Library screen (or a controller forwarding a bare
            ``self``) whose Notes source decides whether the Notes route is
            the database one this shell hosts.
        kind: A ``LibraryShellState.canvas_kind``.

    Returns:
        ``"media"``, ``"notes"``, or ``None`` when the kind belongs to a
        route that still owns its own reader shell.
    """
    if kind == "media":
        return LIBRARY_BROWSE_ROUTE_MEDIA
    if (
        kind == LIBRARY_CANVAS_KIND_NOTES
        and screen._notes_state.source == LIBRARY_NOTES_SOURCE_DATABASE
    ):
        return LIBRARY_BROWSE_ROUTE_NOTES
    return None


def build_library_notes_source_strip(
    screen: "LibraryScreen", canvas_kind: str
) -> Horizontal:
    """Build the Notes Database/Files source strip.

    Shared by ``compose_content`` and the route swap so the strip a switch
    mounts is the strip a recompose would have composed -- the swap adds and
    removes exactly this widget, which is the whole structural delta between
    the two browse routes.

    Args:
        screen: The Library screen.
        canvas_kind: The destination shell state's canvas kind.

    Returns:
        The composed source strip, unmounted.
    """
    adaptive_database_notes = (
        canvas_kind in LIBRARY_NOTES_SOURCE_STRIP_CANVAS_KINDS
        and screen._notes_state.source == LIBRARY_NOTES_SOURCE_DATABASE
    )
    wide_focused_task = (
        not adaptive_database_notes
        and not screen._notes_state.compact
        and screen._library_notes_focused_task_active()
    )
    task_return = Button(
        "‹ Library / Notes",
        id="library-notes-task-return",
        compact=True,
    )
    task_return.display = wide_focused_task
    database_source = Button(
        "Library notes",
        id="library-notes-source-database",
        compact=True,
    )
    database_source.set_class(
        screen._notes_state.source == LIBRARY_NOTES_SOURCE_DATABASE, "-selected"
    )
    database_source.display = not wide_focused_task
    source_separator = Static(
        "|",
        id="library-notes-source-separator",
        markup=False,
    )
    source_separator.display = not wide_focused_task
    files_source = Button(
        "Folder files",
        id="library-notes-source-files",
        compact=True,
    )
    files_source.set_class(
        screen._notes_state.source == LIBRARY_NOTES_SOURCE_FILES, "-selected"
    )
    files_source.display = not wide_focused_task
    return Horizontal(
        task_return,
        database_source,
        source_separator,
        files_source,
        id="library-notes-source-strip",
    )


def _build_library_browse_list_child(
    screen: "LibraryScreen", route: str
) -> Widget:
    """Build the canvas-host child the destination route shows in list view."""
    if route == LIBRARY_BROWSE_ROUTE_MEDIA:
        return screen._build_library_media_active_child()
    if not screen._library_loaded and not screen._library_lookup_error:
        return Static(
            "Loading local Library sources…",
            id="library-canvas-loading",
            classes="destination-purpose",
            markup=False,
        )
    if screen._library_lookup_error:
        return screen._library_canvas_error_widget()
    return LibraryNotesCanvas(
        **screen._library_notes_list_canvas_kwargs(),
        id="library-notes-canvas",
    )


def _build_library_browse_work_pane(
    screen: "LibraryScreen", route: str
) -> Widget:
    """Build the destination route's work pane.

    The work pane is 2-3 widgets and every route entry already rebuilt it, so
    it is swapped rather than made resident -- residency is spent where it
    was measured to pay (the 29-widget Media canvas subtree).
    """
    if route == LIBRARY_BROWSE_ROUTE_MEDIA:
        return screen._build_library_media_reader()
    return LibraryNoteWorkPane(
        **screen._library_note_work_pane_kwargs(),
        id="library-note-work-pane",
    )


async def _apply_library_notes_source_strip(
    screen: "LibraryScreen", canvas_kind: str
) -> None:
    """Mount or remove the Notes source strip to match the destination."""
    mounted = screen.query("#library-notes-source-strip")
    needed = canvas_kind in LIBRARY_NOTES_SOURCE_STRIP_CANVAS_KINDS
    if needed and not mounted:
        await screen.mount(
            build_library_notes_source_strip(screen, canvas_kind),
            before=screen.query_one("#library-shell-grid", Widget),
        )
    elif mounted and not needed:
        await mounted.first(Widget).remove()


async def _adopt_library_browse_canvas(
    screen: "LibraryScreen",
    canvas_host: Vertical,
    route: str,
) -> bool:
    """Show the destination canvas, keeping the other route's resident.

    Mechanism C: a canvas already in the host is repainted in place through
    its own ``sync_state`` (a canvas-scoped rebuild, not a screen one) and a
    canvas absent from the host is mounted for the first time. Everything
    that is neither is removed.

    Returns:
        True when the destination is showing current state -- a freshly
        mounted canvas (nothing to repaint) or a resident one whose own
        ``sync_state`` accepted the repaint. False when the resident repaint
        FAILED: ``_sync_library_canvas`` was told not to take the whole-screen
        fallback itself (``allow_screen_fallback=False``), so this Boolean is
        the ONLY signal that the caller must. Swallowing it (Qodo #7) let a
        failed return-switch report success and keep showing stale content.
    """
    list_canvas_id = _ROUTE_LIST_CANVAS_ID[route]
    resident = next(
        (child for child in canvas_host.children if child.id == list_canvas_id),
        None,
    )
    destination: Widget = resident
    if resident is None:
        destination = _build_library_browse_list_child(screen, route)
        # Only the two list canvases are resident; a loading/error child
        # keeps its pre-phase-C lifetime (mounted on entry, removed on exit).
        await canvas_host.mount(destination)
    stale = [
        child
        for child in canvas_host.children
        if child is not destination and child.id not in LIBRARY_RESIDENT_CANVAS_IDS
    ]
    if stale:
        for child in stale:
            child.display = False
        await canvas_host.remove_children(stale)
    for child in canvas_host.children:
        child.display = child is destination
    if resident is None:
        # A first-time mount shows freshly built state; nothing to repaint,
        # so adoption is unconditionally current.
        return True
    # The resident canvas has been off-route (and, by the route-ownership
    # guard, deliberately un-synced) since the last visit, so switching
    # back MUST repaint it from current state. Propagate the repaint's own
    # verdict: a False here is a resident canvas still showing stale content.
    return _sync_library_canvas(
        screen,
        LIBRARY_BROWSE_ROUTE_SYNC_KIND[route],
        allow_screen_fallback=False,
    )


async def swap_library_browse_route(
    screen: "LibraryScreen", shell: "LibraryShellState"
) -> bool:
    """Move the Library to a browse route without rebuilding the screen.

    Args:
        screen: The Library screen instance driving the update.
        shell: Latest normalized ``LibraryShellState``.

    Returns:
        True when the targeted update completed; False when the caller should
        use the whole-screen fallback.
    """
    route = library_browse_route_for_canvas_kind(screen, shell.canvas_kind)
    if route is None:
        return False
    if route == LIBRARY_BROWSE_ROUTE_MEDIA and screen._media_state.view != "list":
        return False
    if route == LIBRARY_BROWSE_ROUTE_NOTES and screen._notes_state.view != "list":
        return False
    try:
        shell_widget = screen.query_one(
            "#library-browse-reader-shell", LibraryBrowseReaderShell
        )
    except (NoMatches, QueryError):
        # Arriving from a route that owns its own reader shell (or from the
        # ordinary two-pane shell): the shell itself is the structural delta,
        # so this is the whole-screen seam's case, not residency's.
        return False
    try:
        header = screen.query_one("#library-header-line", Static)
        rail = screen.query_one("#library-rail", LibraryRail)
        canvas_host = screen.query_one("#library-canvas", Vertical)
    except (NoMatches, QueryError):
        return False

    same_route = shell_widget.route == route
    generation = screen._library_snapshot_state_generation
    route_key = screen._library_entry_route_key()
    focus_identity = (
        screen._capture_library_entry_focus() if same_route else None
    )
    if screen.is_running:
        try:
            screen.app.capture_mouse(None)
        except Exception:
            logger.debug(
                "Mouse-capture release before Library route update skipped.",
                exc_info=True,
            )
    header.update(screen._library_header_line(shell.header_line))
    rail.apply_selection(
        shell,
        lifecycle=screen._library_lifecycle,
        onboarding_all_empty=screen._library_onboarding_all_empty,
    )
    await _apply_library_notes_source_strip(screen, shell.canvas_kind)
    if not same_route:
        await shell_widget.swap_work(
            _build_library_browse_work_pane(screen, route)
        )
        shell_widget.apply_route(route)
        shell_widget.adopt_route_layout(
            screen._media_state.reader_layout
            if route == LIBRARY_BROWSE_ROUTE_MEDIA
            else screen._notes_state.reader_layout
        )
    adopted = await _adopt_library_browse_canvas(screen, canvas_host, route)
    if not adopted:
        # The resident canvas refused its in-place repaint, and we suppressed
        # its own whole-screen fallback (allow_screen_fallback=False). Report
        # the swap as not done so ``_select_library_rail_row_after_source_
        # admission`` takes its ``recompose()`` recovery, rather than leaving
        # stale content behind a name-clean "success" (Qodo #7).
        return False
    screen._apply_library_notes_stage_visibility()
    screen._apply_library_notes_footer_context()
    screen._hide_library_adaptive_reader_rail_collapse()
    screen.call_after_refresh(
        screen._sync_library_media_reader_layout_from_shell
        if route == LIBRARY_BROWSE_ROUTE_MEDIA
        else screen._sync_library_notes_reader_layout_from_shell
    )
    if focus_identity is not None:
        screen._restore_library_entry_focus(
            focus_identity,
            generation=generation,
            route_key=route_key,
        )
    return True
