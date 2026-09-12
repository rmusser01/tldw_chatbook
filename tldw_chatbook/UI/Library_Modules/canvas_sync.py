"""Library screen targeted canvas-sync and row-toggle helpers.

Moved verbatim out of ``tldw_chatbook/UI/Screens/library_screen.py`` by PR 0a
of the Library screen decomposition
(``.superpowers/sdd/2026-09-01-library-decomposition-foundation``; see
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``).
``library_screen.py`` re-exports every name here so its import surface is
unchanged; later decomposition tasks import directly from this module.
"""
from __future__ import annotations

import operator
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Library.library_notes_state import LibraryNotesFocusIdentity
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_DELETE_SELECTED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_TOOLTIP,
    LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_REVIEW_SELECTED_TOOLTIP,
    build_library_shell_state,
    library_disabled_action_label,
)
from ...Widgets.Library import (
    LibraryConversationsCanvas,
    LibraryExportCanvas,
    LibraryIngestCanvas,
    LibraryLandingCanvas,
    LibraryMediaCanvas,
    LibraryMediaTrashCanvas,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
    LibraryPromptWorkPane,
    LibraryPromptsListCanvas,
    LibrarySearchRagPanel,
    LibrarySkillWorkPane,
    LibrarySkillsListCanvas,
    LibraryStudyHandoffCanvas,
)
from ...Widgets.Library.library_canvas_sync import PostRecomposeCallback
from .screen_constants import _LIBRARY_LIST_ROW_CLASSES

if TYPE_CHECKING:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _move_library_list_row_focus(focused: Widget | None, key: str) -> bool:
    """Move DOM focus to the previous/next Library list row, in place.

    Up/Down are otherwise unbound in this app (see the module docstring
    above); this fires ONLY when ``focused`` is one of the four Library
    list-row Button classes, so it never intercepts Up/Down anywhere else
    on the screen (rail rows, form Inputs, TextAreas, the RAG evidence
    cards, etc. all keep their native/absent behavior). Siblings sharing a
    row class are read off ``focused.parent.children`` and filtered to
    that class -- the Skills list interleaves a non-row ``Static`` secondary
    line between rows (``library_skills_canvas.py``), so a plain "next
    child" walk would skip a row every other step.

    Args:
        focused: The screen's currently focused widget (``screen.focused``),
            or ``None``.
        key: ``"up"`` or ``"down"``.

    Returns:
        ``True`` when ``focused`` is a Library list row -- the caller
        should treat the key as claimed (``event.stop()`` /
        ``event.prevent_default()``) even at a list boundary, where focus
        deliberately does not wrap. ``False`` when ``focused`` is not a
        Library list row, so the caller must leave the key untouched for
        its normal handling elsewhere.
    """
    if focused is None:
        return False
    if not any(focused.has_class(name) for name in _LIBRARY_LIST_ROW_CLASSES):
        return False
    parent = focused.parent
    if parent is None:
        return False
    siblings = [
        child
        for child in parent.children
        if any(child.has_class(name) for name in _LIBRARY_LIST_ROW_CLASSES)
    ]
    try:
        index = siblings.index(focused)
    except ValueError:
        return True
    step = -1 if key == "up" else 1
    new_index = index + step
    if 0 <= new_index < len(siblings):
        siblings[new_index].focus()
    return True


# --- task-252: targeted (non-recompose) selection-interaction updates ------
#
# Docs/Design/2026-07-16-performance-audit.md §P1 B2: library_screen.py
# called self.refresh(recompose=True) -- a whole-screen remove/remount of
# the nav bar, footer, ~20-row rail, and 50-100-row canvas -- from every
# per-row selection/checkbox handler. These two helpers are the SELECTION
# interaction class's staged fix: Tier 1 patches a toggled row in place;
# Tier 2 routes structural selection changes (browse-mode row pick,
# select-mode enter/exit/select-all/clear) through the canvas widget's own
# sync_state(), a canvas-scoped recompose that never touches the nav bar,
# footer, or rail.
#
# Both are MODULE-LEVEL functions (screen passed explicitly), not
# LibraryScreen methods, so that existing bare-SimpleNamespace-fake unit
# tests (Tests/UI/test_library_multiselect_{conversations,media,notes}.py)
# that stub only `.refresh` -- not a new method name -- keep passing: a
# `self._new_method(...)` call would fail attribute lookup on a fake
# lacking that attribute BEFORE this function's own try/except could ever
# run; calling a module-level function with `screen` as an explicit
# argument has no such attribute-lookup step, so a fake missing
# `query_one`/`app`/etc. correctly falls through to the `except` below and
# reaches the already-stubbed `screen.refresh(...)` fallback instead.
def _patch_library_disabled_marker_label(button: Button) -> None:
    """Rebuild a marker-carrying action label after ``disabled`` flipped.

    task-4023 AC#1 (RC-07): the non-colour "○" disabled marker is part of
    the Button's label, so every in-place patcher that flips ``disabled``
    must rebuild the label too (the recompose-discipline rule). The base
    label is stashed on the button at compose time
    (``_library_disabled_marker_base``) because rendered labels are not a
    safe source to reconstruct from (the PR #665 escape lesson) and the
    notes canvas spells its compact label differently. A missing stash
    (unexpected widget shape) is a silent no-op -- the label simply keeps
    its compose-time marker state until the next canvas sync.

    Args:
        button: The action Button whose ``disabled`` was just updated.

    Returns:
        None.
    """
    base = getattr(button, "_library_disabled_marker_base", None)
    if base is None:
        return
    # task-31635 (critique #5 item 4): a button that opted into
    # marker-width alignment at compose time keeps it here -- this patcher
    # IS the path the first selection takes, so rebuilding the plain label
    # is exactly where the two-cell jump came back.
    button.label = library_disabled_action_label(
        base,
        button.disabled,
        align=getattr(button, "_library_disabled_marker_align", False),
    )


def _apply_library_row_toggle(
    screen: "LibraryScreen", kind: str, button: Button, row_id: str
) -> None:
    """In-place select-mode checkbox toggle for a Library row button.

    Tier 1: after the caller's ``row_selection.toggle(row_id)``, flips the
    pressed row's marker, the "N selected" Static, and the
    export-selected button's disabled state directly -- never a
    screen-level recompose. Any failure (e.g. the select-mode action
    strip isn't mounted because the mode raced) falls back to the old
    ``screen.refresh(recompose=True)`` rather than raising.

    Args:
        screen: The Library screen instance driving the update.
        kind: One of "conversations", "media", "notes" -- selects the
            ``#library-<kind>-selected-count`` /
            ``#library-<kind>-export-selected`` action-strip ids, plus the
            row-selection object: for "conversations" that is the dotted
            path ``screen._conversations_state.row_selection`` (extracted
            to ``LibraryConversationsState``, task 6/9), for "media" the
            dotted path ``screen._media_state.row_selection`` (extracted to
            ``LibraryMediaState``, wave-7) and for "notes" the dotted path
            ``screen._notes_state.row_selection`` (extracted to
            ``LibraryNotesState``, wave-8); for every other,
            not-yet-extracted kind it is still the flat attribute
            ``screen._library_<kind>_row_selection``. ``operator.attrgetter``
            resolves both shapes identically -- see the dispatch comment in
            this function's body.
        button: The pressed row's Button, rendered with the marker at
            position 0 (``f"{marker} {title}..."`` / notes'
            ``f"{marker} {title}..."`` glyph shape) -- only that leading
            character is replaced; the rest of the label (title,
            secondary line) is untouched since a selection toggle never
            changes it.
        row_id: The row's id, already toggled into/out of the same
            row-selection object described under ``kind`` above (dotted
            ``screen._conversations_state.row_selection`` for
            conversations, dotted ``screen._media_state.row_selection`` for
            media, dotted ``screen._notes_state.row_selection`` for notes,
            flat ``screen._library_<kind>_row_selection``
            for every other kind) by the caller -- read back here (single
            source of truth) rather than inferred by flipping the old
            marker text.

    Returns:
        None.
    """
    try:
        # task 9 (Conversations cleanup): every kind except conversations
        # still keeps its row-selection object as a flat screen attribute,
        # so the plain formatted name has always resolved via `getattr`.
        # Conversations' own `row_selection` field moved to
        # `screen._conversations_state.row_selection` (Task 6/9) -- one
        # extra hop a bare `getattr(screen, name)` cannot express.
        # `operator.attrgetter` resolves both a flat name and a dotted
        # path identically, so this is a passthrough for every other,
        # not-yet-extracted, kind. Future subsystem extractions hit this
        # exact same shape (see `_assign_library_reader_preferences_attribute`
        # in `library_screen.py` for the read+write sibling of this fix).
        # (wave-7 task 3) Media is the second kind to need the dotted form:
        # its `row_selection` field now lives on `screen._media_state`, and
        # the f-string spelling below can no longer reach it -- the exact
        # "computed name silently stops resolving" shape this dispatch
        # comment exists to warn about.
        # (wave-8 task 3) Notes is the third and LAST -- every extraction
        # wave is landed, so the f-string leg below is now reached only by
        # kinds that never had a state object at all. `_notes_state` also
        # resolves on `LibraryNotesController` (it declares the accessor
        # property for exactly this reason), because 31 notes movers hand
        # the sibling `_sync_library_canvas` dispatcher a controller `self`.
        row_selection_attribute = (
            "_conversations_state.row_selection"
            if kind == "conversations"
            else "_media_state.row_selection"
            if kind == "media"
            else "_notes_state.row_selection"
            if kind == "notes"
            else f"_library_{kind}_row_selection"
        )
        selection = operator.attrgetter(row_selection_attribute)(screen)
        count_static = screen.query_one(f"#library-{kind}-selected-count", Static)
        export_button = screen.query_one(f"#library-{kind}-export-selected", Button)
        checked = selection.is_selected(row_id)
        marker = "☑" if checked else "☐"
        # Rebuild the label from the RAW remainder the canvas stashed on the
        # button at compose time — NEVER from the mounted label: both
        # ``.plain`` and Textual 8's ``str(Content)`` return RENDERED text,
        # stripping the escape_markup() the canvas applied to user titles,
        # so a title like "[draft] notes" would restyle or drop on toggle
        # (PR #665 review; same escape lesson as the Library redesign). A
        # missing stash (unexpected widget shape) raises into the fallback
        # full recompose below.
        glyph = f"{marker} " if kind == "notes" else marker
        if kind == "notes":
            matching_buttons = tuple(
                candidate
                for candidate in screen.query(".library-notes-row")
                if str(getattr(candidate, "note_id", "") or "") == row_id
            )
        else:
            matching_buttons = (button,)
        for matching_button in matching_buttons:
            label_rest = matching_button._library_row_label_rest
            matching_button.label = f"{glyph}{label_rest}"
            if kind == "media":
                matching_button._library_media_checked = checked
        count_static.update(f"{selection.count} selected")
        export_button.disabled = selection.count == 0
        # F-018: the reason/action tooltip flips in place with `disabled`
        # (this patcher deliberately avoids a recompose, so the compose-
        # time tooltip would otherwise go stale).
        export_button.tooltip = (
            LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
            if export_button.disabled
            else LIBRARY_EXPORT_SELECTED_TOOLTIP
        )
        # task-4023 AC#1 (RC-07): the "○" disabled marker lives in the
        # label, so it must flip here alongside `disabled` (recompose
        # discipline). Rebuilt from the base label the canvas stashed at
        # compose time -- notes spells the compact label differently, so
        # the patcher must not hard-code it.
        _patch_library_disabled_marker_label(export_button)
        # task-2853: Media is the only canvas with a "Delete selected" bulk
        # action today (conversations/notes are out of this task's scope) --
        # flip it in place too, same reason/action tooltip pair as export.
        # Row presses are blocked outright while the bulk-delete confirm row
        # has replaced this button (see ``handle_library_media_row``), so
        # this lookup should always find it when ``kind == "media"``; any
        # surprise (e.g. a future caller that skips that guard) falls
        # through to the same full-recompose fallback as every other
        # failure here.
        if kind == "media":
            delete_button = screen.query_one("#library-media-delete-selected", Button)
            delete_button.disabled = selection.count == 0
            delete_button.tooltip = (
                LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP
                if delete_button.disabled
                else LIBRARY_DELETE_SELECTED_TOOLTIP
            )
            _patch_library_disabled_marker_label(delete_button)
            # task-28242 (Qodo #2335): the "Review selected" bulk action flips
            # in place too, or it stays disabled after the first row toggle
            # until an unrelated full recompose.
            review_button = screen.query_one(
                "#library-media-review-selected", Button
            )
            review_button.disabled = selection.count == 0
            review_button.tooltip = (
                LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP
                if review_button.disabled
                else LIBRARY_REVIEW_SELECTED_TOOLTIP
            )
            _patch_library_disabled_marker_label(review_button)
            # task-32045 (critique #7 P2): the shared Export/Review/Delete
            # "nothing selected" reason line flips in place too -- it is
            # always mounted (never conditionally composed; see the
            # compose-time comment in ``library_media_canvas.py``) for
            # exactly this Tier 1 path to find and toggle without a
            # recompose. ``visibility`` (not ``display``): the media row
            # list is composed AFTER this whole select-mode toolbar block,
            # so dropping the line from LAYOUT (``display``) shifted every
            # row up by one line the instant the first row was checked --
            # ``visibility: hidden`` reserves the same height while
            # painting nothing, so rows never move.
            bulk_reason = screen.query("#library-media-select-bulk-reason")
            if bulk_reason:
                bulk_reason.first().styles.visibility = (
                    "hidden" if selection.count else "visible"
                )
            # task-28007 AC#4: same in-place flip for "Analyze", with the
            # provider gate still outranking the count -- an unready
            # provider keeps the action off wearing its own reason, however
            # many rows are checked.
            analyze_button = screen.query_one(
                "#library-media-analyze-selected", Button
            )
            analysis_reason = screen._library_media_analyze_reason()
            analyze_button.disabled = bool(analysis_reason) or selection.count == 0
            analyze_button.tooltip = analysis_reason or (
                LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP
                if analyze_button.disabled
                else LIBRARY_ANALYZE_SELECTED_TOOLTIP
            )
            _patch_library_disabled_marker_label(analyze_button)
        elif kind == "notes":
            work_panes = screen.query("#library-note-work-pane")
            if work_panes and screen._library_note_session.snapshot is not None:
                work_panes.first(LibraryNoteWorkPane).apply_session_state(
                    screen._library_note_presentation_state()
                )
    except Exception:
        logger.debug(
            f"Library {kind} row toggle in-place update failed; falling back "
            "to full recompose.",
            exc_info=True,
        )
        screen.refresh(recompose=True)


#: TASK-32089 (phase C): the kinds whose canvas is RESIDENT under the browse
#: shell, mapped to the rail rows that own them. Before phase C, calling
#: ``_sync_library_canvas(screen, "media")`` while Media was not the route
#: raised ``NoMatches`` from its ``query_one``, was swallowed by the blanket
#: ``except`` below, and fell back to a whole-screen recompose. With both
#: browse canvases resident that query now SUCCEEDS: measured, the same call
#: returned ``True`` and rebuilt the hidden canvas's 28 children. There is no
#: exception left for the ``except`` to swallow, so the ownership test has to
#: be explicit -- otherwise residency silently redirects the sync storm onto
#: invisible canvases.
#:
#: Deliberately narrow: only these two kinds can be found off-route, so every
#: other kind keeps its pre-phase-C behaviour exactly (query raises, existing
#: fallback runs).
_LIBRARY_RESIDENT_CANVAS_OWNER_ROWS: dict[str, frozenset[str]] = {
    "media": frozenset({LIBRARY_ROW_BROWSE_MEDIA}),
    "notes": frozenset({LIBRARY_ROW_BROWSE_NOTES, LIBRARY_ROW_CREATE_NOTE}),
}

#: Sentinel for "this receiver does not expose the selection at all".
_NO_SELECTED_ROW = object()

#: The same two resident kinds as above, mapped to the canvas id residency
#: parks in the canvas host. Kept beside the owner rows so the two halves of
#: the residency guard cannot drift; ``library_browse_route_swap.
#: LIBRARY_RESIDENT_CANVAS_IDS`` holds the same set from the swap's side and
#: deliberately is NOT imported here (that module imports this one). The two
#: are pinned equal by ``Tests/UI/test_library_phase_c_switch_storm.py::
#: test_the_two_residency_guards_agree_on_which_canvases_are_resident``.
_LIBRARY_RESIDENT_CANVAS_IDS: dict[str, str] = {
    "media": "library-media-canvas",
    "notes": "library-notes-canvas",
}
_LIBRARY_RESIDENT_CANVAS_ID_SET = frozenset(_LIBRARY_RESIDENT_CANVAS_IDS.values())

#: The container residency parks hidden canvases in.
LIBRARY_CANVAS_HOST_ID = "library-canvas"


def _library_resident_canvas_awaits_display(canvas: Widget) -> bool:
    """Whether this canvas is resident but has not been shown yet.

    TASK-32089 (phase C, task 2.5) -- the DISPLAY half of the residency guard;
    its three siblings carry the same marker so a grep finds all four.

    The second half of the phase-C residency guard, and the same argument as
    the route-ownership half above: with both browse canvases permanently
    mounted, a sync can now land on one that is parked in the canvas host with
    ``display`` False. The route STATE flips several tens of milliseconds
    before the route SWAP shows the canvas, so syncs land in that window --
    phase-C task 2.5 measured **five of them on a resident Notes switch,
    25 of its 62 mounts**, every one superseded by the swap's own adopt-sync
    before anything was ever visible.

    Refusing is safe because showing a resident canvas is exactly what
    re-paints it: ``library_browse_route_swap._adopt_library_browse_canvas``
    syncs every resident canvas at the moment it displays it, and the route
    swap is the only writer that leaves a browse canvas **hidden and alive**.
    Three other sites do write ``child.display = False`` over canvas-host
    children (``library_screen.py:10341``, ``:10512``, ``:10959``) -- each one
    microseconds before ``remove_children`` on the same tuple, so the canvas
    they hide does not survive to be refused, and any sync landing in that
    window is caught by the projection-depth block below the refusal and
    replayed through ``_library_canvas_resync_pending``.

    Args:
        canvas: The mounted canvas the dispatcher just resolved.

    Returns:
        True when the sync must be refused and left to the swap.
    """
    if canvas.display or canvas.id not in _LIBRARY_RESIDENT_CANVAS_ID_SET:
        return False
    if getattr(canvas.parent, "id", None) != LIBRARY_CANVAS_HOST_ID:
        return False
    logger.debug(
        f"Library {canvas.id} sync refused: the canvas is resident but not "
        "displayed; the route swap repaints it when it shows it."
    )
    return True


def _library_canvas_kind_owns_route(screen: "LibraryScreen", kind: str) -> bool:
    """Whether ``kind``'s canvas is the one the current route owns.

    Reads ``_library_selected_row_id`` rather than the DOM on purpose: the
    dispatcher is handed EITHER the screen or a bare controller ``self`` by
    its seven forwarding controllers (conversations, ingest, media, notes,
    prompts, rag_search, skills), and all seven expose that field, while DOM
    presence stopped meaning "this route is active" the moment the canvases
    became resident.

    Args:
        screen: The receiver the dispatcher was handed.
        kind: The canvas kind requested.

    Returns:
        True when the sync may proceed.
    """
    owners = _LIBRARY_RESIDENT_CANVAS_OWNER_ROWS.get(kind)
    if owners is None:
        return True
    selected = getattr(screen, "_library_selected_row_id", _NO_SELECTED_ROW)
    if selected is _NO_SELECTED_ROW:
        # Fail open: a receiver without the selection cannot be judged, and
        # refusing would be a silent no-op rather than a guarded one.
        return True
    return selected in owners


def _sync_library_canvas(
    screen: "LibraryScreen",
    kind: str,
    *,
    then: Callable[[], bool | None] | None = None,
    allow_screen_fallback: bool = True,
    notes_focus_identity: LibraryNotesFocusIdentity | None = None,
    deferred_guard: Callable[[], bool] | None = None,
    sync_prompt_work: bool = True,
    projection_owned: bool = False,
) -> bool:
    """Canvas-scoped targeted update for a Library browse canvas (Tier 2).

    Rebuilds the given canvas's fresh display state and hands it to the
    mounted canvas widget's own ``sync_state`` -- a canvas-scoped
    ``recompose`` (the widget rebuilds only its OWN children) that skips
    the nav bar, footer, and rail entirely, unlike a screen-level
    ``screen.refresh(recompose=True)``.

    Releases the app's mouse capture first, mirroring
    ``BaseAppScreen.refresh`` (see that method's docstring for the full
    mouse-capture war story it defends against): this path recomposes the
    canvas directly via ``sync_state``, bypassing ``BaseAppScreen.refresh``
    -- and hence its guard -- entirely, so a canvas row mid mouse-capture
    (e.g. an ``Input`` click/selection whose ``MouseUp`` hasn't arrived
    yet) would otherwise be recomposed away and leave
    ``App.mouse_captured`` referencing a removed widget forever,
    permanently breaking click dispatch app-wide.

    TASK-15457 extends the contract to every high-frequency Library canvas.
    An unsupported ``kind``, like any other failure here (e.g. the canvas
    isn't mounted because a mode switch raced), falls back to the old
    whole-screen recompose rather than raising.

    Args:
        screen: The Library screen instance driving the update.
        kind: Supported Library canvas kind.
        then: Optional zero-argument follow-up (in practice: "focus the
            control the user should now be on"), run once the canvas's new
            children are mounted. It exists because
            ``screen.call_after_refresh`` -- correct for a WHOLE-SCREEN
            recompose, where ``Screen._on_timer_update`` runs the recompose
            before the callbacks -- has no ordering against a recompose
            driven by the CANVAS's own message pump. Reproduced while
            converting the notes strips: the follow-ups ran against removed
            children and stranded DOM focus outside the canvas. Supported
            only for canvases carrying ``PostRecomposeCallback``; passing it
            for another kind raises into the whole-screen fallback rather
            than silently dropping the follow-up. Returning ``False`` vetoes
            an explicitly supplied pre-detach Notes restore.

        allow_screen_fallback: Whether a failed targeted update may request
            the legacy whole-screen recompose.
        notes_focus_identity: Rich Notes identity captured before portable
            entry focus is detached, when automatic reconciliation owns sync.
        deferred_guard: Ownership predicate checked by deferred retained-owner
            DOM work before and after each await.
        sync_prompt_work: Whether a Prompt Items projection should also sync
            the independent retained Work pane. Browse-only settlements pass
            ``False`` so live editor widgets, cursor, and undo state survive.
        projection_owned: Whether this is the projection's own final sync.
            External syncs that land during a swap request one replay after
            the swap; the final sync itself must not request another replay.

    Returns:
        True when the mounted canvas accepted the state; otherwise False.
    """
    if not _library_canvas_kind_owns_route(screen, kind):
        # TASK-32089: an off-route sync for a RESIDENT canvas. Before phase C
        # this could not happen (the canvas was not mounted); now it can, and
        # doing the work would repaint an invisible canvas with state the
        # route it belongs to has not re-entered yet. Refuse, and do NOT take
        # the whole-screen fallback -- there is nothing to recover from.
        logger.debug(
            f"Library {kind} canvas sync refused: the route does not own it."
        )
        return False
    if (
        not projection_owned
        and getattr(screen, "_library_canvas_projection_depth", 0) > 0
    ):
        # Even when the outgoing canvas still exists and accepts this sync,
        # the projection may already hold a replacement built from older
        # state. Reapply current state after the outermost swap completes.
        screen._library_canvas_resync_pending = True
    canvas: Widget | None = None
    note_work: LibraryNoteWorkPane | None = None
    note_work_kwargs: dict[str, Any] = {}
    note_work_surface_changed = False
    notes_editor_owned = False
    prompt_work: LibraryPromptWorkPane | None = None
    prompt_work_kwargs: dict[str, Any] = {}
    skill_work: LibrarySkillWorkPane | None = None
    skill_work_kwargs: dict[str, Any] = {}
    follow_up_canvas: Widget | None = None
    prompt_work_recovered = False
    try:
        sync_args: tuple[Any, ...] = ()
        sync_kwargs: dict[str, Any] = {}
        if kind == "conversations":
            canvas = screen.query_one(
                "#library-conversations-canvas", LibraryConversationsCanvas
            )
            sync_args = (screen._build_library_conversations_state(),)
        elif kind == "media":
            canvas = screen.query_one("#library-media-canvas", LibraryMediaCanvas)
            if _library_resident_canvas_awaits_display(canvas):
                return False
            media_state = screen._build_library_media_state()
            # The state builder RESOLVES the selection -- a requested id the
            # active type filter no longer renders falls back to the first
            # row -- so the screen's pointer has to be re-read from it, the
            # same mirror ``compose_content`` and
            # ``_replace_library_browse_canvas`` perform. Without it,
            # filtering the selected item out left the canvas highlighting
            # row 0 while "Open in viewer" still opened the filtered-out
            # item (task-15457 review round 1, Critical 1).
            screen._media_state.selected_media_id = media_state.selected_id
            sync_args = (media_state,)
            sync_kwargs = screen._library_media_canvas_presentation()
        elif kind == "media-trash":
            # task-4025: the media canvas's Trash view -- same targeted
            # contract, its own mounted widget/state builder.
            canvas = screen.query_one(
                "#library-media-trash-canvas", LibraryMediaTrashCanvas
            )
            new_state = screen._build_library_media_trash_state()
            sync_args = (new_state,)
            sync_kwargs = screen._library_media_trash_canvas_presentation()
        elif kind == "notes":
            canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
            if _library_resident_canvas_awaits_display(canvas):
                return False
            # task-32127 (review round 2): BEFORE the kwargs are built. The
            # canvas composes its toolbar from the Items width this resolves,
            # and the caller has already flipped the notes view, so building
            # first handed the new view the previous one's geometry (measured:
            # the editor's 58 columns for a 77-column list after Back). It sits
            # AFTER the residency refusal above: an off-route canvas has no
            # `.library-notes-route` shell to resolve, and the route swap's
            # adopt-sync re-enters here once the canvas is displayed.
            screen._sync_library_notes_reader_layout_from_shell()
            sync_kwargs = screen._library_notes_list_canvas_kwargs()
            sync_kwargs["deferred_guard"] = deferred_guard
            # (wave-8 task 3) `focus_intent_generation` moved to
            # `LibraryNotesState`; the receiver here is the screen OR the
            # notes controller (26 movers at 31 call sites forward a bare
            # `self`), and `operator.attrgetter` re-reads BOTH hops on every
            # call, exactly as the `partial(getattr, screen, …)` it replaces.
            sync_kwargs["focus_intent_generation"] = partial(
                operator.attrgetter("_notes_state.focus_intent_generation"),
                screen,
            )
            shell = build_library_shell_state(
                screen._build_library_shell_input(),
                selected_row_id=screen._library_selected_row_id,
            )
            rail = screen._active_library_rail()
            if rail is not None:
                rail.apply_selection(
                    shell,
                    lifecycle=screen._library_lifecycle,
                    onboarding_all_empty=screen._library_onboarding_all_empty,
                )
            work_panes = screen.query("#library-note-work-pane")
            if work_panes:
                note_work = work_panes.first(LibraryNoteWorkPane)
                note_work_kwargs = screen._library_note_work_pane_kwargs()
                note_work_surface_changed = (
                    note_work_kwargs.get("mode") != note_work.mode
                )
                # task-32062 fix round 2: the reader's own title/body keeps
                # this pane's children (``sync_state`` skips the rebuild), so
                # the focus restore below has nothing to repair -- and queuing
                # it anyway left it pending until some LATER recompose, which
                # then dragged focus back out of the field the reader had
                # moved to. Same lateness for ``EditorReady``, posted from
                # ``_after_recompose``, whose handler re-lands an untouched
                # create on the title.
                notes_editor_owned = (
                    not note_work_surface_changed and note_work.editor_has_focus()
                )
                note_work.sync_state(**note_work_kwargs)
                if notes_editor_owned:
                    # task-32185 AC#5: that skipped rebuild is the ONE case
                    # ``sync_state`` stores a fresh presentation state and
                    # paints nothing -- so entering select mode beside an
                    # open editor (``bulk_read_only`` flipped, the reader's
                    # focus still in the title) left the pane fully editable
                    # with no "Read-only preview" banner. Every select-mode
                    # handler (toggle, select-all, clear, Escape, filter,
                    # sort) lands here, so the one re-apply lives here rather
                    # than at each of them. Gated on exactly the skip: every
                    # other sync recomposes, and ``_apply_post_compose_state``
                    # applies the stored state to the children it mounts --
                    # re-applying there too would run against children a
                    # pending mode-change recompose has not mounted yet.
                    screen._apply_library_note_presentation_state()
        elif kind == "prompts":
            canvas = screen.query_one(
                "#library-prompts-canvas", LibraryPromptsListCanvas
            )
            sync_kwargs = screen._library_prompts_list_canvas_kwargs()
            if sync_prompt_work:
                work_panes = screen.query("#library-prompt-work-pane")
                if work_panes:
                    prompt_work = work_panes.first(LibraryPromptWorkPane)
                    prompt_work_kwargs = screen._library_prompt_work_pane_kwargs()
        elif kind == "skills":
            canvas = screen.query_one("#library-skills-canvas", LibrarySkillsListCanvas)
            sync_kwargs = screen._library_skills_list_canvas_kwargs()
            work_panes = screen.query("#library-skill-work-pane")
            if work_panes:
                skill_work = work_panes.first(LibrarySkillWorkPane)
                skill_work_kwargs = screen._library_skill_work_pane_kwargs()
        elif kind == "ingest":
            canvas = screen.query_one("#library-ingest-canvas", LibraryIngestCanvas)
            sync_args = (screen._build_library_ingest_state(),)
        elif kind == "search":
            canvas = screen.query_one(
                "#library-search-rag-panel", LibrarySearchRagPanel
            )
            # Deliberately the FLAT name, not `screen._rag_search_state.
            # answer_render_key`: every "search"-kind caller of this
            # dispatcher forwards the CONTROLLER as `screen`
            # (`LibraryRagSearchController.cycle_library_rag_mode`/
            # `toggle_library_rag_scope_source`), and the controller's own
            # permanent generated shim exposes this flat setter -- the
            # controller has no `_rag_search_state` attribute at all. A
            # genuine screen-forwarded call would silently create a dead
            # instance attribute instead of raising (no such caller exists,
            # AST-verified). See recipe §18 ("A genuinely new finding...").
            screen._library_rag_answer_render_key = None
            sync_args = (screen._library_rag_panel_state(),)
        elif kind == "export":
            canvas = screen.query_one("#library-export-canvas", LibraryExportCanvas)
            sync_args = (screen._build_library_export_state(),)
        elif kind == "landing":
            canvas = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
            sync_args = (screen._library_landing_canvas_state(),)
        elif kind == "handoff":
            canvas = screen.query_one(
                "#library-study-handoff-canvas", LibraryStudyHandoffCanvas
            )
            shell = build_library_shell_state(
                screen._build_library_shell_input(),
                selected_row_id=screen._library_selected_row_id,
            )
            sync_args = (
                screen._library_study_handoff_canvas_state(shell.canvas_target),
            )
        else:
            raise ValueError(
                f"Unsupported Library canvas kind for targeted sync: {kind!r}"
            )
        if screen.is_running:
            try:
                screen.app.capture_mouse(None)
            except Exception:
                logger.debug(
                    "Mouse-capture release before Library canvas sync skipped.",
                    exc_info=True,
                )
        # task-15457 review round 1, Critical 2 + Important 3: the DEFAULT
        # follow-up. A whole-screen recompose restores portable Notes focus
        # and the named region's scroll offset through ``LibraryScreen.
        # refresh`` -> ``_rehydrate_library_notes_after_recompose`` ->
        # ``_restore_library_notes_focus_identity``. A canvas-scoped sync
        # bypasses that override entirely, so every converted notes site
        # WITHOUT its own ``then=`` let DOM focus escape to the console rail
        # when its focused child was recomposed away, and (only visible
        # below ``LIBRARY_NOTES_COMPACT_BREAKPOINT``, where the list can
        # actually scroll) dropped the notes list back to the top.
        # Captured here, at the same choke point the footer fix uses, and
        # gated on the notes workflow owning the route -- the identity is
        # Notes-specific, so a media/ingest/search sync must not build one.
        # An explicit ``then`` COMPOSES with this rather than replacing it.
        # Ordinary actions restore Notes first and let their follow-up choose
        # final focus; automatic reconciliation runs its generic focus guard
        # first so a newer user move can veto the stale Notes identity.
        follow_up: Callable[[], bool | None] | None = then
        # Gated on the KIND, not on the screen-level workflow predicate
        # (review m5): the identity is the notes canvas's, and
        # ``_library_notes_workflow_active()`` is also true while a
        # media/ingest/search sync runs from a Notes rail row, which would
        # build and restore an identity for a canvas that is not on screen.
        if (
            kind == "notes"
            and not note_work_surface_changed
            and not notes_editor_owned
            and sync_kwargs.get("mode") == getattr(canvas, "mode", None)
        ):
            # ...and only for an IN-SURFACE sync. On a surface TRANSITION
            # (list -> loading -> editor) the capture is worthless: the
            # handlers flip the notes view before calling this, so
            # ``_capture_library_notes_focus_identity`` already reports the
            # DESTINATION region with an empty semantic role -- restoring it
            # resolves to a fallback target (measured: the rail row), which is
            # worse than leaving focus to the transition's own arming path
            # (``_arm_library_note_editor`` + its explicit editor identity).
            identity = (
                notes_focus_identity
                if notes_focus_identity is not None
                else screen._capture_library_notes_focus_identity()
            )

            def _restore_then_explicit(
                _identity: LibraryNotesFocusIdentity = identity,
                _explicit: Callable[[], bool | None] | None = then,
            ) -> None:
                # Automatic reconciliation passes the pre-detach Notes
                # identity explicitly. Let its generic restore enforce the
                # current-user-focus veto before the richer Notes restore can
                # apply stale focus. Ordinary Notes actions retain their
                # established Notes-first, explicit-action-last ordering.
                if (
                    notes_focus_identity is not None
                    and _explicit is not None
                    and _explicit() is False
                ):
                    return
                screen._restore_library_notes_after_targeted_sync(_identity)
                if _explicit is not None and notes_focus_identity is None:
                    _explicit()

            follow_up = _restore_then_explicit
        elif kind == "notes" and notes_editor_owned:
            # task-32106: the editor-owned skip above protects the WORK pane,
            # but the Items pane beside it is a separate canvas and still
            # recomposes -- and its scroll offset went back to the top with
            # it, mid-sentence (measured at 100x30: a 6-row offset read 0
            # after one sync). Re-apply the offset only. Restoring FOCUS is
            # exactly what the skip exists to prevent, so this path never
            # touches it.
            #
            # Queuing a post-recompose follow-up here is only safe because
            # ``notes_editor_owned`` is measured on ``#library-note-work-
            # pane``, a SIBLING of ``#library-notes-canvas`` in the reader
            # shell -- so the list canvas's own ``editor_has_focus()`` is
            # False and it does recompose. Move the editor inside the list
            # canvas and ``sync_state``'s early return applies to it too,
            # and this callback would fire at some later recompose instead
            # (PR #2571 review, finding 8).
            listing = screen._library_notes_scroll_owner("navigator")
            list_offset = (
                (listing.scroll_offset.x, listing.scroll_offset.y)
                if listing is not None
                else None
            )

            def _restore_list_offset(
                _offset: tuple[int, int] | None = list_offset,
                _explicit: Callable[[], bool | None] | None = then,
            ) -> None:
                if _explicit is not None:
                    _explicit()
                if _offset is not None:
                    # Deferred for the same reason the identity restore
                    # defers its own offset: the recomposed rows have no
                    # layout yet, so the container clamps the offset to 0.
                    screen.call_after_refresh(_apply_offset, _offset)

            def _apply_offset(offset: tuple[int, int]) -> None:
                owner = screen._library_notes_scroll_owner("navigator")
                if owner is not None:
                    owner.scroll_to(
                        x=offset[0],
                        y=offset[1],
                        animate=False,
                        force=True,
                        immediate=True,
                    )

            if list_offset is not None or then is not None:
                follow_up = _restore_list_offset
        elif kind == "media":
            # task-31567: the Media equivalent of the Notes restore above,
            # and for the same reason -- a canvas-scoped sync recomposes the
            # Items pane's children, and the adaptive reader shell's two pane
            # GRIPS are the first focusable widgets Textual then picks. Every
            # media site without its own focus ``then=`` (Select all/Clear,
            # arming and cancelling the bulk-delete confirm, both receipt
            # Dismisses, leaving select mode, the Analyze progress repaints)
            # therefore dropped the user onto a grip, where Space collapses a
            # pane. Captured here, at the same choke point, so a new call site
            # gets it for free; the screen owns the capture/restore pair.
            # An explicit ``then`` runs FIRST and wins -- the restore no-ops
            # unless focus is still missing or on a grip.
            media_focus = screen._capture_library_media_focus_identity()

            def _media_restore(
                _previous: str | None = media_focus,
                _explicit: Callable[[], bool | None] | None = then,
            ) -> None:
                if _explicit is not None:
                    _explicit()
                screen._restore_library_media_focus(_previous)

            if then is not None or not canvas.has_pending_recompose_callback:
                # NEVER clobber a follow-up another sync already queued.
                # ``queue_after_recompose`` REPLACES, and a media mutation
                # queues its ``focus_identity`` follow-up (PR E's "land on
                # Undo") from one sync while a second, target-less sync --
                # the facet reload that rides the same completion -- lands
                # before the canvas has recomposed. Installing this restore
                # there dropped the Undo intent on the floor: live at 100x30
                # the bulk-delete receipt came back with a pane grip focused
                # and `Undo` unhighlighted, where dev focuses `┃ Undo ┃`.
                # With nothing queued there is nothing to lose, and an
                # explicit ``then`` replaces the pending callback here
                # exactly as it did before this branch existed.
                follow_up = _media_restore
        if kind == "landing":
            canvas.set_deferred_sync_guard(deferred_guard)
        follow_up_canvas = canvas
        if (
            note_work is not None
            and not notes_editor_owned
            and note_work_kwargs.get("mode") != "list"
        ):
            # Notes keeps its navigator mounted in Items while editor/load
            # children recompose in the retained Work pane. Follow-ups that
            # arm or focus those children must ride the widget doing that
            # recompose, not the unrelated list canvas.
            follow_up_canvas = note_work
        if prompt_work is not None and (
            prompt_work_kwargs.get("mode") != "list"
            or prompt_work_kwargs.get("import_open")
        ):
            follow_up_canvas = prompt_work
        if skill_work is not None and (
            skill_work_kwargs.get("mode") != "list"
            or skill_work_kwargs.get("import_open")
        ):
            follow_up_canvas = skill_work
        if follow_up is not None:
            follow_up_canvas.queue_after_recompose(follow_up)
        canvas.sync_state(*sync_args, **sync_kwargs)
        if prompt_work is not None:
            try:
                prompt_work.sync_state(**prompt_work_kwargs)
                screen._sync_library_prompts_reader_layout_from_shell()
            except Exception:
                logger.opt(exception=True).debug(
                    "Library prompts work-pane sync failed."
                )
                if (
                    follow_up_canvas is prompt_work or allow_screen_fallback
                ) and isinstance(follow_up_canvas, PostRecomposeCallback):
                    follow_up_canvas.queue_after_recompose(None)
                if allow_screen_fallback:
                    screen.refresh(recompose=True)
                    if then is not None:
                        screen.call_after_refresh(then)
                return False
        if skill_work is not None:
            try:
                skill_work.sync_state(**skill_work_kwargs)
                screen._sync_library_skills_reader_layout_from_shell()
            except Exception:
                logger.opt(exception=True).debug(
                    "Library Skills work-pane sync failed."
                )
                if isinstance(follow_up_canvas, PostRecomposeCallback):
                    follow_up_canvas.queue_after_recompose(None)
                if allow_screen_fallback:
                    screen.refresh(recompose=True)
                    if then is not None:
                        screen.call_after_refresh(then)
                return False
        return True
        # NOTE (task-15457 review): a footer re-derivation was added here and
        # then REMOVED after probing. Both footer tiers that a targeted sync
        # can flip -- the Notes tier (select mode / sort-strip visibility) and
        # the shared choice-strip tier -- are already kept honest by dev's own
        # mechanism: ``LibraryScreen.refresh`` calls
        # ``_apply_library_notes_footer_context()`` on EVERY refresh, not only
        # on ``recompose=True``, and every footer-flipping sync has a focus
        # follow-up whose ``set_focus``/``scroll_visible`` triggers one.
        # Verified by disabling both branches: the Notes select-mode and the
        # media type-strip footers both stayed current. Keeping an
        # unfalsifiable branch is worse than the coupling it guards, so it is
        # gone; the coupling itself is recorded in the task file's residuals.

    except Exception:
        # TASK-32089 (phase C): the design record's second ruling was to NARROW
        # this to ``(NoMatches, QueryError)`` so a genuinely unexpected failure
        # raises instead of becoming a silent whole-screen recompose. It was
        # implemented, measured against the suites, and REVERTED here: the
        # narrowing reds
        # ``test_library_canvas_sync_defects.py::
        #  test_strict_failure_retry_retains_original_semantic_focus``,
        # which forces a ``RuntimeError`` out of ``canvas.sync_state`` and
        # requires this handler to swallow it, recompose, and let the reconcile
        # RETRY the sync (``assert attempts == 2``) with the original focus
        # capture intact. So the swallow is not merely incidental: one existing
        # behaviour is pinned on it. Narrowing therefore needs its own task
        # that decides what that retry path should do with a non-DOM failure --
        # it is not something the residency mechanism requires, and phase C's
        # scope rule is "only as the mechanism requires".
        #
        # What the mechanism DID require is above the ``try``: the
        # route-ownership guard, which is what stops residency from silently
        # redirecting the sync storm onto invisible canvases.
        #
        # Kept from that round: the traceback. It used to be discarded, so an
        # AttributeError inside a state builder left one debug line and no
        # clue -- the exact shape wave 8 spent a round finding.
        logger.opt(exception=True).debug(f"Library {kind} canvas sync failed.")
        if kind == "prompts" and prompt_work is not None:
            try:
                prompt_work.sync_state(**prompt_work_kwargs)
                screen._sync_library_prompts_reader_layout_from_shell()
                prompt_work_recovered = True
            except Exception:
                logger.opt(exception=True).debug(
                    "Library prompts recovery work-pane sync failed."
                )
        # task-21116 review, M3: a targeted canvas projection detaches the
        # canvas host's child for the duration of its remove/mount await.
        # Any sync landing inside that window cannot find its canvas and
        # would take this fallback -- firing the very whole-screen
        # recompose the conversion removed, and racing a fresh
        # same-id canvas into the in-flight ``mount`` (observed:
        # ``DuplicateIds`` on '#library-media-canvas', then a 30 s pilot
        # stall). Reproduced BOTH ways: detaching the host and calling this
        # helper fires exactly one whole-screen recompose, and a real
        # viewer-back whose list reload outlives the swap does the same --
        # the shipped tests missed it only because their in-memory service
        # always won the race.
        #
        # Suppressing is safe rather than lossy: the projection rebuilds
        # the destination child from CURRENT screen state, so the state
        # this sync wanted to paint is picked up by the projection itself.
        # ``then`` is deliberately not run -- it is a focus/scroll
        # follow-up, and the projection owns focus through its own
        # ``then=`` (whose callback is ordered after the new children
        # mount); running it here would target children about to be
        # replaced, the stranded-focus shape task-15457 recorded.
        if (
            not projection_owned
            and getattr(screen, "_library_canvas_projection_depth", 0) > 0
        ):
            logger.debug(
                f"Library {kind} canvas sync suppressed: a targeted "
                "projection owns the canvas host."
            )
            screen._library_canvas_resync_pending = True
            if isinstance(follow_up_canvas, PostRecomposeCallback):
                follow_up_canvas.queue_after_recompose(None)
            return False
        if allow_screen_fallback:
            if isinstance(follow_up_canvas, PostRecomposeCallback):
                follow_up_canvas.queue_after_recompose(None)
            screen.refresh(recompose=True)
            if then is not None:
                screen.call_after_refresh(then)
        elif isinstance(follow_up_canvas, PostRecomposeCallback) and (
            follow_up_canvas is canvas or not prompt_work_recovered
        ):
            follow_up_canvas.queue_after_recompose(None)
        return False
