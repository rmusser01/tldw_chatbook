"""Library Browse ▸ Media canvas: media list, type filter, and preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Audio.meeting_session import normalize_speaker_name
# Re-exported (TASK-31745 moved these out of this module unchanged) so every
# existing importer -- and this canvas's own hidden legend -- keeps working.
from tldw_chatbook.Library.meeting_speaker_rename import (  # noqa: F401
    RENAME_REFUSED_EMPTY_TRANSCRIPT,
    RENAME_REFUSED_NOT_MEETING_CONTENT,
    SpeakerRenameResult,
    _meeting_speaker_legend_rows,
    _read_meeting_transcript_segments,
    _render_meeting_transcript,
    _write_meeting_transcript_row,
    can_rename_meeting_speakers,
    rename_meeting_speaker,
)
from tldw_chatbook.Library.library_pager_state import (
    LibraryPagerDisplay,
    library_pager_layout,
)
from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    MEDIA_SORT_CHOICES,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_DELETE_SELECTED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_EXPORT_SELECTED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_ANALYZE_SELECTED_TOOLTIP,
    LIBRARY_BULK_ACTIONS_NO_SELECTION_REASON,
    LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_REVIEW_SELECTED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_choice_label,
    library_choice_tooltip,
    library_disabled_action_label,
)
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths
from tldw_chatbook.UI.destination_recovery import (
    DestinationRecoveryState,
    load_failure_callout,
)
from tldw_chatbook.Widgets.Library.library_rail import _visible_row_title
from tldw_chatbook.Widgets.Library.library_choice_strip import (
    LibraryChoiceOptionList,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
    library_row_button,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Phase C, task 3: the canvas's ACTIONS collaborator is the media
    # controller itself, named rather than concealed behind a bespoke
    # protocol -- the spec's own ruling ("visible coupling over a concealed
    # facade", which retired the `LibraryScreenHost` facade idea). The
    # import is TYPE_CHECKING-only because `library_media_controller`
    # imports THIS module at runtime.
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )


_MEDIA_ROW_COMPACT_HEIGHT = 1
_MEDIA_ROW_WIDE_HEIGHT = 2

#: task-32213 review, finding 4: the F-018 reason for a "Review these" with
#: no rows behind it. Reason then next step, on one line, like the rest of
#: this canvas's disabled tooltips.
LIBRARY_MEDIA_REVIEW_EMPTY_TOOLTIP = (
    "Nothing here to review · clear the filter or pick another type."
)

# task-30043 (critique 2026-09-03 P1): the items pane sits at ~40-44 cols in
# EVERY real shell layout (3-pane reading shell AND the compact stage), so a
# single six-button row can never render its labels there -- live capture
# showed ``t so E Tr R Se`` and select mode's bulk actions as bare ``○ ○ ○``.
# The multi-row grammar below is therefore THE grammar, not a responsive
# variant: every row's label sum (including the "○ "-prefixed disabled
# forms) is budgeted to fit the pane's 40-col floor.


@dataclass(frozen=True)
class LibraryMediaRowGeometry:
    """One public Textual geometry revision from a Media row-scroll owner."""

    revision: int
    size: Size
    virtual_size: Size
    container_size: Size | None


class LibraryMediaRowGeometryChanged(Message):
    """Report one concrete Media row-scroll owner's revised geometry."""

    def __init__(
        self,
        owner: "LibraryMediaRowScroll",
        geometry: LibraryMediaRowGeometry,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.geometry = geometry


class LibraryMediaRowScroll(VerticalScroll):
    """Publish distinct Resize-derived geometry for the owning Media list."""

    latest_geometry: LibraryMediaRowGeometry | None = None

    def on_resize(self, event: events.Resize) -> None:
        """Publish distinct, monotonically revised owner geometry after reflow."""
        previous = self.latest_geometry
        geometry_values = (event.size, event.virtual_size, event.container_size)
        if previous is not None and geometry_values == (
            previous.size,
            previous.virtual_size,
            previous.container_size,
        ):
            return
        geometry = LibraryMediaRowGeometry(
            revision=1 if previous is None else previous.revision + 1,
            size=event.size,
            virtual_size=event.virtual_size,
            container_size=event.container_size,
        )
        self.latest_geometry = geometry
        self.post_message(LibraryMediaRowGeometryChanged(self, geometry))


def _capped_choice_value(value: str, cap: int = 8) -> str:
    """Bound a data-derived chooser value for its opener label (task-30043).

    Args:
        value: The stored value (e.g. a media type).
        cap: Maximum characters to show before an ellipsis.

    Returns:
        The value, or its first ``cap - 1`` characters plus ``…``.
    """
    value = str(value)
    return value if len(value) <= cap else value[: cap - 1] + "…"


def _media_row_marker(
    *,
    select_mode: bool,
    checked: bool,
    reviewed: bool | None,
    selected: bool,
    compact: bool,
) -> str:
    """Return the row's ONE leading state cell.

    task-28009 (controller ruling 4): a row never carries two slots. Select
    mode owns the cell (☑/☐); otherwise an active review set owns it (``✓``
    reviewed, ``·`` not yet), and only a row outside any active set falls
    through to the wide-mode current-row cue (``▸``). The row that is open
    in the Reader keeps its ``library-media-row-selected`` class either way,
    so nothing about the selection becomes invisible when a set is active.
    """
    if select_mode:
        return "☑" if checked else "☐"
    if reviewed is not None:
        return "✓" if reviewed else "·"
    return "▸" if selected and not compact else " "


def _media_row_label_rest(
    title: str,
    secondary: str,
    *,
    compact: bool,
    loading: bool = False,
    loaded: bool = False,
) -> str:
    """Return the marker-free Media row label for one responsive density.

    task-30044 (critique 2026-09-03 P2): both densities use the SHORT state
    word ("loaded" / "loading") -- the old wide-mode prose ("Loaded in
    Reader            ") consumed ~28 of ~35 label cells and displaced
    titles to "Quart"/"SQLit", so the row that mattered most was the one
    you couldn't identify.
    """
    visible_title = _visible_row_title(title)
    # task-32364 AC#1 (critique #10): the state used to prefix the TITLE
    # ("▸ Loaded · Attention Is All You Need"), so the row's identity was
    # displaced by its status. task-30044's constraint still holds -- the
    # SHORT word, never the old prose -- it just belongs on the fact line.
    state = "Loading" if loading else "Loaded" if loaded else ""
    detail = f"{secondary} · {state.lower()}" if state else secondary
    if compact:
        return f" {visible_title} · {detail}"
    return f" {visible_title}\n    {detail}"


class LibraryMediaCanvas(PostRecomposeCallback, RecomposeCaptureGuard, Vertical):
    """Render the Library media list with a type filter and preview.

    Attributes:
        canvas: Current media canvas display state.
    """

    BUNDLED_CSS = """
    /* task-30043: the multi-row action grammar needs CONTENT-width buttons.
     * Textual Button's 16-cell min-width floor alone would overflow the
     * pane's 40-col floor (six floors = 96 cells); each row's label budget
     * is what keeps task-28025's fit contract true. Baseline geometry lives
     * here so harnesses without the app bundle lay out like the app. */
    .ds-toolbar > .library-canvas-action,
    .ds-toolbar > .library-toolbar-count {
        width: auto;
        min-width: 0;
    }
    /* task-31224: Textual Input defaults to width 100%, which consumed the
     * whole filter row and pushed "Clear filter" off-screen -- the one
     * honest recovery for a filter miss was invisible (live: it never
     * rendered at any width). Share the row instead. */
    #library-media-filter {
        width: 1fr;
    }
    #library-media-filter-clear {
        width: auto;
        min-width: 0;
    }
    /* task-31270: receipts are two rows, full width; the copy wraps and the
     * action row keeps content-width buttons so Undo/Dismiss always paint. */
    .library-media-receipt {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-copy {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-actions {
        width: 100%;
        height: auto;
    }
    /* Task 8: same trap as #library-media-filter above -- Input defaults to
     * width 100%, which inside a row Horizontal would blow the label off
     * to the side. */
    .library-media-speaker-row {
        width: 100%;
        height: auto;
    }
    /* Class-keyed (not `.row Static`): ancestor-scoped bare-type subjects
     * are ratcheted by test_textual_css_fastpath (ADR-097). */
    Static.library-media-speaker-label {
        width: auto;
        min-width: 0;
    }
    Input.library-media-speaker-input {
        width: 1fr;
    }
    """

    def __init__(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        analysis_action_reason: str = "",
        load_failure: DestinationRecoveryState | None = None,
        list_unselectable: bool = False,
        compact: bool = False,
        show_preview: bool = True,
        can_rename_speakers: bool = False,
        media_db: Any = None,
        speaker_rename_media_id: int | None = None,
        actions: "LibraryMediaController | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.canvas = canvas
        # Phase C, task 3 (region ownership): what the 16 canvas-origin
        # `@on` rows below call. `None` is the harness default -- a bare
        # canvas built by a test has no controller and its toolbar is inert,
        # exactly as it was when the screen owned the routing and the test
        # mounted no screen. Both PRODUCTION construction sites bind it, and
        # that is pinned by
        # `Tests/UI/test_library_phase_c_region_ownership.py::
        # test_both_media_canvas_construction_sites_bind_the_actions_
        # collaborator`, because a missed one has no runtime symptom short
        # of a silently dead control.
        self.actions = actions
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.analysis_action_reason = analysis_action_reason
        self.load_failure = load_failure
        # task-31635 fix round 1: the screen's OWN "the load failed leaving
        # nothing to select" predicate (`_library_media_list_unselectable`),
        # not re-derived here -- a page failure that retained rows and a
        # facet-only failure both keep `load_failure` set over rows that
        # export fine.
        self.list_unselectable = list_unselectable
        self.compact = compact
        self.show_preview = show_preview
        # Task 8 (meeting diarization spec): True only when the selected
        # item's meeting folder still holds a `meeting.json`
        # (`can_rename_meeting_speakers`, computed by the caller). `media_db`
        # + `speaker_rename_media_id` are the real `MediaDatabase` and the
        # selected item's backing id -- needed here (breaking this canvas's
        # otherwise pure-state design on purpose) so the legend below can
        # read the meeting folder and actually call `rename_meeting_speaker`
        # rather than just showing an inert control. Absent, not merely
        # disabled, when False/None -- there is nothing to rename.
        self.can_rename_speakers = can_rename_speakers
        self.media_db = media_db
        self.speaker_rename_media_id = speaker_rename_media_id
        # Fill the (already 13fr) canvas host, not an independent 13fr --
        # ``LibraryMediaViewer`` documented this trap first: an `fr` width
        # here resolves against the HOST's content width per fraction, so
        # 13fr laid this canvas out ~13x wider than visible (measured 1703
        # cols on a 170-col terminal) and children clipped instead of
        # ellipsizing. task-14900's side-by-side split needs the panes to
        # divide the REAL width, so the canvas must be bounded like the
        # viewer already is.
        self.styles.width = "1fr"
        # task-32060: and NO min-width. task-30043 lowered this floor from 40
        # to 36 because a floor above the slot overflows it and clips every
        # child instead of ellipsizing -- but 36 has the same defect one step
        # down: the Items pane spends 4 cells on its own padding, so a 36-cell
        # pane hands this canvas 32 and the resolver's real floor (a 32-cell
        # pane, ITEMS_MIN_WIDTH) hands it 28. At the floor the armed
        # bulk-delete sentence was cut mid-word ("Delete 2 selected items?
        # You c") and keyword rows lost their ellipsis. The floor existed only
        # to bound the 13fr trap described above, and the `1fr` that replaced
        # it is bounded by the pane already, so there is nothing left to floor.

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Refuse row/action presses while this canvas is resident but hidden.

        Phase C keeps both browse canvases mounted and toggles ``display``
        (``UI/Library_Modules/library_browse_route_swap.py``). Textual 8.2.8's
        ``Button.press()`` consults the BUTTON's own ``disabled``/``display``
        and nothing above it, so a press aimed at a hidden canvas's row still
        bubbles to the screen and would act on a route the user has left --
        the design record's verified finding #3. Stop it here, at the canvas
        that owns the residency state, rather than in each of the screen's
        row handlers.

        **Scope, stated because it is narrower than it looks.** This gates
        ``Button.Pressed`` and nothing else. Still ungated, deliberately:
        ``Input.Changed`` / ``Input.Submitted`` (a hidden widget is not in the
        focus chain, so a user cannot type into one, and no code drives these
        programmatically off-route), and ``LibraryMediaRowGeometryChanged``,
        which a hidden row owner can still post -- that one is rejected
        screen-side by the return-settlement fence instead, pinned by
        ``test_library_media_return_settlement.py::
        test_route_change_rejects_later_geometry_settlement``. If a future
        change makes a hidden canvas focusable or drives its Inputs from code,
        this gate does NOT cover it.

        **And it does not cover this canvas's OWN ``@on`` rows** (phase C,
        task 3). Textual 8.2.8 dispatches a node's decorated handlers BEFORE
        that node's ``on_<message>`` convention method
        (``MessagePump._get_dispatch_methods`` yields ``_decorated_handlers``
        first, per MRO class), and ``event.stop()`` only sets
        ``_stop_propagation``, which is read AFTER the whole dispatch loop --
        so nothing here can un-run a handler on this same node. The migrated
        rows below therefore take the refusal themselves, through
        ``_media_actions_for_press``. This method stays because it is still
        the only thing refusing presses aimed at the 20 canvas-origin rows
        the screen still owns.

        Args:
            event: The bubbling press.

        Returns:
            None.
        """
        if not self.display:
            event.stop()
            event.prevent_default()

    # ---- Phase C, task 3: the canvas-origin rows this region owns --------
    #
    # Sixteen ``@on`` rows moved here from ``LibraryScreen``'s routing table.
    # The rule is ORIGIN: every control below is composed by this canvas's
    # own ``compose``, and ``LibraryMediaRowGeometryChanged`` is posted by
    # ``LibraryMediaRowScroll``, defined in this module -- so the message
    # passes through this widget on its way up. The Reader's, the Trash
    # canvas's and the adaptive shell's rows can only be caught at the
    # screen and stay there permanently; the full three-way census (16
    # migrated / 20 deferred / 43 permanent) is pinned by
    # ``Tests/UI/test_library_phase_c_region_ownership.py``.
    #
    # Behaviour does not move with the routing: each row forwards to the
    # same-named ``LibraryMediaController`` method, unchanged. The One Rule
    # keeps the work in the controller; what the region widget gains is
    # ownership of its own events.

    def _media_actions_for_press(
        self, event: Button.Pressed
    ) -> "LibraryMediaController | None":
        """Resolve the actions collaborator for a press, or refuse.

        The residency refusal for the migrated rows, mirroring
        ``on_button_pressed``'s scope EXACTLY (presses, and nothing else) so
        the migration changes no behaviour: a hidden resident canvas already
        swallowed these presses before they moved here, and it still does --
        just one dispatch step earlier, which is where it now has to happen.

        Args:
            event: The press being routed.

        Returns:
            The media controller, or ``None`` when this canvas is parked
            off-route (or was built without a controller, as in a bare-canvas
            test).
        """
        if not self.display:
            event.stop()
            event.prevent_default()
            return None
        return self.actions

    @on(Input.Changed, "#library-media-filter")
    def handle_library_media_filter_changed(self, event: Input.Changed) -> None:
        """Route the filter box's debounce tick to the media controller."""
        if self.actions is not None:
            self.actions.handle_library_media_filter_changed(event)

    @on(Input.Submitted, "#library-media-filter")
    def handle_library_media_filter_submitted(self, event: Input.Submitted) -> None:
        """Route an explicit filter submit to the media controller."""
        if self.actions is not None:
            self.actions.handle_library_media_filter_submitted(event)

    @on(Button.Pressed, "#library-media-filter-clear")
    @on(Button.Pressed, "#library-media-scope-clear")
    def handle_library_media_filter_clear(self, event: Button.Pressed) -> None:
        """Route "Clear filter" and the scope line's Clear to the controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_filter_clear(event)

    @on(Button.Pressed, "#library-media-sort")
    def handle_library_media_sort(self, event: Button.Pressed) -> None:
        """Route the sort chooser's opener to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_sort(event)

    @on(OptionList.OptionSelected, "#library-media-sort-choices")
    def handle_library_media_sort_choice(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Route a picked sort value to the media controller."""
        if self.actions is not None:
            self.actions.handle_library_media_sort_choice(event)

    @on(Button.Pressed, "#library-media-previous")
    def handle_library_media_previous(self, event: Button.Pressed) -> None:
        """Route the pager's Previous to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_previous(event)

    @on(Button.Pressed, "#library-media-next")
    def handle_library_media_next(self, event: Button.Pressed) -> None:
        """Route the pager's Next to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_next(event)

    @on(Button.Pressed, "#library-media-retry")
    def handle_library_media_retry(self, event: Button.Pressed) -> None:
        """Route Retry -- the pager's and the load-failure callout's -- on."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_retry(event)

    @on(Button.Pressed, "#library-media-select-all")
    def handle_library_media_select_all(self, event: Button.Pressed) -> None:
        """Route select mode's "All" to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_select_all(event)

    @on(Button.Pressed, "#library-media-select-clear")
    def handle_library_media_select_clear(self, event: Button.Pressed) -> None:
        """Route select mode's "None" to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_select_clear(event)

    @on(Button.Pressed, "#library-media-open-viewer")
    def handle_library_media_open_viewer(self, event: Button.Pressed) -> None:
        """Route the preview's "Open in viewer" to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_open_viewer(event)

    @on(Button.Pressed, "#library-media-export")
    async def handle_library_media_export(self, event: Button.Pressed) -> None:
        """Route the toolbar's Export to the media controller.

        The one ``async`` row of the sixteen: the controller's Export handler
        awaits a modal, so a sync forwarder here would build a coroutine and
        drop it.
        """
        actions = self._media_actions_for_press(event)
        if actions is not None:
            await actions.handle_library_media_export(event)

    @on(Button.Pressed, "#library-media-review")
    def handle_library_media_review_these(self, event: Button.Pressed) -> None:
        """Route "Review these" to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_review_these(event)

    @on(Button.Pressed, "#library-media-review-selected")
    def handle_library_media_review_selected(self, event: Button.Pressed) -> None:
        """Route select mode's "Review selected" to the media controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_review_selected(event)

    @on(Button.Pressed, "#library-media-review-sets")
    def handle_library_media_review_sets(self, event: Button.Pressed) -> None:
        """Route the title row's review-sets control to the controller."""
        actions = self._media_actions_for_press(event)
        if actions is not None:
            actions.handle_library_media_review_sets(event)

    @on(LibraryMediaRowGeometryChanged)
    def _handle_library_media_row_geometry_changed(
        self, event: LibraryMediaRowGeometryChanged
    ) -> None:
        """Route one row-scroll owner's settled geometry to the controller.

        Deliberately NOT behind ``_media_actions_for_press``: this message is
        not a press, and the hidden-canvas case is already rejected by the
        controller's own return-settlement fence (``_library_media_
        settlement_tree`` owner check), which is where task 2 recorded that
        refusal as living.
        """
        if self.actions is not None:
            self.actions._handle_library_media_row_geometry_changed(event)

    def sync_state(
        self,
        canvas: LibraryMediaCanvasState,
        *,
        pager: LibraryPagerDisplay | None = None,
        type_options: tuple[str | None, ...] | None = None,
        stale_action_reason: str = "",
        mutation_action_reason: str = "",
        analysis_action_reason: str = "",
        load_failure: DestinationRecoveryState | None = None,
        list_unselectable: bool = False,
        compact: bool = False,
        show_preview: bool = True,
        can_rename_speakers: bool = False,
        media_db: Any = None,
        speaker_rename_media_id: int | None = None,
    ) -> None:
        """Refresh the canvas from new state.

        Args:
            canvas: Latest media canvas display state.

        Returns:
            None.
        """
        self.canvas = canvas
        self.pager = pager
        self.type_options = (
            canvas.type_options if type_options is None else type_options
        )
        self.stale_action_reason = stale_action_reason
        self.mutation_action_reason = mutation_action_reason
        self.analysis_action_reason = analysis_action_reason
        self.load_failure = load_failure
        self.list_unselectable = list_unselectable
        self.compact = compact
        self.show_preview = show_preview
        self.can_rename_speakers = can_rename_speakers
        self.media_db = media_db
        self.speaker_rename_media_id = speaker_rename_media_id
        self.refresh(recompose=True)

    # ---- Task 8 (meeting diarization spec): inline speaker rename ---------
    _SPEAKER_INPUT_PREFIX = "library-media-speaker-input-"

    @on(Input.Submitted, "#library-media-speaker-legend Input")
    def _handle_speaker_rename_submitted(self, event: Input.Submitted) -> None:
        """Rename the submitted row's speaker and refresh the shown transcript.

        Mirrors the live Meetings screen's Task 7 `_apply_rename`: the
        rename itself is unconditional (a submit racing teardown should
        still persist), and only the widget refresh afterwards is
        `is_mounted`-guarded.
        """
        event.stop()
        widget_id = event.input.id or ""
        if not widget_id.startswith(self._SPEAKER_INPUT_PREFIX):
            return
        cluster_id = widget_id[len(self._SPEAKER_INPUT_PREFIX):]
        name = normalize_speaker_name(event.value)
        event.input.value = ""
        if self.media_db is None or self.speaker_rename_media_id is None:
            return
        # Qodo Q3: the rename reads the transcript file, runs several DB
        # writes, FTS maintenance and a post-ingest dispatch -- all of which
        # would freeze the UI on a large transcript or a busy database.
        self.run_worker(
            lambda: self._rename_speaker_off_thread(cluster_id, name),
            group="library-media-speaker-rename",
            thread=True,
            exit_on_error=False,
        )

    def _rename_speaker_off_thread(self, cluster_id: str, name: str) -> None:
        """Persist one rename on a worker thread, then refresh on the UI one."""
        try:
            outcome = rename_meeting_speaker(
                self.media_db, self.speaker_rename_media_id, cluster_id, name
            )
        except Exception as exc:  # noqa: BLE001 - a rename must not crash the canvas
            # `rename_meeting_speaker` reads/writes `meeting.json`; a
            # filesystem failure's `str()` embeds the meeting folder path
            # (task-9 diagnostic inventory review) -- redact it, mirroring
            # `meetings_screen.py`'s own rename-persist failure log.
            logger.warning("Library media speaker rename failed: {}", redact_user_paths(str(exc)))
            outcome = SpeakerRenameResult(False, f"unexpected error ({type(exc).__name__})")
        self.app.call_from_thread(self._apply_speaker_rename_outcome, cluster_id, outcome)

    def _apply_speaker_rename_outcome(
        self, cluster_id: str, outcome: SpeakerRenameResult
    ) -> None:
        """Report a refused/failed rename, or repaint after a successful one.

        Qodo Q15: a rename that changed nothing used to leave only a debug
        log, so the user saw the old name and no explanation.
        """
        if not outcome.ok:
            self.app.notify(f"Couldn't rename this speaker: {outcome.reason}.", severity="warning")
            return
        if not self.is_mounted:
            return
        self._refresh_after_speaker_rename(cluster_id)

    def _refresh_after_speaker_rename(self, cluster_id: str) -> None:
        """Re-read the rewritten `Media.content` and patch the preview text
        plus the just-renamed row's own legend label in place."""
        row = self.media_db.get_media_by_id(self.speaker_rename_media_id)
        content = row["content"] if row else ""
        try:
            self.query_one("#library-media-preview-lines", Static).update(content)
        except NoMatches:
            pass
        try:
            speaker_rows = dict(
                _meeting_speaker_legend_rows(self.media_db, self.speaker_rename_media_id)
            )
            label_widget = self.query_one(
                f"#library-media-speaker-label-{cluster_id}", Static
            )
            label_widget.update(speaker_rows.get(cluster_id, cluster_id))
        except Exception:  # noqa: BLE001 - legend label refresh is best-effort
            pass

    def apply_compact_presentation(self, compact: bool) -> None:
        """Patch mounted Media density and preview participation in place."""
        self.compact = compact
        select_mode = getattr(self.canvas, "select_mode", False)
        row_height = (
            _MEDIA_ROW_COMPACT_HEIGHT if compact else _MEDIA_ROW_WIDE_HEIGHT
        )
        for button in self.query(".library-media-row"):
            title = button._library_media_title
            secondary = button._library_media_secondary
            label_rest = _media_row_label_rest(
                title,
                secondary,
                compact=compact,
                loading=button._library_media_loading,
                loaded=button._library_media_loaded,
            )
            button._library_row_label_rest = label_rest
            marker = _media_row_marker(
                select_mode=select_mode,
                checked=button._library_media_checked,
                reviewed=button._library_media_reviewed,
                selected=button._library_media_selected,
                compact=compact,
            )
            button.label = f"{marker}{label_rest}"
            button.set_class(
                button._library_media_selected and not compact and not select_mode,
                "library-media-row-selected",
            )
            button.styles.height = row_height
            button.styles.min_height = row_height
            self._gate_mutation_action(button, label_rest.lstrip())
        try:
            preview = self.query_one("#library-media-preview")
            open_viewer = self.query_one("#library-media-open-viewer", Button)
        except NoMatches:
            return
        preview.display = self.show_preview and self._has_preview and not compact
        open_viewer.can_focus = self.show_preview and not compact

    def apply_reader_state(self, canvas: LibraryMediaCanvasState) -> None:
        """Patch Reader row state without replacing row widgets.

        Args:
            canvas: Fresh media canvas state carrying selection and load flags.
        """
        self.canvas = canvas
        rows = {row.media_id: row for row in canvas.rows}
        select_mode = getattr(canvas, "select_mode", False)
        for button in self.query(".library-media-row"):
            row = rows.get(str(button.media_id))
            if row is None:
                continue
            button._library_media_selected = row.selected
            button._library_media_checked = row.checked
            button._library_media_loading = row.loading
            button._library_media_loaded = row.loaded
            button._library_media_reviewed = row.reviewed
            label_rest = _media_row_label_rest(
                row.title,
                row.secondary,
                compact=self.compact,
                loading=row.loading,
                loaded=row.loaded,
            )
            button._library_row_label_rest = label_rest
            marker = _media_row_marker(
                select_mode=select_mode,
                checked=row.checked,
                reviewed=row.reviewed,
                selected=row.selected,
                compact=self.compact,
            )
            button.label = f"{marker}{label_rest}"
            button.set_class(
                row.selected and not self.compact and not select_mode,
                "library-media-row-selected",
            )

    def _gate_stale_action(self, button: Button, base_label: str) -> Button:
        """Apply the controller's stale-OR-mutation gate to one unsafe action.

        Final review M-1: despite the name (and despite reading as the
        symmetric partner of ``_gate_mutation_action`` below, which gates on
        write-in-flight only), this disables on EITHER input -- a stale page
        OR a write actually in flight. Do not assume a mutation ending
        leaves these controls live if the page is still stale.
        """
        reason = self.mutation_action_reason or self.stale_action_reason
        if reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = reason
        return button

    def _select_all_button(self, rendered_count: int) -> Button:
        """Build the Select-all bulk action (summary-row sibling of the count)."""
        label = f"Select all {rendered_count} shown"
        select_all = Button(
            label,
            id="library-media-select-all",
            classes="library-canvas-action",
            compact=True,
        )
        return self._gate_stale_action(select_all, label)

    def _clear_selection_button(self) -> Button:
        """Build the Clear bulk action."""
        clear = Button(
            "Clear",
            id="library-media-select-clear",
            classes="library-canvas-action",
            compact=True,
        )
        return self._gate_stale_action(clear, "Clear")

    def _bulk_action_button(
        self,
        base: str,
        widget_id: str,
        disabled_tooltip: str,
        enabled_tooltip: str,
        *,
        danger: bool = False,
    ) -> Button:
        """Build one Export/Review/Delete bulk action (task-30043).

        Short REAL words ("Export" / "Review" / "Delete") -- the mode and the
        adjacent count make them unambiguous, the F-018 tooltips carry the
        full sentences, and a disabled action can never collapse to a bare
        "○" marker the way the full labels did at the pane's 40-col floor.

        task-4023 AC#1 (RC-07): "○" disabled marker -- these are the very
        buttons the user entered Select mode looking for, previously
        colour-only at a measured 1.39:1. The base label is stashed so
        `_apply_library_row_toggle`'s in-place patch can rebuild it when
        the selection count crosses 0.
        """
        bulk_disabled = self.canvas.selected_count == 0
        classes = "library-canvas-action"
        if danger:
            classes += " library-media-action-danger"
        button = Button(
            library_disabled_action_label(base, bulk_disabled, align=True),
            id=widget_id,
            classes=classes,
            compact=True,
        )
        button._library_disabled_marker_base = base
        # task-31635 (critique #5 item 4): these four flip disabled IN PLACE
        # (the selection count crossing 0), so they visibly jumped two cells
        # when the marker left the label. The enabled spelling reserves the
        # marker's width, and the in-place patcher reads this flag so the
        # label it rebuilds holds the same column
        # (``_patch_library_disabled_marker_label``).
        #
        # task-31959 carried the same reservation to the sibling canvases
        # (`library_conversations_canvas.py`, `library_notes_canvas.py`,
        # `library_prompts_canvas.py`), so every select-mode action that
        # flips with the selection count now holds its column.
        button._library_disabled_marker_align = True
        button.disabled = bulk_disabled
        # F-018: a disabled action says why.
        button.tooltip = disabled_tooltip if bulk_disabled else enabled_tooltip
        return self._gate_stale_action(button, base)

    def _select_mode_bulk_buttons(self) -> ComposeResult:
        """Yield the Export and Review bulk actions (Delete rides its own row)."""
        yield self._bulk_action_button(
            "Export",
            "library-media-export-selected",
            LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_EXPORT_SELECTED_TOOLTIP,
        )
        # task-28242: "Review selected" pins the selection as an ordered
        # review set.
        yield self._bulk_action_button(
            "Review",
            "library-media-review-selected",
            LIBRARY_REVIEW_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_REVIEW_SELECTED_TOOLTIP,
        )

    def _analyze_selected_button(self) -> Button:
        """Build the "Analyze" bulk action (task-28007 AC#4).

        Rides its OWN row rather than joining Clear/Export/Review: those
        three already measure 33 of the ~36 cells a narrow Items pane hands
        this canvas, so a fourth 13-cell action clipped every label on that
        row. Same
        multi-row grammar the danger row uses. When no analysis provider is
        configured the resolver's own sentence replaces the F-018 tooltip,
        so the disabled control says WHY, not just that it is off (AC#5's
        rule, applied to the bulk gesture).
        """
        button = self._bulk_action_button(
            "Analyze",
            "library-media-analyze-selected",
            LIBRARY_ANALYZE_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_ANALYZE_SELECTED_TOOLTIP,
        )
        if self.analysis_action_reason:
            button.label = library_disabled_action_label("Analyze", True)
            button.disabled = True
            button.tooltip = self.analysis_action_reason
        return button

    def _delete_selected_button(self) -> Button:
        """Build the isolated danger action (task-2853's far-end rule, upgraded)."""
        return self._bulk_action_button(
            "Delete",
            "library-media-delete-selected",
            LIBRARY_DELETE_SELECTED_DISABLED_TOOLTIP,
            LIBRARY_DELETE_SELECTED_TOOLTIP,
            danger=True,
        )

    def _gate_failed_action(self, button: Button, base_label: str) -> Button:
        """Disable a list-wide action whose list failed with nothing in it.

        task-31635 (critique #5 item 6): with the FIRST load failed,
        "Export…" -- which exports the whole filtered list -- stayed live
        and colour-normal beside the recovery callout, while "Select" had
        already gone to its "○" marker with a reason. Fix round 1: the
        predicate is failure AND nothing retained (the screen's
        ``_library_media_list_unselectable``), not the callout's broader
        one: a later-page failure keeps its rows (that retention is the
        callout's whole point) and a facet-only failure never touches them,
        and those rows export fine.

        Applied BEFORE ``_gate_stale_action`` at each call site, so a write
        in flight or a stale page still wins the tooltip: those are the
        more immediate blocker, and PR E's precedence is untouched.

        task-31960 (J final review M1) settled the one asymmetry this gate
        had: "Review these" pins the whole filtered list as an ordered
        review set -- the same shape of action as "Export…" -- and it alone
        stayed outside this gate, standing live beside a dimmed Export on a
        failed first page. It was defensible (its worker re-fetches and
        notifies on failure), but the asymmetry was unexplained at the
        surface and symmetry cost one line, so both whole-list actions now
        gate here. Still deliberately NOT gated: "Trash" (a route into a
        view with its own fetch, callout and Retry) and the callout's own
        Retry (``_gate_mutation_action`` only) -- both are how a reader
        gets out of a failed list.

        Args:
            button: The list-wide action to gate.
            base_label: The action's plain enabled label.

        Returns:
            The same button, gated when the list failed with no rows behind
            it.
        """
        failure = self.load_failure
        if failure is not None and self.list_unselectable:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = failure.disabled_tooltip
        return button

    def _gate_mutation_action(self, button: Button, base_label: str) -> Button:
        """Disable even recovery controls only while a write is unsettled."""
        if self.mutation_action_reason:
            button.label = library_disabled_action_label(base_label, True)
            button.disabled = True
            button.tooltip = self.mutation_action_reason
        return button

    def compose(self) -> ComposeResult:
        """Render the header/filter, status line, media rows, and preview.

        Returns:
            ComposeResult for the media canvas.
        """
        title_count = self.pager.title_count if self.pager is not None else self.canvas.count
        title = "Media" if title_count is None else f"Media ({title_count})"
        select_mode = getattr(self.canvas, "select_mode", False)
        fresh_zero = (
            self.pager is not None
            and title_count == 0
            and not self.canvas.rows
            and not select_mode
            and not self.canvas.delete_receipt_count
            and not self.stale_action_reason
            and not self.mutation_action_reason
            and not self.pager.status_copy
            and not self.pager.retry_visible
        )
        # task-28243: the "Sets" picker opener lives on the TITLE row, not the
        # action toolbar -- that toolbar already overflows the narrow items
        # pane (task-28025) and one more button pushed a squeezed Button into
        # rich's zero-width chop_cells crash (live-verified 2026-09-02). The
        # title row carries ~9 chars in a min-width-40 pane, so both widgets
        # always render at full width. Auto-width Static + fixed compact
        # Button only (task-4023's render-safe grammar: no 1fr sibling).
        # Hidden in select mode like the other list-level actions.
        # task-31635 (critique #5 item 7): it survives the fresh-empty page
        # too. It used to be composed under the same gate as the page's ONE
        # recovery action, so filtering to zero rows removed the only route
        # back to a saved review set exactly when the list had nothing else
        # to offer -- and Sets is navigation, not a result. It is never
        # disabled here: the picker opens over any list (it carries its own
        # empty copy, and "Read later" needs no saved set at all). The
        # recovery-action budget is unaffected -- that count is about the
        # empty page's own body, and this lives on the title row.
        title_row = Horizontal(id="library-media-title-row")
        title_row.styles.height = "auto"
        with title_row:
            title_static = Static(title, id="library-media-title")
            # A Static defaults to 1fr inside a Horizontal and would swallow
            # the whole row, pushing the button out of view (live-verified).
            title_static.styles.width = "auto"
            yield title_static
            sets_btn = Button(
                "Sets",
                id="library-media-review-sets",
                classes="library-canvas-action",
                compact=True,
                tooltip="Resume, switch, or dismiss saved review sets.",
            )
            sets_btn.display = not select_mode
            yield sets_btn
        # task-32350: the applied scope, stated. The filter Input below is a
        # draft until submitted, so it cannot be trusted to describe the
        # rows; this line is projected from the applied scope in
        # build_library_media_browse_state and can only ever agree with them.
        # A state built by the legacy ``build_library_media_state`` carries no
        # scope_line; an empty Static would just spend a row saying nothing.
        if self.canvas.scope_line:
            scope_row = Horizontal(id="library-media-scope-row")
            scope_row.styles.height = "auto"
            with scope_row:
                # Width comes from `.library-media-scope-line` (1fr +
                # ellipsis), not an inline style: an auto-width Static
                # pushed its own Clear off the pane edge once the Reader
                # narrowed Items.
                yield Static(
                    self.canvas.scope_line,
                    id="library-media-scope-line",
                    classes="library-media-scope-line",
                    markup=False,
                )
                if self.canvas.scope_clearable:
                    yield Button(
                        "Clear",
                        id="library-media-scope-clear",
                        classes="library-canvas-action",
                        compact=True,
                        tooltip="Clear the filter and type this line states.",
                    )
        filter_row = Horizontal(classes="ds-toolbar")
        filter_row.styles.height = "auto"
        with filter_row:
            yield Input(
                value=self.canvas.query,
                # task-31274: say that keywords match too. Kept to 14 cells
                # because Textual word-wraps the placeholder and paints only
                # its first line: at the default 38-col Items pane "Filter by
                # title, content or keyword…" rendered as "Filter by" (live,
                # 235x52). The empty state names the full field set.
                placeholder="Title/keyword…",
                # The placeholder is truncated by design (see above), so the
                # long form lives here rather than nowhere.
                tooltip="Filter by title, content or keyword",
                id="library-media-filter",
            )
            clear_filter = Button(
                "Clear filter",
                id="library-media-filter-clear",
                compact=True,
            )
            clear_filter.disabled = not bool(self.canvas.query)
            yield clear_filter
        if fresh_zero:
            yield Static(
                self.canvas.empty_copy,
                id="library-media-status",
                markup=False,
            )
            # task-31224: a FILTER miss must not suggest importing -- the
            # query-echoing status copy plus the (now visible) Clear filter
            # control above are the honest recovery. Import/Show-all stay
            # the recovery for a genuinely empty source only.
            #
            # task-32213 (critique #9 row 10): this block used to RETURN,
            # taking `type:`, `sort:`, Export…, Trash, Select and Review
            # these with it -- so the facet that produced the empty page
            # could neither be read nor reset from the canvas. Only the
            # recovery BUTTON is conditional now; composition falls through
            # to the toolbar. The recovery budget is unchanged (still at
            # most one action here), and `select_disabled` below already
            # disables Select on a zero-row page.
            if self.canvas.active_type is not None:
                yield Button(
                    "Show all types",
                    id="library-media-empty-clear-type",
                    classes="library-canvas-action",
                    compact=True,
                )
            elif not self.canvas.query:
                yield Button(
                    "Import media",
                    id="library-media-empty-import",
                    classes="library-canvas-action",
                    compact=True,
                )
        # Gate/label off the RENDERED rows, not ``canvas.count`` -- the latter
        # is the pre-filter total across ALL media types, so with a media-type
        # filter active it overstates what's shown (and stays > 0 when the
        # filter renders nothing). ``handle_library_media_select_all`` already
        # selects only the rendered rows, so this keeps the copy/gate honest.
        # Also portable to the conversations canvas state, which has no
        # ``.count`` field.
        rendered_count = len(self.canvas.rows)
        # task-4023 AC#5: one toolbar grammar across the list canvases --
        # these three actions used to stack VERTICALLY (one full-width
        # button per row) while Notes/Prompts/Skills lay theirs out in
        # horizontal ``ds-toolbar`` rows. Same render-safe shape as those
        # canvases: fixed-width compact Buttons only, never mixed with a
        # 1fr sibling.
        # task-14902: while the type choice strip is open it REPLACES this
        # toolbar row (the Notes Sort precedent -- browse actions hide while
        # the chooser is showing), keeping the vertical budget flat.
        type_choices_visible = getattr(self.canvas, "type_choices_visible", False)
        sort_choices_visible = getattr(self.canvas, "sort_choices_visible", False)
        toolbar_visible = not (type_choices_visible or sort_choices_visible)
        sort_labels = dict(MEDIA_SORT_CHOICES)
        current_sort = getattr(self.canvas, "sort_by", "last_modified_desc")
        type_filter = Button(
            # task-14902: a chooser-opener, no longer a cycler -- press
            # opens the direct-pick strip below instead of advancing.
            # Qodo #2350: the VALUE is data and can be long; the label caps
            # it so the chooser row's budget holds (the tooltip and the
            # chooser strip itself carry the full value).
            library_choice_label(
                "type",
                "All types"
                if self.canvas.active_type is None
                else _capped_choice_value(self.canvas.active_type),
            ),
            id="library-media-type-filter",
            classes="library-canvas-action",
            compact=True,
            tooltip=library_choice_tooltip(
                "media type",
                tuple(
                    "All types" if value is None else value
                    for value in self.type_options
                ),
            ),
        )
        self._gate_mutation_action(type_filter, str(type_filter.label))
        # task-28013: sort chooser opener -- hidden in select mode like
        # Export/Trash (Select's toolbar acts on the selection).
        sort_btn = Button(
            library_choice_label(
                "sort", sort_labels.get(current_sort, "Newest")
            ),
            id="library-media-sort",
            classes="library-canvas-action",
            compact=True,
            tooltip=library_choice_tooltip(
                "the sort order", tuple(label for _, label in MEDIA_SORT_CHOICES)
            ),
        )
        sort_btn.display = not select_mode
        self._gate_stale_action(sort_btn, str(sort_btn.label))
        export_btn = Button(
            "Export…",
            id="library-media-export",
            classes="library-canvas-action",
            compact=True,
        )
        export_btn.display = not select_mode
        self._gate_failed_action(export_btn, "Export…")
        self._gate_stale_action(export_btn, "Export…")
        # task-4025: the browsable Trash surface's entry point -- a
        # plain navigation action (never a `type:` cycle value: `type:`
        # cycles CONTENT types derived from the records, and trash is a
        # STATE). Always enabled: the trash count isn't known until its
        # view fetches, and an empty Trash shows its honest empty copy
        # rather than this button lying disabled. task-31635 fix round 1
        # keeps that even under a failed Media load -- Trash is a route into
        # a view with its OWN fetch, callout and Retry, so disabling it
        # would remove the only way to reach deleted items exactly when the
        # store is unhappy. Hidden in select mode like "Export…" -- Select's
        # toolbar is for acting on the selection, not navigating away from
        # it.
        trash_btn = Button(
            "Trash",
            id="library-media-trash-open",
            classes="library-canvas-action",
            compact=True,
            tooltip="Browse and restore deleted media.",
        )
        trash_btn.display = not select_mode
        # task-28242: "Review these" pins the WHOLE filtered result as an
        # ordered review set and walks it in the Reader. A list-level
        # action, hidden in select mode like Export/Trash.
        # task-32213 review, finding 4: with the toolbar now surviving a
        # 0-result page, "Review these" would paint live next to a
        # "○ Select" that says there is nothing to select -- and pressing
        # it only raised a toast ("No media items to review."). A state
        # that cannot act says so ON the control, in this canvas's own
        # grammar (the "○" marker plus an F-018 reason), never only in a
        # toast. Same predicate as ``select_disabled`` below.
        review_disabled = rendered_count == 0
        review_btn = Button(
            library_disabled_action_label("Review these", review_disabled),
            id="library-media-review",
            classes="library-canvas-action",
            compact=True,
            tooltip=(
                LIBRARY_MEDIA_REVIEW_EMPTY_TOOLTIP
                if review_disabled
                else "Review every item in this list, one by one."
            ),
        )
        review_btn.disabled = review_disabled
        review_btn.display = not select_mode
        # task-31960: "Review these" pins the WHOLE filtered list, exactly
        # like "Export…" -- so it takes the same failed-list gate, in the
        # same order (failed first, stale second, so a write in flight or a
        # stale page still wins the tooltip). Before this it was the one
        # list-wide action outside that gate and stood live and
        # colour-normal beside a dimmed Export on a failed first page.
        self._gate_failed_action(review_btn, "Review these")
        self._gate_stale_action(review_btn, "Review these")
        # Disable only when there's nothing to select AND we're not
        # already in select mode -- in select mode the button is "Done"
        # and must always be pressable so the user can exit even if the
        # rows dropped to zero (e.g. a background snapshot refresh
        # emptied the list).
        select_disabled = rendered_count == 0 and not select_mode
        select_btn = Button(
            # task-4023 AC#1 (RC-07): disabled carries the non-colour
            # "○" marker; the F-018 reason tooltip below says why.
            library_disabled_action_label(
                "Done" if select_mode else "Select", select_disabled
            ),
            id="library-media-select-toggle",
            classes="library-canvas-action",
            compact=True,
        )
        select_btn.disabled = select_disabled
        if select_disabled:
            select_btn.tooltip = LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP
        self._gate_stale_action(
            select_btn, "Done" if select_mode else "Select"
        )
        # task-30043 (critique P1): at the items pane's ~40-col real width
        # one Horizontal cannot render six labels (live capture: ``t so E Tr
        # R Se``), so the browse actions split into rows of readable labels.
        # In select mode most of these hide, so ``type:`` keeps that row to
        # itself and Done moves out of the toolbar entirely (see the Done row
        # at the end of the select-mode block below).
        if not select_mode:
            toolbar_rows: tuple[tuple[str, tuple[Button, ...]], ...] = (
                ("library-media-toolbar-choosers", (type_filter, sort_btn)),
                (
                    "library-media-toolbar-actions",
                    (export_btn, trash_btn, select_btn),
                ),
                ("library-media-toolbar-review", (review_btn,)),
            )
        else:
            toolbar_rows = (
                (
                    "library-media-toolbar",
                    (
                        type_filter,
                        sort_btn,
                        export_btn,
                        trash_btn,
                        review_btn,
                    ),
                ),
            )
        for row_id, row_buttons in toolbar_rows:
            row = Horizontal(id=row_id, classes="ds-toolbar")
            row.styles.height = "auto"
            row.display = toolbar_visible
            with row:
                for button in row_buttons:
                    yield button
        if type_choices_visible:
            options: list[Option] = []
            highlighted = 0
            for index, value in enumerate(self.type_options):
                display = "All types" if value is None else value
                option = Option(
                    f"✓ {display}" if value == self.canvas.active_type else display,
                    id=f"library-media-type-option-{index}",
                )
                option.choice_value = value
                options.append(option)
                if value == self.canvas.active_type:
                    highlighted = index
            # task-32210: the house `█` cursor on the highlighted option --
            # the plain OptionList marked it by background alone (1.09:1).
            choices = LibraryChoiceOptionList(
                *options,
                id="library-media-type-choices",
                compact=True,
                markup=False,
            )
            choices.highlighted = highlighted
            choices.styles.height = min(8, max(1, len(options)))
            yield choices
        if sort_choices_visible:
            # task-31235 (critique #3 P1): a vertical OptionList exactly like
            # the type chooser above -- the horizontal choice strip clipped
            # "Title A-Z" and rendered "Title Z-A" nowhere at the items
            # pane's real width, while keyboard selection could still pick
            # the invisible option.
            sort_options: list[Option] = []
            sort_highlighted = 0
            for index, (value, label) in enumerate(MEDIA_SORT_CHOICES):
                option = Option(
                    f"✓ {label}" if value == current_sort else label,
                    id=f"library-media-sort-option-{index}",
                )
                option.choice_value = value
                sort_options.append(option)
                if value == current_sort:
                    sort_highlighted = index
            # task-32210: same `█` cursor as the type chooser above.
            sort_choices = LibraryChoiceOptionList(
                *sort_options,
                id="library-media-sort-choices",
                compact=True,
                markup=False,
            )
            sort_choices.highlighted = sort_highlighted
            sort_choices.styles.height = min(8, max(1, len(sort_options)))
            yield sort_choices
        confirming_bulk_delete = getattr(self.canvas, "confirming_bulk_delete", False)
        if select_mode:
            if confirming_bulk_delete:
                # A single full-width Static above the toolbar, not inside it
                # -- mixing a long sentence Static with the toolbar's fixed-
                # width Buttons in one Horizontal is the known non-rendering
                # failure mode (see LibraryMediaViewer.compose's delete-
                # confirm copy, the same pattern this mirrors). The short
                # "N selected" Static below is unaffected -- it is already
                # proven to render alongside Buttons in this exact row.
                # task-4025 AC3 (ADR-055 Pattern A): the confirm copy names
                # the durable recovery path -- the Trash view this task
                # built (the list toolbar's own "Trash" action) -- on top
                # of the receipt's immediate Undo. Supersedes task-4022
                # AC3's honest "there's no Trash view" copy, which was
                # true only until this surface existed.
                item_word = "item" if self.canvas.selected_count == 1 else "items"
                confirm_copy = Static(
                    f"Delete {self.canvas.selected_count} selected {item_word}? "
                    "You can undo right away, or restore later from Trash.",
                    id="library-media-bulk-delete-confirm-copy",
                    markup=False,
                )
                # task-30043: bound the copy to the pane so the safety
                # sentence WRAPS -- unbounded, it clipped mid-word ("You can
                # und / restore later from Tr") at the narrow pane width.
                confirm_copy.styles.width = "1fr"
                confirm_copy.styles.height = "auto"
                yield confirm_copy
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                # Bug found via task-2853's OWN live tmux verification
                # (reproduced against pre-task-8 HEAD too, so it predates
                # this task, and against the Conversations canvas too, the
                # identical pattern -- see review round 2): with no
                # explicit width, this Static resolved as unbounded inside
                # the ``ds-toolbar`` ``Horizontal`` -- live capture showed
                # it claiming ~1700 columns on a 170-column terminal,
                # pushing every sibling Button entirely off-screen
                # (invisible, though still present in the DOM -- which is
                # why headless ``query_one`` pilot tests never caught it).
                # Fixed as the general rule via the shared
                # ``library-toolbar-count`` class (css/components/
                # _agentic_terminal.tcss), not a per-widget Python
                # one-off, so every canvas's counter is covered by one
                # declaration.
                yield Static(
                    f"{self.canvas.selected_count} selected",
                    id="library-media-selected-count",
                    classes="library-toolbar-count",
                    markup=False,
                )
                if confirming_bulk_delete:
                    confirm = Button(
                        "Delete",
                        id="library-media-bulk-delete-confirm",
                        classes="library-canvas-action library-media-action-danger",
                        compact=True,
                    )
                    yield self._gate_stale_action(confirm, "Delete")
                    cancel = Button(
                        "Cancel",
                        id="library-media-bulk-delete-cancel",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(cancel, "Cancel")
                else:
                    yield self._select_all_button(rendered_count)
            if not confirming_bulk_delete:
                # task-30043: the bulk actions get their own row of short
                # REAL words ("○ Export", never a bare marker) -- one shared
                # row rendered them as ``○ ○ ○`` at the pane's 40-col floor.
                actions_row = Horizontal(
                    id="library-media-select-actions", classes="ds-toolbar"
                )
                actions_row.styles.height = "auto"
                with actions_row:
                    yield self._clear_selection_button()
                    yield from self._select_mode_bulk_buttons()
                # task-32045 (critique #7 P2): Export/Review/Delete all gate
                # on ``selected_count == 0`` (see ``_bulk_action_button``)
                # and previously dimmed with only the "○" marker -- no
                # inline reason, unlike Analyze below. One shared line (not
                # one per button, since all three share the one gate) using
                # Analyze's own always-visible-reason grammar
                # (``.library-media-action-reason``, task-31981). Excluded
                # when the list itself failed to load with nothing to select
                # (``_gate_failed_action``'s predicate) -- that state already
                # explains itself via the recovery callout, and "select
                # items" would be the wrong reason. task-32085 AC#1 (Qodo #7):
                # also gated on ``rendered_count > 0`` -- select mode surviving
                # a refresh to a SUCCESSFUL empty list has nothing to select,
                # so the line paints nothing there too (still mounted, so the
                # in-place toggle below can only ever fire with rows present).
                #
                # Always yielded (visibility toggled, not conditionally
                # composed, and NOT ``display`` either): a single row-press
                # toggle takes the Tier 1 in-place patch
                # (``_apply_library_row_toggle``), which deliberately never
                # recomposes (task-252 perf discipline) -- so this widget
                # must already exist for that patch to flip it alongside
                # Export/Review/Delete's ``disabled``. ``display=False``
                # was tried first and regressed
                # ``test_every_click_on_a_media_row_toggles_it_in_select_mode``:
                # it drops the line's row from layout entirely, so the
                # in-place flip on the FIRST check shifted the media row
                # list (composed further down this same method) up by one
                # line under a screen coordinate a caller had already
                # computed. ``visibility: hidden`` reserves the same
                # height while painting nothing, so the row list never
                # moves.
                bulk_list_failed = (
                    self.load_failure is not None and self.list_unselectable
                )
                bulk_reason_line = Static(
                    LIBRARY_BULK_ACTIONS_NO_SELECTION_REASON,
                    id="library-media-select-bulk-reason",
                    classes="library-media-action-reason",
                    markup=False,
                )
                bulk_reason_line.styles.visibility = (
                    "visible"
                    if (
                        self.canvas.selected_count == 0
                        and rendered_count > 0
                        and not bulk_list_failed
                    )
                    else "hidden"
                )
                yield bulk_reason_line
                # task-28007 AC#4: Analyze gets its own row -- see
                # ``_analyze_selected_button`` for the measurement that put
                # it here rather than beside Export/Review.
                analyze_row = Horizontal(
                    id="library-media-select-analyze", classes="ds-toolbar"
                )
                analyze_row.styles.height = "auto"
                with analyze_row:
                    yield self._analyze_selected_button()
                if self.analysis_action_reason:
                    # task-31981: surface the blocker inline (below its own
                    # row), not only on the hover tooltip -- the same grammar
                    # the Reader's Generate gate and the Export gate use, so a
                    # keyboard-first user sees WHY Analyze is off.
                    yield Static(
                        self.analysis_action_reason,
                        id="library-media-analyze-selected-reason",
                        classes="library-media-action-reason",
                        markup=False,
                    )
                # task-2853's danger-isolation rule, upgraded: Delete gets a
                # whole row, so it is never adjacent to any other action.
                danger_row = Horizontal(
                    id="library-media-select-danger", classes="ds-toolbar"
                )
                danger_row.styles.height = "auto"
                with danger_row:
                    yield self._delete_selected_button()
            # task-31631 AC#3: Done closes the select-mode block on its own
            # row. It used to ride the toolbar row directly after ``type:``,
            # which is the exact cell range ``sort: Newest`` holds in browse
            # mode (measured at 235x52: Done x=63..71 inside sort's
            # x=63..79), so the habitual click on the sort chooser silently
            # became "leave select mode and discard the selection".
            #
            # Every other slot in the pane's top three rows is likewise a
            # browse control's (type:/sort:, Export…/Trash, Review these),
            # and the "N selected / Select all N shown" summary row already
            # measures 33 of the pane's 36 cells with a two-digit count and
            # a "○ " disabled marker -- a trailing Done clips there. The row
            # AFTER the select-mode block is the only slot that collides
            # with nothing. Rendered outside the ``confirming_bulk_delete``
            # branch above, because Done must stay pressable mid-confirm.
            done_row = Horizontal(
                id="library-media-select-done", classes="ds-toolbar"
            )
            done_row.styles.height = "auto"
            with done_row:
                yield select_btn

        # task-4022 AC2: a completed bulk delete's receipt, naming the
        # count with an Undo affordance right at the point of action --
        # mirrors the ingest queue's own done-row grammar ("✓ done · file
        # · 1s" + a jump action) rather than a toast, which this canvas
        # has none of on the success path today. Rendered OUTSIDE
        # select_mode: a full-success delete exits select mode, so this is
        # the only place left to show it. Uses the same
        # ``library-toolbar-count`` class as "N selected" above -- proven
        # safe for a short Static sharing a ``ds-toolbar`` Horizontal with
        # Buttons (see the comment on that Static; an earlier long-
        # sentence Static in this same row went unbounded and pushed every
        # Button off-screen).
        receipt_count = getattr(self.canvas, "delete_receipt_count", 0)
        if receipt_count:
            receipt_word = "item" if receipt_count == 1 else "items"
            # task-31220 (critique #5): a receipt may only claim success
            # while its Undo can actually run. A failed restore retitles it
            # with the same ✗ glyph the Analyze receipt uses for a run
            # where nothing succeeded, and Undo becomes a retry over the
            # still-failed ids ``receipt_count`` now names.
            undo_failure = getattr(self.canvas, "delete_receipt_undo_failure", "")
            # task-31270 (critique #4 P1): two rows -- copy, then actions --
            # at full width. A single content-width Horizontal clipped Undo
            # to "Und" at the Items pane's real width; same multi-row
            # grammar as the toolbars (task-30043).
            receipt = Vertical(
                id="library-media-bulk-delete-receipt",
                classes="library-media-receipt",
            )
            receipt.styles.height = "auto"
            with receipt:
                yield Static(
                    # task-4025 (ADR-055 Pattern A): the receipt names the
                    # durable path too -- "· in Trash" points at the Trash
                    # view that outlives this receipt's Undo/Dismiss.
                    f"✗ undo failed · {undo_failure}"
                    if undo_failure
                    else f"✓ deleted · {receipt_count} {receipt_word} · in Trash",
                    id="library-media-bulk-delete-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                actions.styles.height = "auto"
                with actions:
                    # task-31220: NOT ``_gate_stale_action``. Undo restores
                    # exactly the ids this receipt names, so it is the
                    # receipt's own recovery and a stale PAGE behind it
                    # cannot invalidate it -- disabling it here broke the
                    # confirmation's "You can undo right away" promise at
                    # the one moment it mattered (critique #5). The shared
                    # write interlock still applies, so a second mutation
                    # can never be claimed while one is in flight.
                    undo_label = "Retry undo" if undo_failure else "Undo"
                    undo = Button(
                        undo_label,
                        id="library-media-bulk-delete-undo",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(undo, undo_label)
                    dismiss = Button(
                        "Dismiss",
                        id="library-media-bulk-delete-receipt-dismiss",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(dismiss, "Dismiss")

        # task-31236: a dismissed review set's undo receipt -- the same
        # grammar as the bulk-delete receipt above, because a one-click
        # dismissal of a mid-walk set (with its done-marks) must be
        # recoverable right where the user lands after the picker closes.
        dismissed_set_name = getattr(
            self.canvas, "review_dismiss_receipt_name", ""
        )
        if dismissed_set_name:
            dismiss_receipt = Vertical(
                id="library-media-review-dismiss-receipt",
                classes="library-media-receipt",
            )
            dismiss_receipt.styles.height = "auto"
            with dismiss_receipt:
                yield Static(
                    f"✓ dismissed · {dismissed_set_name}",
                    id="library-media-review-dismiss-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                set_actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                set_actions.styles.height = "auto"
                with set_actions:
                    # Final review I-3: NOT ``_gate_stale_action``, for the
                    # same reason the bulk-delete receipt's Undo above is
                    # exempt -- this Undo restores exactly the one set its
                    # own copy names, so a stale PAGE behind it cannot
                    # invalidate it. Before this branch both receipts' Undo
                    # were gated identically; leaving this one on the stale
                    # gate let it sit disabled beside a live sibling
                    # receipt's Undo with no rule the user could infer.
                    undo_set = Button(
                        "Undo",
                        id="library-media-review-dismiss-undo",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(undo_set, "Undo")
                    close_receipt = Button(
                        "Dismiss",
                        id="library-media-review-dismiss-receipt-close",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(close_receipt, "Dismiss")

        # task-28007 AC#3/AC#4: the bulk-Analyze receipt -- PR A's two-row
        # grammar (copy, then actions) again, because a set-level run must
        # report per-item outcomes where the gesture happened rather than
        # in a toast that outlives nothing. Three states, one block: the
        # AC#3 Skip/Overwrite choice (nothing has run yet), the live run,
        # and the settled run with Retry failed/Dismiss.
        analyze_choice = getattr(self.canvas, "analyze_choice_count", 0)
        analyze_total = getattr(self.canvas, "analyze_receipt_total", 0)
        if analyze_choice or analyze_total:
            analyze_done = getattr(self.canvas, "analyze_receipt_done", 0)
            analyze_failed = getattr(self.canvas, "analyze_receipt_failed", 0)
            analyze_running = getattr(self.canvas, "analyze_receipt_running", False)
            failed_copy = f" · {analyze_failed} failed" if analyze_failed else ""
            if analyze_choice:
                # R3: no dangling dash -- the buttons are on the row BELOW,
                # so a trailing "— " pointed at nothing. ``analyze_total``
                # is the pressed selection's own size on this path.
                analyze_copy = (
                    f"{analyze_choice} of {analyze_total} already analyzed"
                )
            elif analyze_running:
                # 1-based position of the item being analyzed right now.
                position = min(analyze_done + analyze_failed + 1, analyze_total)
                analyze_copy = f"Analyzing {position} of {analyze_total}{failed_copy}"
            else:
                # A run where NOTHING succeeded must not lead with a tick.
                # ✗ (U+2717) is this repo's failure glyph (see
                # UI/Evals/library_rail.py), paired with the ✓ it
                # replaces here.
                glyph = "✓" if analyze_done else "✗"
                analyze_copy = (
                    f"{glyph} analyzed · {analyze_done} of {analyze_total}"
                    f"{failed_copy}"
                )
            analyze_receipt = Vertical(
                id="library-media-analyze-receipt",
                classes="library-media-receipt",
            )
            analyze_receipt.styles.height = "auto"
            with analyze_receipt:
                yield Static(
                    analyze_copy,
                    id="library-media-analyze-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                analyze_actions = Horizontal(
                    classes="ds-toolbar library-media-receipt-actions"
                )
                analyze_actions.styles.height = "auto"
                with analyze_actions:
                    if analyze_choice:
                        skip = Button(
                            "Skip them",
                            id="library-media-analyze-skip",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(skip, "Skip them")
                        overwrite = Button(
                            "Overwrite",
                            id="library-media-analyze-overwrite",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(overwrite, "Overwrite")
                    elif not analyze_running and analyze_failed:
                        retry = Button(
                            "Retry failed",
                            id="library-media-analyze-retry",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_stale_action(retry, "Retry failed")
                    if not analyze_running and not analyze_choice:
                        # A run in flight has nothing to dismiss yet: the
                        # counts are still moving and Dismiss would either
                        # lie (the run continues) or imply a cancel this
                        # gesture does not offer.
                        #
                        # (final review, I-1) The armed CHOICE has no
                        # Dismiss either: three 13-cell buttons overflow
                        # the Items pane at its 36-cell floor (the row
                        # painted "Skip them  Overwrite  Dism", the same
                        # clipping task-31270 fixed for "Und"), and
                        # "Skip them" already IS the change-nothing
                        # outcome -- it retires the card when there is
                        # nothing left to run.
                        analyze_dismiss = Button(
                            "Dismiss",
                            id="library-media-analyze-receipt-dismiss",
                            classes="library-canvas-action",
                            compact=True,
                        )
                        yield self._gate_mutation_action(analyze_dismiss, "Dismiss")

        # task-31632 (critique #5 P1): ONE recovery callout for a failed
        # load -- what failed, why, and the Retry that recovers it, INSIDE
        # the callout. Measured before this: "Couldn't load page 1." painted
        # as a bare sentence with no reason at all, and its only Retry sat
        # 33 rows below in the pager strip (15 rows at 100x30). The
        # ``.ds-recovery-callout`` grammar is the Library hub's own "Needs
        # attention" row; ``.is-blocked`` is the repo-wide error tint that
        # overrides the base warning tint, so a timeout (recoverable by a
        # later attempt) and a hard failure never paint alike.
        failure = self.load_failure
        if failure is not None:
            # PR M carry I1: the widget is the shared
            # ``load_failure_callout`` -- the landing hub and the Library
            # browse row paint the same one, from the same builder. This
            # canvas keeps its own action styling and its write-in-flight
            # gate (even recovery controls wait for an unsettled write).
            yield load_failure_callout(
                failure,
                id="library-media-load-failure",
                copy_id="library-media-load-failure-copy",
                retry_id="library-media-retry",
                retry_classes="library-canvas-action",
                gate=self._gate_mutation_action,
            )

        if fresh_zero:
            # task-32213: the fresh-zero branch near the top of this method
            # owns the page from here down -- it already yielded THIS id
            # (with the same ``empty_copy``) and the one recovery action.
            # Its old early return took the TOOLBAR with it, which is the
            # bug; everything BELOW is list furniture (row viewport,
            # "No media item selected." placeholder, "Item 0-0 of 0" pager)
            # a zero-row page has never shown and still has no use for.
            return
        status_text = (
            self.pager.status_copy
            if self.pager is not None and self.pager.status_copy
            else self.canvas.status_copy or self.canvas.empty_copy
        )
        if failure is not None and status_text.startswith(failure.unavailable_what):
            # The callout above IS this sentence with its reason attached
            # (the controller derives both halves in the same branch, so
            # they can never name different failures) -- painting the bare
            # form again directly under it is the duplicate the callout
            # exists to remove. A STALE page's own gate copy ("Media
            # changed…", "Couldn't retry · <reason>") names a different
            # event and survives untouched.
            status_text = ""
        status = Static(
            status_text,
            id="library-media-status",
            markup=False,
        )
        status.display = bool(status_text)
        yield status

        # task-2853 AC4: while Select mode is active, the preview must never
        # show an item outside the current (multi-item) selection context --
        # ``canvas.selected_id``/``preview_lines`` still carry whatever was
        # focused before Select was entered (the UAT's "bottom preview pane
        # meanwhile shows a previously-selected different item" finding), so
        # the whole block is hidden entirely rather than tracking a second,
        # separate "focused row" concept select mode has no use for.
        has_preview = self.show_preview and (
            not select_mode
            and bool(self.canvas.selected_id and self.canvas.preview_lines)
        )
        self._has_preview = has_preview

        # task-14900: the list and its preview share a workbench container
        # (Collections' `#library-collections-workbench` grammar). Above the
        # screen's one measured width regime it lays them out side by side
        # (this Horizontal's default); below it, the host's existing
        # `library-notes-compact` class gives the list the full canvas and
        # suppresses the preview via CSS -- the conditional is keyed off a
        # class the screen already maintains at compose time AND on every
        # resize crossing, so no compose branch here can drift from an
        # in-place updater. Geometry (heights/overflow) moved from inline
        # styles into the same CSS tiers, because inline styles outrank the
        # class-flipped rules.
        workbench = Horizontal(id="library-media-workbench")
        workbench.set_class(has_preview, "has-preview")
        with workbench:
            media_list = Vertical(id="library-media-list")
            with media_list:
                with LibraryMediaRowScroll(id="library-media-row-scroll"):
                    row_height = (
                        _MEDIA_ROW_COMPACT_HEIGHT
                        if self.compact
                        else _MEDIA_ROW_WIDE_HEIGHT
                    )
                    for index, row in enumerate(self.canvas.rows):
                        marker = _media_row_marker(
                            select_mode=select_mode,
                            checked=row.checked,
                            reviewed=row.reviewed,
                            selected=row.selected,
                            compact=self.compact,
                        )
                        # task-281 (PR #665 review): the in-place toggle needs the
                        # marker-less RAW label to rebuild from -- reading it back
                        # off the mounted Button un-escapes user titles (both
                        # ``.plain`` and Textual 8's ``str(Content)`` return
                        # rendered text), so the raw remainder is stashed here at
                        # the single point of truth.
                        label_rest = _media_row_label_rest(
                            row.title,
                            row.secondary,
                            compact=self.compact,
                            loading=row.loading,
                            loaded=row.loaded,
                        )
                        # task-31631 AC#2 / task-31945: the whole row is
                        # the toggle target, and the shared helper drops
                        # Textual's 0.2s press flash so a second click on
                        # the same row (☐ then its title -- what critique
                        # #5 did) is not swallowed by ``Button._on_click``.
                        # Browse mode wants it too: it stops a fast
                        # double-click on a browse row being lost, and the
                        # feedback there is the item loading into the
                        # Reader, not the flash.
                        button = library_row_button(
                            f"{marker}{label_rest}",
                            id=f"library-media-row-{index}",
                            classes="library-media-row",
                            compact=True,
                        )
                        button.media_id = row.media_id
                        button._library_row_label_rest = label_rest
                        button._library_media_title = row.title
                        button._library_media_secondary = row.secondary
                        button._library_media_selected = row.selected
                        button._library_media_checked = row.checked
                        button._library_media_loading = row.loading
                        button._library_media_loaded = row.loaded
                        button._library_media_reviewed = row.reviewed
                        button.tooltip = escape_markup(row.title)
                        button.set_class(
                            row.selected and not self.compact and not select_mode,
                            "library-media-row-selected",
                        )
                        button.styles.height = row_height
                        button.styles.min_height = row_height
                        # task-31220: a row OPEN is a read, so it is gated
                        # only while a write is actually unsettled -- never by
                        # the stale gate the open is how you recover from.
                        # Only the mutating actions (Select/Export/sort/
                        # Delete/Undo) stay behind ``_gate_stale_action``.
                        yield self._gate_mutation_action(
                            button, label_rest.lstrip()
                        )
                if self.pager is not None:
                    yield from self._compose_pager(self.pager)

            preview = Vertical(id="library-media-preview")
            preview.display = has_preview and not self.compact
            with preview:
                yield Static(
                    "\n".join(self.canvas.preview_lines),
                    id="library-media-preview-lines",
                    markup=False,
                )
                toolbar = Horizontal(classes="ds-toolbar")
                toolbar.styles.height = "auto"
                with toolbar:
                    # Opens the selected item in the IN-LIBRARY media viewer
                    # (nav stays on Library), distinct from the full viewer's
                    # own action row (`#library-media-open`, `LibraryMediaViewer`
                    # -- "Open in Library ▸ Media", task-2857), which posts a
                    # fresh ``NavigateToScreen`` for the "media" route.
                    open_viewer = Button(
                        "Open in viewer",
                        id="library-media-open-viewer",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    open_viewer.can_focus = self.show_preview and not self.compact
                    yield self._gate_stale_action(open_viewer, "Open in viewer")

                # Task 8 (meeting diarization spec): a finished meeting
                # recording's speakers can still be renamed after the fact --
                # one legend row per speaker (mirroring the live Meetings
                # screen's Task 7 legend), surfaced ONLY while the caller's
                # `can_rename_meeting_speakers` says the meeting folder is
                # still there (absent, not disabled, otherwise: a non-meeting
                # item has nothing to rename).
                # ... and only while the pane is actually visible: building
                # the rows parses the whole transcript.jsonl, which is pure
                # waste on a hidden pane (final review, MINOR).
                if (
                    self.can_rename_speakers
                    and has_preview
                    and not self.compact
                    and self.media_db is not None
                    and self.speaker_rename_media_id is not None
                ):
                    try:
                        speaker_rows = _meeting_speaker_legend_rows(
                            self.media_db, self.speaker_rename_media_id
                        )
                    except Exception:  # noqa: BLE001 - a bad read just means no legend
                        speaker_rows = []
                    if speaker_rows:
                        legend = Vertical(id="library-media-speaker-legend")
                        with legend:
                            for cluster_id, label in speaker_rows:
                                yield Horizontal(
                                    Static(
                                        label,
                                        id=f"library-media-speaker-label-{cluster_id}",
                                        markup=False,
                                        classes="library-media-speaker-label",
                                    ),
                                    Input(
                                        placeholder="Rename…",
                                        id=f"library-media-speaker-input-{cluster_id}",
                                        classes="library-media-speaker-input",
                                    ),
                                    classes="library-media-speaker-row",
                                )

            # task-14900: the wide split's detail half never sits blank --
            # when the preview is hidden (Select mode, or an empty list) a
            # placeholder explains the pane, Collections' own detail-pane
            # grammar ("No Collection selected."). CSS-only visibility
            # (never a Python ``display`` write, which would outrank the
            # compact rule that hides it in the preserved stacked layout):
            # hidden while the workbench carries ``has-preview``, and hidden
            # entirely below the breakpoint.
            detail_empty = Static(
                (
                    "No preview in Select mode."
                    if select_mode
                    else "No media item selected."
                ),
                id="library-media-detail-empty",
                markup=False,
            )
            detail_empty.display = self.show_preview
            yield detail_empty

    def _compose_pager(self, pager: LibraryPagerDisplay) -> ComposeResult:
        """Render the controller-owned Media pager below the row viewport."""
        # task-28016: a single-page result has nowhere to page to, so the
        # "Page 1 of 1" counter and the boundary reasons ("Already on the
        # first page.", "No more results.") are pure noise. Show only the item
        # range and keep the (disabled) controls; both return the moment a
        # second page exists.
        # task-31632: while a load failure renders its own callout, the ONE
        # Retry lives there, next to the reason -- not down here. The stale
        # gate has no callout (its copy is a different event) and keeps its
        # Retry in this strip.
        retry_visible = pager.retry_visible and self.load_failure is None
        # task-32104: the rule itself lives in ``library_pager_layout``,
        # shared with every other Library pager.
        layout = library_pager_layout(pager, retry_visible=retry_visible)
        with Vertical(id="library-media-pager", classes="library-source-pager"):
            yield Static(
                " · ".join(layout.status_parts),
                id="library-media-page-status",
                classes="library-source-pager-status",
                markup=False,
            )
            if layout.boundary_reasons:
                yield Static(
                    " · ".join(layout.boundary_reasons),
                    id="library-media-disabled-reason",
                    classes="library-source-pager-status",
                    markup=False,
                )
            # task-31237 (supersedes task-28016's keep-the-disabled-controls
            # choice, critique #3 ruling): a single-page result renders NO
            # pager controls -- two dead "○ Previous ○ Next" forms under
            # every short list were pure noise. The range Static above
            # stays; the controls return the moment a second page exists.
            # A stale page still needs its Retry here even on one page
            # (task-31632 moved a FAILED fetch's Retry into the callout,
            # which is why the layout above reads the gated
            # ``retry_visible``).
            if layout.controls_hidden:
                return
            with Horizontal(classes="library-source-pager-controls"):
                previous = Button(
                    library_disabled_action_label(
                        "Previous", pager.previous_disabled
                    ),
                    id="library-media-previous",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.previous_disabled,
                )
                if pager.previous_disabled:
                    previous.tooltip = pager.previous_reason
                yield self._gate_mutation_action(previous, "Previous")
                if retry_visible:
                    retry = Button(
                        "Retry",
                        id="library-media-retry",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(retry, "Retry")
                next_page = Button(
                    library_disabled_action_label("Next", pager.next_disabled),
                    id="library-media-next",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=pager.next_disabled,
                )
                if pager.next_disabled:
                    next_page.tooltip = pager.next_reason
                yield self._gate_mutation_action(next_page, "Next")
