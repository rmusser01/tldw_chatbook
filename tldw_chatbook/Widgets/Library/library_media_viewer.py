"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from loguru import logger
from rich.color import Color
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ItemGrid, Vertical, VerticalGroup
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from tldw_chatbook.Audio.meeting_session import (
    is_widget_safe_cluster_id,
    normalize_speaker_name,
)
from tldw_chatbook.Library.meeting_speaker_rename import (
    RENAME_REFUSED_EMPTY_TRANSCRIPT,
    RENAME_REFUSED_NOT_MEETING_CONTENT,
    SpeakerRenameResult,
    _meeting_speaker_legend_rows,
    rename_meeting_speaker,
)
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths
from tldw_chatbook.Library.library_shell_state import library_disabled_action_label
from tldw_chatbook.Library.library_media_viewer_state import (
    analysis_find_unavailable_reason,
    LibraryMediaHighlightRow,
    LibraryMediaViewerState,
    find_content_matches,
    looks_like_markdown_content,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
    LibraryMediaContentSearchControls,
)


#: task-31635 (critique #5 item 12): when the Media list load failed and
#: left NO rows behind (see ``_library_media_list_unselectable``), there is
#: nothing to select, so the invitation to select something was the one line
#: on screen contradicting the recovery callout beside it. A failure that
#: retained rows, or one that only hit the type facets, keeps the ordinary
#: copy -- those rows are still painted and still pressable.
READER_EMPTY_COPY = "Select a media item to read it here."
READER_EMPTY_FAILED_COPY = "Nothing loaded — the list could not be loaded."

#: task-31635 (critique #5 item 13): a non-Markdown text item simply dropped
#: the Rendered|Raw strip, so nothing said whether a rendered view existed.
#: task-31958: the note is gated on CONTENT, not on the media type -- an
#: allowlist of article/document left a plaintext, video or audio item that
#: failed the Markdown sniff with the same silent blank slot, while an empty
#: article got the note above "No stored content.", explaining nothing.
#: The copy is type-neutral for the same reason (batch-3 review ruling 1):
#: naming the types a rendered view IS for read as a contradiction above a
#: plain-prose transcript -- "for transcripts", withheld from a transcript.
#: It now says why THIS item has none.
RENDERED_VIEW_NOTE = "No Markdown formatting to render — showing the stored text"

#: task-32365 review finding 1: only the Raw view marks a match (rendered
#: mode mounts the Markdown widget alone, so there is no raw widget to
#: restyle and the scroll-to-match is a SOURCE line index). An active query
#: therefore forces Raw on the Analysis tab; this says so and names the way
#: back, rather than leaving a Rendered button that repaints the same body.
ANALYSIS_RENDERED_BLOCKED_BY_SEARCH = (
    "Showing the stored text so matches can be marked · clear the search to "
    "read it rendered."
)


def empty_reader_copy(*, loading: bool, list_failed: bool) -> str:
    """Return the empty Reader's placeholder copy.

    Args:
        loading: Whether a detail request is pending.
        list_failed: Whether the Media browse controller carries a failure.

    Returns:
        The one line the empty Reader paints.
    """
    if loading:
        return "Loading media…"
    return READER_EMPTY_FAILED_COPY if list_failed else READER_EMPTY_COPY


class LibraryMediaViewer(PostRecomposeCallback, Vertical):
    """Render the full Library media item: metadata, content, and actions.

    DEFAULT_CSS pins the Rendered|Raw toggle's "|" separator to a width of
    1 -- a bare ``Static`` has no width rule of its own (only
    ``height: auto``), so it inherits Textual's base ``1fr`` default and
    silently expands to consume the Horizontal's remaining space, pushing
    the "Raw" button off past the right edge of the screen (found live in
    a 170-column terminal: the button existed in the DOM with a correct
    label -- passing an existence/label-only query -- while its region was
    ``x=184`` against a 170-column screen, entirely off-screen). Mirrors
    ``LibraryScreen``'s own ``#library-notes-source-separator`` rule for
    the "Database | Files" strip this toggle's shape was modeled on.

    Attributes:
        viewer: Current media viewer display state.
        editing: Whether the metadata edit form should render in place of
            the read-only metadata block and action row.
        confirming_delete: Whether the inline delete-confirmation affordance
            should render in place of the normal action row.
        highlights: Reading highlights for this media item, in display order.
        editing_analysis: Whether the analysis edit form (a prefilled
            ``TextArea`` + Save/Cancel) should render in place of the
            read-only analysis text and its "Edit analysis" action.
        content_query: Current in-content search query, or "" when no
            search is active.
        content_match_index: Index into ``find_content_matches``' result
            for the currently focused match (wrapped mod the match count
            by the screen before it is passed in here).
        content_mode: ``"rendered"`` shows ``viewer.content`` through the
            same ``Markdown`` render path Notes Preview uses (LIB-13);
            ``"raw"`` shows the plain/highlighted text ``Static`` (the
            pre-existing behavior). Only meaningful -- and only offered as
            a toggle -- when ``viewer.is_markdown`` is true; the screen is
            responsible for defaulting this per item and never showing
            ``"rendered"`` for a non-markdown item.
        analysis_content_mode: The same choice for the Analysis tab's own
            body (task-32365). Separate from ``content_mode`` because the
            two tabs hold different text: an item's transcript can be plain
            while its generated analysis is Markdown, and vice versa. Owned
            by this widget -- the toggle handler below flips it and
            recomposes -- so it survives the screen's viewer syncs.
    """

    DEFAULT_CSS = """
    LibraryMediaViewer #library-media-content-mode-separator {
        width: 1;
        min-width: 1;
        max-width: 1;
    }
    """

    def __init__(
        self,
        viewer: LibraryMediaViewerState,
        *,
        editing: bool = False,
        confirming_delete: bool = False,
        highlights: Sequence[LibraryMediaHighlightRow] = (),
        editing_analysis: bool = False,
        generating_analysis: bool = False,
        analysis_provider_reason: str = "",
        content_query: str = "",
        content_match_index: int = 0,
        content_mode: str = "raw",
        analysis_content_mode: str = "rendered",
        find_open: bool = False,
        find_focus_pending: bool = False,
        loading: bool = False,
        loading_message: str = "Loading media…",
        error_message: str = "",
        reader_mode: str = "read",
        more_open: bool = False,
        external_detail: bool = False,
        console_representation: str = "Complete stored text excerpt",
        image_preview: Widget | None = None,
        image_preview_status: str = "",
        image_preview_hidden: bool = False,
        image_preview_available: bool = False,
        image_preview_source: Any = None,
        review_banner: str = "",
        back_visible: bool = True,
        list_failed: bool = False,
        trash_list_open: bool = False,
        media_db: Any = None,
        speaker_rename_media_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Hold the viewer's compose inputs.

        Args:
            viewer: Pure display state for the loaded item.
            analysis_provider_reason: Why the Analysis tab's Generate
                action cannot run (no configured provider, an unready
                one), or "" when it can. The screen resolves it through
                the same seam the handler and the ingest path use
                (task-28007 AC#5), so the label and the post-click
                refusal can never disagree.
            find_focus_pending: One-shot token from the Find gesture; the
                bar it mounts takes focus, then the token is spent here so
                later syncs never re-take focus (task-31269).
            find_open: Whether the content Find bar renders (task-31237:
                collapsed until the Find action opens it -- a permanently
                open input duplicated Find and spent 3 rows per item).
            review_banner: One-line active review-set banner ("Reviewing:
                <name> — X of M · N reviewed · ✓ reviewed"), or "" when no
                set is active (task-30045). Rendered as literal text (set
                names derive from user input).
            back_visible: Whether the "‹ Back" control renders. False in the
                three-pane shell, where the Items pane already shows the
                list so Back changed no pixels while revoking every Reader
                binding gated on the view flag (task-31272); the screen
                decides from the shell's effective layout.
            list_failed: Whether the Media browse controller carries a
                failure state (task-31635). Only the EMPTY Reader reads
                it, to say that nothing can be selected rather than
                inviting a selection that cannot be made.
            trash_list_open: Whether the Items pane beside this Reader is
                showing the Trash list (task-31635, critique #5 item 11).
                The Reader keeps the LIVE item it was on, so it says which
                list that item belongs to -- the cheaper honest option than
                clearing a reading position the user is coming back to.
            media_db: The real ``MediaDatabase`` and, with
                ``speaker_rename_media_id``, the selected item's backing id
                (TASK-31745). Breaks this canvas's otherwise pure-state
                design on purpose -- exactly as ``LibraryMediaCanvas``
                does -- so the speaker legend can actually persist a rename
                instead of showing an inert control.
            speaker_rename_media_id: See ``media_db``.
        """
        super().__init__(**kwargs)
        self.viewer = viewer
        self.editing = editing
        self.confirming_delete = confirming_delete
        self.highlights = tuple(highlights)
        self.editing_analysis = editing_analysis
        self.generating_analysis = generating_analysis
        self.analysis_provider_reason = analysis_provider_reason
        self.content_query = content_query
        self.content_match_index = content_match_index
        self.content_mode = content_mode
        self.analysis_content_mode = analysis_content_mode
        self.find_open = find_open
        self.find_focus_pending = find_focus_pending
        self.loading = loading
        self.loading_message = loading_message
        self.error_message = error_message
        self.reader_mode = reader_mode
        self.more_open = more_open
        self.external_detail = external_detail
        self.console_representation = console_representation
        self.image_preview = image_preview
        self.image_preview_status = image_preview_status
        self.image_preview_hidden = image_preview_hidden
        self.image_preview_available = image_preview_available
        self.image_preview_source = image_preview_source
        self.review_banner = review_banner
        self.back_visible = back_visible
        self.list_failed = list_failed
        self.trash_list_open = trash_list_open
        self.media_db = media_db
        self.speaker_rename_media_id = speaker_rename_media_id
        # Fill the (already 13fr) canvas host, not an independent 13fr: an `fr`
        # width here breaks width:100% child resolution so long lines (analysis
        # summary, a long URL) clip instead of wrapping. 1fr fills the same
        # space and lets the text bodies wrap.
        self.styles.width = "1fr"
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        """Render the back control, title, metadata, content, and actions.

        Uses only render-verified widgets (``Static``/``VerticalScroll``/
        ``Button``) stacked full-width in this ``Vertical`` — horizontal rows
        that mix a ``1fr`` sibling with a fixed-width widget are the known
        non-rendering failure mode, so every row here is either a single
        full-width widget, the plain ``ds-toolbar`` action row (already
        proven to render by the conversations/media list canvases), or the
        Rendered|Raw toggle strip (``_compose_content_mode_toggle``) — a
        THIRD instance of that exact failure mode was found live while
        building it (a bare ``Static`` separator with no width rule of its
        own silently inherits Textual's base ``1fr`` default), fixed via
        this class's ``DEFAULT_CSS`` pinning that one widget's width to 1.

        Returns:
            ComposeResult for the media viewer canvas.
        """
        if self.error_message:
            yield Static(
                self.error_message,
                id="library-media-viewer-error",
                classes="destination-purpose",
                markup=False,
            )
            yield Button(
                "Retry",
                id="library-media-reader-retry",
                classes="library-canvas-action",
                compact=True,
            )
        if not self.viewer.media_id:
            yield Static(
                empty_reader_copy(
                    loading=self.loading, list_failed=self.list_failed
                ),
                id="library-media-reader-empty",
                classes="destination-purpose",
                markup=False,
            )
            return
        # task-22207: the pending banner is a PERSISTENT, display-gated
        # widget rather than a conditional child. Traversal keystrokes flip
        # only the loading state, and rebuilding the whole viewer (with a
        # fresh full-document body) just to add/remove this one Static was
        # the dominant per-keystroke cost. ``sync_loading_state`` patches
        # its copy and visibility in place.
        banner = Static(
            self.loading_message,
            id="library-media-viewer-loading",
            classes="destination-purpose",
            markup=False,
        )
        banner.display = self.loading
        yield banner
        # task-31277 (critique #4 P2): only a SERVER item needs an identity
        # line. "Local Media item" restated what the Media list beside it
        # already said, at the cost of the top row of the reading surface on
        # every local open.
        # task-31635 (critique #5 item 11): ...and so does a local item
        # sitting beside the TRASH list, where the list no longer says it.
        # Same slot, same grammar, never both -- a server item is not in
        # the local Media list at all, which is the stronger statement.
        identity_line = (
            "Server item · not in local Media list"
            if self.external_detail
            else "Showing a Media item · not in Trash"
            if self.trash_list_open
            else ""
        )
        if identity_line:
            yield Static(
                identity_line,
                id="library-media-reader-identity",
                markup=False,
            )
        if self.review_banner:
            # task-30045 (critique P2): the active review set is a workflow
            # object -- its name, live progress, and the loaded item's own
            # reviewed state frame the Reader, not just a footer string.
            # markup=False: the set name derives from user input.
            yield Static(
                self.review_banner,
                id="library-media-review-banner",
                markup=False,
            )
        if self.back_visible:
            yield Button("‹ Back", id="library-media-back", compact=True)
        yield Static(
            "Edit media details" if self.editing else self.viewer.title,
            id="library-media-viewer-title",
            markup=False,
        )
        if not self.editing:
            byline = next(
                (line.removeprefix("Author: ") for line in self.viewer.metadata_lines
                 if line.startswith("Author: ")),
                "",
            ) or next(
                (line.removeprefix("URL: ") for line in self.viewer.metadata_lines
                 if line.startswith("URL: ")),
                "",
            )
            # task-31277: an item with neither an author nor a URL spent a
            # row of the Reader header painting nothing at all.
            if byline:
                yield Static(
                    byline,
                    id="library-media-reader-byline",
                    markup=False,
                )
        yield from self._compose_primary_toolbar()
        yield from self._compose_mode_toolbar()

        if self.confirming_delete and not self.editing:
            # A single full-width Static above the toolbar, not inside it --
            # mixing a Static with the toolbar's Buttons is the known
            # non-rendering failure mode called out on ``compose`` above.
            # task-14901 (ADR-055): single delete leaves the same Undo
            # receipt as "Delete selected". task-4025 AC3: the Trash view
            # now exists (the media list toolbar's "Trash" action), so the
            # copy names the durable recovery path exactly like the bulk
            # confirm does -- one promise, two entry points.
            yield Static(
                "Delete this media? You can undo right away, or restore "
                "later from Trash.",
                id="library-media-delete-confirm-copy",
                markup=False,
            )
            with Horizontal(classes="ds-toolbar"):
                yield Button(
                    "Delete", id="library-media-delete-confirm", compact=True
                )
                yield Button(
                    "Cancel", id="library-media-delete-cancel", compact=True
                )

        yield from self._compose_active_body()

    def _compose_primary_toolbar(self) -> ComposeResult:
        """Render the always-reachable Reader actions."""
        with Horizontal(classes="ds-toolbar", id="library-media-reader-primary-toolbar"):
            # Qodo on #2378: an Analysis tab with nothing to search has no
            # bar to mount -- say why Find is off instead of toggling silently.
            find_reason = analysis_find_unavailable_reason(
                mode=self.reader_mode,
                analysis=self.viewer.analysis,
                generating=self.generating_analysis,
                editing=self.editing_analysis,
                # Qodo 4 on #2602: an external detail composes the READ body
                # whatever mode the session carried in, so the gate needs to
                # know -- without this the KEY opened Find on a server
                # document while this button stayed disabled, the exact
                # key/control divergence one shared gate exists to prevent.
                external=self.external_detail,
            )
            find = Button(
                library_disabled_action_label("Find", bool(find_reason)),
                id="library-media-reader-find",
                compact=True,
            )
            if find_reason:
                find.disabled = True
                find.tooltip = find_reason
            yield find
            if not self.external_detail:
                yield Button(
                    "Remove later" if self.viewer.read_later else "Read later",
                    id="library-media-read-later",
                    compact=True,
                )
            yield Button("Use in Console", id="library-media-use-in-chat", compact=True)
            if not self.external_detail or self.viewer.original_source:
                # task-31633 AC#3: the glyph is the disclosure state. The
                # actions render as ONE toolbar row under this one, so
                # nothing else on screen says whether More is open.
                yield Button(
                    "More \u25b4" if self.more_open else "More",
                    id="library-media-reader-more",
                    compact=True,
                )
        # The same condition the More button above is composed under: a stale
        # ``more_open`` carried onto a server-only detail would otherwise paint
        # an empty actions row under a button that is no longer there.
        if self.more_open and (not self.external_detail or self.viewer.original_source):
            # task-31633 AC#3 (critique #5, capture 10): this was a bare
            # Vertical, and an unstyled Vertical defaults to 1fr -- it took
            # 19 rows for three one-row buttons and pushed the tab row and
            # the whole reading body off the fold. A second ds-toolbar row
            # costs exactly one row at the wide size. ItemGrid rather than
            # Horizontal because the four labels need ~60 cells and the
            # Reader is only ~46 wide at 100x30, where a Horizontal clips
            # the fourth action off the pane outright; the grid reflows it
            # onto a second row instead.
            with ItemGrid(
                id="library-media-reader-more-actions",
                classes="ds-toolbar",
                # task-32237: the column has to hold the longest label (13),
                # the Button's own two auto-width cells, and the danger
                # action's 2-cell separation (`.library-media-action-danger`,
                # task-31980) -- 16 cells cut "Move to trash" to "Move to"
                # because that margin is taken out of the button's box.
                min_column_width=17,
                max_column_width=17,
            ):
                if not self.external_detail:
                    yield Button("Edit metadata", id="library-media-edit", compact=True)
                if self.viewer.original_source:
                    yield Button("Open original", id="library-media-open-original", compact=True)
                if not self.external_detail:
                    yield Button("Open manager", id="library-media-open", compact=True)
                    # task-31980 (critique #6 P2): the one destructive action
                    # in this strip takes the Library's quiet-danger class --
                    # muted ink + a left margin (the more-actions rule zeroes
                    # button margins, so the class's own is restored by an
                    # id-scoped rule) -- so it reads apart from the neutral
                    # actions instead of ending the row unmarked and flush.
                    yield Button(
                        "Move to trash",
                        id="library-media-delete",
                        classes="library-media-action-danger",
                        compact=True,
                    )

    def _compose_mode_toolbar(self) -> ComposeResult:
        """Render one explicit mode selector; external detail remains read-only."""
        if self.external_detail:
            return
        with Horizontal(classes="ds-toolbar", id="library-media-reader-mode-toolbar"):
            for mode, label in (
                ("read", "Read"),
                ("analysis", "Analysis"),
                ("highlights", "Highlights"),
                ("info", "Info"),
            ):
                yield Button(
                    f"{label} (selected)" if self.reader_mode == mode else label,
                    id=f"library-media-reader-select-{mode}",
                    classes="library-media-reader-mode",
                    compact=True,
                )

    def _compose_active_body(self) -> ComposeResult:
        """Compose exactly the selected Reader body; never mount hidden modes."""
        if self.external_detail or self.reader_mode == "read":
            # task-31277 (critique #4 P2, AC#3): no section header here --
            # the mode row directly above already reads "Read (selected)",
            # so the header spent a row of the reading surface saying that
            # word twice. Analysis and Highlights lost theirs the same way.
            with Vertical(id="library-media-reader-mode-read"):
                if self.image_preview is not None and not self.image_preview_hidden:
                    with Vertical(id="library-media-image-preview"):
                        yield self.image_preview
                if self.image_preview_status:
                    yield Static(
                        self.image_preview_status,
                        id="library-media-image-preview-status",
                        markup=False,
                    )
                if self.image_preview_available:
                    yield Button(
                        "Show preview" if self.image_preview_hidden else "Hide preview",
                        id="library-media-image-preview-toggle",
                        compact=True,
                    )
                elif self.image_preview_status:
                    yield Button(
                        "Retry preview",
                        id="library-media-image-preview-retry",
                        compact=True,
                    )
                yield from self._compose_content_mode_toggle()
            # Keep the search controls and content body as direct children of
            # the Reader, siblings of the mode marker -- the mode row is the
            # Find bar's anchor (task-31276 retired the dock that moved an
            # active bar to the viewport top).
            # task-31237: the Find bar is collapsed until the Find action
            # opens it (or a query is applied); a permanently open
            # "Search content…" input duplicated Find and spent 3 rows on
            # every fresh item.
            if self.find_open or self.content_query:
                matches = find_content_matches(
                    self.viewer.content, self.content_query
                )
                yield LibraryMediaContentSearchControls(
                    is_markdown=self.viewer.is_markdown,
                    query=self.content_query,
                    matches=matches,
                    match_index=self.content_match_index,
                    focus_on_mount=self.find_focus_pending,
                    id="library-media-content-search-controls",
                )
                # task-31269: the gesture token is spent on this mount.
                self.find_focus_pending = False
            yield LibraryMediaContentBody(
                content=self.viewer.content,
                is_markdown=self.viewer.is_markdown,
                mode=self.content_mode,
                query=self.content_query,
                match_index=self.content_match_index,
                id="library-media-viewer-content",
            )
            yield from self._compose_speaker_legend()
            return
        if self.reader_mode == "analysis":
            with Vertical(id="library-media-reader-mode-analysis"):
                yield from self._compose_analysis()
            return
        if self.reader_mode == "highlights":
            with Vertical(id="library-media-reader-mode-highlights"):
                yield from self._compose_highlights()
            return
        with Vertical(id="library-media-reader-mode-info"):
            if self.editing:
                yield from self._compose_edit_form()
            else:
                yield Static("\n".join(self.viewer.metadata_lines), id="library-media-viewer-meta", markup=False)
                # task-32068: why this item has no Rendered view -- asked
                # and answered here, once, rather than banner-ed over every
                # read of every plain item.
                if not self.viewer.is_markdown and self.viewer.has_content:
                    yield Static(
                        RENDERED_VIEW_NOTE,
                        id="library-media-content-mode-note",
                        classes="destination-purpose",
                        markup=False,
                    )
                yield Static(
                    "\n".join((
                        f"Backend: {self.viewer.backend}",
                        f"Canonical ID: {self.viewer.canonical_id}",
                        f"Original source: {self.viewer.original_source or 'None recorded'}",
                        f"Stored representation: {self.viewer.stored_representation}",
                        f"Use in Console sends: {self.console_representation}",
                    )),
                    id="library-media-reader-provenance",
                    markup=False,
                )

    def _compose_content_mode_toggle(
        self,
        *,
        is_markdown: bool | None = None,
        mode: str | None = None,
        prefix: str = "library-media",
        blocked_reason: str = "",
    ) -> ComposeResult:
        """Render the Rendered|Raw toggle, for an item that can render.

        task-32365: the Analysis tab renders the same pair over its own
        text, so the three things that differ per tab -- whether there is
        anything to render, which view is selected, and the button ids the
        press handler keys on -- are arguments, defaulting to the Read
        tab's. The strip and separator ids stay fixed: only one Reader body
        composes at a time, so they are still unique, and they carry the
        width rules (this class's ``DEFAULT_CSS`` and
        ``#library-media-content-mode-strip``) that keep the row from
        hitting Textual's bare-``1fr`` non-rendering trap.

        The toggle is offered only when ``self.viewer.is_markdown`` is true
        -- a non-markdown item always shows the plain Raw view (no behavior
        change from before LIB-13) and gets nothing here. The one-line note
        that explains WHY it has no toggle (task-31635, widened to every
        media type by task-31958) lives in the Info tab since task-32068:
        it is a fact about the item, and as a banner over the reading
        surface it greeted nearly every open. An item with no stored
        content explains nothing anywhere -- the body already says "No
        stored content.", and there is no rendered view to explain the
        absence of. Mirrors the
        screen's own "Database (selected) | Files" source-strip idiom
        exactly (``library_screen.py``'s notes-source strip): a plain
        ``Horizontal`` of two compact, unstyled ``Button``s with a "|"
        ``Static`` separator, each label suffixed "(selected)" for the
        active mode -- the state-in-text idiom, not a color/class alone,
        so the current mode reads correctly even without extra CSS.

        Returns:
            ComposeResult for the toggle strip, or nothing at all for a
            non-markdown item (see the Info branch of
            ``_compose_active_body``).
        """
        if is_markdown is None:
            is_markdown = self.viewer.is_markdown
        if mode is None:
            mode = self.content_mode
        if not is_markdown:
            # task-32068: the note is a FACT ABOUT THE ITEM, so it belongs to
            # Info (``_compose_active_body``'s info branch), not above the
            # text on every read. Most items are plain, so critique #8 met it
            # on nearly every open -- a line about a view the reader never
            # asked for, above the one they did. Text only either way: there
            # is no rendered view to offer, so a control here would be an
            # affordance for nothing (task-31635, critique #5 item 13).
            return
        with Horizontal(id="library-media-content-mode-strip"):
            rendered_selected = mode == "rendered"
            rendered_button = Button(
                library_disabled_action_label(
                    "Rendered (selected)" if rendered_selected else "Rendered",
                    bool(blocked_reason),
                ),
                id=f"{prefix}-content-mode-rendered",
                compact=True,
            )
            rendered_button.set_class(rendered_selected, "-selected")
            if blocked_reason:
                # Never a pressable control that changes nothing: while a
                # query forces the Raw view (see ``_compose_analysis``) the
                # strip reports the state and says why, and NEITHER half
                # accepts a press -- Qodo on #2602 found that pressing the
                # already-selected Raw wrote the stored preference, so
                # clearing the query no longer restored Rendered and the
                # line beneath became a false promise.
                rendered_button.disabled = True
                rendered_button.tooltip = blocked_reason
            yield rendered_button
            yield Static("|", id="library-media-content-mode-separator", markup=False)
            raw_selected = not rendered_selected
            raw_button = Button(
                "Raw (selected)" if raw_selected else "Raw",
                id=f"{prefix}-content-mode-raw",
                compact=True,
            )
            raw_button.set_class(raw_selected, "-selected")
            if blocked_reason:
                raw_button.disabled = True
                raw_button.tooltip = blocked_reason
            yield raw_button

    @on(Button.Pressed, "#library-media-analysis-content-mode-rendered")
    @on(Button.Pressed, "#library-media-analysis-content-mode-raw")
    def _handle_analysis_content_mode(self, event: Button.Pressed) -> None:
        """Flip the Analysis tab between its rendered and raw views.

        Handled here rather than on the screen (where the Read tab's twin
        lives) because the choice is this widget's own state: the screen
        has no analysis view-mode to keep in step, and a recompose is the
        whole update -- the Read tab patches in place only to avoid
        re-parsing a full document on every traversal keystroke, which a
        deliberate toggle press is not.

        Args:
            event: The Rendered or Raw press from the Analysis toggle strip.
        """
        event.stop()
        self.analysis_content_mode = (
            "rendered" if str(event.button.id).endswith("-rendered") else "raw"
        )
        self.refresh(recompose=True)

    # ---- TASK-31745: rename a finished meeting's speakers, from the reader --
    #: What each refusal means in the user's terms, keyed by the reason
    #: ``rename_meeting_speaker`` returns. Static copy -- never a path, a
    #: name, or transcript text.
    _RENAME_REFUSAL_COPY = {
        RENAME_REFUSED_NOT_MEETING_CONTENT: (
            "This transcript came from ingest; rename the live transcript in Meetings."
        ),
        RENAME_REFUSED_EMPTY_TRANSCRIPT: (
            "This meeting's local transcript is missing or empty; nothing to rename."
        ),
    }
    _SPEAKER_INPUT_PREFIX = "library-media-speaker-input-"

    class SpeakerRenamed(Message):
        """A meeting speaker was renamed on ``media_id``; its detail is stale.

        The reader repaints itself immediately (below), but the SCREEN's
        viewer state is memoized per detail ARRIVAL and still built from the
        pre-rename content -- the next viewer sync would repaint that over
        the new name. The screen re-reads the item on this message.
        """

        def __init__(self, media_id: int) -> None:
            super().__init__()
            self.media_id = media_id

    def _compose_speaker_legend(self) -> ComposeResult:
        """Render one rename row per speaker of a finished meeting recording.

        Absent, not disabled, for anything else: a non-meeting item has no
        speakers to rename (the screen resolves that into
        ``viewer.can_rename_speakers``).

        ``VerticalGroup``, never a bare ``Vertical``: Textual's ``Vertical``
        defaults to ``height: 1fr``, so as a direct sibling of the ``1fr``
        content body this legend would claim HALF the reading pane (the
        task-31222/31276 trap). ``VerticalGroup`` is ``height: auto`` in
        upstream's own CSS, so the section costs exactly its rows and needs
        no rule here.

        Label above input, each full-width -- the shape ``_compose_edit_form``
        uses; the labels reuse its ``.library-media-edit-label`` styling. A
        ``Horizontal`` row mixing an auto-width ``Static`` with a ``1fr``
        ``Input`` is this canvas's known non-rendering failure mode, and
        re-keying that would spend two more ancestor-scoped bare-type rules
        against ADR-097's ratchet.

        Returns:
            ComposeResult for the legend, or nothing when there is none.
        """
        if not self.viewer.can_rename_speakers or not self.viewer.speaker_legend_rows:
            return
        with VerticalGroup(id="library-media-speaker-legend"):
            yield Static(
                "Rename speakers",
                id="library-media-speaker-legend-title",
                classes="library-media-edit-label",
                markup=False,
            )
            for cluster_id, label in self.viewer.speaker_legend_rows:
                # A hand-edited transcript.jsonl can carry an id that is not
                # a legal Textual widget id ("S 1"); interpolating it would
                # raise out of compose() and take the screen down.
                if not is_widget_safe_cluster_id(cluster_id):
                    continue
                yield Static(
                    label,
                    id=f"library-media-speaker-label-{cluster_id}",
                    markup=False,
                    classes="library-media-edit-label library-media-speaker-label",
                )
                yield Input(
                    placeholder="Rename…",
                    id=f"{self._SPEAKER_INPUT_PREFIX}{cluster_id}",
                    classes="library-media-speaker-input",
                )

    @on(Input.Submitted, ".library-media-speaker-input")
    def _handle_speaker_rename_submitted(self, event: Input.Submitted) -> None:
        """Persist the submitted row's rename off the UI thread.

        Mirrors ``LibraryMediaCanvas``' legend: the rename itself is
        unconditional (a submit racing teardown should still persist) and
        only the repaint afterwards is ``is_mounted``-guarded.
        """
        event.stop()
        widget_id = event.input.id or ""
        if not widget_id.startswith(self._SPEAKER_INPUT_PREFIX):
            return
        cluster_id = widget_id[len(self._SPEAKER_INPUT_PREFIX):]
        name = normalize_speaker_name(event.value)
        event.input.value = ""
        media_id = self.speaker_rename_media_id
        if self.media_db is None or media_id is None:
            return
        # The rename reads the transcript file, runs several DB writes, FTS
        # maintenance and a post-ingest dispatch -- all of which would freeze
        # the reader on a large transcript or a busy database.
        # ``exclusive`` keeps two fast submits from piling up in this group.
        # (Textual cannot interrupt a THREAD worker mid-flight, so a genuine
        # overlap still ends at the row's optimistic lock -- which fails safe,
        # writing nothing and reporting the conflict.)
        # The id is captured here, not read in the worker: a selection change
        # mid-rename must not retarget the write. The legend the user typed
        # into belongs to this id, so the rename lands on it either way.
        self.run_worker(
            lambda: self._rename_speaker_off_thread(media_id, cluster_id, name),
            group="library-media-speaker-rename",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _rename_speaker_off_thread(
        self, media_id: int, cluster_id: str, name: str
    ) -> None:
        """Rename on a worker thread, then repaint on the UI one.

        The post-rename re-reads (content + legend labels) happen HERE, on
        the worker, so the UI-thread callback only assigns and recomposes.
        """
        content = ""
        rows: tuple[tuple[str, str], ...] = ()
        try:
            outcome = rename_meeting_speaker(
                self.media_db, media_id, cluster_id, name
            )
            if outcome.ok:
                row = self.media_db.get_media_by_id(media_id)
                content = (row["content"] if row else "") or ""
                rows = tuple(_meeting_speaker_legend_rows(self.media_db, media_id))
        except Exception as exc:  # noqa: BLE001 - a rename must not crash the reader
            # A filesystem failure's ``str()`` embeds the meeting folder path.
            logger.warning(
                "Library media reader speaker rename failed: {}",
                redact_user_paths(str(exc)),
            )
            outcome = SpeakerRenameResult(False, f"unexpected error ({type(exc).__name__})")
        self.app.call_from_thread(
            self._apply_speaker_rename_outcome, media_id, outcome, content, rows
        )

    def _apply_speaker_rename_outcome(
        self,
        media_id: int,
        outcome: SpeakerRenameResult,
        content: str,
        rows: tuple[tuple[str, str], ...],
    ) -> None:
        """Explain a refused/failed rename, or repaint after a successful one."""
        if not outcome.ok:
            # ``reason`` is documented static, user-safe copy, so an
            # unmapped one (a failure, not a refusal) is shown as it stands.
            detail = self._RENAME_REFUSAL_COPY.get(outcome.reason, outcome.reason)
            self.app.notify(
                f"Couldn't rename this speaker. {detail}", severity="warning"
            )
            return
        if not self.is_mounted:
            return
        # Recompose, not an in-place patch: the content body holds an
        # immutable document (a content change builds a new body by design),
        # and the same pass repaints the legend's labels from ``rows``.
        # A selection that moved on during the write self-corrects: the screen
        # rebuilds this viewer from the NEW item's detail, and the message
        # below names the id that was actually renamed.
        self.viewer = dataclasses.replace(
            self.viewer, content=content, speaker_legend_rows=rows
        )
        self.refresh(recompose=True)
        self.post_message(self.SpeakerRenamed(media_id))

    def sync_loading_state(self, *, loading: bool, message: str) -> None:
        """Patch the mounted loading placeholder without rebuilding the body.

        task-22207: a traversal keystroke flips only the pending-request
        state; recomposing the viewer for that re-parses the full document
        being LEFT purely to paint "Loading…". This patches the persistent
        banner (or the empty-reader placeholder) in place instead.
        Display-gating a widget composed once -- rather than mounting and
        unmounting it here -- is deliberate: an async mount seam on this
        surface is the TASK-21116 M3 ``DuplicateIds`` race class.

        Args:
            loading: Whether a detail request is pending without error.
            message: Banner copy for the pending request.

        Returns:
            None.
        """
        self.loading = loading
        self.loading_message = message
        if not self.viewer.media_id:
            try:
                empty = self.query_one("#library-media-reader-empty", Static)
            except (NoMatches, QueryError):
                # Not composed yet -- compose() reads the attributes above.
                return
            copy = empty_reader_copy(
                loading=loading, list_failed=self.list_failed
            )
            if str(empty.content) != copy:
                empty.update(copy)
            return
        try:
            banner = self.query_one("#library-media-viewer-loading", Static)
        except (NoMatches, QueryError):
            # Not composed yet -- compose() reads the attributes above.
            return
        if loading and str(banner.content) != message:
            banner.update(message)
        if banner.display != loading:
            banner.display = loading

    def sync_list_failed(self, list_failed: bool) -> None:
        """Repaint the EMPTY Reader's placeholder when the list's health changes.

        task-31635: the Media browse controller owns this fact, and only the
        empty Reader reads it. Patched through the same in-place seam the
        loading placeholder uses -- a recompose here would re-parse the
        document of a LOADED Reader because the list beside it failed.

        Args:
            list_failed: Whether the browse controller carries a failure.

        Returns:
            None.
        """
        if self.list_failed == list_failed:
            return
        self.list_failed = list_failed
        self.sync_loading_state(loading=self.loading, message=self.loading_message)

    def sync_query_state(
        self, *, query: str, matches: tuple[int, ...], match_index: int
    ) -> None:
        """Synchronize a submitted query without rebuilding the viewer.

        Args:
            query: Submitted content-search query.
            matches: Source-line indexes matching ``query``.
            match_index: Zero-based index of the active match.

        Returns:
            None.
        """
        forces_analysis_raw = (
            self.reader_mode == "analysis"
            and bool(self.content_query) != bool(query)
            and looks_like_markdown_content(self.viewer.analysis)
        )
        self.content_query = query
        self.content_match_index = match_index
        if forces_analysis_raw:
            # task-32365 review finding 1: this seam patches in place, so the
            # composed mode would never follow the query and the marked Raw
            # view would never appear. Rebuild instead -- the Find bar hands
            # its own caret back through the existing focus token, so the
            # swap costs the user nothing.
            self.find_focus_pending = True
            self.refresh(recompose=True)
            return
        self.query_one(
            "#library-media-content-search-controls",
            LibraryMediaContentSearchControls,
        ).sync_query_state(
            is_markdown=self.viewer.is_markdown,
            query=query,
            matches=matches,
            match_index=match_index,
        )
        self.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        ).sync_search(query, match_index)

    def sync_match_index(
        self, *, matches: tuple[int, ...], match_index: int
    ) -> None:
        """Synchronize match navigation without rebuilding viewer children.

        Args:
            matches: Source-line indexes matching the active query.
            match_index: Zero-based index of the active match.

        Returns:
            None.
        """
        self.content_match_index = match_index
        self.query_one(
            "#library-media-content-search-controls",
            LibraryMediaContentSearchControls,
        ).sync_match_index(matches=matches, match_index=match_index)
        self.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        ).sync_search(self.content_query, match_index)

    async def sync_mode(self, mode: str) -> None:
        """Synchronize toggle state and reuse the persistent content views.

        Args:
            mode: Requested content mode, either ``"raw"`` or ``"rendered"``.

        Returns:
            None.

        Raises:
            ValueError: If ``mode`` is not a supported content mode.
        """
        self.content_mode = mode
        rendered_selected = mode == "rendered"
        rendered_button = self.query_one(
            "#library-media-content-mode-rendered", Button
        )
        raw_button = self.query_one("#library-media-content-mode-raw", Button)
        rendered_button.label = (
            "Rendered (selected)" if rendered_selected else "Rendered"
        )
        raw_button.label = "Raw" if rendered_selected else "Raw (selected)"
        rendered_button.set_class(rendered_selected, "-selected")
        raw_button.set_class(not rendered_selected, "-selected")
        rendered_button.refresh(layout=True)
        raw_button.refresh(layout=True)
        await self.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        ).sync_mode(mode)

    def _compose_edit_form(self) -> ComposeResult:
        """Render the metadata edit inputs, prefilled from ``viewer.edit_fields``.

        Stacked full-width ``Input`` widgets in a plain ``Vertical`` --
        matching the render-verified pattern already used by the Library
        Collections create/rename form.

        Returns:
            ComposeResult for the metadata edit form.
        """
        with Vertical(id="library-media-edit-form"):
            for label, field, placeholder, field_id in (
                ("Title", "title", "Title", "library-media-edit-title"),
                ("Author", "author", "Author", "library-media-edit-author"),
                ("URL", "url", "URL", "library-media-edit-url"),
                (
                    "Keywords",
                    "keywords",
                    "Keywords (comma-separated)",
                    "library-media-edit-keywords",
                ),
            ):
                # Persistent field label so each input stays identifiable even
                # when its value is cleared (a bare prefilled input is only
                # readable by its current text).
                yield Static(
                    label,
                    classes="library-media-edit-label",
                    markup=False,
                )
                yield Input(
                    value=self.viewer.edit_fields.get(field, ""),
                    placeholder=placeholder,
                    id=field_id,
                )
            with Horizontal(classes="ds-toolbar"):
                yield Button("Save", id="library-media-edit-save", compact=True)
                yield Button("Cancel", id="library-media-edit-cancel", compact=True)

    def _compose_analysis(self) -> ComposeResult:
        """Render the Analysis section: read-only text + Edit/Generate, or a form.

        Always renders (mirroring the Content section's always-present
        placeholder) so the actions are reachable even when no analysis
        exists yet. "Edit"/"Add" hand-edits text via ``save_analysis_version``;
        "Generate" (task-28006) runs the configured analysis provider and
        persists the result the same way. While a generation is in flight
        the section shows a progress line instead of the actions.

        Returns:
            ComposeResult for the Analysis section.
        """
        if self.editing_analysis:
            yield from self._compose_analysis_edit_form()
            return
        if self.generating_analysis:
            yield Static(
                self.viewer.analysis or "No analysis yet.",
                id="library-media-viewer-analysis-text",
                markup=False,
            )
            yield Static(
                "Generating analysis…",
                id="library-media-analysis-generating",
                classes="destination-purpose",
                markup=False,
            )
            return
        if self.viewer.analysis:
            # task-28026: render the analysis in the SAME searchable/
            # highlightable widgets the Read tab uses, so the in-item find
            # bar works over the analysis text. The screen's search corpus
            # (_library_media_content_matches) is mode-aware, so the query,
            # match count, Prev/Next, and Enter-advance all follow the
            # active tab.
            analysis_is_markdown = looks_like_markdown_content(self.viewer.analysis)
            matches = find_content_matches(self.viewer.analysis, self.content_query)
            # task-32365 review finding 1: only the Raw view can MARK a match
            # -- rendered mode mounts the Markdown widget alone, so
            # ``LibraryMediaContentBody.sync_search`` has no raw widget to
            # restyle and the scroll-to-match applies a SOURCE line index to
            # a rendered scroller. Before this task the analysis was always
            # raw, so Find worked; making it renderable would have shipped
            # "Match 3 of 11" over a body with nothing highlighted. An active
            # query therefore shows the view that can mark, and the toggle
            # strip below reads "Raw (selected)" so the swap is stated, not
            # silent. Clearing the query hands Rendered straight back.
            analysis_mode = "raw" if self.content_query else self.analysis_content_mode
            # task-31269: like Read, the bar is collapsed until Find opens
            # it -- an always-mounted bar stole focus on every item load and
            # swallowed the walk keys (critique #4 P0).
            blocked = (
                ANALYSIS_RENDERED_BLOCKED_BY_SEARCH if self.content_query else ""
            )
            yield from self._compose_content_mode_toggle(
                is_markdown=analysis_is_markdown,
                mode=analysis_mode,
                prefix="library-media-analysis",
                blocked_reason=blocked,
            )
            if blocked and analysis_is_markdown:
                # The reason reaches a keyboard-first reader, not only a
                # mouse tooltip -- the same inline-reason grammar the
                # Generate gate below uses (task-31981).
                yield Static(
                    blocked,
                    id="library-media-analysis-content-mode-reason",
                    classes="library-media-action-reason",
                    markup=False,
                )
            if self.find_open or self.content_query:
                yield LibraryMediaContentSearchControls(
                    is_markdown=analysis_is_markdown,
                    query=self.content_query,
                    matches=matches,
                    match_index=self.content_match_index,
                    focus_on_mount=self.find_focus_pending,
                    id="library-media-content-search-controls",
                )
                self.find_focus_pending = False
            yield LibraryMediaContentBody(
                content=self.viewer.analysis,
                # task-32365 (critique #10): this body used to pin
                # ``is_markdown=False, mode="raw"`` because an analysis was
                # assumed to be plain text. Generated analyses are Markdown
                # in practice ("## Key contributions" painted as source, A
                # cap 43), and task-32234 already made the CONTENT sniff --
                # not a type guess -- the Read tab's authority. Same sniff,
                # same widget, same Raw toggle.
                is_markdown=analysis_is_markdown,
                mode=analysis_mode,
                query=self.content_query,
                match_index=self.content_match_index,
                id="library-media-viewer-content",
            )
        else:
            yield Static(
                "No analysis yet.",
                id="library-media-viewer-analysis-text",
                markup=False,
            )
        # task-32217 reverses task-31237's `height: 1fr` fill for THIS tab
        # (see the analysis-scoped rules in _agentic_terminal.tcss): the fill
        # pinned this row to the pane floor, ~33 empty rows under a 2-line
        # analysis; the box now hugs its text and these actions follow it.
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Edit analysis" if self.viewer.analysis else "Add analysis",
                id="library-media-analysis-edit",
                classes="library-canvas-action",
                compact=True,
            )
            # task-28006: LLM generation, in the reading flow (no detour to
            # the manager). "Regenerate" when an analysis already exists.
            # task-28007 AC#5: with no callable provider it says so at the
            # control, in PR A's "○"-with-reason grammar, instead of
            # accepting the click and answering with a toast.
            reason = self.analysis_provider_reason
            generate = Button(
                library_disabled_action_label(
                    "Regenerate" if self.viewer.analysis else "Generate",
                    bool(reason),
                ),
                id="library-media-analysis-generate",
                classes="library-canvas-action",
                compact=True,
            )
            if reason:
                generate.disabled = True
                generate.tooltip = reason
            yield generate
        if reason:
            # task-31981: the reason must reach a keyboard-first user, not
            # only a mouse tooltip. Same inline-reason grammar as the Export
            # gate's "No destination chosen" line under its blocked button
            # (library_export_canvas.py). The tooltip above stays as a bonus.
            yield Static(
                reason,
                id="library-media-analysis-generate-reason",
                classes="library-media-action-reason",
                markup=False,
            )

    def _compose_analysis_edit_form(self) -> ComposeResult:
        """Render the analysis edit ``TextArea`` prefilled with the current analysis.

        ``TextArea`` renders cleanly full-width in a plain ``Vertical``
        (verified when this canvas's rendering approach was chosen), so
        this follows the same stacked, render-safe shape as
        ``_compose_edit_form``.

        Returns:
            ComposeResult for the analysis edit form.
        """
        with Vertical(id="library-media-analysis-edit-form"):
            yield TextArea(
                self.viewer.analysis,
                id="library-media-analysis-edit-text",
            )
            toolbar = Horizontal(classes="ds-toolbar")
            toolbar.styles.height = "auto"
            with toolbar:
                yield Button(
                    "Save",
                    id="library-media-analysis-save",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-media-analysis-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )

    @staticmethod
    def _renderable_color(color: str) -> str | None:
        """Return ``color`` if it is renderable as a Rich style color, else None.

        Highlight colors are free-text (the add form's "Color (optional)"),
        so a value like "highlighter pink" cannot be shown as a swatch.

        Validated with Rich's ``Color.parse`` -- the SAME grammar that
        consumes the value in ``_highlight_quote_text`` (``Text.append(...,
        style=color)``). Textual's color grammar is a superset (it accepts
        ``transparent``/``hsl(...)``/``rgba(...)``/``ansi_*`` which Rich
        rejects), so validating with Textual's parser would let those pass
        here and then raise ``rich.errors.MissingStyle`` inside Textual's
        layout at render time -- a persistent, data-triggered crash.

        Args:
            color: The stored highlight color string.

        Returns:
            The color string when Rich can render it as a style color,
            otherwise None (the caller then shows it as plain text instead).
        """
        if not color:
            return None
        try:
            Color.parse(color)
        except Exception:
            return None
        return color

    def _highlight_quote_text(self, highlight: LibraryMediaHighlightRow) -> Text:
        """Build the quote line, led by a swatch tinted to the highlight color.

        Color is the language of a highlighting feature, so a parseable
        color shows as a tinted "●" marker before the quote rather than as
        the bare word "yellow". Built as a Rich ``Text`` (only the marker is
        styled; the quote is appended as a raw slice) so quote content can
        never inject styles.

        Args:
            highlight: The highlight row to render.

        Returns:
            A Rich ``Text`` of the (optionally swatched) quote.
        """
        text = Text()
        swatch = self._renderable_color(highlight.color)
        if swatch:
            text.append("● ", style=swatch)
        text.append(f"“{highlight.quote}”")
        return text

    def _highlight_meta_text(self, highlight: LibraryMediaHighlightRow) -> str:
        """Build the highlight's secondary line (note, and color only if not swatched).

        The color is shown here as text only when it is not renderable as a
        swatch (so no information is lost for exotic color strings); a
        renderable color is already conveyed by the quote's tinted marker.

        Args:
            highlight: The highlight row to render.

        Returns:
            The secondary line text, or "" when there is nothing to show.
        """
        parts: list[str] = []
        if highlight.color and not self._renderable_color(highlight.color):
            parts.append(f"Color: {highlight.color}")
        if highlight.note:
            parts.append(f"Note: {highlight.note}")
        return " · ".join(parts)

    def _compose_highlights(self) -> ComposeResult:
        """Render the highlights section: existing rows, then the collapsed add form.

        Each highlight is its own indented card ``Vertical`` holding the
        quote ``Static`` (led by a swatch tinted to the highlight color), an
        optional meta ``Static`` (note, and color-as-text only when it is not
        a renderable swatch), and a compact "✕ Delete" ``Button`` -- so a
        per-row delete is unambiguously tied to one highlight. All children
        are stacked full-width inside the card, matching the render-safety
        rule on ``compose`` above. The delete button carries the highlight's
        id as a plain attribute (mirroring ``LibraryMediaCanvas`` setting
        ``button.media_id``) so the screen's class-selector handler can read
        it back.

        The highlight list always renders in full above the add form. The
        add form itself (the three inputs + "Add highlight" button) is
        nested inside a collapsed-by-default ``Collapsible`` -- it was
        dominating the section with three large empty inputs even when a
        user just wants to read existing highlights, so it now stays out of
        the way until explicitly opened. All add-form widget ids are
        unchanged; only their container changed.

        Returns:
            ComposeResult for the highlights section.
        """
        if not self.highlights:
            yield Static(
                "No highlights yet.",
                id="library-media-viewer-highlights-empty",
                markup=False,
            )
        else:
            for index, highlight in enumerate(self.highlights):
                # Each highlight is its own indented card (quote, optional
                # meta, its delete) so a per-row delete is unambiguously tied
                # to one highlight -- a flat list of identical "Delete
                # highlight" buttons could not say which it removed.
                with Vertical(classes="library-media-highlight-row"):
                    yield Static(
                        self._highlight_quote_text(highlight),
                        id=f"library-media-highlight-{index}",
                        markup=False,
                    )
                    meta_text = self._highlight_meta_text(highlight)
                    if meta_text:
                        yield Static(
                            meta_text,
                            classes="library-media-highlight-meta",
                            markup=False,
                        )
                    delete_button = Button(
                        "✕ Delete",
                        id=f"library-media-highlight-delete-{index}",
                        classes="library-canvas-action library-media-highlight-delete",
                        compact=True,
                    )
                    delete_button.highlight_id = highlight.highlight_id
                    yield delete_button
        with Collapsible(
            title="Add highlight",
            collapsed=True,
            id="library-media-highlight-add-collapsible",
        ):
            yield Input(
                placeholder="Quote",
                id="library-media-highlight-quote",
            )
            yield Input(
                placeholder="Note (optional)",
                id="library-media-highlight-note",
            )
            yield Input(
                placeholder="Color (optional)",
                id="library-media-highlight-color",
            )
            yield Button(
                "Add highlight",
                id="library-media-highlight-add",
                classes="library-canvas-action",
                compact=True,
            )
