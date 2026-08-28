"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

from typing import Any, Sequence

from rich.color import Color
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from tldw_chatbook.Library.library_media_viewer_state import (
    LibraryMediaHighlightRow,
    LibraryMediaViewerState,
    find_content_matches,
)
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
    LibraryMediaContentSearchControls,
)


class LibraryMediaViewer(Vertical):
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
        content_query: str = "",
        content_match_index: int = 0,
        content_mode: str = "raw",
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.viewer = viewer
        self.editing = editing
        self.confirming_delete = confirming_delete
        self.highlights = tuple(highlights)
        self.editing_analysis = editing_analysis
        self.content_query = content_query
        self.content_match_index = content_match_index
        self.content_mode = content_mode
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
                "Loading media…"
                if self.loading
                else "Select a media item to read it here.",
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
        yield Static(
            "Server item · not in local Media list"
            if self.external_detail
            else "Local Media item",
            id="library-media-reader-identity",
            markup=False,
        )
        yield Button("‹ Back", id="library-media-back", compact=True)
        yield Static(
            "Edit media details" if self.editing else self.viewer.title,
            id="library-media-viewer-title",
            markup=False,
        )
        if not self.editing:
            yield Static(
                next(
                    (line.removeprefix("Author: ") for line in self.viewer.metadata_lines
                     if line.startswith("Author: ")),
                    "",
                )
                or next(
                    (line.removeprefix("URL: ") for line in self.viewer.metadata_lines
                     if line.startswith("URL: ")),
                    "",
                ),
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
            yield Button("Find", id="library-media-reader-find", compact=True)
            if not self.external_detail:
                yield Button(
                    "Remove later" if self.viewer.read_later else "Read later",
                    id="library-media-read-later",
                    compact=True,
                )
            yield Button("Use in Console", id="library-media-use-in-chat", compact=True)
            if not self.external_detail or self.viewer.original_source:
                yield Button("More", id="library-media-reader-more", compact=True)
        if self.more_open:
            with Vertical(id="library-media-reader-more-actions"):
                if not self.external_detail:
                    yield Button("Edit metadata", id="library-media-edit", compact=True)
                if self.viewer.original_source:
                    yield Button("Open original", id="library-media-open-original", compact=True)
                if not self.external_detail:
                    yield Button("Open manager", id="library-media-open", compact=True)
                    yield Button("Move to trash", id="library-media-delete", compact=True)

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
            with Vertical(id="library-media-reader-mode-read"):
                yield Static("Read", classes="destination-section")
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
            # the scrolling Reader. Textual docks relative to the immediate
            # container, so nesting these under the mode marker pins an active
            # Find bar below the Reader header instead of at the viewport top.
            matches = find_content_matches(self.viewer.content, self.content_query)
            yield LibraryMediaContentSearchControls(
                is_markdown=self.viewer.is_markdown,
                query=self.content_query,
                matches=matches,
                match_index=self.content_match_index,
                id="library-media-content-search-controls",
            )
            yield LibraryMediaContentBody(
                content=self.viewer.content,
                is_markdown=self.viewer.is_markdown,
                mode=self.content_mode,
                query=self.content_query,
                match_index=self.content_match_index,
                id="library-media-viewer-content",
            )
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
            yield Static("Info", classes="destination-section")
            if self.editing:
                yield from self._compose_edit_form()
            else:
                yield Static("\n".join(self.viewer.metadata_lines), id="library-media-viewer-meta", markup=False)
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

    def _compose_content_mode_toggle(self) -> ComposeResult:
        """Render the Rendered|Raw content-view toggle for markdown-typed media.

        Only rendered when ``self.viewer.is_markdown`` is true -- a
        non-markdown item never offers a toggle and always shows the plain
        Raw view (no behavior change from before LIB-13). Mirrors the
        screen's own "Database (selected) | Files" source-strip idiom
        exactly (``library_screen.py``'s notes-source strip): a plain
        ``Horizontal`` of two compact, unstyled ``Button``s with a "|"
        ``Static`` separator, each label suffixed "(selected)" for the
        active mode -- the state-in-text idiom, not a color/class alone,
        so the current mode reads correctly even without extra CSS.

        Returns:
            ComposeResult for the toggle strip, or nothing for non-markdown
            media.
        """
        if not self.viewer.is_markdown:
            return
        with Horizontal(id="library-media-content-mode-strip"):
            rendered_selected = self.content_mode == "rendered"
            rendered_button = Button(
                "Rendered (selected)" if rendered_selected else "Rendered",
                id="library-media-content-mode-rendered",
                compact=True,
            )
            rendered_button.set_class(rendered_selected, "-selected")
            yield rendered_button
            yield Static("|", id="library-media-content-mode-separator", markup=False)
            raw_selected = not rendered_selected
            raw_button = Button(
                "Raw (selected)" if raw_selected else "Raw",
                id="library-media-content-mode-raw",
                compact=True,
            )
            raw_button.set_class(raw_selected, "-selected")
            yield raw_button

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
            copy = (
                "Loading media…"
                if loading
                else "Select a media item to read it here."
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
        self.content_query = query
        self.content_match_index = match_index
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
        """Render the Analysis section: read-only text + Edit toggle, or the edit form.

        Always renders (mirroring the Content section's always-present
        placeholder) so "Edit analysis" is reachable even when no analysis
        exists yet -- editing an empty analysis simply creates the first
        one via ``save_analysis_version``. Analysis (re)generation via an
        LLM is explicitly out of scope; this only edits existing text.

        Returns:
            ComposeResult for the Analysis section.
        """
        yield Static(
            "Analysis",
            id="library-media-viewer-analysis-title",
            classes="destination-section",
        )
        if self.editing_analysis:
            yield from self._compose_analysis_edit_form()
        else:
            yield Static(
                self.viewer.analysis or "No analysis yet.",
                id="library-media-viewer-analysis-text",
                markup=False,
            )
            yield Button(
                "Edit analysis" if self.viewer.analysis else "Add analysis",
                id="library-media-analysis-edit",
                classes="library-canvas-action",
                compact=True,
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
        yield Static(
            "Highlights",
            id="library-media-viewer-highlights-title",
            classes="destination-section",
        )
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
