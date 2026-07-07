"""Library media viewer canvas: full metadata + content, with a Back control."""

from __future__ import annotations

from typing import Any, Sequence

from rich.text import Text
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from tldw_chatbook.Library.library_media_viewer_state import (
    LibraryMediaHighlightRow,
    LibraryMediaViewerState,
    find_content_matches,
)


class LibraryMediaViewer(Vertical):
    """Render the full Library media item: metadata, content, and actions.

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
        # Fill the (already 13fr) canvas host, not an independent 13fr: an `fr`
        # width here breaks width:100% child resolution so long lines (analysis
        # summary, a long URL) clip instead of wrapping. 1fr fills the same
        # space and lets the text bodies wrap.
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        """Render the back control, title, metadata, content, and actions.

        Uses only render-verified widgets (``Static``/``VerticalScroll``/
        ``Button``) stacked full-width in this ``Vertical`` — horizontal rows
        that mix a ``1fr`` sibling with a fixed-width widget are the known
        non-rendering failure mode, so every row here is either a single
        full-width widget or the plain ``ds-toolbar`` action row (already
        proven to render by the conversations/media list canvases).

        Returns:
            ComposeResult for the media viewer canvas.
        """
        yield Button(
            "‹ Back to list",
            id="library-media-back",
            classes="library-canvas-action",
            compact=True,
        )
        yield Static(
            "Edit media details" if self.editing else self.viewer.title,
            id="library-media-viewer-title",
            markup=False,
        )
        if self.editing:
            yield from self._compose_edit_form()
        else:
            yield Static(
                "\n".join(self.viewer.metadata_lines),
                id="library-media-viewer-meta",
                markup=False,
            )
        yield Static(
            "Content",
            id="library-media-viewer-content-title",
            classes="destination-section",
        )
        yield from self._compose_content_search()
        with VerticalScroll(id="library-media-viewer-content"):
            yield Static(
                self._content_renderable(),
                id="library-media-viewer-content-text",
                markup=False,
            )
        yield from self._compose_analysis()

        yield from self._compose_highlights()

        if self.confirming_delete and not self.editing:
            # A single full-width Static above the toolbar, not inside it --
            # mixing a Static with the toolbar's Buttons is the known
            # non-rendering failure mode called out on ``compose`` above.
            yield Static(
                "Delete this media? This moves it to trash.",
                id="library-media-delete-confirm-copy",
                markup=False,
            )

        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            if self.editing:
                yield Button(
                    "Save",
                    id="library-media-edit-save",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-media-edit-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )
            elif self.confirming_delete:
                yield Button(
                    "Delete",
                    id="library-media-delete-confirm",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Cancel",
                    id="library-media-delete-cancel",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                # Object/primary actions first, then the escape hatch to the
                # legacy screen, then the destructive Delete pushed to the far
                # end (CSS margin) so it is not adjacent to Edit -- avoids the
                # classic Edit/Delete misclick trap.
                yield Button(
                    "Edit",
                    id="library-media-edit",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Use in Chat",
                    id="library-media-use-in-chat",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Remove from read-it-later" if self.viewer.read_later else "Read it later",
                    id="library-media-read-later",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Open in Media",
                    id="library-media-open",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "Delete",
                    id="library-media-delete",
                    classes="library-canvas-action library-media-action-danger",
                    compact=True,
                )

    def _compose_content_search(self) -> ComposeResult:
        """Render the in-content search box, and its status/prev-next only while active.

        The search ``Input`` always renders, full-width above the content
        ``VerticalScroll``. The match-count status ``Static`` and the
        prev/next ``ds-toolbar`` only render while ``self.content_query``
        is non-empty -- with no active search there is nothing to page
        through, so the status line and toolbar are omitted entirely
        rather than left showing as empty/orphaned chrome. When present,
        Input and Static are each their own row and prev/next live in a
        plain ``ds-toolbar`` of buttons only, matching the render-safety
        rule on ``compose`` above (never mix a ``1fr`` sibling with a
        fixed-width widget in one ``Horizontal``).

        Returns:
            ComposeResult for the content search row and, when a query is
            active, the status line and prev/next action toolbar.
        """
        yield Input(
            value=self.content_query,
            placeholder="Search content…",
            id="library-media-content-search",
        )
        if not self.content_query:
            return
        matches = find_content_matches(self.viewer.content, self.content_query)
        yield Static(
            self._content_search_status_text(matches),
            id="library-media-content-search-status",
            markup=False,
        )
        search_toolbar = Horizontal(classes="ds-toolbar")
        search_toolbar.styles.height = "auto"
        with search_toolbar:
            yield Button(
                "◀ Prev",
                id="library-media-content-search-prev",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Next ▶",
                id="library-media-content-search-next",
                classes="library-canvas-action",
                compact=True,
            )

    def _content_search_status_text(self, matches: tuple[int, ...]) -> str:
        """Build the in-content search status line text.

        Args:
            matches: Ordered line indices matching ``self.content_query``,
                as returned by ``find_content_matches``.

        Returns:
            "" when the query is blank (no search active), "No matches"
            when a non-blank query has no hits, otherwise
            "Match {i} of {n} matches" for the current (wrapped) index.
        """
        if not self.content_query:
            return ""
        if not matches:
            return "No matches"
        index = self.content_match_index % len(matches)
        return f"Match {index + 1} of {len(matches)} matches"

    def _content_renderable(self) -> Text | str:
        """Return the content body, with every query occurrence highlighted while searching.

        With no active search this is the plain content string. While a
        search is active, occurrences of the query (case-insensitive) are
        marked with a reverse style so the matches the status line counts
        are actually visible in the body -- otherwise "Match 1 of N" points
        at hits the reader cannot see. Built as a Rich ``Text`` from raw
        slices (never markup) so arbitrary content cannot inject styles.

        Returns:
            The plain content ``str`` when idle, or a Rich ``Text`` with
            highlighted matches while searching.
        """
        content = self.viewer.content or "No stored content."
        query = (self.content_query or "").strip()
        if not query or not self.viewer.content:
            return content
        needle = query.lower()
        haystack = content.lower()
        span = len(needle)
        text = Text()
        cursor = 0
        while True:
            hit = haystack.find(needle, cursor)
            if hit == -1:
                text.append(content[cursor:])
                break
            text.append(content[cursor:hit])
            text.append(content[hit : hit + span], style="reverse")
            cursor = hit + span
        return text

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
                "Edit analysis",
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
        """Return ``color`` if it parses as a renderable color, else None.

        Highlight colors are free-text (the add form's "Color (optional)"),
        so a value like "highlighter pink" cannot be shown as a swatch.

        Args:
            color: The stored highlight color string.

        Returns:
            The color string when Textual can parse it, otherwise None.
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
