"""Virtualized Raw content view for the Library media reader (TASK-22500).

Renders only the rows in view. The whole-document ``Static`` this replaces
cost 1051 ms at first paint and 684 ms on every ``update()`` -- that is,
on every search keystroke and every match-navigation click.
"""

from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.text import Text
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.selection import Selection
from textual.strip import Strip

from tldw_chatbook.Utils.text_wrap_index import WrapIndex

MATCH_STYLE = Style(reverse=True)
ACTIVE_MATCH_STYLE = Style(reverse=True, bold=True)
EMPTY_CONTENT_MESSAGE = "No stored content."


class VirtualizedRawContent(ScrollView):
    """Scrollable raw text that renders one row at a time.

    Wraps and paginates a document through :class:`WrapIndex` so that only
    the rows currently in the viewport are rendered on each repaint,
    regardless of document size.
    """

    # Test-only instrumentation; asserting harness wall time is meaningless
    # because pilot.pause() costs ~30 ms per call.
    RENDER_LINE_CALLS: dict[str, int] = {"n": 0}

    def __init__(
        self,
        *,
        content: str,
        query: str,
        match_index: int,
        max_visible_rows: int = 18,
        **kwargs: Any,
    ) -> None:
        """Initialize the virtualized raw content view.

        Args:
            content: The full document text to render, source lines joined
                by ``"\\n"``.
            query: The initial search query to highlight, if any.
            match_index: The index of the initially active match within the
                set of matches for ``query``.
            max_visible_rows: The maximum number of rows the widget will
                request via its CSS ``height`` (the pane still scrolls past
                this to reach the full document).
            **kwargs: Additional keyword arguments forwarded to
                :class:`~textual.scroll_view.ScrollView`.
        """
        super().__init__(**kwargs)
        self.source_lines = (content or EMPTY_CONTENT_MESSAGE).split("\n")
        self.wrap_index: WrapIndex | None = None
        self._indexed_width: int | None = None
        self._query = query.strip()
        self._match_index = match_index
        self._max_visible_rows = max_visible_rows
        self._match_lines: tuple[int, ...] = ()

    def on_resize(self, _event: Any = None) -> None:
        """Reindex the document if the available width has changed.

        Args:
            _event: The resize event (unused; the current size is read
                directly from the widget).
        """
        self._reindex_if_width_changed()

    def on_mount(self) -> None:
        """Build the initial wrap index once the widget has a size."""
        self._reindex_if_width_changed()

    def _reindex_if_width_changed(self) -> None:
        """Rebuild the wrap index when the rendering width changes.

        No-ops if the width is unchanged or not yet known (zero), so a
        resize that doesn't affect wrapping is cheap.
        """
        width = self.scrollable_content_region.width or self.size.width
        if width <= 0 or width == self._indexed_width:
            return
        self.wrap_index = WrapIndex.build(self.source_lines, width)
        self._indexed_width = width
        self.virtual_size = Size(width, self.wrap_index.virtual_height)
        self.styles.height = min(self.wrap_index.virtual_height, self._max_visible_rows)
        self.refresh()

    def sync_search(self, query: str, match_index: int) -> None:
        """Restyle the visible rows for a new query or active match.

        Args:
            query: The new search query to highlight.
            match_index: The index of the active match within the set of
                matches for ``query``.
        """
        self._query = query.strip()
        self._match_index = match_index
        self.refresh()

    def scroll_to_source_line(self, line_index: int) -> None:
        """Scroll so a SOURCE line is visible, mapping through the index.

        The screen previously scrolled to the source-line index as if it
        were a screen row, which drifts once any line wraps.

        Args:
            line_index: The index of a source line (0-based) to scroll to.
        """
        if self.wrap_index is None:
            return
        self.scroll_to(y=self.wrap_index.line_start_row(line_index), animate=False)

    def render_line(self, y: int) -> Strip:
        """Render a single visible row by mapping it through the wrap index.

        Args:
            y: The row number relative to the top of the viewport (not the
                document); the document row is derived by adding the
                current vertical scroll offset.

        Returns:
            A :class:`~textual.strip.Strip` for the requested row, blank if
            the row is out of range or the widget has no width yet.
        """
        type(self).RENDER_LINE_CALLS["n"] += 1
        width = self.scrollable_content_region.width or self.size.width
        if self.wrap_index is None or width <= 0:
            return Strip.blank(max(width, 0))
        row = y + int(self.scroll_offset.y)
        if row < 0 or row >= self.wrap_index.virtual_height:
            return Strip.blank(width)
        line_index, segment_index = self.wrap_index.row_to_line(row)
        segments = self.wrap_index.segments(line_index)
        piece = segments[segment_index] if segment_index < len(segments) else ""
        text = Text(piece, no_wrap=True, end="")
        if self._query:
            hit = piece.lower().find(self._query.lower())
            if hit >= 0:
                active = (
                    self._match_lines
                    and line_index
                    == self._match_lines[self._match_index % len(self._match_lines)]
                )
                text.stylize(
                    ACTIVE_MATCH_STYLE if active else MATCH_STYLE,
                    hit,
                    hit + len(self._query),
                )
        # Visual selection highlight: Static gets this for free from
        # Visual.to_strips reading widget.text_selection; a hand-rolled
        # render_line has to do it explicitly or a drag copies the right
        # text (get_selection, above) while showing no visible feedback at
        # all. Follows the same precedent as Textual's own Log widget
        # (textual/widgets/_log.py, ScrollView + hand-rolled render_line):
        # Selection.get_span(row) already returns the right sub-range for
        # every case -- full row for SELECT_ALL, `(x, -1)`/`(0, x)` for a
        # row that is only partially covered (the first/last row of a
        # multi-row selection), `(0, -1)` for a fully-covered middle row --
        # so no separate wrap-boundary-vs-line-boundary logic is needed
        # here; `row` is already the same per-wrapped-row domain
        # `apply_offsets` below uses.
        selection = self.text_selection
        if selection is not None:
            span = selection.get_span(row)
            if span is not None:
                select_start, select_end = span
                if select_end == -1:
                    select_end = len(piece)
                selection_style = self.screen.get_component_rich_style(
                    "screen--selection"
                )
                text.stylize(selection_style, select_start, select_end)
        rendered = list(text.render(self.app.console))
        strip = Strip(rendered, len(piece))
        # Embed a per-cell content offset (column-in-piece, document row) so
        # Textual's mouse hit-testing (Compositor.get_widget_and_offset_at)
        # can resolve a real (x, y) inside this row. A ScrollView that
        # renders its own Strips gets none of this for free the way a
        # Content-backed Static does -- without it, every drag over this
        # widget resolves to a `None` content offset and Textual silently
        # degrades the selection to "select this whole widget"
        # (`Selection(None, None)`), losing the old Static's precise
        # drag-select rather than raising anything visible.
        strip = strip.apply_offsets(0, row)
        return strip.adjust_cell_length(width)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Map a screen selection back to SOURCE text.

        Rows in ``selection`` are wrap-index rows -- the same document-space
        row ``render_line`` embeds via ``Strip.apply_offsets`` -- not source
        line indices, and each endpoint's column is relative to that row's
        own wrapped segment (``render_line`` resets the column to 0 at the
        start of every row). A selection spanning a wrap boundary inside one
        source line must rejoin the segments it covers without inserting a
        newline the document does not contain; a selection spanning two
        source lines must contain exactly one newline between them.

        Args:
            selection: The selection to extract. Either endpoint may be
                ``None``, meaning "start of document" / "end of document"
                respectively -- ``Selection(None, None)`` selects
                everything, matching :data:`textual.selection.SELECT_ALL`.

        Returns:
            A tuple of the extracted text and a trailing delimiter (mirrors
            ``Static``/``Log``'s ``"\\n"`` convention), or ``None`` if the
            widget has no content to select from yet.
        """
        if self.wrap_index is None:
            return None
        height = self.wrap_index.virtual_height
        if height <= 0:
            return None

        start = selection.start
        end = selection.end
        first_row, start_col = (0, 0) if start is None else (start.y, start.x)
        last_row, end_col = (height - 1, None) if end is None else (end.y, end.x)
        first_row = max(0, min(first_row, height - 1))
        last_row = max(0, min(last_row, height - 1))

        collected: list[str] = []
        previous_line: int | None = None
        for row in range(first_row, last_row + 1):
            line_index, segment_index = self.wrap_index.row_to_line(row)
            segments = self.wrap_index.segments(line_index)
            piece = segments[segment_index] if segment_index < len(segments) else ""
            if row == first_row and row == last_row:
                piece = piece[start_col:end_col]
            elif row == first_row:
                piece = piece[start_col:]
            elif row == last_row:
                piece = piece[:end_col]
            if previous_line is not None and line_index != previous_line:
                collected.append("\n")
            collected.append(piece)
            previous_line = line_index
        return "".join(collected), "\n"

    def set_match_lines(self, match_lines: tuple[int, ...]) -> None:
        """Record which SOURCE lines match, for active-match styling.

        Args:
            match_lines: The source line indices (0-based) that contain a
                match for the current search query, in match order.
        """
        self._match_lines = match_lines
        self.refresh()
