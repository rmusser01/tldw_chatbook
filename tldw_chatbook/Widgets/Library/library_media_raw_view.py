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
        rendered = list(text.render(self.app.console))
        return Strip(rendered, len(piece)).adjust_cell_length(width)

    def set_match_lines(self, match_lines: tuple[int, ...]) -> None:
        """Record which SOURCE lines match, for active-match styling.

        Args:
            match_lines: The source line indices (0-based) that contain a
                match for the current search query, in match order.
        """
        self._match_lines = match_lines
        self.refresh()
