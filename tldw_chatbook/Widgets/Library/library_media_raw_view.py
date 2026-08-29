"""Virtualized Raw content view for the Library media reader (TASK-22500).

Renders only the rows in view. The whole-document ``Static`` this replaces
cost 1051 ms at first paint and 684 ms on every ``update()`` -- that is,
on every search keystroke and every match-navigation click.
"""

from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.text import Text
from textual import events
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.selection import Selection
from textual.strip import Strip
from textual.timer import Timer

from tldw_chatbook.Utils.text_wrap_index import WrapIndex

MATCH_STYLE = Style(reverse=True)
ACTIVE_MATCH_STYLE = Style(reverse=True, bold=True)
PLAIN_STYLE = Style()
EMPTY_CONTENT_MESSAGE = "No stored content."
# Matches Textual's Content.expand_tabs default (textual/content.py) -- the
# Static this widget replaces renders unstyled content through Content, whose
# expand_tabs takes the plain str.expandtabs(8) fast path for markup=False,
# span-free text. Expanding once here, before the wrap index is built, keeps
# both wrap points and rendered columns aligned with what Static painted.
TAB_SIZE = 8


class VirtualizedRawContent(ScrollView):
    """Scrollable raw text that renders one row at a time.

    Wraps and paginates a document through :class:`WrapIndex` so that only
    the rows currently in the viewport are rendered on each repaint,
    regardless of document size.
    """

    # Test-only instrumentation; asserting harness wall time is meaningless
    # because pilot.pause() costs ~30 ms per call.
    RENDER_LINE_CALLS: dict[str, int] = {"n": 0}

    # Rebuilding the wrap index costs ~125-155 ms on a 2.5 MB document and
    # ~400 ms at 6.5 MB (measured). Textual fires many `Resize` events
    # during a single drag of the pane edge, so `on_resize` coalesces a
    # burst into one rebuild after this quiet period, mirroring TASK-22211's
    # hysteresis precedent in the Watchlists layout.
    REINDEX_DEBOUNCE_SECONDS: float = 0.12

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
        # Two parallel line lists, index-aligned (expandtabs never adds or
        # removes a "\n", so line counts match): `source_lines` (expanded)
        # drives wrapping and painting, matching what Content/Static wraps
        # and paints against; `_raw_lines` (literal tabs intact) is what
        # get_selection maps copies back to, matching what Static's own
        # get_selection actually returns (Widget.get_selection reads
        # Content.plain, which expand_tabs never mutates -- expansion is a
        # transient copy made inside render_strips).
        self._raw_lines = (content or EMPTY_CONTENT_MESSAGE).split("\n")
        self.source_lines = [line.expandtabs(TAB_SIZE) for line in self._raw_lines]
        self.wrap_index: WrapIndex | None = None
        self._indexed_width: int | None = None
        self._query = query.strip()
        self._match_index = match_index
        self._max_visible_rows = max_visible_rows
        self._match_lines: tuple[int, ...] = ()
        self._pending_reindex_width: int | None = None
        self._reindex_timer: Timer | None = None
        # FINDING 1 fix: the first-occurrence offset of the active query
        # within a SOURCE line, keyed by line index and invalidated on any
        # query change. Static (the widget this replaces) ran exactly one
        # `find` per SOURCE line, letting Rich wrap that single span across
        # rows on its own; matching that here means searching the source
        # line once, not the rendered segment on every row/repaint -- a
        # heavily wrapped line (thousands of segments) would otherwise pay
        # a full-line `find` on every visible row of that line, every time
        # the widget repaints.
        # Unbounded on purpose, unlike WrapIndex's segment cache: an entry is
        # one small int per SOURCE line actually rendered while a query is
        # active, and the whole dict is dropped on any query change. Worst
        # case is bounded by the document's line count.
        self._match_hit_cache: dict[int, int] = {}
        self._match_hit_cache_query: str | None = None

    def on_resize(self, _event: events.Resize | None = None) -> None:
        """Reindex on a width change, debounced after the first layout pass.

        ``on_mount`` runs before this widget has been given a real size
        (``width`` is still 0 at that point), so the FIRST ``Resize`` --
        the one that reports the initial layout's width -- is the one that
        actually delivers first paint and must build immediately, exactly
        like ``on_mount`` would if it could. Only resizes AFTER that first
        real build (a user dragging the pane edge, etc.) are debounced via
        :meth:`_request_reindex`, so a burst collapses into one rebuild
        without delaying the initial paint.

        Args:
            _event: The resize event (unused; the current size is read
                directly from the widget).
        """
        width = self.scrollable_content_region.width or self.size.width
        if self._indexed_width is None:
            self._build_index_now(width)
        else:
            self._request_reindex(width)
        # Re-run unconditionally: the height cap depends on the PARENT's
        # content region, which settles a layout pass after this widget
        # first claims a height. `_build_index_now` no-ops when the width is
        # unchanged, so without this the cap would keep its pre-settle value.
        self._apply_height_cap()

    def on_mount(self) -> None:
        """Build the initial wrap index synchronously, if already sized.

        Unlike a debounced resize, this must NOT go through the debounce --
        a debounced first index would leave the reader showing an empty
        body for the debounce interval on every open, which is a visible
        regression. In practice the widget is not yet sized at mount time
        (``on_resize`` delivers the real first-layout width, handled
        synchronously there too -- see :meth:`on_resize`), but this call is
        cheap and correct either way: a zero width no-ops, and a nonzero
        one here means one less event round-trip before first paint.
        """
        width = self.scrollable_content_region.width or self.size.width
        self._build_index_now(width)

    def on_unmount(self) -> None:
        """Cancel any pending debounced reindex so it never fires into a
        detached widget.

        Stopping the timer explicitly is needed even though
        :meth:`_fire_pending_reindex` also guards on attachment, since an
        unstopped ``Timer`` otherwise keeps running until it fires.
        """
        if self._reindex_timer is not None:
            self._reindex_timer.stop()
            self._reindex_timer = None

    def _request_reindex(self, width: int) -> None:
        """Arm (or re-arm) a single debounce timer to rebuild at ``width``.

        Any previously pending rebuild is superseded -- both its timer and
        the width it was going to use -- so a burst of resize events pays
        the rebuild cost exactly once, for the final width, after the
        burst goes quiet for :data:`REINDEX_DEBOUNCE_SECONDS`.

        Args:
            width: The candidate rendering width to reindex at once the
                debounce interval elapses.
        """
        self._pending_reindex_width = width
        if self._reindex_timer is not None:
            self._reindex_timer.stop()
            self._reindex_timer = None
        if self.REINDEX_DEBOUNCE_SECONDS <= 0:
            # `set_timer(0.0)` never fires in Textual 8 -- a
            # ZeroDivisionError is raised inside the timer's own task and
            # swallowed, silently disabling the rebuild. Guard against ever
            # relying on that path.
            self._fire_pending_reindex()
            return
        self._reindex_timer = self.set_timer(
            self.REINDEX_DEBOUNCE_SECONDS, self._fire_pending_reindex
        )

    def _fire_pending_reindex(self) -> None:
        """Consume the pending width and rebuild, unless already detached.

        This is the timer callback armed by :meth:`_request_reindex`. It
        can legitimately fire after this widget has been removed from the
        DOM (the timer was already running when removal happened), so it
        must check attachment rather than assume it is still safe to touch
        widget state.

        ``is_attached``, not ``is_mounted``: this repo has previously hit a
        widget whose ``is_mounted`` stayed True after ``remove()``, while
        ``is_attached`` correctly reflects whether it still has a path to
        the DOM root.
        """
        self._reindex_timer = None
        width = self._pending_reindex_width
        if width is None or not self.is_attached:
            return
        self._build_index_now(width)

    def _build_index_now(self, width: int) -> None:
        """Rebuild the wrap index for ``width`` immediately, no debounce.

        No-ops if the width is unchanged or not yet known (zero), so a
        resize that doesn't affect wrapping is cheap.

        Args:
            width: The rendering width to index against.
        """
        if width <= 0 or width == self._indexed_width:
            return
        self.wrap_index = WrapIndex.build(self.source_lines, width)
        self._indexed_width = width
        # Width 0, deliberately: this reader wraps, so there is never
        # anything to reach by scrolling horizontally. Claiming the indexed
        # width here grew a horizontal scrollbar the moment the vertical one
        # appeared; that scrollbar consumed a row, shrank the render width
        # below the width the index was built at, and every wrapped row was
        # then silently truncated by the difference (2 columns, measured)
        # rather than re-flowed. `on_resize` does not fire for that, because
        # the widget's own size never changed.
        self.virtual_size = Size(0, self.wrap_index.virtual_height)
        self._apply_height_cap()
        self.refresh()

    def _visible_row_cap(self) -> int:
        """Return how many rows this widget may occupy.

        Derived from the parent's own content region rather than assumed,
        because the container's ``max-height`` is an OUTER bound: with
        ``border: solid`` the two border rows come out of it, so a widget
        that claims the full ``max-height`` overflows by exactly the border
        and has its last rows clipped -- while ``ScrollView`` still computes
        ``max_scroll_y`` against the unclipped height it thinks it has, so
        the tail of the document becomes unreachable by scrolling.

        Returns:
            The parent's available content height, or the configured
            fallback when the parent is not yet sized.
        """
        parent = self.parent
        available = getattr(parent, "content_region", None)
        if available is not None and available.height > 0:
            return available.height
        return self._max_visible_rows

    def _apply_height_cap(self) -> None:
        """Size this widget to its content, bounded by the room available."""
        if self.wrap_index is None:
            return
        self.styles.height = min(self.wrap_index.virtual_height, self._visible_row_cap())

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

    def _hit_for_line(self, line_index: int) -> int:
        """Return the first-occurrence offset of the active query in a source line.

        Mirrors ``build_raw_content_highlight_plan``'s original behavior --
        one ``str.find`` per SOURCE line, first occurrence only -- not per
        rendered (wrapped) segment. The result is cached per line index and
        invalidated whenever ``self._query`` changes, so scrolling a heavily
        wrapped matching line (many rows, one source line) does not re-scan
        the same line text on every row and every repaint.

        Args:
            line_index: The index of a source line (0-based) to search.

        Returns:
            The 0-based character offset of the query's first occurrence in
            ``self.source_lines[line_index]`` (case-insensitive), or ``-1``
            if it does not occur there.
        """
        if self._match_hit_cache_query != self._query:
            self._match_hit_cache_query = self._query
            self._match_hit_cache = {}
        hit = self._match_hit_cache.get(line_index)
        if hit is None:
            hit = self.source_lines[line_index].lower().find(self._query.lower())
            self._match_hit_cache[line_index] = hit
        return hit

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
            return Strip.blank(max(width, 0), PLAIN_STYLE)
        if width != self._indexed_width:
            # The render width can change without a Resize on this widget --
            # a scrollbar appearing inside it shrinks the content region
            # while `size` stays put. Re-arm the (debounced) rebuild so the
            # index converges on the width actually being painted instead of
            # truncating every row by the difference. Costs one rebuild, and
            # self-limits: once rebuilt, the widths match.
            self._request_reindex(width)
        row = y + int(self.scroll_offset.y)
        if row < 0 or row >= self.wrap_index.virtual_height:
            return Strip.blank(width, PLAIN_STYLE)
        line_index, segment_index = self.wrap_index.row_to_line(row)
        segments = self.wrap_index.segments(line_index)
        piece = segments[segment_index] if segment_index < len(segments) else ""
        text = Text(piece, no_wrap=True, end="")
        if self._query:
            # FINDING 1 fix: search the SOURCE line (first occurrence only,
            # exactly like the retired Static-backed highlighter), then map
            # that source-character range onto THIS segment by intersecting
            # it with the segment's own source-character span. A match
            # straddling a wrap boundary is styled -- partially, where
            # clipped -- on every row it covers, instead of only on the row
            # whose OWN substring happens to contain the whole needle (which
            # is "no row" when the needle itself straddles the boundary).
            hit = self._hit_for_line(line_index)
            if hit >= 0:
                match_end = hit + len(self._query)
                segment_start = self.wrap_index.segment_start(line_index, segment_index)
                segment_end = segment_start + len(piece)
                local_start = max(hit, segment_start) - segment_start
                local_end = min(match_end, segment_end) - segment_start
                if local_start < local_end:
                    active = (
                        self._match_lines
                        and line_index
                        == self._match_lines[self._match_index % len(self._match_lines)]
                    )
                    text.stylize(
                        ACTIVE_MATCH_STYLE if active else MATCH_STYLE,
                        local_start,
                        local_end,
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
        # No explicit cell length: `len(piece)` is a CHARACTER count, and
        # Strip's second argument is a CELL count. They diverge on any
        # 2-cell glyph (CJK, emoji), and because the declared length came
        # out SHORT, `adjust_cell_length` below padded instead of
        # truncating -- emitting 43-46 cell rows into a 40 cell screen on a
        # document containing wide characters. Letting Strip measure the
        # segments itself restores byte-identical output to the Static this
        # widget replaces.
        strip = Strip(rendered)
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
        # Textual's no-color filter assumes every segment has a Rich Style.
        # ``adjust_cell_length`` otherwise pads a short row with style=None,
        # which crashes monochrome terminals while painting the reader.
        return strip.adjust_cell_length(width, PLAIN_STYLE)

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

        The returned text is read from ``_raw_lines`` (literal tabs, never
        expanded), not from the wrap index's expanded segments, so a copy of
        a tab-delimited document round-trips its tabs -- matching what
        ``Static``'s own ``get_selection`` returns (it reads
        ``Content.plain``, which ``expand_tabs`` never mutates in place).
        The row/column math still walks the EXPANDED segments, because that
        is the coordinate space ``render_line``'s embedded offsets and
        wrap points use; each column is then reinterpreted as an offset into
        the matching raw line. That reuse is deliberate, not an
        approximation: Static does the exact same thing (the compositor's
        embedded per-cell offset is an expanded-column count, and
        ``Selection.extract`` indexes the raw string with it unchanged), so
        replicating it -- rather than building a "smarter" corrected
        mapping -- is what makes a partial selection landing inside an
        expanded tab run byte-identical to Static, including the case where
        release lands on the tab's second expanded cell and the selection
        reads one raw character past the tab.

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
            # Offset of this wrapped segment's start within the full
            # EXPANDED line -- the same cumulative column count Static's
            # own per-cell offsets carry across a wrap boundary. Read from
            # the index's prefix sums: re-summing it per row made a
            # select-all copy quadratic (5.7 s on a 2.5 MB single-line
            # document).
            segment_start = self.wrap_index.segment_start(line_index, segment_index)
            row_end = segment_start + len(piece)
            abs_start = segment_start + start_col if row == first_row else segment_start
            if row == last_row:
                abs_end = segment_start + end_col if end_col is not None else row_end
            else:
                abs_end = row_end
            raw_line = self._raw_lines[line_index]
            raw_piece = raw_line[abs_start:abs_end]
            if previous_line is not None and line_index != previous_line:
                collected.append("\n")
            collected.append(raw_piece)
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
