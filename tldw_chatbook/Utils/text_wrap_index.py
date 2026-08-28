"""Exact wrap arithmetic for virtualized text views (TASK-22500).

Pure: Rich only, no Textual, no Library package import (``Library/__init__``
eagerly pulls a 66-module service stack -- TASK-22223).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from rich.console import Console
from rich.text import Text

try:  # pragma: no cover - exercised by the agreement test
    from rich._wrap import divide_line as _rich_divide_line
except ImportError:  # pragma: no cover - fallback path
    _rich_divide_line = None

_FALLBACK_CONSOLE = Console(width=80)


def divide_source_line(line: str, width: int) -> list[int]:
    """Return the offsets at which ``line`` wraps at ``width``.

    Uses Rich's ``divide_line`` when available -- that is the function
    ``Static`` itself wraps with, so its offsets are what makes this widget
    paint identically to the ``Static`` it replaces. Those offsets are
    pinned exactly by ``test_divide_source_line_offsets_match_the_private_
    api_exactly``.

    Falls back to the public ``Text.wrap`` API if that private API ever
    disappears. The fallback is deliberately best-effort about WHERE lines
    break: ``Text.wrap`` returns each divided line already rstripped, and
    cannot distinguish whitespace absorbed after a word break (which Rich
    keeps on the preceding segment) from whitespace after a hard mid-word
    fold (which Rich gives its own segment). What the fallback does
    guarantee -- and what is tested -- is that no character is dropped and
    no segment carries real content past ``width``.

    Args:
        line: The source text to divide.
        width: The maximum width in cells before wrapping occurs.

    Returns:
        A list of character offsets where ``line`` wraps. Empty if the line
        fits entirely within ``width`` or if ``width`` is invalid.
    """
    if width <= 0 or not line:
        return []
    if len(line) <= width and line.isascii():
        # A short pure-ASCII line cannot wrap: every ASCII character occupies
        # exactly one cell, so its cell width is at most its character count.
        # Skipping the call here is what keeps the index build off the
        # >100 ms "needs a worker" threshold (repo performance rule): most
        # lines in a real document are short and ASCII, and the measured
        # build for a 2.5 MB document drops 134.5 ms -> 1.1 ms with heights
        # unchanged. Deliberately conservative -- any non-ASCII character may
        # be wide (or zero-width), so those still go through Rich.
        return []
    if _rich_divide_line is not None:
        return list(_rich_divide_line(line, width))
    # ``Text.wrap`` rstrips each divided line, so summing the rendered
    # segment lengths under-counts by exactly the whitespace Rich stripped
    # at each wrap point. Left uncorrected the error accumulates and the
    # final segment runs long -- ``["aaa ", "  bb", "b   ", "ccc   ddd"]``
    # for width 4 -- and ``adjust_cell_length`` then truncates it, dropping
    # text off the end of the document. Skipping the stripped whitespace
    # after each division restores Rich's own offsets, which keep the
    # trailing run attached to the segment that precedes the break.
    lines = Text(line).wrap(_FALLBACK_CONSOLE, width)
    offsets: list[int] = []
    running = 0
    for segment in lines[:-1]:
        running += len(segment.plain)
        skipped = running
        while skipped < len(line) and line[skipped].isspace():
            skipped += 1
        # Rich only folds a whitespace run into the preceding segment when
        # more content follows it; a line that ENDS in whitespace keeps that
        # run as its own trailing segment, so skipping to the end here would
        # break the last offset.
        if skipped < len(line):
            running = skipped
        offsets.append(running)
    return offsets


class WrapIndex:
    """Maps virtual rows to (source line, wrapped segment) at one width.

    Precomputes text wrapping boundaries for efficient virtualization.
    """

    __slots__ = ("_lines", "_width", "_starts", "virtual_height", "_segment_cache")

    _SEGMENT_CACHE_LIMIT = 512

    def __init__(self, lines: Sequence[str], width: int, starts: list[int], height: int):
        """Initialize a wrap index (use :meth:`build` instead).

        Args:
            lines: Sequence of source lines to index.
            width: The wrapping width in cells.
            starts: Precomputed row start positions for each line.
            height: Total virtual height (sum of wrapped rows).
        """
        self._lines = lines
        self._width = width
        self._starts = starts
        self.virtual_height = height
        self._segment_cache: dict[int, tuple[list[str], list[int]]] = {}

    @classmethod
    def build(cls, lines: Sequence[str], width: int) -> "WrapIndex":
        """Build a wrap index for the given lines at a fixed width.

        Args:
            lines: Sequence of source lines to index.
            width: The wrapping width in cells.

        Returns:
            A WrapIndex instance mapping virtual rows to line/segment positions.
        """
        starts: list[int] = []
        running = 0
        for line in lines:
            starts.append(running)
            running += len(divide_source_line(line, width)) + 1
        return cls(lines, width, starts, max(running, 1))

    def row_to_line(self, row: int) -> tuple[int, int]:
        """Map a virtual row to its source line and wrapped segment index.

        Args:
            row: The virtual row number (0-based, spans wrapped segments).

        Returns:
            A tuple (line_index, segment_index) identifying which source line
            and which wrapped segment of that line contains the virtual row.
        """
        line_index = bisect_right(self._starts, row) - 1
        if line_index < 0:
            return (0, 0)
        return (line_index, row - self._starts[line_index])

    def line_start_row(self, line_index: int) -> int:
        """Return the virtual row where the given source line begins.

        Args:
            line_index: The index of a source line (0-based).

        Returns:
            The virtual row number where that source line's first segment starts.
        """
        if not self._starts:
            return 0
        clamped = max(0, min(line_index, len(self._starts) - 1))
        return self._starts[clamped]

    def segments(self, line_index: int) -> list[str]:
        """Return the wrapped segments of a source line.

        Segments are cached up to a bounded size.

        Args:
            line_index: The index of a source line (0-based).

        Returns:
            A list of strings, one per wrapped segment of the source line.
        """
        return self._segments_and_starts(line_index)[0]

    def segment_start(self, line_index: int, segment_index: int) -> int:
        """Return a wrapped segment's start offset within its source line.

        Reads a precomputed prefix sum rather than re-adding the lengths of
        every preceding segment. Summing on demand made both callers
        quadratic in the segment index: copying a 2.5 MB single-line
        document took 5.7 s, and highlighting a row 20,000 segments deep
        cost 0.368 ms against 0.010 ms at the top of the same line.

        Args:
            line_index: The index of a source line (0-based).
            segment_index: The index of a wrapped segment within that line.

        Returns:
            The 0-based character offset at which the segment begins inside
            its source line, clamped to the line's own segment range.
        """
        starts = self._segments_and_starts(line_index)[1]
        if not starts:
            return 0
        clamped = max(0, min(segment_index, len(starts) - 1))
        return starts[clamped]

    def _segments_and_starts(self, line_index: int) -> tuple[list[str], list[int]]:
        """Return a source line's wrapped segments and their start offsets.

        Args:
            line_index: The index of a source line (0-based).

        Returns:
            A tuple of the segment strings and their start offsets within
            the source line; both lists are the same length.
        """
        cached = self._segment_cache.get(line_index)
        if cached is not None:
            return cached
        line = self._lines[line_index]
        breaks = divide_source_line(line, self._width)
        segments: list[str] = []
        starts: list[int] = []
        start = 0
        for offset in (*breaks, len(line)):
            starts.append(start)
            segments.append(line[start:offset])
            start = offset
        if not segments:
            segments = [""]
            starts = [0]
        # Bounded: one pathological 500k-character line costs ~9.4 ms per
        # divide_line call, which render_line would otherwise pay per row.
        if len(self._segment_cache) >= self._SEGMENT_CACHE_LIMIT:
            self._segment_cache.clear()
        self._segment_cache[line_index] = (segments, starts)
        return (segments, starts)
