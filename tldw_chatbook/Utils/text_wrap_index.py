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

    Uses Rich's ``divide_line`` when available. Falls back to the public
    ``Text.wrap`` API if the private API is unavailable.

    Args:
        line: The source text to divide.
        width: The maximum width in cells before wrapping occurs.

    Returns:
        A list of character offsets where ``line`` wraps. Empty if the line
        fits entirely within ``width`` or if ``width`` is invalid.
    """
    if width <= 0 or not line:
        return []
    if _rich_divide_line is not None:
        return list(_rich_divide_line(line, width))
    lines = Text(line).wrap(_FALLBACK_CONSOLE, width)
    offsets: list[int] = []
    running = 0
    for segment in lines[:-1]:
        running += len(segment.plain)
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
        self._segment_cache: dict[int, list[str]] = {}

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
        cached = self._segment_cache.get(line_index)
        if cached is not None:
            return cached
        line = self._lines[line_index]
        breaks = divide_source_line(line, self._width)
        segments: list[str] = []
        start = 0
        for offset in (*breaks, len(line)):
            segments.append(line[start:offset])
            start = offset
        if not segments:
            segments = [""]
        # Bounded: one pathological 500k-character line costs ~9.4 ms per
        # divide_line call, which render_line would otherwise pay per row.
        if len(self._segment_cache) >= self._SEGMENT_CACHE_LIMIT:
            self._segment_cache.clear()
        self._segment_cache[line_index] = segments
        return segments
