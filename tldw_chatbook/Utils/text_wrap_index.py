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

    Uses Rich's ``divide_line`` when available -- it is private, so the
    fallback re-derives the same breaks through the public ``Text.wrap``
    and ``Tests/Utils/test_text_wrap_index.py`` pins that the two agree.
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
    """Maps virtual rows to (source line, wrapped segment) at one width."""

    __slots__ = ("_lines", "_width", "_starts", "virtual_height", "_segment_cache")

    _SEGMENT_CACHE_LIMIT = 512

    def __init__(self, lines: Sequence[str], width: int, starts: list[int], height: int):
        self._lines = lines
        self._width = width
        self._starts = starts
        self.virtual_height = height
        self._segment_cache: dict[int, list[str]] = {}

    @classmethod
    def build(cls, lines: Sequence[str], width: int) -> "WrapIndex":
        starts: list[int] = []
        running = 0
        for line in lines:
            starts.append(running)
            running += len(divide_source_line(line, width)) + 1
        return cls(lines, width, starts, max(running, 1))

    def row_to_line(self, row: int) -> tuple[int, int]:
        line_index = bisect_right(self._starts, row) - 1
        if line_index < 0:
            return (0, 0)
        return (line_index, row - self._starts[line_index])

    def line_start_row(self, line_index: int) -> int:
        if not self._starts:
            return 0
        clamped = max(0, min(line_index, len(self._starts) - 1))
        return self._starts[clamped]

    def segments(self, line_index: int) -> list[str]:
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
