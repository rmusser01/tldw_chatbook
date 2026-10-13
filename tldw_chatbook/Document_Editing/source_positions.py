"""Scalar source coordinates and cooperative pure-engine resource bounds."""

import re
from dataclasses import dataclass, field
from time import monotonic

MAX_BYTES = 1024 * 1024
MAX_DEPTH = 100
MAX_DIAGNOSTICS = 100
DEADLINE_SECONDS = 2.0


def scalar_offset(text: str, line: int, column: int) -> int:
    """Convert zero-based logical line/column to a bounded scalar offset.

    CRLF counts as one line terminator; tabs and combining marks count as
    individual characters. Python parser absolute indices already use scalars.
    """
    starts = [0, *(match.end() for match in re.finditer(r"\r\n|[\r\n]", text))]
    if line >= len(starts):
        return len(text)
    line = max(0, line)
    start = starts[line]
    end = starts[line + 1] if line < len(starts) - 1 else len(text)
    while end > start and text[end - 1] in "\r\n":
        end -= 1
    return min(start + max(0, column), end)


class DocumentError(Exception):
    """A source-free engine refusal; never wrap a parser's exception text."""

    def __init__(self, code: str, message: str, start: int = 0, end: int | None = None):
        super().__init__(message)
        self.code = code
        self.start = start
        self.end = start if end is None else end


@dataclass
class Budget:
    """Cooperative checks only; the worker owner must enforce hard termination."""

    deadline: float = field(default_factory=lambda: monotonic() + DEADLINE_SECONDS)

    def check(self, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            raise DocumentError("resource_limited", "Nesting exceeds the depth limit.")
        if monotonic() >= self.deadline:
            raise DocumentError(
                "resource_limited", "The processing deadline was reached."
            )

    def source(self, text: str) -> None:
        if len(text) > MAX_BYTES or len(text.encode("utf-8")) > MAX_BYTES:
            raise DocumentError(
                "resource_limited", "Source exceeds the UTF-8 size limit."
            )
        self.check()
