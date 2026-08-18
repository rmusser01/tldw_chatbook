"""Console transcript text-selection state (TASK: console selection phase 1).

Pure logic, no Textual imports: the transcript widget feeds it mouse events,
row widgets render whatever it produces. Selection domain is a row's
displayed plain text; single-row only (spec 2026-08-14 §1).
"""

from __future__ import annotations

from dataclasses import dataclass

SELECTION_QUOTE_CAP = 4000
_TRUNCATION_MARKER = "\n… [truncated]"


@dataclass(frozen=True)
class TextSelection:
    row_key: str
    start: int
    end: int

    @property
    def is_empty(self) -> bool:
        return self.end <= self.start


@dataclass(frozen=True)
class SelectionState:
    active: bool
    selection: TextSelection | None


class SelectionManager:
    """Tracks one in-progress or finished selection on a single row."""

    def __init__(self) -> None:
        self._origin_row: str | None = None
        self._origin_offset: int = 0
        self._current_offset: int = 0
        self._active: bool = False
        self._finished: TextSelection | None = None
        self._just_finished: bool = False
        self._release_click_pending: bool = False

    @property
    def state(self) -> SelectionState:
        selection = None
        if self._active or self._finished is not None:
            start, end = sorted((self._origin_offset, self._current_offset))
            selection = TextSelection(row_key=self._origin_row or "", start=start, end=end)
        return SelectionState(active=self._active, selection=selection)

    @property
    def just_finished(self) -> bool:
        return self._just_finished

    def consume_just_finished(self) -> None:
        self._just_finished = False

    def consume_release_click(self) -> bool:
        """Consume the pending drag-release click, if any.

        Live-spike round 3: Textual synthesizes the release Click for a
        drag LATE -- it can dispatch after an intervening press already
        consumed ``just_finished``. This one-shot token is consumed ONLY
        by the row-click suppression, so the release Click is suppressed
        exactly once no matter when it arrives.
        """
        pending = self._release_click_pending
        self._release_click_pending = False
        return pending

    def begin_drag(self, row_key: str, offset: int) -> None:
        self._release_click_pending = False
        self._origin_row = row_key
        self._origin_offset = max(0, offset)
        self._current_offset = self._origin_offset
        self._active = True
        self._finished = None

    def extend_drag(self, row_key: str, offset: int) -> None:
        if not self._active or row_key != self._origin_row:
            return  # cross-row drags clamp to the origin row
        self._current_offset = max(0, offset)

    def finish_drag(self) -> TextSelection | None:
        if not self._active:
            return None
        state = self.state
        self._active = False
        self._finished = None if state.selection is None or state.selection.is_empty else state.selection
        self._just_finished = True
        if self._finished is not None:
            self._release_click_pending = True
        return self._finished

    def cancel(self) -> None:
        self._origin_row = None
        self._active = False
        self._finished = None
        self._just_finished = False


def cap_quote(text: str) -> str:
    if len(text) <= SELECTION_QUOTE_CAP:
        return text
    return text[: SELECTION_QUOTE_CAP - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def offset_for_cell(text: str, cell_x: int) -> int:
    """Map a horizontal cell offset to a character offset on one line.

    v1 maps ``cell_x`` directly to a character offset on the unwrapped line
    (plain rows render unwrapped long lines clipped, so the mapping is
    monotone and clamped to ``[0, len(text)]``).
    """
    return max(0, min(cell_x, len(text)))

# --- keyboard motion helpers (phase 5) -----------------------------------------


def _clamp(text: str, offset: int) -> int:
    """Clamp offset to [0, len(text)]."""
    return max(0, min(offset, len(text)))


def word_forward_offset(text: str, offset: int) -> int:
    """Vim-w: the start of the next word (whitespace-delimited), else end."""
    i = _clamp(text, offset)
    n = len(text)
    while i < n and not text[i].isspace():
        i += 1
    while i < n and text[i].isspace():
        i += 1
    return i


def word_back_offset(text: str, offset: int) -> int:
    """Vim-b: the start of the previous word, else 0."""
    i = _clamp(text, offset)
    while i > 0 and text[i - 1].isspace():
        i -= 1
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return i


def line_start_offset(text: str, offset: int) -> int:
    """Vim-0: start of the line containing ``offset``."""
    i = _clamp(text, offset)
    return text.rfind("\n", 0, i) + 1


def line_end_offset(text: str, offset: int) -> int:
    """Vim-$: end of the line containing ``offset`` (before its newline)."""
    i = _clamp(text, offset)
    nl = text.find("\n", i)
    return len(text) if nl == -1 else nl


def next_line_offset(text: str, offset: int) -> int:
    """One line down, preserving the column where the next line allows."""
    i = _clamp(text, offset)
    column = i - line_start_offset(text, i)
    end = line_end_offset(text, i)
    if end >= len(text):
        return len(text)
    nstart = end + 1
    return min(nstart + column, line_end_offset(text, nstart))


def prev_line_offset(text: str, offset: int) -> int:
    """One line up, preserving the column where the previous line allows."""
    i = _clamp(text, offset)
    start = line_start_offset(text, i)
    if start == 0:
        return 0
    column = i - start
    pstart = line_start_offset(text, start - 1)
    return min(pstart + column, line_end_offset(text, pstart))
