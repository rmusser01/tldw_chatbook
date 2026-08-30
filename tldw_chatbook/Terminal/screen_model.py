"""Safe immutable projections for persistent terminal output."""

from __future__ import annotations

import codecs
from collections import deque
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Callable

import pyte
from pyte import modes as pyte_modes
from wcwidth import wcwidth, wcswidth

from tldw_chatbook.Terminal.contracts import (
    MAX_COLUMNS,
    MAX_ROWS,
    MAX_SCROLLBACK_BYTES,
    MAX_SCROLLBACK_LINES,
    MIN_COLUMNS,
    MIN_ROWS,
    TerminalReason,
)
from tldw_chatbook.Terminal.protocol_gate import TerminalProtocolGate


MAX_CELL_SCALARS = 32
MAX_CELL_UTF8_BYTES = 256
MAX_CURSOR_SAVEPOINTS = 16
MAX_REPLY_BYTES = 256
MAX_PENDING_REPLY_BYTES = 4 * 1024
STYLE_RUN_ACCOUNTING_BYTES = 32
LINE_ACCOUNTING_BYTES = 16
_OVERFLOW_CELL = "\udfff"

_ALLOWED_PRIVATE_MODES = frozenset((1, 5, 6, 7, 25, 2004))
_ALLOWED_STANDARD_MODES = frozenset((4, 20))
_DEC_ALTERNATE_SCREEN_MODE = 1049
_SAFE_COLOR = re.compile(r"\A[0-9A-Za-z#_-]{1,32}\Z")
_CURSOR_REPLY = re.compile(r"\A\x1b\[[1-9][0-9]{0,4};[1-9][0-9]{0,4}R\Z")


@dataclass(frozen=True, slots=True)
class SafeTerminalStyle:
    """Renderer-safe style values for one terminal run."""

    fg: str = "default"
    bg: str = "default"
    bold: bool = False
    italics: bool = False
    underscore: bool = False
    strikethrough: bool = False
    reverse: bool = False
    blink: bool = False


@dataclass(frozen=True, slots=True)
class SafeTerminalCell:
    """One bounded renderer-safe terminal cell."""

    text: str = ""
    width: int = 0


@dataclass(frozen=True, slots=True)
class SafeTerminalRun:
    """Adjacent safe cells sharing one style."""

    cells: tuple[SafeTerminalCell, ...] = ()
    style: SafeTerminalStyle = SafeTerminalStyle()

    @property
    def text(self) -> str:
        """Return the run's safe text."""
        return "".join(cell.text for cell in self.cells)


@dataclass(frozen=True, slots=True)
class SafeTerminalLine:
    """One run-compressed safe terminal line."""

    runs: tuple[SafeTerminalRun, ...] = ()
    accounted_bytes: int = LINE_ACCOUNTING_BYTES

    @property
    def text(self) -> str:
        """Return the line's safe text."""
        return "".join(run.text for run in self.runs)

    @property
    def column_width(self) -> int:
        """Return the retained terminal-cell width."""
        return sum(cell.width for run in self.runs for cell in run.cells)


@dataclass(frozen=True, slots=True)
class TerminalScreenSnapshot:
    """Immutable screen projection safe for Textual consumers."""

    lines: tuple[SafeTerminalLine, ...]
    scrollback: tuple[SafeTerminalLine, ...] = ()
    cursor_row: int = 1
    cursor_column: int = 1
    cursor_visible: bool = True
    cursor_savepoints: int = 0
    in_alternate: bool = False
    generation: int = 0
    dirty_lines: tuple[int, ...] = ()
    scrollback_bytes: int = 0
    cell_overflow_count: int = 0
    failure_reason: TerminalReason | None = None


class _BoundedScreen(pyte.Screen):
    """pyte screen with bounded mutable state and content-free callbacks."""

    def __init__(
        self,
        columns: int,
        lines: int,
        *,
        on_scroll: Callable[[Any], None] | None,
        on_reply: Callable[[bytes], None],
        on_cell_overflow: Callable[[], None],
    ) -> None:
        self._on_scroll = on_scroll
        self._on_reply = on_reply
        self._on_cell_overflow = on_cell_overflow
        self._join_cell: tuple[int, int, int] | None = None
        super().__init__(columns, lines)

    def reset(self) -> None:
        """Reset the screen and every code-owned mutable extension."""
        super().reset()
        self.savepoints.clear()
        self._join_cell = None

    def save_cursor(self) -> None:
        """Save the cursor while retaining only the newest 16 entries."""
        super().save_cursor()
        if len(self.savepoints) > MAX_CURSOR_SAVEPOINTS:
            del self.savepoints[: len(self.savepoints) - MAX_CURSOR_SAVEPOINTS]

    def index(self) -> None:
        """Capture a line only when the full normal viewport scrolls."""
        top, bottom = self.margins or pyte.screens.Margins(0, self.lines - 1)
        full_viewport_scroll = (
            self.cursor.y == bottom and top == 0 and bottom == self.lines - 1
        )
        if full_viewport_scroll and self._on_scroll is not None:
            self._on_scroll(self.buffer[top])
        super().index()
        self._join_cell = None

    def set_mode(self, *modes: int, **kwargs: Any) -> None:
        """Apply only the finite mode allowlist used by the safe adapter."""
        allowed = (
            _ALLOWED_PRIVATE_MODES
            if kwargs.get("private") is True
            else _ALLOWED_STANDARD_MODES
        )
        retained = tuple(mode for mode in modes if mode in allowed)
        if retained:
            super().set_mode(*retained, **kwargs)

    def reset_mode(self, *modes: int, **kwargs: Any) -> None:
        """Reset only modes from the same finite allowlist."""
        allowed = (
            _ALLOWED_PRIVATE_MODES
            if kwargs.get("private") is True
            else _ALLOWED_STANDARD_MODES
        )
        retained = tuple(mode for mode in modes if mode in allowed)
        if retained:
            super().reset_mode(*retained, **kwargs)

    def draw(self, data: str) -> None:
        """Draw Unicode text while bounding combining and joiner cells."""
        data = data.translate(self.g1_charset if self.charset else self.g0_charset)
        for character in data:
            width = wcwidth(character)
            if width == 0:
                self._append_zero_width(character)
                continue
            if width < 0:
                self._join_cell = None
                continue
            if self._append_joined_character(character):
                continue
            self._draw_advancing_character(character, min(width, 2))
        self.dirty.add(self.cursor.y)

    def _append_zero_width(self, character: str) -> None:
        position = self._previous_base_cell()
        if position is None:
            return
        row, column = position
        line = self.buffer[row]
        previous = line[column]
        if previous.data == _OVERFLOW_CELL:
            return
        value = unicodedata.normalize("NFC", previous.data + character)
        position = self._replace_cell_text(position, value)
        if character == "\u200d":
            self._join_cell = (*position, self.cursor.x)

    def _append_joined_character(self, character: str) -> bool:
        join = self._join_cell
        self._join_cell = None
        if join is None:
            return False
        row, column, expected_cursor = join
        if row != self.cursor.y or expected_cursor != self.cursor.x:
            return False
        position = (row, column)
        line = self.buffer[row]
        previous = line[column]
        if previous.data == _OVERFLOW_CELL:
            return True
        value = unicodedata.normalize("NFC", previous.data + character)
        self._replace_cell_text(position, value)
        return True

    def _draw_advancing_character(self, character: str, width: int) -> None:
        if self.cursor.x == self.columns:
            if pyte_modes.DECAWM in self.mode:
                self.dirty.add(self.cursor.y)
                self.carriage_return()
                self.linefeed()
            else:
                self.cursor.x = max(0, self.cursor.x - width)

        if width == 2 and self.cursor.x == self.columns - 1:
            if pyte_modes.DECAWM in self.mode:
                self.dirty.add(self.cursor.y)
                self.carriage_return()
                self.linefeed()
            else:
                character = "\ufffd"
                width = 1

        if pyte_modes.IRM in self.mode:
            self.insert_characters(width)

        row, column = self.cursor.y, self.cursor.x
        line = self.buffer[row]
        self._clear_intersecting_wide_cells(line, column, width)
        line[column] = self.cursor.attrs._replace(data=character)
        if width == 2 and column + 1 < self.columns:
            line[column + 1] = self.cursor.attrs._replace(data="")
        self.cursor.x = min(self.cursor.x + width, self.columns)
        self._join_cell = None

    def _clear_intersecting_wide_cells(
        self, line: Any, column: int, width: int
    ) -> None:
        """Clear both halves of every wide cell touched by a new write."""
        clear_columns: set[int] = set()
        for target in range(column, min(column + width, self.columns)):
            if line[target].data == "" and target > 0:
                clear_columns.update((target - 1, target))
            elif target + 1 < self.columns and line[target + 1].data == "":
                clear_columns.update((target, target + 1))
        for target in clear_columns:
            line[target] = self.default_char

    def _previous_base_cell(self) -> tuple[int, int] | None:
        row = self.cursor.y
        column = self.cursor.x - 1
        if column < 0:
            return None
        line = self.buffer[row]
        while column >= 0:
            if line[column].data:
                return row, column
            column -= 1
        return None

    def _bounded_cell(self, value: str) -> str:
        if (
            len(value) <= MAX_CELL_SCALARS
            and len(value.encode("utf-8")) <= MAX_CELL_UTF8_BYTES
        ):
            return value
        self._on_cell_overflow()
        return _OVERFLOW_CELL

    def _replace_cell_text(
        self, position: tuple[int, int], value: str
    ) -> tuple[int, int]:
        """Replace one cell and atomically reconcile a grapheme width change."""
        row, column = position
        line = self.buffer[row]
        previous = line[column]
        old_width = (
            2 if column + 1 < self.columns and line[column + 1].data == "" else 1
        )
        bounded = self._bounded_cell(value)
        new_width = 1 if bounded == _OVERFLOW_CELL else max(1, min(wcswidth(value), 2))

        if new_width == 2 and column + 1 >= self.columns:
            if (
                pyte_modes.DECAWM in self.mode
                and row == self.cursor.y
                and self.cursor.x == self.columns
            ):
                line[column] = previous._replace(data=" ")
                self.dirty.add(row)
                self.carriage_return()
                self.linefeed()
                row, column = self.cursor.y, self.cursor.x
                line = self.buffer[row]
                line[column] = previous._replace(data=bounded)
                line[column + 1] = previous._replace(data="")
                self.cursor.x = column + 2
                self.dirty.add(row)
                return row, column
            bounded = "\ufffd"
            new_width = 1

        line[column] = previous._replace(data=bounded)
        if old_width == 1 and new_width == 2:
            line[column + 1] = previous._replace(data="")
        elif old_width == 2 and new_width == 1:
            line[column + 1] = previous._replace(data=" ")
        if row == self.cursor.y and self.cursor.x == column + old_width:
            self.cursor.x = min(column + new_width, self.columns)
        self.dirty.add(row)
        return row, column

    def write_process_input(self, data: str) -> None:
        """Queue only fixed or bounded code-owned replies."""
        if (
            data not in ("\x1b[0n", "\x1b[?6c")
            and _CURSOR_REPLY.fullmatch(data) is None
        ):
            return
        encoded = data.encode("ascii")
        if len(encoded) <= MAX_REPLY_BYTES:
            self._on_reply(encoded)

    def set_title(self, _: str) -> None:
        """Ignore host title changes."""

    def set_icon_name(self, _: str) -> None:
        """Ignore host icon changes."""

    def bell(self) -> None:
        """Ignore host bell requests."""

    def debug(self, *args: Any, **kwargs: Any) -> None:
        """Ignore unsupported operations without retaining their content."""


class _AlternateScreenAdapter:
    """Route DEC 1049 operations between isolated bounded screens."""

    def __init__(
        self,
        columns: int,
        lines: int,
        *,
        on_scroll: Callable[[Any], None],
        on_reply: Callable[[bytes], None],
        on_cell_overflow: Callable[[], None],
    ) -> None:
        self.primary = _BoundedScreen(
            columns,
            lines,
            on_scroll=on_scroll,
            on_reply=on_reply,
            on_cell_overflow=on_cell_overflow,
        )
        self.alternate = _BoundedScreen(
            columns,
            lines,
            on_scroll=None,
            on_reply=on_reply,
            on_cell_overflow=on_cell_overflow,
        )
        self.active = self.primary
        self.in_alternate = False

    def set_mode(self, *modes: int, **kwargs: Any) -> None:
        """Enter alternate screen for private DEC mode 1049."""
        private = kwargs.get("private") is True
        remaining = tuple(
            mode
            for mode in modes
            if not (private and mode == _DEC_ALTERNATE_SCREEN_MODE)
        )
        if private and _DEC_ALTERNATE_SCREEN_MODE in modes and not self.in_alternate:
            self.primary.save_cursor()
            self.alternate.reset()
            self.active = self.alternate
            self.in_alternate = True
            self.active.dirty.update(range(self.active.lines))
        if remaining:
            self.active.set_mode(*remaining, **kwargs)

    def reset_mode(self, *modes: int, **kwargs: Any) -> None:
        """Leave alternate screen for private DEC mode 1049."""
        private = kwargs.get("private") is True
        remaining = tuple(
            mode
            for mode in modes
            if not (private and mode == _DEC_ALTERNATE_SCREEN_MODE)
        )
        if remaining:
            self.active.reset_mode(*remaining, **kwargs)
        if private and _DEC_ALTERNATE_SCREEN_MODE in modes and self.in_alternate:
            self.active = self.primary
            self.primary.restore_cursor()
            self.in_alternate = False
            self.active.dirty.update(range(self.active.lines))

    def reset(self) -> None:
        """Reset both buffers and leave alternate-screen mode coherently."""
        self.primary.reset()
        self.alternate.reset()
        self.active = self.primary
        self.in_alternate = False

    def resize(self, *, columns: int, rows: int) -> None:
        """Resize both isolated buffers while preserving their cursor state."""
        for screen in (self.primary, self.alternate):
            screen.resize(lines=rows, columns=columns)
            screen._join_cell = None
            screen.dirty.update(range(rows))

    def _forward(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name != "draw":
            self.active._join_cell = None
        return getattr(self.active, name)(*args, **kwargs)


def _install_dynamic_screen_forwarders() -> None:
    """Install forwarding methods before pyte statically binds its events."""

    def make_forwarder(name: str) -> Callable[..., Any]:
        def forward(self: _AlternateScreenAdapter, *args: Any, **kwargs: Any) -> Any:
            return self._forward(name, *args, **kwargs)

        forward.__name__ = name
        return forward

    for event in pyte.Stream.events - {"set_mode", "reset_mode"}:
        if not hasattr(_AlternateScreenAdapter, event):
            setattr(_AlternateScreenAdapter, event, make_forwarder(event))


_install_dynamic_screen_forwarders()


class TerminalScreenModel:
    """Own bounded parser state and expose only safe immutable values."""

    def __init__(
        self,
        *,
        columns: int,
        rows: int,
        scrollback_line_limit: int = MAX_SCROLLBACK_LINES,
        scrollback_byte_limit: int = MAX_SCROLLBACK_BYTES,
    ) -> None:
        if not MIN_COLUMNS <= columns <= MAX_COLUMNS:
            raise ValueError("terminal columns outside contract")
        if not MIN_ROWS <= rows <= MAX_ROWS:
            raise ValueError("terminal rows outside contract")
        if not 0 <= scrollback_line_limit <= MAX_SCROLLBACK_LINES:
            raise ValueError("scrollback line limit outside contract")
        if not 0 <= scrollback_byte_limit <= MAX_SCROLLBACK_BYTES:
            raise ValueError("scrollback byte limit outside contract")

        self.columns = columns
        self.rows = rows
        self._scrollback_line_limit = scrollback_line_limit
        self._scrollback_byte_limit = scrollback_byte_limit
        self._scrollback: deque[SafeTerminalLine] = deque()
        self._scrollback_bytes = 0
        self._pending_replies: deque[bytes] = deque()
        self._pending_reply_bytes = 0
        self._cell_overflow_count = 0
        self._failure_reason: TerminalReason | None = None
        self._generation = 0
        self._gate = TerminalProtocolGate()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._screens = _AlternateScreenAdapter(
            columns,
            rows,
            on_scroll=self._retain_scrollback_line,
            on_reply=self._queue_reply,
            on_cell_overflow=self._count_cell_overflow,
        )
        self._stream = pyte.Stream(self._screens)

    def feed(self, data: bytes) -> None:
        """Feed one bounded terminal-output chunk."""
        if self._failure_reason is not None:
            return
        admitted = self._gate.feed(data)
        if not admitted:
            return
        decoded = self._decoder.decode(admitted, final=False)
        if decoded:
            self._feed_parser(decoded)

    def finish(self) -> None:
        """Finalize incremental decoding and discard incomplete controls."""
        if self._failure_reason is not None:
            return
        self._gate.finish()
        decoded = self._decoder.decode(b"", final=True)
        if decoded:
            self._feed_parser(decoded)

    def resize(self, *, columns: int, rows: int) -> None:
        """Resize both terminal buffers within the persistent-session bounds.

        Args:
            columns: New viewport width in terminal cells.
            rows: New viewport height in terminal rows.

        Raises:
            ValueError: If either dimension is outside the terminal contract.
        """
        if not MIN_COLUMNS <= columns <= MAX_COLUMNS:
            raise ValueError("terminal columns outside contract")
        if not MIN_ROWS <= rows <= MAX_ROWS:
            raise ValueError("terminal rows outside contract")
        self._screens.resize(columns=columns, rows=rows)
        self.columns = columns
        self.rows = rows
        self._generation += 1

    def snapshot(self) -> TerminalScreenSnapshot:
        """Return an immutable renderer-safe screen projection."""
        active = self._screens.active
        lines = tuple(
            self._project_line(active.buffer[row]) for row in range(self.rows)
        )
        return TerminalScreenSnapshot(
            lines=lines,
            scrollback=tuple(self._scrollback),
            cursor_row=active.cursor.y + 1,
            cursor_column=active.cursor.x + 1,
            cursor_visible=not active.cursor.hidden,
            cursor_savepoints=len(active.savepoints),
            in_alternate=self._screens.in_alternate,
            generation=self._generation,
            dirty_lines=tuple(sorted(row + 1 for row in active.dirty)),
            scrollback_bytes=self._scrollback_bytes,
            cell_overflow_count=self._cell_overflow_count,
            failure_reason=self._failure_reason,
        )

    def visible_text(self) -> str:
        """Return visible safe text without trailing blank rows."""
        lines = [line.text for line in self.snapshot().lines]
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def pending_replies(self) -> tuple[bytes, ...]:
        """Return queued code-owned terminal replies."""
        return tuple(self._pending_replies)

    def take_pending_replies(self) -> tuple[bytes, ...]:
        """Take queued code-owned terminal replies."""
        replies = tuple(self._pending_replies)
        self._pending_replies.clear()
        self._pending_reply_bytes = 0
        return replies

    def _feed_parser(self, decoded: str) -> None:
        try:
            self._screens.active.dirty.clear()
            self._stream.feed(decoded)
        except Exception:
            # Parser exceptions may contain terminal output. Never copy or log
            # them; the category is the complete diagnostic contract.
            self._failure_reason = TerminalReason.TERMINAL_PROTOCOL_FAILED
            return
        self._generation += 1

    def _retain_scrollback_line(self, line: Any) -> None:
        retained = self._project_line(line)
        self._scrollback.append(retained)
        self._scrollback_bytes += retained.accounted_bytes
        while self._scrollback and (
            len(self._scrollback) > self._scrollback_line_limit
            or self._scrollback_bytes > self._scrollback_byte_limit
        ):
            evicted = self._scrollback.popleft()
            self._scrollback_bytes -= evicted.accounted_bytes

    def _count_cell_overflow(self) -> None:
        self._cell_overflow_count += 1

    def _queue_reply(self, reply: bytes) -> None:
        if self._pending_reply_bytes + len(reply) > MAX_PENDING_REPLY_BYTES:
            return
        self._pending_replies.append(reply)
        self._pending_reply_bytes += len(reply)

    def _project_line(self, line: Any) -> SafeTerminalLine:
        last_column = -1
        for column in range(self.columns):
            character = line[column]
            if character.data != "" and (
                character.data != " " or _safe_style(character) != SafeTerminalStyle()
            ):
                last_column = column
        if last_column < 0:
            return SafeTerminalLine()

        runs: list[SafeTerminalRun] = []
        run_cells: list[SafeTerminalCell] = []
        run_style: SafeTerminalStyle | None = None
        column = 0
        while column <= last_column:
            character = line[column]
            if character.data == "":
                column += 1
                continue
            text = _safe_cell_text(character.data)
            width = (
                2 if column + 1 < self.columns and line[column + 1].data == "" else 1
            )
            if width == 1 and wcswidth(text) > 1:
                text = "\ufffd"
            cell = SafeTerminalCell(text=text, width=width)
            style = _safe_style(character)
            if run_style is not None and style != run_style:
                runs.append(SafeTerminalRun(cells=tuple(run_cells), style=run_style))
                run_cells = []
            run_style = style
            run_cells.append(cell)
            column += width

        if run_style is not None:
            runs.append(SafeTerminalRun(cells=tuple(run_cells), style=run_style))
        accounted_bytes = (
            sum(len(run.text.encode("utf-8")) for run in runs)
            + len(runs) * STYLE_RUN_ACCOUNTING_BYTES
            + LINE_ACCOUNTING_BYTES
        )
        return SafeTerminalLine(runs=tuple(runs), accounted_bytes=accounted_bytes)


def _safe_cell_text(value: str) -> str:
    """Strip renderer-active controls from a parser-owned cell value."""
    safe = "".join(
        character
        if not (
            unicodedata.category(character) in {"Cc", "Cs"}
            or 0x80 <= ord(character) <= 0x9F
        )
        else "\ufffd"
        for character in value
    )
    return unicodedata.normalize("NFC", safe)


def _safe_style(character: Any) -> SafeTerminalStyle:
    """Copy a pyte character style into primitive immutable values."""

    def color(value: Any) -> str:
        candidate = str(value)
        return candidate if _SAFE_COLOR.fullmatch(candidate) is not None else "default"

    return SafeTerminalStyle(
        fg=color(character.fg),
        bg=color(character.bg),
        bold=bool(character.bold),
        italics=bool(character.italics),
        underscore=bool(character.underscore),
        strikethrough=bool(character.strikethrough),
        reverse=bool(character.reverse),
        blink=bool(character.blink),
    )
