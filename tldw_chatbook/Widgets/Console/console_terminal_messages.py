"""Lightweight Console Terminal messages that are safe at first paint."""

from __future__ import annotations

from typing import Literal

from textual.message import Message


class ConsoleTerminalInputRequested(Message):
    """Carry one freshly encoded key event toward the Terminal controller."""

    def __init__(self, data: bytes) -> None:
        super().__init__()
        self.data = data


TerminalAction = Literal[
    "return",
    "open-settings",
    "arm",
    "new",
    "select",
    "rename",
    "focus",
    "close",
    "retry",
    "jump-live",
]


class ConsoleTerminalActionRequested(Message):
    """Request one direct-user Terminal workspace action."""

    def __init__(self, action: TerminalAction, session_id: str | None = None) -> None:
        super().__init__()
        self.action = action
        self.session_id = session_id


class ConsoleTerminalResizeRequested(Message):
    """Report the viewport's final painted allocation to its controller."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        min_columns: int,
        max_columns: int,
        min_rows: int,
        max_rows: int,
    ) -> None:
        super().__init__()
        self.painted_width = width
        self.painted_height = height
        self.clamped = width > max_columns or height > max_rows
        self.columns = min(max_columns, max(min_columns, width or 80))
        self.rows = min(max_rows, max(min_rows, height or 24))
