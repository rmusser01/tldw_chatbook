"""Native Console transcript widget."""

from __future__ import annotations

import asyncio
from typing import Iterable

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import Click, Key
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage


CONSOLE_TRANSCRIPT_ACTION_ROW = "Copy | Edit | Save as... | ♻ | ---> | 👍/👎                 🗑"


def _message_role_label(message: ConsoleChatMessage) -> str:
    role = message.role.value if hasattr(message.role, "value") else str(message.role)
    return role.title()


def _message_body(message: ConsoleChatMessage) -> str:
    if message.status in {"streaming", "stopped", "failed"}:
        return f"{message.content} [{message.status}]".strip()
    return message.content


class ConsoleTranscriptMessage(Static):
    """Clickable native Console transcript message row."""

    can_focus = False

    def __init__(self, message: ConsoleChatMessage, *, selected: bool = False) -> None:
        self.message_id = message.id
        classes = "console-transcript-message"
        if selected:
            classes = f"{classes} console-transcript-message-selected"
        super().__init__(
            f"{_message_role_label(message)}\n{_message_body(message)}",
            id=f"console-message-{message.id}",
            classes=classes,
        )

    def on_click(self, event: Click) -> None:
        event.stop()
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
            transcript.select_message(self.message_id)


class ConsoleTranscript(VerticalScroll):
    """Focusable native Console transcript with compact rule-separated messages."""

    can_focus = True
    BINDINGS = [
        ("down,j", "select_next", "Next message"),
        ("up,k", "select_previous", "Previous message"),
        ("enter", "confirm_selection", "Show actions"),
        ("escape", "clear_selection", "Clear selection"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._messages: list[ConsoleChatMessage] = []
        self.selected_message_id: str | None = None
        self._refresh_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield from self._message_widgets()

    def set_messages(self, messages: Iterable[ConsoleChatMessage]) -> None:
        """Replace transcript messages and refresh mounted rows when possible."""
        self._messages = list(messages)
        message_ids = {message.id for message in self._messages}
        if self.selected_message_id not in message_ids:
            self.selected_message_id = None

    async def refresh_messages(self) -> None:
        """Rebuild mounted message rows from the current transcript state."""
        async with self._refresh_lock:
            await self.remove_children()
            for widget in self._message_widgets():
                await self.mount(widget)

    def select_message(self, message_id: str) -> None:
        """Select one message and show its contextual action row."""
        if message_id not in {message.id for message in self._messages}:
            return
        self.selected_message_id = message_id
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def to_plain_text(self, width: int = 80) -> str:
        """Return a terminal-readable transcript rendering for tests and exports."""
        rule = "─" * max(1, width)
        lines: list[str] = []
        for message in self._messages:
            lines.extend(
                [
                    rule,
                    _message_role_label(message),
                    _message_body(message),
                ]
            )
            if message.id == self.selected_message_id:
                lines.append(CONSOLE_TRANSCRIPT_ACTION_ROW)
        if self._messages:
            lines.append(rule)
        return "\n".join(lines)

    def action_select_next(self) -> None:
        self._select_relative(1)

    def action_select_previous(self) -> None:
        self._select_relative(-1)

    def action_confirm_selection(self) -> None:
        if self.selected_message_id is None and self._messages:
            self.select_message(self._messages[0].id)

    def action_clear_selection(self) -> None:
        self.selected_message_id = None
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def on_key(self, event: Key) -> None:
        if event.key in {"down", "j"}:
            self.action_select_next()
            event.stop()
        elif event.key in {"up", "k"}:
            self.action_select_previous()
            event.stop()
        elif event.key == "enter":
            self.action_confirm_selection()
            event.stop()
        elif event.key == "escape":
            self.action_clear_selection()
            event.stop()

    def _select_relative(self, offset: int) -> None:
        if not self._messages:
            return
        if self.selected_message_id is None:
            index = 0 if offset >= 0 else len(self._messages) - 1
        else:
            current = next(
                (
                    index
                    for index, message in enumerate(self._messages)
                    if message.id == self.selected_message_id
                ),
                0,
            )
            index = min(max(current + offset, 0), len(self._messages) - 1)
        self.select_message(self._messages[index].id)

    def _message_widgets(self) -> list[Static]:
        widgets: list[Static] = []
        for message in self._messages:
            widgets.append(Static("─" * 72, classes="console-transcript-rule"))
            widgets.append(
                ConsoleTranscriptMessage(
                    message,
                    selected=message.id == self.selected_message_id,
                )
            )
            if message.id == self.selected_message_id:
                widgets.append(
                    Static(
                        CONSOLE_TRANSCRIPT_ACTION_ROW,
                        classes="console-transcript-action-row",
                    )
                )
        if self._messages:
            widgets.append(Static("─" * 72, classes="console-transcript-rule"))
        else:
            widgets.append(
                Static(
                    "Empty transcript\nNo messages yet. Send a prompt or attach context.",
                    classes="console-transcript-empty-state",
                )
            )
        return widgets
