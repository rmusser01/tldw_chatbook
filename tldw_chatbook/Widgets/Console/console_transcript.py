"""Native Console transcript widget."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import Click, Key
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageAction, ConsoleMessageActionService


CONSOLE_TRANSCRIPT_RULE = "─" * 200
EMPTY_TRANSCRIPT_COPY = "Ready. Ask a question, run a command, or attach context."
_ACTION_TOOLTIPS = {
    "copy": "Copy this message to the clipboard.",
    "edit": "Edit this message before continuing the thread.",
    "save-as": "Choose a destination for this message, such as Chatbook or Note.",
    "retry": "Retry the failed response.",
    "regenerate": "Generate another assistant variant for this turn.",
    "continue": "Continue and extend the selected message.",
    "feedback-up": "Mark this response as helpful.",
    "feedback-down": "Mark this response as not helpful.",
    "delete": "Delete this message from the Console transcript.",
    "variant-previous": "Show the previous regenerated variant.",
    "variant-next": "Show the next regenerated variant.",
}


def _message_role_label(message: ConsoleChatMessage) -> str:
    role = message.role.value if hasattr(message.role, "value") else str(message.role)
    return role.title()


def _message_body(message: ConsoleChatMessage) -> str:
    if message.variants is not None:
        content = message.variants.current.content
    else:
        content = message.content
    if message.status in {"streaming", "stopped", "failed"}:
        return f"{content} [{message.status}]".strip()
    return content


@dataclass(frozen=True)
class _TranscriptRow:
    key: str
    kind: Literal["rule", "message", "actions", "empty"]
    signature: tuple
    message: ConsoleChatMessage | None = None
    selected: bool = False
    renderable: str = ""


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

    def sync_message(self, message: ConsoleChatMessage, *, selected: bool = False) -> None:
        """Update row content and selection styling without remounting the row."""
        self.message_id = message.id
        self.update(f"{_message_role_label(message)}\n{_message_body(message)}")
        if selected:
            self.add_class("console-transcript-message-selected")
        else:
            self.remove_class("console-transcript-message-selected")

    def on_click(self, event: Click) -> None:
        event.stop()
        transcript = self.parent
        while transcript is not None and not isinstance(transcript, ConsoleTranscript):
            transcript = transcript.parent
        if isinstance(transcript, ConsoleTranscript):
            transcript.select_message(self.message_id)


class ConsoleTranscriptActionButton(Button):
    """Message action button that supports Enter activation in transcript focus mode."""

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            self.action_activate_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "tab":
            self.action_focus_next_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "shift+tab":
            self.action_focus_previous_action()
            event.stop()
            event.prevent_default()
            return
        if event.key == "escape":
            self.action_clear_message_selection()
            event.stop()
            event.prevent_default()

    def action_activate_action(self) -> None:
        """Activate the focused message action."""
        self.press()

    def action_focus_next_action(self) -> None:
        """Move focus to the next visible action in the selected-message row."""
        self._focus_relative_action(1)

    def action_focus_previous_action(self) -> None:
        """Move focus to the previous visible action in the selected-message row."""
        self._focus_relative_action(-1)

    def action_clear_message_selection(self) -> None:
        """Clear the transcript selection from a focused action button."""
        transcript = self._parent_transcript()
        if transcript is not None:
            transcript.action_clear_selection()

    def _parent_transcript(self) -> ConsoleTranscript | None:
        parent = self.parent
        while parent is not None and not isinstance(parent, ConsoleTranscript):
            parent = parent.parent
        return parent if isinstance(parent, ConsoleTranscript) else None

    def _focus_relative_action(self, offset: int) -> None:
        parent = self.parent
        if parent is None:
            return
        action_buttons = [
            child
            for child in parent.children
            if isinstance(child, ConsoleTranscriptActionButton) and not child.disabled
        ]
        if not action_buttons:
            return
        try:
            current_index = action_buttons.index(self)
        except ValueError:
            return
        action_buttons[(current_index + offset) % len(action_buttons)].focus()


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
        self.empty_state_copy = EMPTY_TRANSCRIPT_COPY
        self._row_widgets: dict[str, Widget] = {}
        self._row_signatures: dict[str, tuple] = {}
        self._row_build_counts: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        self._row_widgets.clear()
        self._row_signatures.clear()
        self._row_build_counts.clear()
        for row in self._transcript_rows():
            widget = self._build_row_widget(row, track=True)
            self._row_widgets[row.key] = widget
            self._row_signatures[row.key] = row.signature
            yield widget

    def set_messages(self, messages: Iterable[ConsoleChatMessage]) -> None:
        """Replace transcript messages and refresh mounted rows when possible."""
        self._messages = list(messages)
        message_ids = {message.id for message in self._messages}
        if self.selected_message_id not in message_ids:
            self.selected_message_id = None

    def sync_empty_state(self, copy: str = "") -> None:
        """Refresh the empty transcript copy while preserving message exports."""
        next_copy = copy.strip() or EMPTY_TRANSCRIPT_COPY
        if self.empty_state_copy == next_copy:
            return
        self.empty_state_copy = next_copy
        if self.is_mounted and not self._messages:
            self.call_later(self.refresh_messages)

    async def refresh_messages(self) -> None:
        """Reconcile mounted message rows from the current transcript state."""
        async with self._refresh_lock:
            await self._reconcile_rows(self._transcript_rows())

    def row_build_counts(self) -> dict[str, int]:
        """Return row build counts for focused reconciliation tests."""
        return dict(self._row_build_counts)

    def row_render_signatures(self) -> dict[str, tuple]:
        """Return active row signatures for focused reconciliation tests."""
        return dict(self._row_signatures)

    def select_message(self, message_id: str) -> None:
        """Select one message and show its contextual action row."""
        if message_id not in {message.id for message in self._messages}:
            return
        self.selected_message_id = message_id
        if self.is_mounted:
            self.call_later(self.refresh_messages)
            self.call_later(self._notify_selection_changed)

    def focus_action(self, message_id: str, action_id: str) -> None:
        """Focus a selected-message action button by message/action ID."""
        if self.selected_message_id != message_id:
            self.select_message(message_id)
        self.call_later(self._focus_action_button, message_id, action_id)

    def select_next_variant(self, message_id: str) -> None:
        """Select the next rendered variant for a message when available."""
        message = self._message_by_id(message_id)
        if message is None or message.variants is None or not message.variants.can_go_next:
            return
        message.variants.selected_index += 1
        if self.is_mounted:
            self.call_later(self.refresh_messages)

    def select_previous_variant(self, message_id: str) -> None:
        """Select the previous rendered variant for a message when available."""
        message = self._message_by_id(message_id)
        if message is None or message.variants is None or not message.variants.can_go_previous:
            return
        message.variants.selected_index -= 1
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
                lines.append(self._plain_action_row(message))
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
            self.call_later(self._notify_selection_changed)

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

    def _message_by_id(self, message_id: str) -> ConsoleChatMessage | None:
        return next((message for message in self._messages if message.id == message_id), None)

    def _notify_selection_changed(self) -> None:
        """Let the owning screen refresh inspector/control surfaces after selection changes."""
        sync_console_control_bar = getattr(self.screen, "_sync_console_control_bar", None)
        if callable(sync_console_control_bar):
            sync_console_control_bar()

    def _transcript_rows(self) -> list[_TranscriptRow]:
        rows: list[_TranscriptRow] = []
        for message in self._messages:
            selected = message.id == self.selected_message_id
            rows.append(
                _TranscriptRow(
                    key=f"rule:{message.id}",
                    kind="rule",
                    signature=("rule", message.id),
                    renderable=CONSOLE_TRANSCRIPT_RULE,
                )
            )
            rows.append(
                _TranscriptRow(
                    key=f"message:{message.id}",
                    kind="message",
                    signature=self._message_row_signature(message, selected=selected),
                    message=message,
                    selected=selected,
                )
            )
            if selected:
                rows.append(
                    _TranscriptRow(
                        key=f"actions:{message.id}",
                        kind="actions",
                        signature=self._action_row_signature(message),
                        message=message,
                    )
                )
        if self._messages:
            rows.append(
                _TranscriptRow(
                    key="rule:end",
                    kind="rule",
                    signature=("rule", "end"),
                    renderable=CONSOLE_TRANSCRIPT_RULE,
                )
            )
        else:
            rows.append(
                _TranscriptRow(
                    key="empty",
                    kind="empty",
                    signature=("empty", self.empty_state_copy),
                    renderable=self.empty_state_copy,
                )
            )
        return rows

    def _message_widgets(self) -> list[Widget]:
        return [self._build_row_widget(row, track=False) for row in self._transcript_rows()]

    async def _reconcile_rows(self, rows: list[_TranscriptRow]) -> None:
        desired_keys = [row.key for row in rows]
        desired_key_set = set(desired_keys)

        for stale_key in [key for key in self._row_widgets if key not in desired_key_set]:
            stale_widget = self._row_widgets.pop(stale_key)
            self._row_signatures.pop(stale_key, None)
            self._row_build_counts.pop(stale_key, None)
            await stale_widget.remove()

        previous_widget: Widget | None = None
        for index, row in enumerate(rows):
            widget = self._row_widgets.get(row.key)
            row_was_mounted = False
            if widget is None:
                widget = self._build_row_widget(row, track=True)
                if previous_widget is None:
                    await self.mount(widget, before=0 if self.children else None)
                else:
                    await self.mount(widget, after=previous_widget)
                row_was_mounted = True
                self._row_widgets[row.key] = widget
                self._row_signatures[row.key] = row.signature
            elif self._row_signatures.get(row.key) != row.signature:
                updated_widget = self._update_row_widget(widget, row)
                if updated_widget is widget:
                    self._row_signatures[row.key] = row.signature
                else:
                    await widget.remove()
                    widget = updated_widget
                    if previous_widget is None:
                        await self.mount(widget, before=0 if self.children else None)
                    else:
                        await self.mount(widget, after=previous_widget)
                    row_was_mounted = True
                    self._row_widgets[row.key] = widget
                    self._row_signatures[row.key] = row.signature

            if not row_was_mounted:
                if previous_widget is None:
                    self.move_child(widget, before=0)
                else:
                    self.move_child(widget, after=previous_widget)
            previous_widget = widget

    def _build_row_widget(self, row: _TranscriptRow, *, track: bool) -> Widget:
        if track:
            self._row_build_counts[row.key] = self._row_build_counts.get(row.key, 0) + 1
        if row.kind == "rule":
            return Static(
                row.renderable,
                id=self._row_widget_id(row),
                classes="console-transcript-rule",
            )
        if row.kind == "empty":
            return Static(
                row.renderable,
                id=self._row_widget_id(row),
                classes="console-transcript-empty-state",
            )
        if row.kind == "message" and row.message is not None:
            return ConsoleTranscriptMessage(row.message, selected=row.selected)
        if row.kind == "actions" and row.message is not None:
            return self._action_row(row.message)
        raise ValueError(f"Unsupported transcript row: {row}")

    def _update_row_widget(self, widget: Widget, row: _TranscriptRow) -> Widget:
        if row.kind == "message" and row.message is not None and isinstance(widget, ConsoleTranscriptMessage):
            widget.sync_message(row.message, selected=row.selected)
            return widget
        if row.kind == "empty" and isinstance(widget, Static):
            widget.update(row.renderable)
            return widget
        return self._build_row_widget(row, track=True)

    @staticmethod
    def _row_widget_id(row: _TranscriptRow) -> str:
        return "console-transcript-row-" + row.key.replace(":", "-")

    @staticmethod
    def _message_row_signature(message: ConsoleChatMessage, *, selected: bool) -> tuple:
        variants_signature = None
        if message.variants is not None:
            variants_signature = (
                message.variants.selected_index,
                tuple(variant.id for variant in message.variants.variants),
            )
        return (
            "message",
            _message_role_label(message),
            _message_body(message),
            message.status,
            selected,
            variants_signature,
        )

    @staticmethod
    def _action_row_signature(message: ConsoleChatMessage) -> tuple:
        actions = []
        for action in ConsoleMessageActionService().available_actions(message):
            if action.action_id == "feedback":
                actions.append(("feedback-up", "👍", True, ""))
                actions.append(("feedback-down", "👎", True, ""))
                continue
            actions.append(
                (
                    action.action_id,
                    action.label,
                    action.enabled,
                    action.disabled_reason or "",
                )
            )
        return ("actions", message.id, tuple(actions))

    def _action_row(self, message: ConsoleChatMessage) -> Horizontal:
        buttons: list[Button] = []
        for action in ConsoleMessageActionService().available_actions(message):
            if action.action_id == "feedback":
                buttons.append(self._action_button(message, ConsoleMessageAction("feedback-up", "👍")))
                buttons.append(self._action_button(message, ConsoleMessageAction("feedback-down", "👎")))
                continue
            buttons.append(self._action_button(message, action))
        return Horizontal(*buttons, id=f"console-message-actions-{message.id}", classes="console-transcript-action-row")

    @staticmethod
    def _plain_action_row(message: ConsoleChatMessage) -> str:
        return ConsoleMessageActionService().plain_action_row(message)

    @staticmethod
    def _action_button(message: ConsoleChatMessage, action: ConsoleMessageAction) -> Button:
        button = ConsoleTranscriptActionButton(
            action.label,
            id=f"console-message-action-{action.action_id}-{message.id}",
            classes="console-transcript-action-button",
            disabled=not action.enabled,
        )
        if action.disabled_reason:
            button.tooltip = action.disabled_reason
        else:
            button.tooltip = _ACTION_TOOLTIPS.get(action.action_id)
        return button

    def _focus_action_button(self, message_id: str, action_id: str) -> None:
        try:
            self.query_one(f"#console-message-action-{action_id}-{message_id}", Button).focus()
        except Exception:
            return
