"""Native Console transcript widget."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click, Key
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageAction, ConsoleMessageActionService
from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_SETUP_CARD_TITLE,
    ConsoleSetupCardState,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested


CONSOLE_TRANSCRIPT_RULE = "─" * 200
EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL = "Choose model"
EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP = (
    "Choose the provider and model for this Console session."
)
SELECTED_MESSAGE_ACTION_GUIDE = (
    "Guide: ♻ Regenerate  ---> Continue  👍/👎 Rate  🗑 Delete"
)
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


def _coerce_card_state(value: object) -> ConsoleSetupCardState:
    """Guard against a transiently non-``ConsoleSetupCardState`` value.

    A flaky resume race can hand the empty panel a stale/incorrect value
    (observed as a bare ``str``) before the real card state lands. Fall back
    to the quiet copy rather than raising ``AttributeError`` deep in
    ``compose()``.
    """
    if isinstance(value, ConsoleSetupCardState):
        return value
    return ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)


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


def _message_render_text(message: ConsoleChatMessage, *, selected: bool) -> str:
    """Return compact transcript copy for the visible message row."""
    role_label = _message_role_label(message)
    body = _message_body(message)
    if not selected and "\n" not in body and len(body) <= 120:
        return f"{role_label}  {body}"
    return f"{role_label}\n{body}"


@dataclass(frozen=True)
class _TranscriptRow:
    key: str
    kind: Literal["rule", "message", "actions", "action-help", "empty"]
    signature: tuple
    message: ConsoleChatMessage | None = None
    selected: bool = False
    renderable: str = ""
    action_label: str = EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL
    action_tooltip: str = EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP
    card_state: ConsoleSetupCardState | None = None


class ConsoleTranscriptMessage(Static):
    """Clickable native Console transcript message row."""

    can_focus = False

    def __init__(self, message: ConsoleChatMessage, *, selected: bool = False) -> None:
        self.message_id = message.id
        classes = "console-transcript-message"
        if selected:
            classes = f"{classes} console-transcript-message-selected"
        super().__init__(
            _message_render_text(message, selected=selected),
            id=f"console-message-{message.id}",
            classes=classes,
        )

    def sync_message(self, message: ConsoleChatMessage, *, selected: bool = False) -> None:
        """Update row content and selection styling without remounting the row."""
        self.message_id = message.id
        self.update(_message_render_text(message, selected=selected))
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
        """Activate the focused message action.

        Presses the currently focused transcript action button.
        """
        self.press()

    def action_focus_next_action(self) -> None:
        """Move focus to the next visible action.

        Advances focus within the selected-message action row.
        """
        self._focus_relative_action(1)

    def action_focus_previous_action(self) -> None:
        """Move focus to the previous visible action.

        Moves focus backward within the selected-message action row.
        """
        self._focus_relative_action(-1)

    def action_clear_message_selection(self) -> None:
        """Clear transcript selection from a focused action button.

        Delegates to the parent transcript when the action row owns focus.
        """
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


class ConsoleTranscriptEmptyAction(Button):
    """Activation button shown when the Console transcript has no messages."""

    def __init__(
        self,
        label: str,
        *,
        action_id: str,
        tooltip: str,
        id: str,
    ) -> None:
        super().__init__(
            label,
            id=id,
            classes="console-transcript-empty-action",
            compact=True,
        )
        self._workbench_action_id = action_id
        self.tooltip = tooltip

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route empty-state activation through the owning Workbench screen."""
        action_id = getattr(self, "_workbench_action_id", "")
        if not action_id:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested(action_id))


class ConsoleTranscriptEmptyPanel(Vertical):
    """Actionable Console transcript empty state, driven by a setup card state."""

    def __init__(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str,
        provider_action_tooltip: str,
    ) -> None:
        super().__init__(
            id="console-transcript-empty-state",
            classes="console-transcript-empty-panel",
        )
        self.card_state = _coerce_card_state(card_state)
        self.provider_action_label = provider_action_label
        self.provider_action_tooltip = provider_action_tooltip

    def compose(self) -> ComposeResult:
        # The blocking setup card (title + numbered steps + primary action) now
        # lives in ``ConsoleSetupModal``; while setup is incomplete
        # (``mode == "card"``) this in-transcript panel shows only the quiet
        # empty line, dimmed under the overlay. ``ready_line``/``quiet`` render
        # as before.
        title = Static(
            CONSOLE_SETUP_CARD_TITLE,
            id="console-empty-title",
            classes="console-transcript-empty-title",
        )
        title.styles.display = "none"
        yield title

        body = Static(
            self.card_state.body_copy or CONSOLE_QUIET_EMPTY_COPY,
            id="console-empty-body",
            classes="console-transcript-empty-body console-transcript-empty-state",
        )
        yield body

        action_row = Horizontal(
            ConsoleTranscriptEmptyAction(
                self.provider_action_label,
                action_id="provider-recovery",
                tooltip=self.provider_action_tooltip,
                id="console-empty-choose-model",
            ),
            ConsoleTranscriptEmptyAction(
                "Attach context",
                action_id="attach-context",
                tooltip="Open context sources and attach workspace material.",
                id="console-empty-attach-context",
            ),
            ConsoleTranscriptEmptyAction(
                "Run Library RAG",
                action_id="run-library-rag",
                tooltip="Search Library sources before sending.",
                id="console-empty-run-library-rag",
            ),
            id="console-empty-action-row",
            classes="console-transcript-empty-action-row",
        )
        # The setup action row is owned by ``ConsoleSetupModal`` now; the
        # in-transcript row is retained (hidden) only so legacy selectors keep
        # resolving.
        action_row.styles.display = "none"
        yield action_row

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str,
        provider_action_tooltip: str,
    ) -> None:
        """Refresh the onboarding surface from a new card state."""
        self.card_state = _coerce_card_state(card_state)
        self.provider_action_label = provider_action_label
        self.provider_action_tooltip = provider_action_tooltip
        self.refresh(recompose=True)


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
        self._empty_card_state = ConsoleSetupCardState(
            mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY
        )
        self.empty_state_action_label = EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL
        self.empty_state_action_tooltip = EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP
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

    def sync_empty_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str = "",
        provider_action_tooltip: str = "",
    ) -> None:
        """Refresh the empty transcript state while preserving message exports."""
        next_card_state = _coerce_card_state(card_state)
        next_action_label = (
            provider_action_label.strip() or EMPTY_TRANSCRIPT_PROVIDER_ACTION_LABEL
        )
        next_action_tooltip = (
            provider_action_tooltip.strip() or EMPTY_TRANSCRIPT_PROVIDER_ACTION_TOOLTIP
        )
        if (
            self._empty_card_state == next_card_state
            and self.empty_state_action_label == next_action_label
            and self.empty_state_action_tooltip == next_action_tooltip
        ):
            return
        self._empty_card_state = next_card_state
        self.empty_state_action_label = next_action_label
        self.empty_state_action_tooltip = next_action_tooltip
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
                lines.append(SELECTED_MESSAGE_ACTION_GUIDE)
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
                rows.append(
                    _TranscriptRow(
                        key=f"action-help:{message.id}",
                        kind="action-help",
                        signature=("action-help", SELECTED_MESSAGE_ACTION_GUIDE),
                        renderable=SELECTED_MESSAGE_ACTION_GUIDE,
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
                    signature=(
                        "empty",
                        self._empty_card_state,
                        self.empty_state_action_label,
                        self.empty_state_action_tooltip,
                    ),
                    action_label=self.empty_state_action_label,
                    action_tooltip=self.empty_state_action_tooltip,
                    card_state=self._empty_card_state,
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
            assert row.card_state is not None
            return ConsoleTranscriptEmptyPanel(
                row.card_state,
                provider_action_label=row.action_label,
                provider_action_tooltip=row.action_tooltip,
            )
        if row.kind == "action-help":
            return Static(
                row.renderable,
                id=self._row_widget_id(row),
                classes="console-transcript-action-guide",
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
        if row.kind == "empty" and isinstance(widget, ConsoleTranscriptEmptyPanel):
            assert row.card_state is not None
            widget.sync_card_state(
                row.card_state,
                provider_action_label=row.action_label,
                provider_action_tooltip=row.action_tooltip,
            )
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
            _message_render_text(message, selected=selected),
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
