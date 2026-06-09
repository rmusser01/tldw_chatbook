"""Pure selected-message action contracts for the native Console transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage


ConsoleActionStatus = Literal[
    "completed",
    "wip",
    "blocked",
    "continue_requested",
    "edit_requested",
]


@dataclass(frozen=True)
class ConsoleMessageAction:
    """One visible action in the selected-message action row."""

    action_id: str
    label: str
    enabled: bool = True
    disabled_reason: str = ""


@dataclass(frozen=True)
class ConsoleActionResult:
    """Result of dispatching a Console selected-message action."""

    action_id: str
    status: ConsoleActionStatus
    visible_copy: str
    clipboard_text: str | None = None
    target_message_id: str | None = None
    target_content: str | None = None


@dataclass(frozen=True)
class ConsoleSaveDestination:
    """One Save as destination shown in the Console save modal."""

    label: str
    available: bool
    reason: str = ""


class ConsoleMessageActionService:
    """Resolve and dispatch safe Console selected-message actions."""

    FEEDBACK_PLAIN_LABELS: tuple[str, str] = ("Good", "Bad")

    _COMPLETED_ACTIONS: tuple[tuple[str, str], ...] = (
        ("copy", "Copy"),
        ("edit", "Edit"),
        ("save-as", "Save as..."),
        ("regenerate", "Regen"),
        ("continue", "--->"),
        ("feedback", "Feedback"),
        ("delete", "Del"),
    )

    def __init__(self, *, available_save_destinations: set[str] | None = None) -> None:
        self.available_save_destinations = available_save_destinations or {"Chatbook"}

    def available_actions(self, message: ConsoleChatMessage) -> list[ConsoleMessageAction]:
        """Return canonical selected-message actions for a transcript message."""
        disabled_reason = self._disabled_reason(message)
        completed_actions = list(self._COMPLETED_ACTIONS)
        if message.variants is not None:
            completed_actions = [
                ("copy", "Copy"),
                ("edit", "Edit"),
                ("save-as", "Save as..."),
                ("variant-previous", "<"),
                ("variant-next", ">"),
                ("regenerate", "Regen"),
                ("continue", "--->"),
                ("feedback", "Feedback"),
                ("delete", "Del"),
            ]
        if message.status == "failed":
            return [
                ConsoleMessageAction("copy", "Copy"),
                ConsoleMessageAction("edit", "Edit"),
                ConsoleMessageAction("save-as", "Save as..."),
                ConsoleMessageAction("retry", "Try"),
                ConsoleMessageAction("regenerate", "Regen"),
                ConsoleMessageAction("continue", "--->"),
                ConsoleMessageAction("feedback", "Feedback"),
                ConsoleMessageAction("delete", "Del"),
            ]
        return [
            ConsoleMessageAction(
                action_id=action_id,
                label=label,
                enabled=disabled_reason == "" and self._variant_action_enabled(action_id, message),
                disabled_reason=disabled_reason or self._variant_disabled_reason(action_id, message),
            )
            for action_id, label in completed_actions
        ]

    def plain_action_labels(self, message: ConsoleChatMessage) -> list[str]:
        """Return terminal-width labels for a message action row."""
        return self.expand_plain_action_labels(self.available_actions(message))

    def plain_action_row(self, message: ConsoleChatMessage) -> str:
        """Return a terminal-readable action row for plain transcript exports."""
        return " ".join(self.plain_action_labels(message))

    @classmethod
    def expand_plain_action_labels(cls, actions: list[ConsoleMessageAction]) -> list[str]:
        """Expand grouped UI actions into the labels shown in plain text."""
        labels: list[str] = []
        for action in actions:
            if action.action_id == "feedback":
                labels.extend(cls.FEEDBACK_PLAIN_LABELS)
            else:
                labels.append(action.label)
        return labels

    def save_as_destinations(self, message: ConsoleChatMessage) -> list[ConsoleSaveDestination]:
        """Return Save as destinations, including explicit WIP/unavailable entries."""
        _ = message
        labels = ("Chatbook", "Note", "Media", "Prompt")
        return [
            ConsoleSaveDestination(
                label=label,
                available=label in self.available_save_destinations,
                reason="" if label in self.available_save_destinations else f"WIP: save as {label} is not wired yet.",
            )
            for label in labels
        ]

    def dispatch(self, action_id: str, message: ConsoleChatMessage) -> ConsoleActionResult:
        """Dispatch a pure action result without touching UI or persistence."""
        if message.status in {"pending", "streaming"}:
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy=self._disabled_reason(message),
            )
        if action_id == "copy":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Copied message to clipboard.",
                clipboard_text=message.content,
            )
        if action_id == "retry" and message.status == "failed":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Retrying failed response.",
            )
        if action_id == "edit":
            target_content = (
                message.variants.current.content
                if message.variants is not None
                else message.content
            )
            return ConsoleActionResult(
                action_id=action_id,
                status="edit_requested",
                visible_copy="Opened Edit Message.",
                target_message_id=message.id,
                target_content=target_content,
            )
        if action_id in {"feedback-up", "feedback-down"}:
            feedback = "up" if action_id == "feedback-up" else "down"
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy=f"Marked message feedback: {feedback}.",
                target_message_id=message.id,
                target_content=feedback,
            )
        if action_id == "delete":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Deleted message from transcript.",
                target_message_id=message.id,
            )
        if action_id in {"variant-previous", "variant-next"}:
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Selected response variant.",
            )
        if action_id == "continue":
            target_content = (
                message.variants.current.content
                if message.variants is not None
                else message.content
            )
            return ConsoleActionResult(
                action_id=action_id,
                status="continue_requested",
                visible_copy="Continuing from selected message.",
                target_message_id=message.id,
                target_content=target_content,
            )
        return ConsoleActionResult(
            action_id=action_id,
            status="wip",
            visible_copy=f"WIP: {action_id} is not wired yet.",
        )

    @staticmethod
    def _disabled_reason(message: ConsoleChatMessage) -> str:
        if message.status in {"pending", "streaming"}:
            return "Wait for response to finish before using message actions."
        return ""

    @staticmethod
    def _variant_action_enabled(action_id: str, message: ConsoleChatMessage) -> bool:
        if action_id == "variant-previous":
            return message.variants is not None and message.variants.can_go_previous
        if action_id == "variant-next":
            return message.variants is not None and message.variants.can_go_next
        return True

    @staticmethod
    def _variant_disabled_reason(action_id: str, message: ConsoleChatMessage) -> str:
        if action_id in {"variant-previous", "variant-next"} and not ConsoleMessageActionService._variant_action_enabled(action_id, message):
            return "No response variant in that direction."
        return ""
