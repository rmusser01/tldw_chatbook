"""Pure selected-message action contracts for the native Console transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)


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

    FEEDBACK_PLAIN_LABELS: tuple[str, str] = ("👍", "👎")

    _COMPLETED_ACTIONS: tuple[tuple[str, str], ...] = (
        ("copy", "Copy"),
        ("speak", "🔊"),
        ("edit", "Edit"),
        ("save-as", "Save as..."),
        ("regenerate", "♻"),
        ("continue", "--->"),
        ("feedback", "Feedback"),
        ("delete", "🗑"),
    )
    _VARIANT_NAV_ACTIONS: tuple[tuple[str, str], ...] = (
        ("variant-previous", "<"),
        ("variant-next", ">"),
    )
    _KEEP_ACTION: tuple[tuple[str, str], ...] = (("keep", "keep"),)
    _SPEAK_STOP_ACTION: tuple[str, str] = ("speak-stop", "⏹")
    _FAILED_RETRY_ACTIONS: tuple[tuple[str, str], ...] = (("retry", "Try"),)
    _IMAGE_VIEW_ACTIONS: tuple[tuple[str, str], ...] = (("toggle-image-view", "View"),)
    _SAVE_IMAGE_ACTIONS: tuple[tuple[str, str], ...] = (("save-image", "Save Image"),)
    _VIEW_ORIGINAL_ATTEMPT_ACTION: tuple[tuple[str, str], ...] = (
        ("view-original-attempt", "View original attempt"),
    )

    @staticmethod
    def _has_image(message: ConsoleChatMessage) -> bool:
        return message.image_data is not None or bool(message.image_mime_type)

    @staticmethod
    def _speak_visible(message: ConsoleChatMessage) -> bool:
        """Spec §1a: speak is offered for any completed message with
        non-empty text (any role, generation-card marker text included) --
        absent (not merely disabled) for a failed message or one with no
        text yet (e.g. a still-pending assistant turn)."""
        return message.status != "failed" and bool(message.content.strip())

    def __init__(
        self,
        *,
        available_save_destinations: set[str] | None = None,
        unavailable_save_reasons: dict[str, str] | None = None,
    ) -> None:
        self.available_save_destinations = set(available_save_destinations or ())
        self.unavailable_save_reasons = dict(unavailable_save_reasons or {})

    @classmethod
    def _base_actions_with(
        cls, inserted: tuple[tuple[str, str], ...]
    ) -> list[tuple[str, str]]:
        """Return the base action row with extra actions inserted before regenerate."""
        actions: list[tuple[str, str]] = []
        for action_id, label in cls._COMPLETED_ACTIONS:
            if action_id == "regenerate":
                actions.extend(inserted)
            actions.append((action_id, label))
        return actions

    def available_actions(
        self,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        speaking_message_id: str | None = None,
        original_attempt_available: bool = False,
    ) -> list[ConsoleMessageAction]:
        """Return canonical selected-message actions for a transcript message.

        Args:
            message: Transcript message to resolve actions for.
            generation_variant_count: Number of image-generation variants
                carried by this message (0 for a non-generation message).
                When > 0 this gates `<`/`>`/Keep INSTEAD of the text-sibling
                ``sibling_count``/``sibling_index`` fields on ``message`` --
                an image-variant set and a text-sibling set are mutually
                exclusive shapes (spec §5.1). Defaults to 0 so existing
                callers that don't pass these kwargs see byte-identical
                behavior.
            generation_browsed_index: Currently browsed variant index for a
                generation message (ignored when ``generation_variant_count``
                is 0).
            speaking_message_id: id of the Console message currently driving
                TTS playback, if any (task-559 unit 2). When it matches
                ``message.id`` the row's 🔊 speak action swaps to a ⏹
                speak-stop action in the same slot -- mirrors how the
                generation card's browsed index swaps in "Keep". Defaults to
                ``None`` so existing callers see byte-identical behavior.
            original_attempt_available: Whether this completed assistant has
                a current-session original-attempt preview. Defaults to false;
                plain/export helpers intentionally never pass it.
        """
        disabled_reason = self._disabled_reason(message)
        is_generation_message = generation_variant_count > 0
        completed_actions = list(self._COMPLETED_ACTIONS)
        extra_actions: list[tuple[str, str]] = []
        if (
            original_attempt_available
            and message.status == "complete"
            and self._is_assistant_message(message)
        ):
            extra_actions.extend(self._VIEW_ORIGINAL_ATTEMPT_ACTION)
        if is_generation_message:
            if generation_variant_count > 1:
                extra_actions.extend(self._VARIANT_NAV_ACTIONS)
            if generation_browsed_index != 0:
                extra_actions.extend(self._KEEP_ACTION)
        elif message.sibling_count > 1:
            extra_actions.extend(self._VARIANT_NAV_ACTIONS)
        if extra_actions:
            completed_actions = self._base_actions_with(tuple(extra_actions))
        if self._has_image(message):
            completed_actions = (
                completed_actions
                + list(self._IMAGE_VIEW_ACTIONS)
                + list(self._SAVE_IMAGE_ACTIONS)
            )
        if not self._speak_visible(message):
            completed_actions = [
                (action_id, label)
                for action_id, label in completed_actions
                if action_id != "speak"
            ]
        elif speaking_message_id == message.id:
            completed_actions = [
                (self._SPEAK_STOP_ACTION[0], self._SPEAK_STOP_ACTION[1])
                if action_id == "speak"
                else (action_id, label)
                for action_id, label in completed_actions
            ]
        if message.status == "failed" and self._is_assistant_message(message):
            # Retry regenerates a failed ASSISTANT response. A failed USER row —
            # e.g. the TASK-457(a) optimistic echo rejected before any provider
            # send — has nothing to regenerate, so it must not offer retry (the
            # user re-sends from the composer instead). Speak is also absent
            # here (spec §1a) -- a failed row's content is not a completed
            # response worth reading aloud.
            return [
                ConsoleMessageAction(action_id, label)
                for action_id, label in self._base_actions_with(
                    self._FAILED_RETRY_ACTIONS
                )
                if action_id != "speak"
            ]
        return [
            ConsoleMessageAction(
                action_id=action_id,
                label=label,
                enabled=disabled_reason == ""
                and self._action_enabled(
                    action_id,
                    message,
                    generation_variant_count=generation_variant_count,
                    generation_browsed_index=generation_browsed_index,
                ),
                disabled_reason=disabled_reason
                or self._action_disabled_reason(
                    action_id,
                    message,
                    generation_variant_count=generation_variant_count,
                    generation_browsed_index=generation_browsed_index,
                ),
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
    def expand_plain_action_labels(
        cls, actions: list[ConsoleMessageAction]
    ) -> list[str]:
        """Expand grouped UI actions into the labels shown in plain text."""
        labels: list[str] = []
        for action in actions:
            if action.action_id == "feedback":
                labels.extend(cls.FEEDBACK_PLAIN_LABELS)
            else:
                labels.append(action.label)
        return labels

    def save_as_destinations(
        self, message: ConsoleChatMessage
    ) -> list[ConsoleSaveDestination]:
        """Return Save as destinations, including explicit unavailable entries."""
        _ = message
        labels = ("Chatbook", "Note", "Media", "Prompt")
        destinations: list[ConsoleSaveDestination] = []
        for label in labels:
            available = label in self.available_save_destinations
            reason = ""
            if not available:
                reason = (
                    self.unavailable_save_reasons.get(label)
                    or f"Save as {label} is not available in this session."
                )
            destinations.append(
                ConsoleSaveDestination(label=label, available=available, reason=reason)
            )
        return destinations

    def dispatch(
        self, action_id: str, message: ConsoleChatMessage
    ) -> ConsoleActionResult:
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
        if action_id == "view-original-attempt":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Toggled original attempt preview.",
                target_message_id=message.id,
            )
        if action_id == "speak":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Speaking message.",
                target_message_id=message.id,
                target_content=message.content,
            )
        if action_id == "speak-stop":
            # task-559 unit 2: stop is safe to request unconditionally --
            # the app-level TTSPlaybackEvent(action="stop") handler already
            # no-ops when nothing is playing/cached for this message id.
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Stopped speaking.",
                target_message_id=message.id,
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
        if action_id == "keep":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Kept this variant as the message's canonical image.",
                target_message_id=message.id,
            )
        if (
            action_id == "regenerate"
            and not ConsoleMessageActionService._is_assistant_message(message)
        ):
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy="Only assistant messages can be regenerated.",
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
        if action_id == "toggle-image-view":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Toggled image view.",
                target_message_id=message.id,
            )
        if action_id == "save-image":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Saving image to disk.",
                target_message_id=message.id,
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
    def _variant_action_enabled(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
    ) -> bool:
        if generation_variant_count > 0:
            # Generation-variant boundary check takes precedence over the
            # text-sibling fields for these two ids (spec §7) -- a
            # generation message never carries text siblings.
            if action_id == "variant-previous":
                return generation_browsed_index > 0
            if action_id == "variant-next":
                return generation_browsed_index < generation_variant_count - 1
            return True
        if action_id == "variant-previous":
            return message.sibling_index > 0
        if action_id == "variant-next":
            return message.sibling_index < message.sibling_count - 1
        return True

    @staticmethod
    def _action_enabled(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
    ) -> bool:
        if action_id == "regenerate":
            return ConsoleMessageActionService._is_assistant_message(message)
        return ConsoleMessageActionService._variant_action_enabled(
            action_id,
            message,
            generation_variant_count=generation_variant_count,
            generation_browsed_index=generation_browsed_index,
        )

    @staticmethod
    def _action_disabled_reason(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
    ) -> str:
        if (
            action_id == "regenerate"
            and not ConsoleMessageActionService._is_assistant_message(message)
        ):
            return "Only assistant messages can be regenerated."
        if action_id in {
            "variant-previous",
            "variant-next",
        } and not ConsoleMessageActionService._variant_action_enabled(
            action_id,
            message,
            generation_variant_count=generation_variant_count,
            generation_browsed_index=generation_browsed_index,
        ):
            return "No response variant in that direction."
        return ""

    @staticmethod
    def _is_assistant_message(message: ConsoleChatMessage) -> bool:
        role = getattr(message.role, "value", message.role)
        return str(role).lower() == ConsoleMessageRole.ASSISTANT.value
