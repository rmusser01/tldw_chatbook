"""Trusted, process-local Console speech request contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleMessageStatus,
)
from tldw_chatbook.TTS.profile_types import CharacterRef


class ConsoleSpeechSnapshotRejectionCode(str, Enum):
    """Bounded reasons an issued Console speech snapshot can be rejected."""

    MISSING_MESSAGE = "missing_message"
    SESSION_CHANGED = "session_changed"
    MESSAGE_CHANGED = "message_changed"
    MESSAGE_NOT_SPEAKABLE = "message_not_speakable"
    PERSISTED_VERSION_UNAVAILABLE = "persisted_version_unavailable"
    PERSISTED_VERSION_CHANGED = "persisted_version_changed"
    AUTHORSHIP_CHANGED = "authorship_changed"


class ConsoleSpeechSnapshotRejected(ValueError):
    """Reject a stale or unverifiable snapshot without carrying private data."""

    USER_COPY = "Message changed before speech started; select Speak again."

    def __init__(self, code: ConsoleSpeechSnapshotRejectionCode) -> None:
        if type(code) is not ConsoleSpeechSnapshotRejectionCode:
            raise ValueError("rejection code must be bounded")
        self.code = code
        super().__init__(self.USER_COPY)


@dataclass(frozen=True, slots=True)
class TTSMessageSpeechSnapshot:
    """Immutable identity and content captured for one Console Speak action."""

    session_id: str
    message_id: str
    persisted_conversation_id: str | None
    persisted_message_id: str | None
    raw_content: str = field(repr=False)
    selected_variant_id: str
    speech_revision: int
    persisted_message_version: int | None
    role: ConsoleMessageRole
    status: ConsoleMessageStatus
    assistant_kind: str | None
    character_ref: CharacterRef | None = field(repr=False)

    def __post_init__(self) -> None:
        """Reject malformed snapshots at their sole construction boundary."""
        for name in ("session_id", "message_id", "selected_variant_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(name)
        for name in ("persisted_conversation_id", "persisted_message_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(name)
        if type(self.raw_content) is not str:
            raise ValueError("raw_content")
        if type(self.speech_revision) is not int or self.speech_revision < 0:
            raise ValueError("speech_revision")
        if self.persisted_message_version is not None and (
            type(self.persisted_message_version) is not int
            or self.persisted_message_version < 1
        ):
            raise ValueError("persisted_message_version")
        if type(self.role) is not ConsoleMessageRole:
            raise ValueError("role")
        if self.status not in {
            "complete",
            "pending",
            "streaming",
            "stopped",
            "failed",
        }:
            raise ValueError("status")
        if self.assistant_kind is not None and type(self.assistant_kind) is not str:
            raise ValueError("assistant_kind")
        if (
            self.character_ref is not None
            and type(self.character_ref) is not CharacterRef
        ):
            raise ValueError("character_ref")
