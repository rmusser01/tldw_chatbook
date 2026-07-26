"""Memory-only ownership for revisioned destination handoffs."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from enum import StrEnum
import threading
from typing import Any, Generic, TypeVar

from ...ACP_Interop.runtime_session import ACP_SESSION_RECORD_PREFIX
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.console_live_work import ConsoleLiveWorkLaunch
from ..Screens.study_scope_models import (
    STUDY_INITIAL_SECTIONS,
    StudyScopeContext,
)


ARTIFACT_CHATBOOK_RECORD_PREFIX = "local:chatbook:"


class HandoffChannel(StrEnum):
    """Typed single-slot channels owned by the application."""

    CHAT = "chat"
    CONSOLE_LIVE_WORK = "console_live_work"
    CONSOLE_PROMPT_INSERT = "console_prompt_insert"
    STUDY_SCOPE = "study_scope"
    STUDY_INITIAL_SECTION = "study_initial_section"
    ARTIFACT_CHATBOOK_TARGET = "artifact_chatbook_target"
    ACP_SESSION_TARGET = "acp_session_target"


class HandoffValueError(ValueError):
    """A staged value could not be normalized and structurally detached."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class HandoffClaim(Generic[T]):
    """Opaque claim token and detached value for one consumer attempt."""

    channel: HandoffChannel
    revision: int
    value: T = field(repr=False, compare=False)


@dataclass(slots=True)
class _InFlight:
    claim: HandoffClaim[Any]
    retained_value: Any


@dataclass(slots=True)
class _Slot:
    revision: int = 0
    pending: tuple[int, Any] | None = None
    in_flight: _InFlight | None = None


class PendingHandoffStore:
    """Own one latest pending value and one claim per typed channel."""

    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._slots = {channel: _Slot() for channel in HandoffChannel}

    def stage(self, channel: HandoffChannel, value: Any) -> int:
        """Normalize and replace the latest pending value for a channel."""
        self._assert_owner_thread()
        slot = self._slot_for(channel)
        normalized = self._detached_value(channel, value)
        slot.revision += 1
        slot.pending = (slot.revision, normalized)
        return slot.revision

    def clear_pending(self, channel: HandoffChannel) -> int:
        """Advance a channel and remove its latest unclaimed value."""
        self._assert_owner_thread()
        slot = self._slot_for(channel)
        slot.revision += 1
        slot.pending = None
        return slot.revision

    def claim(self, channel: HandoffChannel) -> HandoffClaim[Any] | None:
        """Claim the pending value when no other consumer is in flight."""
        self._assert_owner_thread()
        slot = self._slot_for(channel)
        if slot.in_flight is not None or slot.pending is None:
            return None
        revision, retained_value = slot.pending
        delivered_value = self._detached_value(channel, retained_value)
        claim = HandoffClaim(
            channel=channel,
            revision=revision,
            value=delivered_value,
        )
        slot.pending = None
        slot.in_flight = _InFlight(
            claim=claim,
            retained_value=retained_value,
        )
        return claim

    def acknowledge(self, claim: HandoffClaim[Any]) -> bool:
        """Settle only the exact claim currently in flight."""
        self._assert_owner_thread()
        slot = self._slot_for_claim(claim)
        current = slot.in_flight
        if current is None or current.claim is not claim:
            return False
        slot.in_flight = None
        return True

    def release(self, claim: HandoffClaim[Any]) -> bool:
        """Release an exact claim without overwriting a newer revision."""
        self._assert_owner_thread()
        slot = self._slot_for_claim(claim)
        current = slot.in_flight
        if current is None or current.claim is not claim:
            return False
        slot.in_flight = None
        if slot.revision == claim.revision:
            slot.pending = (claim.revision, current.retained_value)
        return True

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("pending handoff store access requires the owner thread")

    def _slot_for(self, channel: HandoffChannel) -> _Slot:
        if not isinstance(channel, HandoffChannel):
            raise TypeError("handoff channel must be a HandoffChannel")
        return self._slots[channel]

    def _slot_for_claim(self, claim: HandoffClaim[Any]) -> _Slot:
        if not isinstance(claim, HandoffClaim):
            raise TypeError("handoff settlement requires a HandoffClaim")
        return self._slot_for(claim.channel)

    @classmethod
    def _detached_value(cls, channel: HandoffChannel, value: Any) -> Any:
        try:
            return cls._copy_value(channel, value)
        except MemoryError:
            raise
        except Exception:
            raise HandoffValueError("handoff value could not be normalized") from None

    @staticmethod
    def _copy_value(channel: HandoffChannel, value: Any) -> Any:
        if channel is HandoffChannel.CHAT:
            if not isinstance(value, (ChatHandoffPayload, Mapping)):
                raise TypeError("Chat handoff must be a payload or mapping")
            copied = ChatHandoffPayload.from_dict(value)
            if copied is None:
                raise ValueError("invalid Chat handoff")
            return copied
        if channel is HandoffChannel.CONSOLE_LIVE_WORK:
            copied = ConsoleLiveWorkLaunch.from_pending(value)
            if copied is None:
                raise ValueError("invalid Console launch")
            return copied
        if channel is HandoffChannel.CONSOLE_PROMPT_INSERT:
            if not isinstance(value, str):
                raise TypeError("Console prompt must be text")
            if not value.strip():
                raise ValueError("Console prompt must be non-empty text")
            return value
        if channel is HandoffChannel.STUDY_SCOPE:
            if not isinstance(value, StudyScopeContext):
                raise TypeError("Study scope must be a StudyScopeContext")
            return copy.deepcopy(value)
        if channel is HandoffChannel.STUDY_INITIAL_SECTION:
            if not isinstance(value, str):
                raise TypeError("Study section must be text")
            normalized = value.strip()
            if normalized not in STUDY_INITIAL_SECTIONS:
                raise ValueError("invalid Study section")
            return normalized
        if channel is HandoffChannel.ARTIFACT_CHATBOOK_TARGET:
            return PendingHandoffStore._canonical_target(
                value,
                prefix=ARTIFACT_CHATBOOK_RECORD_PREFIX,
            )
        if channel is HandoffChannel.ACP_SESSION_TARGET:
            return PendingHandoffStore._canonical_target(
                value,
                prefix=ACP_SESSION_RECORD_PREFIX,
            )
        raise ValueError("unsupported handoff channel")

    @staticmethod
    def _canonical_target(value: Any, *, prefix: str) -> str:
        if not isinstance(value, str):
            raise TypeError("handoff target must be text")
        normalized = value.strip()
        if not normalized.startswith(prefix):
            raise ValueError("invalid handoff target prefix")
        suffix = normalized.removeprefix(prefix).strip()
        if not suffix:
            raise ValueError("handoff target must include an identifier")
        return f"{prefix}{suffix}"
