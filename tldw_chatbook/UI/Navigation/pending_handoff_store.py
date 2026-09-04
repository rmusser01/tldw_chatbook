"""Memory-only ownership for revisioned destination handoffs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
import threading
import time
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from ...ACP_Interop.runtime_session import ACP_SESSION_RECORD_PREFIX
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.console_chat_models import ConsoleFleetCompletionTarget
from ...Chat.console_live_work import ConsoleLiveWorkLaunch
from ...Chat.provider_readiness import provider_config_key
from ...Prompt_Management.prompt_variables import PromptVariableApplication
from .audio_cpp_model_handoff import (
    AudioCppModelLibraryRequest,
    AudioCppModelLibraryResult,
)
from .conversation_settings_navigation import ConversationSettingsReturnIntent
from ..Screens.study_scope_models import (
    STUDY_INITIAL_SECTIONS,
    STUDY_ORIGINS,
    StudyScopeContext,
)


ARTIFACT_CHATBOOK_RECORD_PREFIX = "local:chatbook:"
_PROVIDER_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]{0,127}")


@dataclass(frozen=True, slots=True)
class ConsoleProviderIntent:
    """Memory-only request to select one normalized Console provider."""

    provider: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str):
            raise TypeError("Console provider must be text")
        normalized = provider_config_key(self.provider)
        if not normalized:
            raise ValueError("Console provider must be non-empty")
        if _PROVIDER_IDENTIFIER_PATTERN.fullmatch(normalized) is None:
            raise ValueError("Console provider identifier is invalid")
        object.__setattr__(self, "provider", normalized)


@dataclass(frozen=True, slots=True)
class ConsoleFirstChatIntent:
    """Secret-free request to activate one exact first-run Console session."""

    session_id: str
    provider: str
    model: str
    config_revision: int

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not str
            or not self.session_id
            or self.session_id != self.session_id.strip()
            or len(self.session_id) > 256
        ):
            raise ValueError("Console first-chat session is invalid")
        if type(self.provider) is not str:
            raise TypeError("Console first-chat provider must be text")
        normalized_provider = provider_config_key(self.provider)
        if (
            not normalized_provider
            or _PROVIDER_IDENTIFIER_PATTERN.fullmatch(normalized_provider) is None
        ):
            raise ValueError("Console first-chat provider is invalid")
        if (
            type(self.model) is not str
            or not self.model
            or self.model != self.model.strip()
            or len(self.model) > 512
        ):
            raise ValueError("Console first-chat model is invalid")
        if type(self.config_revision) is not int or self.config_revision < 1:
            raise ValueError("Console first-chat config revision is invalid")
        object.__setattr__(self, "provider", normalized_provider)


class HandoffChannel(StrEnum):
    """Typed single-slot channels owned by the application."""

    CHAT = "chat"
    CONSOLE_LIVE_WORK = "console_live_work"
    CONSOLE_PROMPT_INSERT = "console_prompt_insert"
    CONSOLE_PROVIDER = "console_provider"
    #: PR3a-2 Task 4: a background sub-agent completion's deep link --
    #: staged by the fleet drain consumer while Console is not the active
    #: screen; the next Console mount claims it and switches to the
    #: settled conversation's session (and Task 5's mount-claim reads the
    #: same channel for wake delivery).
    CONSOLE_FLEET_COMPLETION = "console_fleet_completion"
    CONSOLE_FIRST_CHAT = "console_first_chat"
    STUDY_SCOPE = "study_scope"
    STUDY_INITIAL_SECTION = "study_initial_section"
    STUDY_ORIGIN = "study_origin"
    ARTIFACT_CHATBOOK_TARGET = "artifact_chatbook_target"
    ACP_SESSION_TARGET = "acp_session_target"
    AUDIO_CPP_MODEL_LIBRARY_REQUEST = "audio_cpp_model_library_request"
    AUDIO_CPP_MODEL_LIBRARY_RESULT = "audio_cpp_model_library_result"
    CONVERSATION_SETTINGS_RETURN = "conversation_settings_return"


class HandoffValueError(ValueError):
    """A staged value could not be normalized and structurally detached."""


T = TypeVar("T")
HandoffClaimStatus: TypeAlias = Literal["ready", "expired"]
HandoffRevisionStatus: TypeAlias = Literal[
    "pending", "in_flight", "settled", "superseded"
]


@dataclass(frozen=True, slots=True)
class HandoffClaim(Generic[T]):
    """Opaque claim token and detached value for one consumer attempt."""

    channel: HandoffChannel
    revision: int
    value: T = field(repr=False, compare=False)
    status: HandoffClaimStatus = "ready"

    def __post_init__(self) -> None:
        if self.status not in ("ready", "expired"):
            raise ValueError("handoff claim status is invalid")


@dataclass(slots=True)
class _InFlight:
    claim: HandoffClaim[Any]
    retained_value: Any


@dataclass(slots=True)
class _Slot:
    revision: int = 0
    pending: tuple[int, Any] | None = None
    in_flight: _InFlight | None = None
    reserved_revisions: set[int] = field(default_factory=set)


class PendingHandoffStore:
    """Own one latest pending value and one claim per typed channel."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(monotonic_clock):
            raise TypeError("pending handoff clock must be callable")
        self._owner_thread_id = threading.get_ident()
        self._monotonic_clock = monotonic_clock
        self._lock = threading.RLock()
        self._slots = {channel: _Slot() for channel in HandoffChannel}

    def stage(self, channel: HandoffChannel, value: Any) -> int:
        """Normalize and replace the latest pending value for a channel."""
        return self._stage(channel, value, reserves_new_session=False)

    def stage_reserved_console_first_chat(
        self,
        intent: ConsoleFirstChatIntent,
    ) -> int:
        """Stage a first-chat intent whose absent exact target may be created."""

        return self._stage(
            HandoffChannel.CONSOLE_FIRST_CHAT,
            intent,
            reserves_new_session=True,
        )

    def _stage(
        self,
        channel: HandoffChannel,
        value: Any,
        *,
        reserves_new_session: bool,
    ) -> int:
        self._assert_owner_thread()
        normalized = self._detached_value(channel, value)
        with self._lock:
            slot = self._slot_for(channel)
            if slot.pending is not None:
                slot.reserved_revisions.discard(slot.pending[0])
            slot.revision += 1
            slot.pending = (slot.revision, normalized)
            if reserves_new_session:
                slot.reserved_revisions.add(slot.revision)
            return slot.revision

    def clear_pending(self, channel: HandoffChannel) -> int:
        """Advance a channel and remove its latest unclaimed value."""
        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for(channel)
            if slot.pending is not None:
                slot.reserved_revisions.discard(slot.pending[0])
            slot.revision += 1
            slot.pending = None
            return slot.revision

    def discard_pending_exact(
        self,
        channel: HandoffChannel,
        revision: int,
        value: Any,
    ) -> bool:
        """Discard only one exact pending slot, even while another claim is active."""

        self._assert_owner_thread()
        if type(revision) is not int or revision < 1:
            raise ValueError("handoff revision must be a positive exact integer")
        slot = self._slot_for(channel)
        normalized = self._detached_value(channel, value)
        if slot.pending != (revision, normalized):
            return False
        slot.pending = None
        return True

    def claim(self, channel: HandoffChannel) -> HandoffClaim[Any] | None:
        """Claim the pending value when no other consumer is in flight."""
        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for(channel)
            if slot.in_flight is not None or slot.pending is None:
                return None
            revision, retained_value = slot.pending
            status: HandoffClaimStatus = "ready"
            if channel is HandoffChannel.CONSOLE_PROMPT_INSERT:
                now = self._monotonic_now()
                if retained_value.is_expired(now_monotonic=now):
                    status = "expired"
            delivered_value = self._detached_value(channel, retained_value)
            claim = HandoffClaim(
                channel=channel,
                revision=revision,
                value=delivered_value,
                status=status,
            )
            slot.pending = None
            slot.in_flight = _InFlight(
                claim=claim,
                retained_value=retained_value,
            )
            return claim

    def has_pending(self, channel: HandoffChannel) -> bool:
        """Return whether a channel has an unclaimed value without exposing it."""
        self._assert_owner_thread()
        with self._lock:
            return self._slot_for(channel).pending is not None

    def exact_revision_status(
        self,
        channel: HandoffChannel,
        revision: int,
    ) -> HandoffRevisionStatus:
        """Describe one revision's ownership without exposing its value."""

        self._assert_owner_thread()
        if type(revision) is not int or revision < 1:
            raise ValueError("handoff revision must be a positive exact integer")
        with self._lock:
            slot = self._slot_for(channel)
            if slot.pending is not None and slot.pending[0] == revision:
                return "pending"
            if (
                slot.in_flight is not None
                and slot.in_flight.claim.revision == revision
            ):
                return "in_flight"
            if slot.revision > revision:
                return "superseded"
            return "settled"

    def is_current_claim(self, claim: HandoffClaim[Any]) -> bool:
        """Return whether a claim still owns the channel's latest revision."""

        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for_claim(claim)
            current = slot.in_flight
            return (
                current is not None
                and current.claim is claim
                and slot.revision == claim.revision
            )

    def acknowledge(self, claim: HandoffClaim[Any]) -> bool:
        """Settle only the exact claim currently in flight."""
        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for_claim(claim)
            current = slot.in_flight
            if current is None or current.claim is not claim:
                return False
            slot.in_flight = None
            slot.reserved_revisions.discard(claim.revision)
            return True

    def acknowledge_current(self, claim: HandoffClaim[Any]) -> bool:
        """Settle a claim only while it still owns the latest revision."""

        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for_claim(claim)
            current = slot.in_flight
            if (
                current is None
                or current.claim is not claim
                or slot.revision != claim.revision
            ):
                return False
            slot.in_flight = None
            slot.reserved_revisions.discard(claim.revision)
            return True

    def settle_transferred_claim(self, claim: HandoffClaim[Any]) -> bool:
        """Atomically terminally settle one transferred Settings return.

        The exact claim may still be in flight or may have been requeued by a
        partial prior release. Already settled and superseded revisions are
        terminal successes. A different pending or in-flight owner is never
        mutated, and no handoff value is returned.

        Args:
            claim: The opaque Conversation settings return claim whose draft
                has transferred to its destination modal.

        Returns:
            ``True`` when the exact revision is terminal, or ``False`` when a
            different owner currently holds that revision.

        Raises:
            RuntimeError: If called outside the owning thread.
            TypeError: If ``claim`` is not a :class:`HandoffClaim`.
            ValueError: If ``claim`` belongs to another channel or has an
                invalid revision.
        """

        self._assert_owner_thread()
        slot = self._slot_for_claim(claim)
        if claim.channel is not HandoffChannel.CONVERSATION_SETTINGS_RETURN:
            raise ValueError(
                "Conversation settings transfer settlement requires its return channel"
            )
        if type(claim.revision) is not int or claim.revision < 1:
            raise ValueError("handoff revision must be a positive exact integer")
        normalized = self._detached_value(claim.channel, claim.value)
        with self._lock:
            current = slot.in_flight
            if current is not None:
                if current.claim is claim:
                    slot.in_flight = None
                    slot.reserved_revisions.discard(claim.revision)
                    return True
                return slot.revision > claim.revision
            if slot.pending is not None and slot.pending[0] == claim.revision:
                if slot.pending != (claim.revision, normalized):
                    return False
                slot.pending = None
                slot.reserved_revisions.discard(claim.revision)
                return True
            return slot.revision >= claim.revision

    def claim_reserves_new_console_session(
        self,
        claim: HandoffClaim[ConsoleFirstChatIntent],
    ) -> bool:
        """Return reservation metadata only for the exact in-flight claim."""

        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for_claim(claim)
            if claim.channel is not HandoffChannel.CONSOLE_FIRST_CHAT:
                raise ValueError("reservation metadata requires a first-chat claim")
            current = slot.in_flight
            return (
                current is not None
                and current.claim is claim
                and claim.revision in slot.reserved_revisions
            )

    def release(self, claim: HandoffClaim[Any]) -> bool:
        """Release an exact claim without overwriting a newer revision."""
        released, _prompt_status = self._release_claim(claim)
        return released

    def release_prompt_claim(
        self,
        claim: HandoffClaim[PromptVariableApplication],
    ) -> HandoffClaimStatus | None:
        """Atomically retry or expire one exact Prompt claim.

        Args:
            claim: The Prompt claim being released after a transient failure.

        Returns:
            ``"ready"`` when the exact claim was requeued, ``"expired"``
            when it was terminally settled by the injected clock, or ``None``
            when the claim was not exact or a newer revision superseded it.

        Raises:
            RuntimeError: If called outside the owning thread.
            TypeError: If ``claim`` is not a :class:`HandoffClaim`.
            ValueError: If ``claim`` does not belong to the Prompt channel.
        """
        self._assert_owner_thread()
        self._slot_for_claim(claim)
        if claim.channel is not HandoffChannel.CONSOLE_PROMPT_INSERT:
            raise ValueError("release_prompt_claim requires a Prompt claim")
        _released, prompt_status = self._release_claim(claim)
        return prompt_status

    def _release_claim(
        self,
        claim: HandoffClaim[Any],
    ) -> tuple[bool, HandoffClaimStatus | None]:
        """Settle one exact claim and report a Prompt retry outcome."""
        self._assert_owner_thread()
        with self._lock:
            slot = self._slot_for_claim(claim)
            current = slot.in_flight
            if current is None or current.claim is not claim:
                return False, None
            slot.in_flight = None
            should_requeue = slot.revision == claim.revision
            prompt_status: HandoffClaimStatus | None = None
            if should_requeue and claim.channel is HandoffChannel.CONSOLE_PROMPT_INSERT:
                should_requeue = claim.status == "ready" and self._prompt_is_unexpired(
                    current.retained_value
                )
                prompt_status = "ready" if should_requeue else "expired"
            if should_requeue:
                slot.pending = (claim.revision, current.retained_value)
            else:
                slot.reserved_revisions.discard(claim.revision)
            return True, prompt_status

    def _prompt_is_unexpired(self, value: PromptVariableApplication) -> bool:
        try:
            now = self._monotonic_now()
        except HandoffValueError:
            return False
        return not value.is_expired(now_monotonic=now)

    def _monotonic_now(self) -> float:
        try:
            value = self._monotonic_clock()
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError
        except MemoryError:
            raise
        except Exception:
            raise HandoffValueError(
                "handoff clock must return a finite number"
            ) from None
        return normalized

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
            if not isinstance(value, PromptVariableApplication):
                raise TypeError("Console prompt handoff must be typed")
            return PromptVariableApplication(
                system_text=value.system_text,
                user_text=value.user_text,
                apply_system=value.apply_system,
                apply_user=value.apply_user,
                destination=value.destination,
                target_session_id=value.target_session_id,
                composer_fingerprint=value.composer_fingerprint,
                system_fingerprint=value.system_fingerprint,
                created_monotonic=value.created_monotonic,
            )
        if channel is HandoffChannel.CONSOLE_PROVIDER:
            if not isinstance(value, ConsoleProviderIntent):
                raise TypeError("Console provider handoff must be typed")
            return ConsoleProviderIntent(provider=value.provider)
        if channel is HandoffChannel.CONSOLE_FLEET_COMPLETION:
            if not isinstance(value, ConsoleFleetCompletionTarget):
                raise TypeError("Console fleet completion handoff must be typed")
            return ConsoleFleetCompletionTarget(
                conversation_id=value.conversation_id,
                session_id=value.session_id,
            )
        if channel is HandoffChannel.CONSOLE_FIRST_CHAT:
            if not isinstance(value, ConsoleFirstChatIntent):
                raise TypeError("Console first-chat handoff must be typed")
            return ConsoleFirstChatIntent(
                session_id=value.session_id,
                provider=value.provider,
                model=value.model,
                config_revision=value.config_revision,
            )
        if channel is HandoffChannel.CONVERSATION_SETTINGS_RETURN:
            if not isinstance(value, ConversationSettingsReturnIntent):
                raise TypeError("Conversation settings return handoff must be typed")
            return ConversationSettingsReturnIntent(
                session_id=value.session_id,
                settings_revision=value.settings_revision,
                active_view=value.active_view,
                focus_control_id=value.focus_control_id,
            )
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
        if channel is HandoffChannel.STUDY_ORIGIN:
            if not isinstance(value, str):
                raise TypeError("Study origin must be text")
            normalized = value.strip()
            if normalized not in STUDY_ORIGINS:
                raise ValueError("invalid Study origin")
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
        if channel is HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST:
            if type(value) is not AudioCppModelLibraryRequest:
                raise TypeError("audio.cpp Model Library request must be exact")
            return AudioCppModelLibraryRequest(
                token=value.token,
                draft_revision=value.draft_revision,
            )
        if channel is HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT:
            if type(value) is not AudioCppModelLibraryResult:
                raise TypeError("audio.cpp Model Library result must be exact")
            return AudioCppModelLibraryResult(
                token=value.token,
                draft_revision=value.draft_revision,
                artifact_id=value.artifact_id,
                revision=value.revision,
                variant=value.variant,
                canonical_root=value.canonical_root,
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
