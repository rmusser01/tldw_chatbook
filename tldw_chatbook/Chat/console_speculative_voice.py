"""Parent compatibility adapter for the dependency-light voice policy reducer.

The app keeps original requests, turn contexts, tool arguments and promotion
outcomes. Only typed policy values and opaque identities enter the audio core.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import inspect
from typing import Any, TypeAlias
from uuid import uuid4

from tldw_chatbook.Audio.duplex_contracts import DuplexMode
from tldw_chatbook.Audio.voice_process_types import (
    RESPONSE_EAGERNESS_DEFAULT_MS,
    AttemptDispatchPrepared as PreparedVoicePolicy,
    AttemptToolPending,
    ProviderAttemptFailed as CoreProviderAttemptFailed,
    VoiceRequestHandle,
    VoiceTerminalDisposition,
    VoiceTurnContextHandle,
)
from tldw_chatbook.Audio.voice_turn_coordinator import (
    AdmittedSpeechFrame,
    AttemptGenerationCompleted,
    AttemptOutputDelta,
    AttemptPlaybackBoundaryKnown,
    AttemptPlaybackFailed,
    AttemptPlaybackStarted,
    AttemptPlaybackTerminal,
    AttemptTtsFailed,
    AudioCapabilityChanged,
    AudioRouteReady,
    AudioRouteRebuildFailed,
    CaptureGated,
    ControlAction,
    ControlKind,
    ManualInterruption,
    ManualRetry,
    SpeculativeTurnCoordinator as _CoreCoordinator,
    SpeculativeVoiceSnapshot,
    SpeculativeVoiceState,
    VoiceEvent as _CoreVoiceEvent,
    _FRAME_TOLERANCE_NS,
    _Scheduler,
)
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.console_voice_attempts import (
    ProviderAttemptFailed,
    VoiceAttemptToolRequest,
)
from tldw_chatbook.Chat.console_voice_eligibility import classify_voice_speculation
from tldw_chatbook.Chat.console_voice_promotion import (
    VoicePromotionOutcome,
    VoicePromotionOutcomeStatus,
)


@dataclass(frozen=True, slots=True)
class AttemptDispatchPrepared:
    """Immutable request and policy context prepared for one active attempt."""

    attempt_epoch: int
    frozen_session_context: ConsoleTurnExecutionContext = field(repr=False)
    prepared_request: PreparedProviderRequest = field(repr=False)
    requires_citation_creation: bool = False
    requires_pre_dispatch_authority: bool = False

    def __post_init__(self) -> None:
        if type(self.attempt_epoch) is not int or self.attempt_epoch < 0:
            raise ValueError("attempt_epoch must be a non-negative integer")
        if not isinstance(
            self.frozen_session_context,
            ConsoleTurnExecutionContext,
        ):
            raise TypeError(
                "frozen_session_context must be a ConsoleTurnExecutionContext"
            )
        if not isinstance(self.prepared_request, PreparedProviderRequest):
            raise TypeError("prepared_request must be a PreparedProviderRequest")
        if type(self.requires_citation_creation) is not bool:
            raise TypeError("requires_citation_creation must be a bool")
        if type(self.requires_pre_dispatch_authority) is not bool:
            raise TypeError("requires_pre_dispatch_authority must be a bool")


VoiceEvent: TypeAlias = (
    _CoreVoiceEvent
    | AttemptDispatchPrepared
    | ProviderAttemptFailed
    | VoiceAttemptToolRequest
)


def _promotion_disposition(result: object) -> VoiceTerminalDisposition:
    """Project the exact original outcome, preserving legacy synchronous success."""

    if result is None:
        return VoiceTerminalDisposition.PROMOTED
    if type(result) is VoicePromotionOutcome:
        if result.status is VoicePromotionOutcomeStatus.PROMOTED:
            return VoiceTerminalDisposition.PROMOTED
        if result.status is VoicePromotionOutcomeStatus.RECOVERY:
            return VoiceTerminalDisposition.RECOVERY
    return VoiceTerminalDisposition.FAILED


class _ParentEffects:
    """Translate the few authority-bearing effects; delegate unchanged operations."""

    def __init__(self, effects: Any) -> None:
        self._effects = effects
        # One current original pair, not an accumulating per-attempt registry.
        # Tool fencing must retain its context until accepted-turn custody transfers.
        self._prepared: tuple[PreparedVoicePolicy, AttemptDispatchPrepared] | None = (
            None
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._effects, name)

    def prepare(self, original: AttemptDispatchPrepared) -> PreparedVoicePolicy:
        policy = PreparedVoicePolicy(
            attempt_epoch=original.attempt_epoch,
            decision=classify_voice_speculation(
                frozen_session_context=original.frozen_session_context,
                prepared_request=original.prepared_request,
                requires_citation_creation=original.requires_citation_creation,
                requires_pre_dispatch_authority=original.requires_pre_dispatch_authority,
            ),
            request_handle=VoiceRequestHandle(uuid4().hex),
            turn_context_handle=VoiceTurnContextHandle(uuid4().hex),
        )
        self._prepared = (policy, original)
        return policy

    def start_prepared_attempt(
        self, attempt_epoch: int, request_handle: VoiceRequestHandle
    ) -> Any:
        prepared = self._prepared
        if (
            type(request_handle) is not VoiceRequestHandle
            or prepared is None
            or prepared[0].attempt_epoch != attempt_epoch
            or prepared[0].request_handle != request_handle
        ):
            raise ValueError("prepared request handle is not current")
        return self._effects.start_prepared_attempt(attempt_epoch)

    def submit_accepted_voice_turn(
        self, exact_user_text: str, turn_context_handle: VoiceTurnContextHandle
    ) -> Any:
        prepared = self._prepared
        if (
            type(turn_context_handle) is not VoiceTurnContextHandle
            or prepared is None
            or prepared[0].turn_context_handle != turn_context_handle
        ):
            raise ValueError("turn context handle is not current")
        return self._effects.submit_accepted_voice_turn(
            exact_user_text, prepared[1].frozen_session_context
        )

    def promote(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> VoiceTerminalDisposition | Awaitable[VoiceTerminalDisposition]:
        result = self._effects.promote(
            turn_id=turn_id,
            attempt_epoch=attempt_epoch,
            transcript=transcript,
            assistant_text=assistant_text,
            terminal_boundary_ns=terminal_boundary_ns,
        )
        if not inspect.isawaitable(result):
            return _promotion_disposition(result)

        async def project() -> VoiceTerminalDisposition:
            return _promotion_disposition(await result)

        return project()

    def release_prepared(self) -> None:
        self._prepared = None


class SpeculativeTurnCoordinator(_CoreCoordinator):
    """Retain the app-facing constructor and events while reducing narrow values."""

    def __init__(
        self,
        *,
        effects: Any,
        scheduler: _Scheduler | None = None,
        response_eagerness_ms: object = RESPONSE_EAGERNESS_DEFAULT_MS,
        frame_tolerance_ns: int = _FRAME_TOLERANCE_NS,
        initial_clock_generation: int = 0,
        initial_duplex_mode: DuplexMode = DuplexMode.FULL_DUPLEX,
        turn_id_factory: Callable[[int], str] | None = None,
        frozen_session_context: ConsoleTurnExecutionContext | None = None,
        prepared_request: PreparedProviderRequest | None = None,
        requires_citation_creation: bool = False,
        requires_pre_dispatch_authority: bool = False,
        deferred_attempt_preparation: bool = False,
    ) -> None:
        if (frozen_session_context is None) is not (prepared_request is None):
            raise ValueError(
                "frozen_session_context and prepared_request must be supplied together"
            )
        if type(deferred_attempt_preparation) is not bool:
            raise TypeError("deferred_attempt_preparation must be a bool")
        if deferred_attempt_preparation and frozen_session_context is not None:
            raise ValueError(
                "deferred attempt preparation cannot use static request context"
            )
        self._parent_effects = _ParentEffects(effects)
        prepared_policy = None
        if frozen_session_context is not None:
            prepared_policy = self._parent_effects.prepare(
                AttemptDispatchPrepared(
                    0,
                    frozen_session_context,
                    prepared_request,
                    requires_citation_creation,
                    requires_pre_dispatch_authority,
                )
            )
        super().__init__(
            effects=self._parent_effects,
            scheduler=scheduler,
            response_eagerness_ms=response_eagerness_ms,
            frame_tolerance_ns=frame_tolerance_ns,
            initial_clock_generation=initial_clock_generation,
            initial_duplex_mode=initial_duplex_mode,
            turn_id_factory=turn_id_factory,
            prepared_policy=prepared_policy,
            deferred_attempt_preparation=deferred_attempt_preparation,
        )

    async def _reduce(self, event: object) -> None:
        if isinstance(event, AttemptDispatchPrepared):
            if (
                not self._deferred_attempt_preparation
                or event.attempt_epoch != self._active_epoch
            ):
                return
            event = self._parent_effects.prepare(event)
        elif isinstance(event, VoiceAttemptToolRequest):
            event = AttemptToolPending(event.attempt_epoch)
        elif isinstance(event, ProviderAttemptFailed):
            event = CoreProviderAttemptFailed(event.attempt_epoch, event.error_class)
        await super()._reduce(event)

    def _clear_turn(self) -> None:
        super()._clear_turn()
        if self._deferred_attempt_preparation:
            self._parent_effects.release_prepared()


__all__ = [
    "AdmittedSpeechFrame",
    "AttemptDispatchPrepared",
    "AttemptGenerationCompleted",
    "AttemptOutputDelta",
    "AttemptPlaybackStarted",
    "AttemptPlaybackBoundaryKnown",
    "AttemptPlaybackFailed",
    "AttemptPlaybackTerminal",
    "AttemptTtsFailed",
    "AudioCapabilityChanged",
    "AudioRouteReady",
    "AudioRouteRebuildFailed",
    "CaptureGated",
    "ControlAction",
    "ControlKind",
    "ManualInterruption",
    "ManualRetry",
    "SpeculativeTurnCoordinator",
    "SpeculativeVoiceSnapshot",
    "SpeculativeVoiceState",
    "VoiceEvent",
]
