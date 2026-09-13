"""Pre-dispatch eligibility for provisional Console voice generation."""

from __future__ import annotations

from tldw_chatbook.Audio.voice_process_types import VoiceSpeculationDecision

from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext


def classify_voice_speculation(
    *,
    frozen_session_context: ConsoleTurnExecutionContext,
    prepared_request: PreparedProviderRequest,
    requires_citation_creation: bool = False,
    requires_pre_dispatch_authority: bool = False,
) -> VoiceSpeculationDecision:
    """Classify only effects required before provisional provider dispatch.

    Tool schemas are deliberately not an effect: an attempt may observe an
    inert provider tool request, but it has no executor or approval authority.
    """

    if not isinstance(frozen_session_context, ConsoleTurnExecutionContext):
        raise TypeError("frozen_session_context must be a ConsoleTurnExecutionContext")
    if not isinstance(prepared_request, PreparedProviderRequest):
        raise TypeError("prepared_request must be a PreparedProviderRequest")
    if (
        frozen_session_context.library_authority.policy.auto_retrieve
        is ConsoleAutoRetrieve.AUTOMATIC
        or requires_citation_creation
        or requires_pre_dispatch_authority
    ):
        return VoiceSpeculationDecision.WAIT_FOR_STABLE_TURN
    return VoiceSpeculationDecision.PROVISIONAL


__all__ = ["VoiceSpeculationDecision", "classify_voice_speculation"]
