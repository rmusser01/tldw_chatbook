"""Live-only Console endpoint ownership for verified session handoffs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


class ConsoleEndpointPolicyState(str, Enum):
    """Whether a session endpoint override may participate in a send."""

    ACTIVE = "active"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ConsoleEphemeralEndpointPolicy:
    """A process-local endpoint override that is never conversation metadata."""

    provider: str
    model: str | None
    base_url: str
    state: ConsoleEndpointPolicyState = ConsoleEndpointPolicyState.ACTIVE

    def effective_settings(
        self,
        settings: ConsoleSessionSettings,
    ) -> ConsoleSessionSettings:
        """Overlay the endpoint only while this policy still owns the selection."""

        if settings.provider != self.provider or settings.model != self.model:
            return settings
        return replace(
            settings,
            base_url=(
                self.base_url
                if self.state is ConsoleEndpointPolicyState.ACTIVE
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ConsoleEndpointAdoptionReceipt:
    """Optimistic proof for restoring exact pre-adoption metadata."""

    conversation_id: str
    before_metadata: object
    written_metadata: str
    written_version: int


class ConsoleEndpointRollbackOutcome(str, Enum):
    """Exact compensation result returned to the handoff coordinator."""

    RESTORED = "restored"
    LOST_SESSION_FENCE = "lost_session_fence"
    BLOCKED_DURABLE_RESTORE = "blocked_durable_restore"
