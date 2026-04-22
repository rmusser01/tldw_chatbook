from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

ActiveSource = Literal["local", "server"]
ServerReachability = Literal["unknown", "reachable", "unreachable"]
ServerAuthState = Literal["unknown", "authenticated", "auth_required", "session_invalid"]
ActionKind = Literal["browse", "detail", "create", "update", "delete", "launch", "observe"]
RequiredSource = ActiveSource
AuthorityOwner = Literal["local", "server", "shared"]
OfflinePolicy = Literal["available", "unavailable", "explicit_fallback"]


@dataclass(frozen=True, slots=True)
class RuntimeSourceState:
    active_source: ActiveSource = "local"
    active_server_id: str | None = None
    server_configured: bool = False
    server_reachability: ServerReachability = "unknown"
    server_reachability_checked_at: datetime | None = None
    server_auth_state: ServerAuthState = "unknown"
    server_auth_checked_at: datetime | None = None
    last_known_server_label: str | None = None

    def normalized_for_policy(self, *, now: datetime, freshness_window: timedelta) -> "RuntimeSourceState":
        from .source_state import normalize_runtime_source_state

        return normalize_runtime_source_state(self, now=now, freshness_window=freshness_window)

    def to_dict(self) -> dict:
        from .source_state import runtime_source_state_to_dict

        return runtime_source_state_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeSourceState":
        from .source_state import runtime_source_state_from_dict

        return runtime_source_state_from_dict(data)


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    action_id: str
    capability_id: str
    domain_id: str
    action_kind: ActionKind
    required_source: RequiredSource
    authority_owner: AuthorityOwner
    offline_policy: OfflinePolicy = "available"
    enabled: bool = True
    default_deny_reason: str = "authority_denied"
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason_code: str | None
    user_message: str
    effective_source: str
    authority_owner: str


class PolicyDeniedError(Exception):
    def __init__(
        self,
        *,
        action_id: str,
        reason_code: str,
        user_message: str,
        effective_source: str,
        authority_owner: str,
    ) -> None:
        super().__init__(user_message)
        self.action_id = action_id
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    def to_decision(self) -> PolicyDecision:
        return PolicyDecision(
            allowed=False,
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )
