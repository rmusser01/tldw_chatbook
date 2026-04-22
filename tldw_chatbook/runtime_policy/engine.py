from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

from .source_state import POLICY_FRESHNESS_WINDOW
from .types import CapabilityEntry, PolicyDecision, RuntimeSourceState


class PolicyEngine:
    def __init__(self, registry: Mapping[str, CapabilityEntry]):
        self._registry = registry

    def evaluate(
        self,
        *,
        action_id: str,
        state: RuntimeSourceState,
        now: datetime | None = None,
        freshness_window: timedelta = POLICY_FRESHNESS_WINDOW,
    ) -> PolicyDecision:
        entry = self._registry.get(action_id)
        if entry is None:
            return self._deny(
                action_id=action_id,
                reason_code="authority_denied",
                user_message=f"Unknown runtime-policy action_id: {action_id}",
                effective_source="unknown",
                authority_owner="unknown",
            )

        current_time = now or datetime.now(timezone.utc)
        normalized_state = state.normalized_for_policy(now=current_time, freshness_window=freshness_window)

        if not entry.enabled:
            return self._deny(
                action_id=action_id,
                reason_code="capability_disabled",
                user_message=f"{action_id} is disabled by runtime policy.",
                effective_source=normalized_state.active_source,
                authority_owner=entry.authority_owner,
            )

        if normalized_state.active_source != entry.required_source:
            return self._deny(
                action_id=action_id,
                reason_code="wrong_source",
                user_message=f"{action_id} requires {entry.required_source} mode.",
                effective_source=normalized_state.active_source,
                authority_owner=entry.authority_owner,
            )

        if entry.required_source == "server":
            if not normalized_state.server_configured:
                return self._deny(
                    action_id=action_id,
                    reason_code="server_not_configured",
                    user_message=f"{action_id} requires a configured server.",
                    effective_source=normalized_state.active_source,
                    authority_owner=entry.authority_owner,
                )

            if normalized_state.server_reachability == "unreachable":
                return self._deny(
                    action_id=action_id,
                    reason_code="server_unreachable",
                    user_message=f"{action_id} cannot run because the server is unreachable.",
                    effective_source=normalized_state.active_source,
                    authority_owner=entry.authority_owner,
                )

            if normalized_state.server_auth_state == "auth_required":
                return self._deny(
                    action_id=action_id,
                    reason_code="server_auth_required",
                    user_message=f"{action_id} requires server authentication.",
                    effective_source=normalized_state.active_source,
                    authority_owner=entry.authority_owner,
                )

            if normalized_state.server_auth_state == "session_invalid":
                return self._deny(
                    action_id=action_id,
                    reason_code="server_session_invalid",
                    user_message=f"{action_id} requires a valid server session.",
                    effective_source=normalized_state.active_source,
                    authority_owner=entry.authority_owner,
                )

        return PolicyDecision(
            allowed=True,
            reason_code=None,
            user_message=f"{action_id} is allowed.",
            effective_source=normalized_state.active_source,
            authority_owner=entry.authority_owner,
        )

    @staticmethod
    def _deny(
        *,
        action_id: str,
        reason_code: str,
        user_message: str,
        effective_source: str,
        authority_owner: str,
    ) -> PolicyDecision:
        return PolicyDecision(
            allowed=False,
            reason_code=reason_code,
            user_message=user_message,
            effective_source=effective_source,
            authority_owner=authority_owner,
        )
