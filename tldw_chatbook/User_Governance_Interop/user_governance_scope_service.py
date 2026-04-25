"""Source-aware routing for remote-owned user-governance state."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class UserGovernanceBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "user_governance.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server user-governance state is unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "user_governance.admin_scopes.server",
        "source": "server",
        "supported": False,
        "reason_code": "out_of_scope_admin_surface",
        "user_message": "Org/team privilege maps, privilege snapshots, exports, and resource-governor policy administration are not exposed in Chatbook.",
        "affected_action_ids": [],
    }
]


class UserGovernanceScopeService:
    """Route consent and privilege-map calls without mirroring server identity state locally."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: UserGovernanceBackend | str | None) -> UserGovernanceBackend:
        if mode is None:
            return UserGovernanceBackend.SERVER
        if isinstance(mode, UserGovernanceBackend):
            return mode
        try:
            return UserGovernanceBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid user-governance backend: {mode}") from exc

    def _require_server_service(self, mode: UserGovernanceBackend) -> Any:
        if mode == UserGovernanceBackend.LOCAL:
            raise ValueError(
                "User governance is server-only; Chatbook does not invent local consent or privilege-map state."
            )
        if self.server_service is None:
            raise ValueError("Server user-governance backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _consent_record_id(mode: UserGovernanceBackend, item: dict[str, Any]) -> str | None:
        source_id = item.get("id")
        if source_id is None:
            purpose = item.get("purpose")
            user_id = item.get("user_id")
            if purpose is None or user_id is None:
                return None
            source_id = f"{user_id}:{purpose}"
        return f"{mode.value}:consent:{source_id}"

    @staticmethod
    def _privilege_record_id(mode: UserGovernanceBackend, item: dict[str, Any], *, user_id: str | None = None) -> str | None:
        privilege_scope_id = item.get("privilege_scope_id")
        method = item.get("method")
        endpoint = item.get("endpoint")
        if privilege_scope_id is None or method is None or endpoint is None:
            return None
        if user_id is None:
            return f"{mode.value}:privilege:{privilege_scope_id}:{method}:{endpoint}"
        return f"{mode.value}:privilege:{user_id}:{privilege_scope_id}:{method}:{endpoint}"

    def _with_consent_record_id(self, mode: UserGovernanceBackend, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        record_id = self._consent_record_id(mode, record)
        if record_id is not None:
            record.setdefault("record_id", record_id)
        return record

    def _normalize_consent_preferences(self, mode: UserGovernanceBackend, result: dict[str, Any]) -> dict[str, Any]:
        payload = dict(result or {})
        payload.setdefault("backend", mode.value)
        user_id = payload.get("user_id")
        if user_id is not None:
            payload.setdefault("record_id", f"{mode.value}:consent_preferences:{user_id}")
        if isinstance(payload.get("consents"), list):
            payload["consents"] = [
                self._with_consent_record_id(mode, item) if isinstance(item, dict) else item
                for item in payload["consents"]
            ]
        return payload

    def _normalize_privilege_map(
        self,
        mode: UserGovernanceBackend,
        result: dict[str, Any],
        *,
        map_id: str,
        item_user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = dict(result or {})
        payload.setdefault("backend", mode.value)
        payload.setdefault("record_id", f"{mode.value}:privilege_map:{map_id}")
        if isinstance(payload.get("items"), list):
            normalized_items: list[Any] = []
            for item in payload["items"]:
                if not isinstance(item, dict):
                    normalized_items.append(item)
                    continue
                record = dict(item)
                record.setdefault("backend", mode.value)
                record_id = self._privilege_record_id(mode, record, user_id=item_user_id)
                if record_id is not None:
                    record.setdefault("record_id", record_id)
                normalized_items.append(record)
            payload["items"] = normalized_items
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: UserGovernanceBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == UserGovernanceBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _call(
        self,
        *,
        mode: UserGovernanceBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[UserGovernanceBackend, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return normalized_mode, result

    async def get_consent_preferences(
        self,
        *,
        mode: UserGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="user_governance.consent.list.server",
            method_name="get_consent_preferences",
        )
        return self._normalize_consent_preferences(normalized_mode, result)

    async def grant_consent(
        self,
        purpose: str,
        *,
        mode: UserGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="user_governance.consent.update.server",
            method_name="grant_consent",
            args=(purpose,),
        )
        return self._with_consent_record_id(normalized_mode, result)

    async def withdraw_consent(
        self,
        purpose: str,
        *,
        mode: UserGovernanceBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="user_governance.consent.update.server",
            method_name="withdraw_consent",
            args=(purpose,),
        )
        return self._with_consent_record_id(normalized_mode, result)

    async def get_self_privilege_map(
        self,
        *,
        mode: UserGovernanceBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="user_governance.privileges.list.server",
            method_name="get_self_privilege_map",
            kwargs=kwargs,
        )
        return self._normalize_privilege_map(normalized_mode, result, map_id="self")

    async def get_user_privilege_map(
        self,
        user_id: str,
        *,
        mode: UserGovernanceBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode, result = await self._call(
            mode=mode,
            action_id="user_governance.privileges.detail.server",
            method_name="get_user_privilege_map",
            args=(user_id,),
            kwargs=kwargs,
        )
        return self._normalize_privilege_map(normalized_mode, result, map_id=user_id, item_user_id=user_id)
