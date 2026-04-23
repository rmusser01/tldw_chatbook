"""Source-aware router for chat conversation metadata operations."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Mapping


class ChatConversationBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ChatConversationScopeService:
    """Route chat conversation actions to the selected backend without source fallback."""

    _ACTION_IDS = {
        "list": "chat.list",
        "detail": "chat.detail",
        "update": "chat.update",
    }

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ChatConversationBackend | str | None) -> ChatConversationBackend:
        if mode is None:
            return ChatConversationBackend.LOCAL
        if isinstance(mode, ChatConversationBackend):
            return mode
        try:
            return ChatConversationBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid chat conversation backend: {mode}") from exc

    def _service_for_mode(self, mode: ChatConversationBackend) -> Any:
        if mode == ChatConversationBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local chat conversation backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server chat conversation backend is unavailable.")
        return self.server_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, mode: ChatConversationBackend, action: str) -> None:
        if self.policy_enforcer is None:
            return
        action_prefix = self._ACTION_IDS.get(action)
        if action_prefix is None:
            return
        self.policy_enforcer.require_allowed(action_id=f"{action_prefix}.{mode.value}")

    @staticmethod
    def _scope_matches(
        record: Mapping[str, Any] | None,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        if record is None:
            return True
        if scope_type is None and workspace_id is None:
            return True

        requested_scope = str(scope_type or ("workspace" if workspace_id else "global")).strip().lower()
        actual_scope = str(record.get("scope_type") or "global").strip().lower()
        if requested_scope == "global":
            return actual_scope == "global"
        if requested_scope != "workspace":
            raise ValueError("scope_type must be 'global' or 'workspace'")

        if actual_scope != "workspace":
            return False
        if workspace_id is None:
            raise ValueError("workspace_id is required when scope_type='workspace'")
        return str(record.get("workspace_id") or "").strip() == str(workspace_id).strip()

    async def list_conversations(
        self,
        *,
        mode: ChatConversationBackend | str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
        scope_type: str | None = None,
        workspace_id: str | None = None,
        include_deleted: bool = False,
        deleted_only: bool = False,
        state: str | None = None,
        topic_label: str | None = None,
        character_id: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "list")
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.list_conversations(
                query=query,
                limit=limit,
                offset=offset,
                scope_type=scope_type,
                workspace_id=workspace_id,
                include_deleted=include_deleted,
                deleted_only=deleted_only,
                state=state,
                topic_label=topic_label,
                character_id=character_id,
            )
        )

    async def get_conversation(
        self,
        conversation_id: str,
        *,
        mode: ChatConversationBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "detail")
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ChatConversationBackend.SERVER:
            return await self._maybe_await(
                service.get_conversation_metadata(
                    conversation_id,
                    scope_type=scope_type,
                    workspace_id=workspace_id,
                )
            )

        record = await self._maybe_await(service.get_conversation_metadata(conversation_id))
        if not self._scope_matches(record, scope_type=scope_type, workspace_id=workspace_id):
            return None
        return record

    async def update_conversation(
        self,
        conversation_id: str,
        update_data: Mapping[str, Any],
        *,
        expected_version: int,
        mode: ChatConversationBackend | str | None = None,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "update")
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == ChatConversationBackend.SERVER:
            return await self._maybe_await(
                service.update_conversation_metadata(
                    conversation_id,
                    dict(update_data),
                    expected_version,
                    scope_type=scope_type,
                    workspace_id=workspace_id,
                )
            )
        return await self._maybe_await(
            service.update_conversation_metadata(
                conversation_id,
                dict(update_data),
                expected_version,
            )
        )

    async def get_conversation_tree(
        self,
        conversation_id: str,
        *,
        mode: ChatConversationBackend | str | None = None,
        root_limit: int = 50,
        root_offset: int = 0,
        max_depth: int = 4,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(normalized_mode, "detail")
        service = self._service_for_mode(normalized_mode)
        kwargs = {
            "root_limit": root_limit,
            "root_offset": root_offset,
            "depth_cap": max_depth,
        }
        if normalized_mode == ChatConversationBackend.SERVER:
            kwargs.update({"scope_type": scope_type, "workspace_id": workspace_id})
        return await self._maybe_await(service.get_conversation_tree(conversation_id, **kwargs))
