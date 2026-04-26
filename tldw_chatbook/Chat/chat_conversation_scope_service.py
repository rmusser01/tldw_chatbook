"""Source-aware chat conversation service seam."""

from __future__ import annotations

import inspect
from typing import Any, Mapping


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "chat.loop.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local chat loop run control is not implemented in Chatbook yet.",
        "affected_action_ids": [
            "chat.loop.launch.local",
            "chat.loop.observe.local",
            "chat.loop.approve.local",
            "chat.loop.cancel.local",
        ],
    },
    {
        "operation_id": "chat.adjunct_controls.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Server chat adjunct controls are not available in local Chatbook mode.",
        "affected_action_ids": [
            "chat.analytics.observe.local",
            "chat.commands.list.local",
            "chat.knowledge.create.local",
            "chat.share_links.create.local",
            "chat.share_links.detail.local",
            "chat.share_links.list.local",
            "chat.share_links.revoke.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "chat.conversation.create.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server chat conversation contract does not expose first-class conversation creation outside chat launch/persist flows.",
        "affected_action_ids": ["chat.create.server"],
    },
    {
        "operation_id": "chat.conversation.delete.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server chat conversation contract does not expose conversation deletion.",
        "affected_action_ids": ["chat.delete.server"],
    },
]


class ChatConversationScopeService:
    """Route chat conversation operations to local or server backends with policy gates."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        normalized = str(mode or "local").strip().lower()
        if normalized not in {"local", "server"}:
            raise ValueError("mode must be 'local' or 'server'")
        return normalized

    @staticmethod
    def _action_id(action: str, mode: str) -> str:
        return f"chat.{action}.{mode}"

    @staticmethod
    def _loop_action_id(action: str, mode: str) -> str:
        return f"chat.loop.{action}.{mode}"

    @staticmethod
    def _commands_action_id(action: str, mode: str) -> str:
        return f"chat.commands.{action}.{mode}"

    @staticmethod
    def _knowledge_action_id(action: str, mode: str) -> str:
        return f"chat.knowledge.{action}.{mode}"

    @staticmethod
    def _share_links_action_id(action: str, mode: str) -> str:
        return f"chat.share_links.{action}.{mode}"

    @staticmethod
    def _analytics_action_id(action: str, mode: str) -> str:
        return f"chat.analytics.{action}.{mode}"

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    def _service_for_mode(self, mode: str) -> Any:
        service = self.server_service if mode == "server" else self.local_service
        if service is None:
            raise ValueError(f"Chat conversation {mode} service is unavailable.")
        return service

    def list_unsupported_capabilities(self, *, mode: str | None = None) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "local":
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_conversations(self, *, mode: str = "local", **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("list", normalized_mode))
        return await self._maybe_await(self._service_for_mode(normalized_mode).list_conversations(**kwargs))

    async def get_conversation(self, conversation_id: str, *, mode: str = "local", **kwargs: Any) -> dict[str, Any] | None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("detail", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == "server":
            return await self._maybe_await(service.get_conversation(conversation_id, **kwargs))
        return await self._maybe_await(service.get_conversation_metadata(conversation_id))

    async def update_conversation(
        self,
        conversation_id: str,
        update_data: Mapping[str, Any],
        *,
        mode: str = "local",
        expected_version: int | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("update", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        payload = dict(update_data)
        if normalized_mode == "server":
            return await self._maybe_await(service.update_conversation(conversation_id, payload, **kwargs))

        version = expected_version if expected_version is not None else payload.pop("version", None)
        if version is None:
            raise ValueError("expected_version or update_data['version'] is required for local conversation updates.")

        keywords = payload.pop("keywords", None)
        metadata_updated = True
        if payload:
            metadata_updated = bool(
                await self._maybe_await(
                    service.update_conversation_metadata(conversation_id, payload, int(version))
                )
            )
        if keywords is not None and metadata_updated:
            await self._maybe_await(service.replace_conversation_keywords(conversation_id, list(keywords)))
        return metadata_updated

    async def get_conversation_tree(self, conversation_id: str, *, mode: str = "local", **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("detail", normalized_mode))
        service_kwargs = dict(kwargs)
        if normalized_mode == "local":
            if "limit" in service_kwargs and "root_limit" not in service_kwargs:
                service_kwargs["root_limit"] = service_kwargs.pop("limit")
            if "offset" in service_kwargs and "root_offset" not in service_kwargs:
                service_kwargs["root_offset"] = service_kwargs.pop("offset")
            if "max_depth" in service_kwargs and "depth_cap" not in service_kwargs:
                service_kwargs["depth_cap"] = service_kwargs.pop("max_depth")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).get_conversation_tree(conversation_id, **service_kwargs)
        )

    async def get_messages_with_context(
        self,
        conversation_id: str,
        *,
        mode: str = "server",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("detail", normalized_mode))
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).get_messages_with_context(conversation_id, **kwargs)
        )

    async def get_citations(self, conversation_id: str, *, mode: str = "server") -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("detail", normalized_mode))
        return await self._maybe_await(self._service_for_mode(normalized_mode).get_citations(conversation_id))

    async def list_commands(self, *, mode: str = "server") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._commands_action_id("list", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat commands are unavailable in local mode.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).list_commands())

    async def save_knowledge(self, *, mode: str = "server", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._knowledge_action_id("create", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat knowledge-save is unavailable in local mode.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).save_knowledge(**kwargs))

    async def create_share_link(
        self,
        conversation_id: str,
        request_data: Mapping[str, Any],
        *,
        mode: str = "server",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._share_links_action_id("create", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat share links are unavailable in local mode.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).create_share_link(conversation_id, request_data, **kwargs)
        )

    async def list_share_links(self, conversation_id: str, *, mode: str = "server", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._share_links_action_id("list", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat share links are unavailable in local mode.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).list_share_links(conversation_id, **kwargs))

    async def revoke_share_link(
        self,
        conversation_id: str,
        share_id: str,
        *,
        mode: str = "server",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._share_links_action_id("revoke", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat share links are unavailable in local mode.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).revoke_share_link(conversation_id, share_id, **kwargs)
        )

    async def resolve_share_token(self, share_token: str, *, mode: str = "server", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._share_links_action_id("detail", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat share links are unavailable in local mode.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).resolve_share_token(share_token, **kwargs))

    async def get_analytics(self, *, mode: str = "server", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._analytics_action_id("observe", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Server chat analytics are unavailable in local mode.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).get_analytics(**kwargs))

    async def start_loop(self, *, mode: str = "server", messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._loop_action_id("launch", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Local chat loop runs are not implemented in Chatbook yet.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).start_loop(messages=messages, **kwargs)
        )

    async def list_loop_events(
        self,
        run_id: str,
        *,
        mode: str = "server",
        after_seq: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._loop_action_id("observe", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Local chat loop runs are not implemented in Chatbook yet.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).list_loop_events(run_id, after_seq=after_seq)
        )

    async def approve_loop_call(
        self,
        run_id: str,
        *,
        mode: str = "server",
        approval_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._loop_action_id("approve", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Local chat loop runs are not implemented in Chatbook yet.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).approve_loop_call(run_id, approval_id=approval_id)
        )

    async def reject_loop_call(
        self,
        run_id: str,
        *,
        mode: str = "server",
        approval_id: str,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._loop_action_id("approve", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Local chat loop runs are not implemented in Chatbook yet.")
        return await self._maybe_await(
            self._service_for_mode(normalized_mode).reject_loop_call(run_id, approval_id=approval_id)
        )

    async def cancel_loop(self, run_id: str, *, mode: str = "server") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._loop_action_id("cancel", normalized_mode))
        if normalized_mode == "local":
            raise NotImplementedError("Local chat loop runs are not implemented in Chatbook yet.")
        return await self._maybe_await(self._service_for_mode(normalized_mode).cancel_loop(run_id))

    async def create_conversation(self, *, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("create", normalized_mode))
        return await self._maybe_await(self._service_for_mode(normalized_mode).create_conversation(**kwargs))

    async def delete_conversation(
        self,
        conversation_id: str,
        *,
        expected_version: int,
        mode: str = "local",
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("delete", normalized_mode))
        return bool(
            await self._maybe_await(
                self._service_for_mode(normalized_mode).delete_conversation(
                    conversation_id,
                    expected_version=expected_version,
                )
            )
        )
