"""Server-backed chat conversation metadata and history service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.tldw_api import (
    ChatKnowledgeSaveRequest,
    ChatLoopApprovalDecisionRequest,
    ChatLoopStartRequest,
    ConversationShareLinkCreateRequest,
    ConversationUpdateRequest,
    TLDWAPIClient,
)


class ServerChatConversationService:
    """Policy-gated wrapper around tldw_server chat conversation endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient] = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any] | None,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatConversationService":
        return cls(
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatConversationService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server chat conversation operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server chat action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _action_id(action: str) -> str:
        return f"chat.{action}.server"

    @staticmethod
    def _update_request(update_data: ConversationUpdateRequest | Mapping[str, Any]) -> ConversationUpdateRequest:
        if isinstance(update_data, ConversationUpdateRequest):
            return update_data
        return ConversationUpdateRequest(**dict(update_data))

    @staticmethod
    def _loop_start_request(messages: list[dict[str, Any]], extras: Mapping[str, Any]) -> ChatLoopStartRequest:
        payload = dict(extras)
        payload["messages"] = messages
        return ChatLoopStartRequest(**payload)

    @staticmethod
    def _knowledge_save_request(payload: Mapping[str, Any]) -> ChatKnowledgeSaveRequest:
        return ChatKnowledgeSaveRequest(**dict(payload))

    @staticmethod
    def _share_link_create_request(
        payload: ConversationShareLinkCreateRequest | Mapping[str, Any],
    ) -> ConversationShareLinkCreateRequest:
        if isinstance(payload, ConversationShareLinkCreateRequest):
            return payload
        return ConversationShareLinkCreateRequest(**dict(payload))

    async def list_conversations(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("list"))
        return await self._require_client().list_chat_conversations(**kwargs)

    async def get_conversation(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("detail"))
        return await self._require_client().get_chat_conversation(conversation_id, **kwargs)

    async def update_conversation(
        self,
        conversation_id: str,
        update_data: ConversationUpdateRequest | Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("update"))
        return await self._require_client().update_chat_conversation(
            conversation_id,
            self._update_request(update_data),
            **kwargs,
        )

    async def get_conversation_tree(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("detail"))
        return await self._require_client().get_chat_conversation_tree(conversation_id, **kwargs)

    async def get_messages_with_context(self, conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self._enforce(self._action_id("detail"))
        return await self._require_client().get_chat_conversation_messages_with_context(conversation_id, **kwargs)

    async def get_citations(self, conversation_id: str) -> dict[str, Any]:
        self._enforce(self._action_id("detail"))
        return await self._require_client().get_chat_conversation_citations(conversation_id)

    async def list_commands(self) -> Any:
        self._enforce("chat.commands.list.server")
        return await self._require_client().list_chat_commands()

    async def save_knowledge(self, **kwargs: Any) -> Any:
        self._enforce("chat.knowledge.create.server")
        return await self._require_client().save_chat_knowledge(self._knowledge_save_request(kwargs))

    async def create_share_link(
        self,
        conversation_id: str,
        request_data: ConversationShareLinkCreateRequest | Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        self._enforce("chat.share_links.create.server")
        return await self._require_client().create_chat_conversation_share_link(
            conversation_id,
            self._share_link_create_request(request_data),
            **kwargs,
        )

    async def list_share_links(self, conversation_id: str, **kwargs: Any) -> Any:
        self._enforce("chat.share_links.list.server")
        return await self._require_client().list_chat_conversation_share_links(conversation_id, **kwargs)

    async def revoke_share_link(self, conversation_id: str, share_id: str, **kwargs: Any) -> Any:
        self._enforce("chat.share_links.revoke.server")
        return await self._require_client().revoke_chat_conversation_share_link(
            conversation_id,
            share_id,
            **kwargs,
        )

    async def resolve_share_token(self, share_token: str, *, limit: int = 200) -> Any:
        self._enforce("chat.share_links.detail.server")
        return await self._require_client().resolve_chat_conversation_share_token(share_token, limit=limit)

    async def get_analytics(self, **kwargs: Any) -> Any:
        self._enforce("chat.analytics.observe.server")
        return await self._require_client().get_chat_analytics(**kwargs)

    async def start_loop(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self._enforce("chat.loop.launch.server")
        return await self._require_client().start_chat_loop_run(
            self._loop_start_request(messages=messages, extras=kwargs)
        )

    async def list_loop_events(self, run_id: str, *, after_seq: int = 0) -> Any:
        self._enforce("chat.loop.observe.server")
        return await self._require_client().list_chat_loop_events(run_id, after_seq=after_seq)

    async def approve_loop_call(self, run_id: str, *, approval_id: str) -> Any:
        self._enforce("chat.loop.approve.server")
        return await self._require_client().approve_chat_loop_call(
            run_id,
            ChatLoopApprovalDecisionRequest(approval_id=approval_id, decision="approve"),
        )

    async def reject_loop_call(self, run_id: str, *, approval_id: str) -> Any:
        self._enforce("chat.loop.approve.server")
        return await self._require_client().reject_chat_loop_call(
            run_id,
            ChatLoopApprovalDecisionRequest(approval_id=approval_id, decision="reject"),
        )

    async def cancel_loop(self, run_id: str) -> Any:
        self._enforce("chat.loop.cancel.server")
        return await self._require_client().cancel_chat_loop_run(run_id)

    async def create_conversation(self, **_kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("create"))
        raise NotImplementedError(
            "tldw_server does not expose first-class conversation create outside chat launch/persist flows."
        )

    async def delete_conversation(self, conversation_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("delete"))
        raise NotImplementedError(
            "tldw_server does not expose conversation delete through the chat conversation contract."
        )
