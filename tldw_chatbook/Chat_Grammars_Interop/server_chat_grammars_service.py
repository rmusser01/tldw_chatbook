"""Server-backed saved chat grammar service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import ChatGrammarCreate, ChatGrammarUpdate
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerChatGrammarsService:
    """Policy-gated access to server saved grammar APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatGrammarsService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatGrammarsService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server chat grammar operations.")

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
                    user_message=getattr(decision, "user_message", None)
                    or "Server chat grammar action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, (dict, list, bool)):
            return response
        return dict(response or {})

    async def create_grammar(
        self,
        *,
        name: str,
        grammar_text: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        self._enforce("chat.grammars.create.server")
        request = ChatGrammarCreate(name=name, description=description, grammar_text=grammar_text)
        return self._dump(await self._require_client().create_chat_grammar(request))

    async def list_grammars(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("chat.grammars.list.server")
        return self._dump(
            await self._require_client().list_chat_grammars(
                include_archived=include_archived,
                limit=limit,
                offset=offset,
            )
        )

    async def get_grammar(self, grammar_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        self._enforce("chat.grammars.detail.server")
        return self._dump(await self._require_client().get_chat_grammar(grammar_id, include_archived=include_archived))

    async def update_grammar(
        self,
        grammar_id: str,
        *,
        version: int | None = None,
        name: str | None = None,
        description: str | None = None,
        grammar_text: str | None = None,
        validation_status: str | None = None,
        validation_error: str | None = None,
        last_validated_at: Any = None,
        is_archived: bool | None = None,
    ) -> dict[str, Any]:
        self._enforce("chat.grammars.update.server")
        request = ChatGrammarUpdate(
            version=version,
            name=name,
            description=description,
            grammar_text=grammar_text,
            validation_status=validation_status,  # type: ignore[arg-type]
            validation_error=validation_error,
            last_validated_at=last_validated_at,
            is_archived=is_archived,
        )
        return self._dump(await self._require_client().update_chat_grammar(grammar_id, request))

    async def delete_grammar(self, grammar_id: str, *, hard_delete: bool = False) -> bool:
        self._enforce("chat.grammars.delete.server")
        return bool(await self._require_client().delete_chat_grammar(grammar_id, hard_delete=hard_delete))
