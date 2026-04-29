"""Server prompt CRUD adapter for source-aware prompt parity."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import PromptCreateRequest, PromptPreviewRequest


class ServerPromptService:
    """Delegate prompt operations to the configured tldw_server API client."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ):
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: dict[str, Any] | None,
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> "ServerPromptService":
        if client_provider is not None:
            return cls(
                client=None,
                client_provider=client_provider,
                policy_enforcer=policy_enforcer,
            )
        from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

        return cls(
            build_runtime_api_client(app_config=app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerPromptService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("Server prompt client is unavailable.")

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
                    user_message=getattr(decision, "user_message", None) or "Server prompt action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _action_id(action: str) -> str:
        return f"prompts.{action}.server"

    @staticmethod
    def _version_action_id(action: str) -> str:
        return f"prompts.versions.{action}.server"

    @staticmethod
    def _subresource_action_id(resource: str, action: str) -> str:
        return f"prompts.{resource}.{action}.server"

    async def list_prompts(self, *, include_deleted: bool = False, **_kwargs: Any) -> Any:
        self._enforce(self._action_id("list"))
        return await self._require_client().list_prompts(include_deleted=include_deleted)

    async def create_prompt(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("create"))
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptCreateRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptCreateRequest(**request_data)
        return await self._require_client().create_prompt(request_data)

    async def preview_prompt(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("preview"))
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptPreviewRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptPreviewRequest(**request_data)
        return await self._require_client().preview_prompt(request_data)

    async def update_prompt(self, prompt_id: int | str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("update"))
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptCreateRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptCreateRequest(**request_data)
        return await self._require_client().update_prompt(prompt_id, request_data)

    async def delete_prompt(self, prompt_id: int | str) -> bool:
        self._enforce(self._action_id("delete"))
        await self._require_client().delete_prompt(prompt_id)
        return True

    async def list_prompt_versions(self, prompt_id: int | str) -> Any:
        self._enforce(self._version_action_id("list"))
        return await self._require_client().list_prompt_versions(prompt_id)

    async def restore_prompt_version(self, prompt_id: int | str, version: int) -> dict[str, Any]:
        self._enforce(self._version_action_id("restore"))
        return await self._require_client().restore_prompt_version(prompt_id, version)

    async def get_prompts_health(self) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("health", "detail"))
        return await self._require_client().get_prompts_health()

    async def get_prompt_sync_log(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("sync_log", "list"))
        return await self._require_client().get_prompt_sync_log(**kwargs)

    async def search_prompts(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("search", "list"))
        return await self._require_client().search_prompts(**kwargs)

    async def create_prompt_keyword(self, keyword_text: str) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("keywords", "create"))
        return await self._require_client().create_prompt_keyword(keyword_text)

    async def list_prompt_keywords(self) -> Any:
        self._enforce(self._subresource_action_id("keywords", "list"))
        return await self._require_client().list_prompt_keywords()

    async def delete_prompt_keyword(self, keyword_text: str) -> Any:
        self._enforce(self._subresource_action_id("keywords", "delete"))
        return await self._require_client().delete_prompt_keyword(keyword_text)

    async def export_prompts(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("transfer", "export"))
        return await self._require_client().export_prompts(**kwargs)

    async def export_prompt_keywords(self) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("keywords", "export"))
        return await self._require_client().export_prompt_keywords()

    async def import_prompts(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("transfer", "import"))
        return await self._require_client().import_prompts(payload)

    async def extract_prompt_template_variables(self, template: str) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("templates", "process"))
        return await self._require_client().extract_prompt_template_variables(template)

    async def render_prompt_template(self, template: str, variables: dict[str, Any]) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("templates", "process"))
        return await self._require_client().render_prompt_template(template, variables)

    async def convert_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("templates", "process"))
        return await self._require_client().convert_prompt(payload)

    async def bulk_delete_prompts(self, prompt_ids: list[int]) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("bulk", "delete"))
        return await self._require_client().bulk_delete_prompts(prompt_ids)

    async def bulk_update_prompt_keywords(
        self,
        prompt_ids: list[int],
        keywords: list[str] | None = None,
        *,
        mode: str = "add",
    ) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("bulk", "update"))
        return await self._require_client().bulk_update_prompt_keywords(prompt_ids, keywords, mode=mode)

    async def record_prompt_usage(self, prompt_identifier: int | str) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("usage", "update"))
        return await self._require_client().record_prompt_usage(prompt_identifier)

    async def create_prompt_collection(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("collections", "create"))
        return await self._require_client().create_prompt_collection(**kwargs)

    async def list_prompt_collections(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("collections", "list"))
        return await self._require_client().list_prompt_collections(**kwargs)

    async def get_prompt_collection(self, collection_id: int) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("collections", "detail"))
        return await self._require_client().get_prompt_collection(collection_id)

    async def update_prompt_collection(self, collection_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._subresource_action_id("collections", "update"))
        return await self._require_client().update_prompt_collection(collection_id, **kwargs)
