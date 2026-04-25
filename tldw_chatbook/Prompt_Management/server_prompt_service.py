"""Server prompt CRUD adapter for source-aware prompt parity."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import PromptCreateRequest, PromptPreviewRequest


class ServerPromptService:
    """Delegate prompt operations to the configured tldw_server API client."""

    def __init__(self, client: Any, *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: dict[str, Any] | None,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerPromptService":
        from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

        return cls(
            build_runtime_api_client(app_config=app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> Any:
        if self.client is None:
            raise ValueError("Server prompt client is unavailable.")
        return self.client

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
