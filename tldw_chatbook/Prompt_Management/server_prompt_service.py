"""Server prompt CRUD adapter for source-aware prompt parity."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.tldw_api.prompt_chatbook_schemas import PromptCreateRequest, PromptPreviewRequest


class ServerPromptService:
    """Delegate prompt operations to the configured tldw_server API client."""

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    def from_config(cls, app_config: dict[str, Any] | None) -> "ServerPromptService":
        from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client

        return cls(build_runtime_api_client(app_config=app_config))

    def _require_client(self) -> Any:
        if self.client is None:
            raise ValueError("Server prompt client is unavailable.")
        return self.client

    async def list_prompts(self, *, include_deleted: bool = False, **_kwargs: Any) -> Any:
        return await self._require_client().list_prompts(include_deleted=include_deleted)

    async def create_prompt(self, **kwargs: Any) -> dict[str, Any]:
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptCreateRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptCreateRequest(**request_data)
        return await self._require_client().create_prompt(request_data)

    async def preview_prompt(self, **kwargs: Any) -> dict[str, Any]:
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptPreviewRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptPreviewRequest(**request_data)
        return await self._require_client().preview_prompt(request_data)

    async def update_prompt(self, prompt_id: int | str, **kwargs: Any) -> dict[str, Any]:
        request_data = kwargs.get("request_data")
        if request_data is None:
            request_data = PromptCreateRequest(**kwargs)
        elif isinstance(request_data, dict):
            request_data = PromptCreateRequest(**request_data)
        return await self._require_client().update_prompt(prompt_id, request_data)

    async def delete_prompt(self, prompt_id: int | str) -> bool:
        await self._require_client().delete_prompt(prompt_id)
        return True
