"""Mode-aware routing for prompt and chatbook parity services."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .prompt_chatbook_normalizers import normalize_chatbook_result, normalize_prompt_result


class PromptChatbookBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class PromptChatbookScopeService:
    """Route prompt/chatbook operations to local or server backends with policy enforcement."""

    def __init__(
        self,
        *,
        local_prompt_service: Any,
        server_prompt_service: Any,
        local_chatbook_service: Any = None,
        server_chatbook_service: Any = None,
        policy_enforcer: Any = None,
    ):
        self.local_prompt_service = local_prompt_service
        self.server_prompt_service = server_prompt_service
        self.local_chatbook_service = local_chatbook_service
        self.server_chatbook_service = server_chatbook_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: PromptChatbookBackend | str | None) -> PromptChatbookBackend:
        if mode is None:
            return PromptChatbookBackend.LOCAL
        if isinstance(mode, PromptChatbookBackend):
            return mode
        try:
            return PromptChatbookBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid prompt/chatbook backend: {mode}") from exc

    def _service_for_mode(self, resource: str, mode: PromptChatbookBackend) -> Any:
        if resource == "prompts":
            service = self.local_prompt_service if mode == PromptChatbookBackend.LOCAL else self.server_prompt_service
        elif resource == "chatbooks":
            service = self.local_chatbook_service if mode == PromptChatbookBackend.LOCAL else self.server_chatbook_service
        else:
            raise ValueError(f"Unknown prompt/chatbook resource: {resource}")
        if service is None:
            raise ValueError(f"{mode.value.title()} {resource} backend is unavailable.")
        return service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _action_id(resource: str, action: str, mode: PromptChatbookBackend) -> str:
        return f"{resource}.{action}.{mode.value}"

    async def _call_service(self, service: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(service, method_name):
            raise NotImplementedError(f"{service.__class__.__name__}.{method_name} is not supported.")
        method = getattr(service, method_name)
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return await self._maybe_await(method(*args, **kwargs))

    async def list_prompts(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("prompts", "list", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("prompts", normalized_mode),
            "list_prompts",
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return normalize_prompt_result(normalized_mode.value, result)

    async def create_prompt(self, *, mode: PromptChatbookBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("prompts", "create", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("prompts", normalized_mode),
            "create_prompt",
            **kwargs,
        )
        return normalize_prompt_result(normalized_mode.value, result)

    async def preview_prompt(self, *, mode: PromptChatbookBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("prompts", "preview", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("prompts", normalized_mode),
            "preview_prompt",
            **kwargs,
        )
        return normalize_prompt_result(normalized_mode.value, result)

    async def update_prompt(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        prompt_id: int | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("prompts", "update", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("prompts", normalized_mode),
            "update_prompt",
            prompt_id,
            **kwargs,
        )
        return normalize_prompt_result(normalized_mode.value, result)

    async def delete_prompt(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        prompt_id: int | str,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("prompts", "delete", normalized_mode))
        return bool(
            await self._call_service(
                self._service_for_mode("prompts", normalized_mode),
                "delete_prompt",
                prompt_id,
            )
        )

    async def preview_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        chatbook_file_path: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "preview_chatbook",
            chatbook_file_path,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook", result)

    async def list_chatbooks(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "list", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "list_chatbooks",
            **kwargs,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook", result)

    async def get_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        chatbook_id: int | str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "get_chatbook",
            chatbook_id,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook", result)

    async def create_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "create", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "create_chatbook",
            **kwargs,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook", result)

    async def update_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        chatbook_id: int | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "update", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "update_chatbook",
            chatbook_id,
            **kwargs,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook", result)

    async def delete_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        chatbook_id: int | str,
    ) -> bool:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "delete", normalized_mode))
        return bool(
            await self._call_service(
                self._service_for_mode("chatbooks", normalized_mode),
                "delete_chatbook",
                chatbook_id,
            )
        )

    async def export_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "export", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "export_chatbook",
            request_data,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook_job", result)

    async def import_chatbook(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        chatbook_file_path: str,
        request_data: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "import", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "import_chatbook",
            chatbook_file_path,
            request_data,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook_job", result)

    async def get_export_job(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "get_export_job",
            job_id,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook_job", result)

    async def get_import_job(
        self,
        *,
        mode: PromptChatbookBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("chatbooks", "detail", normalized_mode))
        result = await self._call_service(
            self._service_for_mode("chatbooks", normalized_mode),
            "get_import_job",
            job_id,
        )
        return normalize_chatbook_result(normalized_mode.value, "chatbook_job", result)
