"""Mode-aware routing for research search provider surfaces."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ResearchSearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class ResearchSearchScopeService:
    """Route research search provider actions to local or server backends."""

    SERVER_PAPER_PROVIDERS = (
        {
            "provider_id": "arxiv",
            "display_name": "arXiv",
            "capabilities": ["paper_search"],
        },
        {
            "provider_id": "semantic_scholar",
            "display_name": "Semantic Scholar",
            "capabilities": ["paper_search"],
        },
    )

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: ResearchSearchBackend | str | None) -> ResearchSearchBackend:
        if mode is None:
            return ResearchSearchBackend.LOCAL
        if isinstance(mode, ResearchSearchBackend):
            return mode
        try:
            return ResearchSearchBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid research search backend: {mode}") from exc

    def _service_for_mode(self, mode: ResearchSearchBackend) -> Any:
        if mode == ResearchSearchBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local research search backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server research search backend is unavailable.")
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
    def _action_id(action: str, mode: ResearchSearchBackend) -> str:
        return f"research.search.providers.{action}.{mode.value}"

    @staticmethod
    def _provider_record(
        *,
        mode: ResearchSearchBackend,
        provider_type: str,
        provider_id: str,
        display_name: str | None = None,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "record_id": f"{mode.value}:research_search_provider:{provider_type}:{provider_id}",
            "backend": mode.value,
            "provider_type": provider_type,
            "provider_id": provider_id,
            "display_name": display_name or provider_id.replace("_", " ").title(),
            "capabilities": capabilities or [provider_type],
            "config_scope": mode.value,
        }

    @staticmethod
    def _with_backend(mode: ResearchSearchBackend, result: Any) -> Any:
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("backend", mode.value)
            return payload
        return result

    async def list_supported_websearch_engines(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
    ) -> list[str]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        return list(await self._maybe_await(service.list_supported_websearch_engines()))

    async def list_provider_catalog(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("list", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        engines = list(await self._maybe_await(service.list_supported_websearch_engines()))
        records = [
            self._provider_record(
                mode=normalized_mode,
                provider_type="websearch",
                provider_id=str(engine),
                capabilities=["websearch"],
            )
            for engine in engines
        ]
        if normalized_mode == ResearchSearchBackend.SERVER:
            records.extend(
                self._provider_record(
                    mode=normalized_mode,
                    provider_type="paper_search",
                    provider_id=provider["provider_id"],
                    display_name=provider["display_name"],
                    capabilities=provider["capabilities"],
                )
                for provider in self.SERVER_PAPER_PROVIDERS
            )
        return records

    async def websearch(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.websearch(**kwargs))
        return self._with_backend(normalized_mode, result)

    async def search_arxiv(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.search_arxiv(**kwargs))
        return self._with_backend(normalized_mode, result)

    async def search_semantic_scholar(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.search_semantic_scholar(**kwargs))
        return self._with_backend(normalized_mode, result)
