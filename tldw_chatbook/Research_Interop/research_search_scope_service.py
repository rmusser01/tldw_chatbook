"""Mode-aware routing for research search provider surfaces."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class ResearchSearchBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "research.search.providers.configure.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local provider configuration CRUD is not exposed by the current research search seam.",
        "affected_action_ids": ["research.search.providers.configure.local"],
    },
    {
        "operation_id": "research.search.providers.observe.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local research search launches are synchronous and do not expose provider observation events.",
        "affected_action_ids": ["research.search.providers.observe.local"],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "research.search.providers.configure.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server research search API does not expose provider configuration CRUD.",
        "affected_action_ids": ["research.search.providers.configure.server"],
    },
    {
        "operation_id": "research.search.providers.observe.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": (
            "The current server research search API is synchronous and does not expose provider observation events."
        ),
        "affected_action_ids": ["research.search.providers.observe.server"],
    },
]


class ResearchSearchScopeService:
    """Route research search provider actions to local or server backends."""

    PAPER_PROVIDER_DEFINITIONS = {
        "arxiv": {
            "display_name": "arXiv",
            "capabilities": ["paper_search"],
        },
        "semantic_scholar": {
            "display_name": "Semantic Scholar",
            "capabilities": ["paper_search"],
        },
        "biorxiv": {
            "display_name": "bioRxiv",
            "capabilities": ["paper_search"],
        },
        "medrxiv": {
            "display_name": "medRxiv",
            "capabilities": ["paper_search"],
        },
        "pubmed": {
            "display_name": "PubMed",
            "capabilities": ["paper_search"],
        },
    }
    SERVER_DETAIL_PAPER_PROVIDERS = {"arxiv", "semantic_scholar", "biorxiv", "medrxiv", "pubmed"}

    def __init__(self, *, local_service: Any = None, server_service: Any = None, policy_enforcer: Any = None):
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

    @classmethod
    def _paper_provider_capabilities(
        cls,
        *,
        mode: ResearchSearchBackend,
        provider_id: str,
        base_capabilities: list[str] | None = None,
    ) -> list[str]:
        capabilities = list(base_capabilities or ["paper_search"])
        if mode == ResearchSearchBackend.SERVER and provider_id in cls.SERVER_DETAIL_PAPER_PROVIDERS:
            if "paper_detail" not in capabilities:
                capabilities.append("paper_detail")
        return capabilities

    @staticmethod
    def _with_backend(mode: ResearchSearchBackend, result: Any, *, entity_kind: str | None = None) -> Any:
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("backend", mode.value)
            if entity_kind is not None:
                payload.setdefault("entity_kind", entity_kind)
            return payload
        return result

    def list_unsupported_capabilities(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchSearchBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

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
        list_paper_providers = getattr(service, "list_supported_paper_providers", None)
        paper_provider_ids = (
            list(await self._maybe_await(list_paper_providers()))
            if callable(list_paper_providers)
            else list(self.PAPER_PROVIDER_DEFINITIONS)
        )
        for provider_id in paper_provider_ids:
            provider_definition = self.PAPER_PROVIDER_DEFINITIONS.get(str(provider_id), {})
            records.append(
                self._provider_record(
                    mode=normalized_mode,
                    provider_type="paper_search",
                    provider_id=str(provider_id),
                    display_name=provider_definition.get("display_name"),
                    capabilities=self._paper_provider_capabilities(
                        mode=normalized_mode,
                        provider_id=str(provider_id),
                        base_capabilities=provider_definition.get("capabilities", ["paper_search"]),
                    ),
                )
            )
        return records

    async def websearch(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == ResearchSearchBackend.LOCAL and self.local_service is None:
            raise ValueError("websearch is server-only when no local research search backend is configured.")
        self._enforce_policy(self._action_id("launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(service.websearch(**kwargs))
        return self._with_backend(normalized_mode, result, entity_kind="research_websearch")

    async def _call_server_only_provider_method(
        self,
        *,
        mode: ResearchSearchBackend | str | None,
        method_name: str,
        action: str,
        entity_kind: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != ResearchSearchBackend.SERVER:
            raise ValueError(f"{method_name} is server-only.")
        self._enforce_policy(self._action_id(action, normalized_mode))
        service = self._service_for_mode(normalized_mode)
        result = await self._maybe_await(getattr(service, method_name)(**kwargs))
        return self._with_backend(normalized_mode, result, entity_kind=entity_kind)

    async def paper_search(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_server_only_provider_method(
            mode=mode,
            method_name="paper_search",
            action="launch",
            entity_kind="paper_search",
            kwargs=kwargs,
        )

    async def paper_detail(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_server_only_provider_method(
            mode=mode,
            method_name="paper_detail",
            action="observe",
            entity_kind="paper_search_detail",
            kwargs=kwargs,
        )

    async def paper_ingest(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_server_only_provider_method(
            mode=mode,
            method_name="paper_ingest",
            action="launch",
            entity_kind="paper_search_ingest",
            kwargs=kwargs,
        )

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

    async def get_arxiv_by_id(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="get_arxiv_by_id", kwargs=kwargs)

    async def get_semantic_scholar_by_id(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(
            mode=mode,
            method_name="get_semantic_scholar_by_id",
            kwargs=kwargs,
        )

    async def _call_provider_method(
        self,
        *,
        mode: ResearchSearchBackend | str | None,
        method_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._action_id("launch", normalized_mode))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, method_name, None)
        if not callable(method):
            raise ValueError(
                f"{method_name} is unavailable for {normalized_mode.value} research search backend."
            )
        result = await self._maybe_await(method(**kwargs))
        return self._with_backend(normalized_mode, result)

    async def search_biorxiv(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="search_biorxiv", kwargs=kwargs)

    async def get_biorxiv_by_doi(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="get_biorxiv_by_doi", kwargs=kwargs)

    async def search_medrxiv(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="search_medrxiv", kwargs=kwargs)

    async def get_medrxiv_by_doi(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="get_medrxiv_by_doi", kwargs=kwargs)

    async def search_pubmed(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="search_pubmed", kwargs=kwargs)

    async def get_pubmed_by_id(
        self,
        *,
        mode: ResearchSearchBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call_provider_method(mode=mode, method_name="get_pubmed_by_id", kwargs=kwargs)
