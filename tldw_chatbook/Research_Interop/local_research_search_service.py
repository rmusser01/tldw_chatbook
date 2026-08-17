"""Local research search provider service."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ..runtime_policy.types import PolicyDeniedError
from .research_source_catalog import catalog_entries


LOCAL_SUPPORTED_WEBSEARCH_ENGINES = {
    "baidu",
    "bing",
    "brave",
    "duckduckgo",
    "exa",
    "google",
    "kagi",
    "searx",
    "serper",
    "tavily",
    "yandex",
}
# task-16792: the paper-provider listing IS the catalog (one source of
# truth -- a hardcoded tuple here drifted from the runnable lanes).
LOCAL_SUPPORTED_PAPER_PROVIDERS = tuple(e.source_id for e in catalog_entries())


class LocalResearchSearchService:
    """Policy-gated local research search provider launcher."""

    def __init__(
        self,
        *,
        websearch_runner: Callable[[str, dict[str, Any]], Any] | None = None,
        aggregate_runner: Callable[
            [dict[str, Any], dict[str, Any], dict[str, Any]], Any
        ]
        | None = None,
        arxiv_runner: Callable[..., Any] | None = None,
        semantic_scholar_runner: Callable[..., Any] | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.websearch_runner = websearch_runner or self._default_websearch_runner
        self.aggregate_runner = aggregate_runner or self._default_aggregate_runner
        self.arxiv_runner = arxiv_runner or self._default_arxiv_runner
        self.semantic_scholar_runner = (
            semantic_scholar_runner or self._default_semantic_scholar_runner
        )
        self.policy_enforcer = policy_enforcer

    @staticmethod
    def _default_websearch_runner(question: str, search_params: dict[str, Any]) -> Any:
        from ..Web_Scraping.WebSearch_APIs import generate_and_search

        return generate_and_search(question, search_params)

    @staticmethod
    async def _default_aggregate_runner(
        web_search_results_dict: dict[str, Any],
        sub_query_dict: dict[str, Any],
        search_params: dict[str, Any],
    ) -> Any:
        from ..Web_Scraping.WebSearch_APIs import analyze_and_aggregate

        return await analyze_and_aggregate(
            web_search_results_dict, sub_query_dict, search_params
        )

    @staticmethod
    def _default_arxiv_runner(
        *,
        query: str | None = None,
        author: str | None = None,
        year: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        # task-16326: httpx + retry/backoff + DOI normalization live in
        # academic_providers; this seam stays so tests can inject runners.
        from .academic_providers import search_arxiv

        return search_arxiv(
            query=query,
            author=author,
            year=year,
            page=page,
            results_per_page=results_per_page,
        )

    @staticmethod
    def _coerce_csv(value: list[str] | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return ",".join(str(item) for item in value)

    @classmethod
    def _default_semantic_scholar_runner(
        cls,
        *,
        query: str,
        fields_of_study: list[str] | str | None = None,
        publication_types: list[str] | str | None = None,
        year_range: str | None = None,
        venue: list[str] | str | None = None,
        min_citations: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        # task-16326: httpx + retry/backoff + config-resolved API key live
        # in academic_providers; this seam stays for runner injection.
        from .academic_providers import (
            resolve_semantic_scholar_api_key,
            search_semantic_scholar,
        )

        return search_semantic_scholar(
            query=query,
            fields_of_study=fields_of_study,
            publication_types=publication_types,
            year_range=year_range,
            venue=venue,
            min_citations=min_citations,
            page=page,
            results_per_page=results_per_page,
            api_key=resolve_semantic_scholar_api_key(),
        )


    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(
            self.policy_enforcer, "require_ui_action_allowed", None
        )
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None)
                    or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Local research search action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None)
                    or "local",
                    authority_owner=getattr(decision, "authority_owner", None)
                    or "local",
                )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _normalize_engine(engine: str) -> str:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.research_search_schemas import WEBSEARCH_ENGINE_ALIASES

        return WEBSEARCH_ENGINE_ALIASES.get(str(engine).lower(), str(engine).lower())

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def list_supported_websearch_engines(self) -> list[str]:
        self._enforce("research.search.providers.list.local")
        return sorted(LOCAL_SUPPORTED_WEBSEARCH_ENGINES)

    async def list_supported_paper_providers(self) -> list[str]:
        self._enforce("research.search.providers.list.local")
        return list(LOCAL_SUPPORTED_PAPER_PROVIDERS)

    async def websearch(
        self,
        *,
        query: str,
        engine: str = "google",
        result_count: int = 10,
        aggregate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import WebSearchRequest

        self._enforce("research.search.providers.launch.local")
        normalized_engine = self._normalize_engine(engine)
        if normalized_engine not in LOCAL_SUPPORTED_WEBSEARCH_ENGINES:
            supported = ", ".join(sorted(LOCAL_SUPPORTED_WEBSEARCH_ENGINES))
            raise ValueError(
                f"Unsupported local websearch engine: {engine}. Supported engines: {supported}"
            )

        request = WebSearchRequest(
            query=query,
            engine=normalized_engine,
            result_count=result_count,
            aggregate=aggregate,
            **kwargs,
        )
        search_params = request.model_dump(exclude_none=True, mode="json")
        search_params.pop("query", None)
        result = self._dump(
            await self._maybe_await(self.websearch_runner(query, search_params))
        )

        if not aggregate:
            return result

        web_results = result.get("web_search_results_dict") or {}
        sub_queries = result.get("sub_query_dict") or {}
        return self._dump(
            await self._maybe_await(
                self.aggregate_runner(web_results, sub_queries, search_params)
            )
        )

    async def search_arxiv(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        year: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.local")
        return self._dump(
            await self._maybe_await(
                self.arxiv_runner(
                    query=query,
                    author=author,
                    year=year,
                    page=page,
                    results_per_page=results_per_page,
                )
            )
        )

    async def search_semantic_scholar(
        self,
        *,
        query: str,
        fields_of_study: list[str] | str | None = None,
        publication_types: list[str] | str | None = None,
        year_range: str | None = None,
        venue: list[str] | str | None = None,
        min_citations: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.local")
        return self._dump(
            await self._maybe_await(
                self.semantic_scholar_runner(
                    query=query,
                    fields_of_study=fields_of_study,
                    publication_types=publication_types,
                    year_range=year_range,
                    venue=venue,
                    min_citations=min_citations,
                    page=page,
                    results_per_page=results_per_page,
                )
            )
        )
