"""Local research search provider service."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import WebSearchRequest
from ..tldw_api.research_search_schemas import WEBSEARCH_ENGINE_ALIASES


LOCAL_SUPPORTED_WEBSEARCH_ENGINES = {
    "baidu",
    "bing",
    "brave",
    "duckduckgo",
    "google",
    "kagi",
    "searx",
    "serper",
    "tavily",
    "yandex",
}


class LocalResearchSearchService:
    """Policy-gated local research search provider launcher."""

    def __init__(
        self,
        *,
        websearch_runner: Callable[[str, dict[str, Any]], Any] | None = None,
        aggregate_runner: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any] | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.websearch_runner = websearch_runner or self._default_websearch_runner
        self.aggregate_runner = aggregate_runner or self._default_aggregate_runner
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

        return await analyze_and_aggregate(web_search_results_dict, sub_query_dict, search_params)

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
                    user_message=getattr(decision, "user_message", None) or "Local research search action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "local",
                    authority_owner=getattr(decision, "authority_owner", None) or "local",
                )

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _normalize_engine(engine: str) -> str:
        return WEBSEARCH_ENGINE_ALIASES.get(str(engine).lower(), str(engine).lower())

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def list_supported_websearch_engines(self) -> list[str]:
        self._enforce("research.search.providers.list.local")
        return sorted(LOCAL_SUPPORTED_WEBSEARCH_ENGINES)

    async def websearch(
        self,
        *,
        query: str,
        engine: str = "google",
        result_count: int = 10,
        aggregate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.local")
        normalized_engine = self._normalize_engine(engine)
        if normalized_engine not in LOCAL_SUPPORTED_WEBSEARCH_ENGINES:
            supported = ", ".join(sorted(LOCAL_SUPPORTED_WEBSEARCH_ENGINES))
            raise ValueError(f"Unsupported local websearch engine: {engine}. Supported engines: {supported}")

        request = WebSearchRequest(
            query=query,
            engine=normalized_engine,
            result_count=result_count,
            aggregate=aggregate,
            **kwargs,
        )
        search_params = request.model_dump(exclude_none=True, mode="json")
        search_params.pop("query", None)
        result = self._dump(await self._maybe_await(self.websearch_runner(query, search_params)))

        if not aggregate:
            return result

        web_results = result.get("web_search_results_dict") or {}
        sub_queries = result.get("sub_query_dict") or {}
        return self._dump(await self._maybe_await(self.aggregate_runner(web_results, sub_queries, search_params)))

    async def search_arxiv(self, **_: Any) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.local")
        raise NotImplementedError("arXiv paper search is server-backed in the current Chatbook parity seam.")

    async def search_semantic_scholar(self, **_: Any) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.local")
        raise NotImplementedError("Semantic Scholar paper search is server-backed in the current Chatbook parity seam.")
