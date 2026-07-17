"""Local research search provider service."""

from __future__ import annotations

import inspect
import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable

from ..runtime_policy.types import PolicyDeniedError


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
LOCAL_SUPPORTED_PAPER_PROVIDERS = ("arxiv", "semantic_scholar")


class LocalResearchSearchService:
    """Policy-gated local research search provider launcher."""

    def __init__(
        self,
        *,
        websearch_runner: Callable[[str, dict[str, Any]], Any] | None = None,
        aggregate_runner: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any] | None = None,
        arxiv_runner: Callable[..., Any] | None = None,
        semantic_scholar_runner: Callable[..., Any] | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.websearch_runner = websearch_runner or self._default_websearch_runner
        self.aggregate_runner = aggregate_runner or self._default_aggregate_runner
        self.arxiv_runner = arxiv_runner or self._default_arxiv_runner
        self.semantic_scholar_runner = semantic_scholar_runner or self._default_semantic_scholar_runner
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

    @staticmethod
    def _default_arxiv_runner(
        *,
        query: str | None = None,
        author: str | None = None,
        year: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        search_parts: list[str] = []
        if query:
            search_parts.append(f"all:{query}")
        if author:
            search_parts.append(f"au:{author}")
        if year:
            search_parts.append(f"submittedDate:{year}01010000 TO {year}12312359")
        search_query = " AND ".join(search_parts) if search_parts else "all:*"
        start = max(page - 1, 0) * results_per_page
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": results_per_page,
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()

        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }
        root = ET.fromstring(payload)
        total_results = int(
            root.findtext("opensearch:totalResults", default="0", namespaces=namespaces) or 0
        )
        items: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", namespaces):
            pdf_url = None
            for link in entry.findall("atom:link", namespaces):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                    break
            authors = [
                str(name).strip()
                for name in (
                    author_node.findtext("atom:name", default="", namespaces=namespaces)
                    for author_node in entry.findall("atom:author", namespaces)
                )
                if str(name).strip()
            ]
            entry_id = (entry.findtext("atom:id", default="", namespaces=namespaces) or "").strip()
            title = " ".join((entry.findtext("atom:title", default="", namespaces=namespaces) or "").split())
            published = (entry.findtext("atom:published", default="", namespaces=namespaces) or "").strip()
            abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=namespaces) or "").split())
            items.append(
                {
                    "id": entry_id or None,
                    "title": title or None,
                    "authors": ", ".join(authors) or None,
                    "published_date": published or None,
                    "abstract": abstract or None,
                    "pdf_url": pdf_url,
                }
            )

        return {
            "query_echo": {
                "query": query,
                "author": author,
                "year": year,
                "page": page,
                "results_per_page": results_per_page,
            },
            "items": items,
            "total_results": total_results,
            "page": page,
            "results_per_page": results_per_page,
            "total_pages": math.ceil(total_results / results_per_page) if results_per_page else 0,
        }

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
        offset = max(page - 1, 0) * results_per_page
        params: dict[str, Any] = {
            "query": query,
            "offset": offset,
            "limit": results_per_page,
            "fields": (
                "paperId,title,abstract,year,citationCount,authors,venue,openAccessPdf,url,"
                "publicationTypes,publicationDate,externalIds"
            ),
        }
        optional_params = {
            "fieldsOfStudy": cls._coerce_csv(fields_of_study),
            "publicationTypes": cls._coerce_csv(publication_types),
            "year": year_range,
            "venue": cls._coerce_csv(venue),
            "minCitationCount": min_citations,
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items = list(payload.get("data") or [])
        total_results = int(payload.get("total") or len(items))
        return {
            "query_echo": {
                "query": query,
                "fields_of_study": fields_of_study,
                "publication_types": publication_types,
                "year_range": year_range,
                "venue": venue,
                "min_citations": min_citations,
                "page": page,
                "results_per_page": results_per_page,
            },
            "items": items,
            "total_results": total_results,
            "offset": offset,
            "limit": results_per_page,
            "next_offset": payload.get("next"),
            "page": page,
            "total_pages": math.ceil(total_results / results_per_page) if results_per_page else 0,
        }

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
                    or "Local research search action is not allowed.",
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
