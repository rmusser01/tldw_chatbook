"""Server-backed research search provider service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    SUPPORTED_WEBSEARCH_ENGINES,
    TLDWAPIClient,
    WebSearchRequest,
)
from ..tldw_api.research_search_schemas import (
    PaperSearchDetailRequest,
    PaperSearchIngestRequest,
    PaperSearchRequest,
)


class ServerResearchSearchService:
    """Policy-gated access to server research search provider endpoints."""

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
    ) -> "ServerResearchSearchService":
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
    ) -> "ServerResearchSearchService":
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
        raise ValueError("TLDW API client is required for server research search operations.")

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
                    or "Server research search action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    @staticmethod
    def _params(**kwargs: Any) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if value is not None}

    async def list_supported_websearch_engines(self) -> list[str]:
        self._enforce("research.search.providers.list.server")
        return sorted(SUPPORTED_WEBSEARCH_ENGINES)

    async def list_supported_paper_providers(self) -> list[str]:
        self._enforce("research.search.providers.list.server")
        return ["arxiv", "semantic_scholar", "biorxiv", "medrxiv", "pubmed"]

    async def websearch(
        self,
        *,
        query: str,
        engine: str = "google",
        result_count: int = 10,
        aggregate: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        request = WebSearchRequest(
            query=query,
            engine=engine,
            result_count=result_count,
            aggregate=aggregate,
            **kwargs,
        )
        client = self._require_client()
        method = getattr(client, "run_research_websearch", None) or getattr(client, "research_websearch")
        return self._dump(await method(request))

    async def paper_search(self, *, endpoint: str, **params: Any) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        request = PaperSearchRequest(endpoint=endpoint, params=self._params(**params))
        client = self._require_client()
        method = getattr(client, "run_paper_search", None)
        if callable(method):
            return self._dump(await method(request))
        response = await client._request(
            "GET",
            f"/api/v1/paper-search/{request.endpoint}",
            params=request.params,
        )
        return self._dump(response)

    async def paper_detail(self, *, endpoint: str, **params: Any) -> dict[str, Any]:
        self._enforce("research.search.providers.observe.server")
        request = PaperSearchDetailRequest(endpoint=endpoint, params=self._params(**params))
        client = self._require_client()
        method = getattr(client, "get_paper_search_detail", None)
        if callable(method):
            return self._dump(await method(request))
        response = await client._request(
            "GET",
            f"/api/v1/paper-search/{request.endpoint}",
            params=request.params,
        )
        return self._dump(response)

    async def paper_ingest(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        request = PaperSearchIngestRequest(endpoint=endpoint, params=self._params(**params), payload=payload)
        client = self._require_client()
        method = getattr(client, "run_paper_search_ingest", None)
        if callable(method):
            return self._dump(await method(request))
        response = await client._request(
            "POST",
            f"/api/v1/paper-search/{request.endpoint}",
            params=request.params,
            json_data=request.payload,
        )
        return self._dump(response)

    async def search_arxiv(
        self,
        *,
        query: str | None = None,
        author: str | None = None,
        year: str | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(
            await self._require_client().search_arxiv_papers(
                query=query,
                author=author,
                year=year,
                page=page,
                results_per_page=results_per_page,
            )
        )

    async def get_arxiv_by_id(self, *, id: str) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(await self._require_client().get_arxiv_paper_by_id(id=id))

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
        self._enforce("research.search.providers.launch.server")
        return self._dump(
            await self._require_client().search_semantic_scholar_papers(
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

    async def get_semantic_scholar_by_id(self, *, paper_id: str) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(await self._require_client().get_semantic_scholar_paper_by_id(paper_id=paper_id))

    async def search_biorxiv(
        self,
        *,
        q: str | None = None,
        server: str = "biorxiv",
        from_date: str | None = None,
        to_date: str | None = None,
        category: str | None = None,
        recent_days: int | None = None,
        recent_count: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(
            await self._require_client().search_biorxiv_papers(
                q=q,
                server=server,
                from_date=from_date,
                to_date=to_date,
                category=category,
                recent_days=recent_days,
                recent_count=recent_count,
                page=page,
                results_per_page=results_per_page,
            )
        )

    async def get_biorxiv_by_doi(
        self,
        *,
        doi: str,
        server: str = "biorxiv",
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(await self._require_client().get_biorxiv_paper_by_doi(doi=doi, server=server))

    async def search_medrxiv(
        self,
        *,
        q: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        category: str | None = None,
        recent_days: int | None = None,
        recent_count: int | None = None,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(
            await self._require_client().search_medrxiv_papers(
                q=q,
                from_date=from_date,
                to_date=to_date,
                category=category,
                recent_days=recent_days,
                recent_count=recent_count,
                page=page,
                results_per_page=results_per_page,
            )
        )

    async def get_medrxiv_by_doi(self, *, doi: str) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(await self._require_client().get_medrxiv_paper_by_doi(doi=doi))

    async def search_pubmed(
        self,
        *,
        q: str,
        from_year: int | None = None,
        to_year: int | None = None,
        free_full_text: bool = False,
        page: int = 1,
        results_per_page: int = 10,
    ) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(
            await self._require_client().search_pubmed_papers(
                q=q,
                from_year=from_year,
                to_year=to_year,
                free_full_text=free_full_text,
                page=page,
                results_per_page=results_per_page,
            )
        )

    async def get_pubmed_by_id(self, *, pmid: str) -> dict[str, Any]:
        self._enforce("research.search.providers.launch.server")
        return self._dump(await self._require_client().get_pubmed_paper_by_id(pmid=pmid))
