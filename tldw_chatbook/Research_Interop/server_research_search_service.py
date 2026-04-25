"""Server-backed research search provider service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    SUPPORTED_WEBSEARCH_ENGINES,
    TLDWAPIClient,
    WebSearchRequest,
)


class ServerResearchSearchService:
    """Policy-gated access to server research search provider endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerResearchSearchService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server research search operations.")
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
                    user_message=getattr(decision, "user_message", None) or "Server research search action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def list_supported_websearch_engines(self) -> list[str]:
        self._enforce("research.search.providers.list.server")
        return sorted(SUPPORTED_WEBSEARCH_ENGINES)

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
        return self._dump(await self._require_client().research_websearch(request))

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
