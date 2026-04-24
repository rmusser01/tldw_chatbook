from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


SUPPORTED_WEBSEARCH_ENGINES = {
    "google",
    "duckduckgo",
    "brave",
    "kagi",
    "tavily",
    "searx",
    "exa",
    "firecrawl",
    "baidu",
    "bing",
    "yandex",
    "sogou",
    "startpage",
    "stract",
    "serper",
    "4chan",
}

WEBSEARCH_ENGINE_ALIASES = {"searxng": "searx"}


class WebSearchRequest(BaseModel):
    query: str
    engine: str = "google"
    result_count: int = Field(default=10, ge=1, le=50)
    content_country: str = "US"
    search_lang: str = "en"
    output_lang: str = "en"
    date_range: str | None = None
    safesearch: str | None = None
    site_blacklist: list[str] | None = None
    exactTerms: str | None = None
    excludeTerms: str | None = None
    filter: str | None = None
    geolocation: str | None = None
    search_result_language: str | None = None
    sort_results_by: str | None = None
    searx_url: str | None = None
    searx_json_mode: bool = False
    google_domain: str | None = None
    boards: list[str] | None = None
    max_threads_per_board: int | None = Field(default=None, ge=1, le=1000)
    max_archived_threads_per_board: int | None = Field(default=None, ge=1, le=500)
    include_archived: bool = False
    subquery_generation: bool = False
    subquery_generation_llm: str | None = None
    user_review: bool = False
    relevance_analysis_llm: str | None = None
    final_answer_llm: str | None = None
    aggregate: bool = False

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, value: str) -> str:
        engine = WEBSEARCH_ENGINE_ALIASES.get(value.lower(), value.lower())
        if engine not in SUPPORTED_WEBSEARCH_ENGINES:
            allowed = ", ".join(sorted(SUPPORTED_WEBSEARCH_ENGINES))
            raise ValueError(f"Unsupported engine '{value}'. Supported engines: {allowed}")
        return engine


class WebSearchFinalAnswer(BaseModel):
    text: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    chunks: list[dict[str, Any]] = Field(default_factory=list)


class WebSearchRawResponse(BaseModel):
    web_search_results_dict: dict[str, Any]
    sub_query_dict: dict[str, Any]


class WebSearchAggregateResponse(BaseModel):
    final_answer: WebSearchFinalAnswer | None = None
    relevant_results: dict[str, Any] | None = None
    web_search_results_dict: dict[str, Any]
    sub_query_dict: dict[str, Any]
