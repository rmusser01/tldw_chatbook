"""Research search provider API schemas."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        engine = WEBSEARCH_ENGINE_ALIASES.get(str(value).lower(), str(value).lower())
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
    final_answer: WebSearchFinalAnswer | None
    relevant_results: dict[str, Any] | None
    web_search_results_dict: dict[str, Any]
    sub_query_dict: dict[str, Any]


PAPER_SEARCH_ENDPOINTS = frozenset(
    {
        "arxiv",
        "biorxiv",
        "medrxiv",
        "semantic-scholar",
        "biorxiv-pubs",
        "pubmed",
        "biorxiv/publisher",
        "biorxiv/pub",
        "biorxiv/funder",
        "biorxiv/reports/summary",
        "biorxiv/reports/usage",
        "ieee",
        "springer",
        "scopus",
        "acm",
        "wiley",
        "chemrxiv/items",
        "earthrxiv",
        "osf",
        "zenodo",
        "figshare",
        "hal",
        "vixra/search",
    }
)

PAPER_SEARCH_DETAIL_ENDPOINTS = frozenset(
    {
        "arxiv/by-id",
        "semantic-scholar/by-id",
        "pubmed/by-id",
        "biorxiv/by-doi",
        "medrxiv/by-doi",
        "biorxiv-pubs/by-doi",
        "repec/by-handle",
        "repec/citations",
        "ieee/by-id",
        "ieee/by-doi",
        "springer/by-doi",
        "scopus/by-doi",
        "acm/by-doi",
        "wiley/by-doi",
        "chemrxiv/items/by-id",
        "chemrxiv/items/by-doi",
        "iacr/conf",
        "earthrxiv/by-id",
        "earthrxiv/by-doi",
        "osf/by-id",
        "osf/by-doi",
        "zenodo/by-id",
        "zenodo/by-doi",
        "figshare/by-id",
        "figshare/by-doi",
        "hal/by-id",
        "vixra/by-id",
    }
)

PAPER_SEARCH_INGEST_ENDPOINTS = frozenset(
    {
        "pmc-oa/ingest-pdf",
        "arxiv/ingest",
        "earthrxiv/ingest",
        "pubmed/ingest",
        "semantic-scholar/ingest",
        "ingest/by-doi",
        "ingest/batch",
        "osf/ingest",
        "zenodo/ingest",
        "figshare/ingest",
        "figshare/ingest-by-doi",
        "hal/ingest",
        "vixra/ingest",
    }
)


def _normalize_paper_search_endpoint(value: str, *, allowed: frozenset[str], label: str) -> str:
    endpoint = str(value or "").strip().strip("/").lower()
    if endpoint not in allowed:
        label_prefix = f"{label} " if label else ""
        raise ValueError(f"Unsupported paper search {label_prefix}endpoint: {value}")
    return endpoint


class PaperSearchRequest(BaseModel):
    """Allowlisted GET route under `/api/v1/paper-search/*` for provider search."""

    allowed_endpoints: ClassVar[frozenset[str]] = PAPER_SEARCH_ENDPOINTS

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _normalize_paper_search_endpoint(value, allowed=cls.allowed_endpoints, label="")


class PaperSearchDetailRequest(BaseModel):
    """Allowlisted GET route under `/api/v1/paper-search/*` for provider detail lookups."""

    allowed_endpoints: ClassVar[frozenset[str]] = PAPER_SEARCH_DETAIL_ENDPOINTS

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _normalize_paper_search_endpoint(value, allowed=cls.allowed_endpoints, label="detail")


class PaperSearchIngestRequest(BaseModel):
    """Allowlisted POST route under `/api/v1/paper-search/*` for server-side paper ingest."""

    allowed_endpoints: ClassVar[frozenset[str]] = PAPER_SEARCH_INGEST_ENDPOINTS

    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _normalize_paper_search_endpoint(value, allowed=cls.allowed_endpoints, label="ingest")


class PaperSearchListResponse(BaseModel):
    """Source-honest provider search response with flexible provider-specific item fields."""

    model_config = ConfigDict(extra="allow")

    query_echo: dict[str, Any] = Field(default_factory=dict)
    items: list[dict[str, Any]] = Field(default_factory=list)
    total_results: int | None = None
    page: int | None = None
    results_per_page: int | None = None
    total_pages: int | None = None


class PaperSearchDetailResponse(BaseModel):
    """Flexible paper/provider detail payload returned by server paper-search endpoints."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    title: str | None = None
    doi: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    provider: str | None = None


class PaperSearchOperationResponse(BaseModel):
    """Wrapper for provider ingest responses whose shape varies by endpoint."""

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def wrap_raw_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "data" not in value:
            return {"data": value, **value}
        return value


class ArxivPaper(BaseModel):
    id: str | None = None
    title: str | None = None
    authors: str | None = None
    published_date: str | None = None
    abstract: str | None = None
    pdf_url: str | None = None


class ArxivSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[ArxivPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int


class BioRxivPaper(BaseModel):
    doi: str
    title: str
    authors: str | None = None
    category: str | None = None
    date: str | None = None
    abstract: str | None = None
    server: str | None = None
    version: int | None = None
    url: str | None = None
    pdf_url: str | None = None


class BioRxivSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[BioRxivPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int


class PubMedPaper(BaseModel):
    pmid: str
    title: str
    authors: str | None = None
    journal: str | None = None
    pub_date: str | None = None
    abstract: str | None = None
    doi: str | None = None
    url: str | None = None
    pmcid: str | None = None
    pmc_url: str | None = None
    pdf_url: str | None = None


class PubMedSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[PubMedPaper]
    total_results: int
    page: int
    results_per_page: int
    total_pages: int


class SemanticScholarAuthor(BaseModel):
    authorId: str | None = None
    name: str


class SemanticScholarOpenAccessPdf(BaseModel):
    url: str
    status: str


class SemanticScholarPaper(BaseModel):
    paperId: str
    title: str
    abstract: str | None = None
    year: int | None = None
    citationCount: int | None = None
    authors: list[SemanticScholarAuthor] | None = Field(default_factory=list)
    venue: str | None = None
    openAccessPdf: SemanticScholarOpenAccessPdf | None = None
    url: str | None = None
    publicationTypes: list[str] | None = None
    publicationDate: str | None = None
    externalIds: dict[str, str] | None = None


class SemanticScholarSearchResponse(BaseModel):
    query_echo: dict[str, Any]
    items: list[SemanticScholarPaper]
    total_results: int
    offset: int
    limit: int
    next_offset: int | None = None
    page: int
    total_pages: int
