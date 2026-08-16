"""Academic search providers (task-16326).

httpx-based arXiv and Semantic Scholar runners replacing the inline urllib
calls in ``local_research_search_service``: configurable timeouts, retry
with exponential backoff (+ jitter) on transport errors and 429/5xx,
constants-driven endpoints, a config-resolved Semantic Scholar API key, and
DOI normalization so papers can dedup against each other in one evidence
pool with web results (``merge_evidence_pools``).
"""

from __future__ import annotations

import math
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import httpx
from loguru import logger

from tldw_chatbook.config import get_cli_setting

__all__ = [
    "ARXIV_API_ENDPOINT",
    "SEMANTIC_SCHOLAR_API_ENDPOINT",
    "AcademicProviderError",
    "merge_evidence_pools",
    "papers_to_evidence",
    "resolve_semantic_scholar_api_key",
    "search_arxiv",
    "search_semantic_scholar",
]

ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_ATOM_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

_ARXIV_ID_RE = re.compile(r"arxiv\.org/abs/([^v/\s]+)", re.IGNORECASE)


class AcademicProviderError(Exception):
    """A provider request failed after exhausting retries (or hit a
    non-retryable HTTP error)."""


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter, capped so a bad provider can never
    wedge a research run for long."""
    time.sleep(min((2**attempt) + random.random(), 8.0))


def _request_with_retries(
    client: Any,
    method: str,
    url: str,
    *,
    timeout: Any,
    max_retries: int,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
) -> Any:
    last_failure: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.request(
                method, url, headers=dict(headers or {}), params=dict(params or {}), timeout=timeout
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if status in _RETRYABLE_STATUS:
                last_failure = AcademicProviderError(f"http {status} from {url}")
                if attempt < max_retries:
                    _sleep_backoff(attempt)
                    continue
                raise last_failure
            if status >= 400:
                raise AcademicProviderError(f"http {status} from {url}")
            return response
        except httpx.TransportError as exc:
            last_failure = exc
            if attempt < max_retries:
                _sleep_backoff(attempt)
                continue
            raise AcademicProviderError(f"{url} failed after {max_retries + 1} attempt(s): {exc}") from exc
    raise AcademicProviderError(f"{url} failed: {last_failure}")


def _arxiv_doi(entry_id: str | None) -> str | None:
    if not entry_id:
        return None
    match = _ARXIV_ID_RE.search(entry_id)
    if not match:
        return None
    return f"10.48550/arxiv.{match.group(1)}"


def search_arxiv(
    *,
    query: str | None = None,
    author: str | None = None,
    year: str | None = None,
    page: int = 1,
    results_per_page: int = 10,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search arXiv over httpx with retry/backoff; returns the legacy runner
    shape with each item additionally carrying ``doi`` and ``source``."""
    search_parts: list[str] = []
    if query:
        search_parts.append(f"all:{query}")
    if author:
        search_parts.append(f"au:{author}")
    if year:
        search_parts.append(f"submittedDate:{year}01010000 TO {year}12312359")
    search_query = " AND ".join(search_parts) if search_parts else "all:*"
    start = max(page - 1, 0) * results_per_page
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": results_per_page,
    }

    owned_client = client is None
    http = client or httpx.Client()
    try:
        response = _request_with_retries(
            http,
            "GET",
            ARXIV_API_ENDPOINT,
            timeout=timeout,
            max_retries=max_retries,
            params=params,
        )
        payload = response.content if hasattr(response, "content") else response.text
    finally:
        if owned_client:
            http.close()

    root = ET.fromstring(payload)
    total_results = int(
        root.findtext(
            "opensearch:totalResults", default="0", namespaces=_ATOM_NAMESPACES
        )
        or 0
    )
    items: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", namespaces=_ATOM_NAMESPACES):
        pdf_url = None
        for link in entry.findall("atom:link", namespaces=_ATOM_NAMESPACES):
            if (
                link.attrib.get("title") == "pdf"
                or link.attrib.get("type") == "application/pdf"
            ):
                pdf_url = link.attrib.get("href")
                break
        authors = [
            str(name).strip()
            for name in (
                author_node.findtext(
                    "atom:name", default="", namespaces=_ATOM_NAMESPACES
                )
                for author_node in entry.findall("atom:author", namespaces=_ATOM_NAMESPACES)
            )
            if str(name).strip()
        ]
        entry_id = (
            entry.findtext("atom:id", default="", namespaces=_ATOM_NAMESPACES) or ""
        ).strip()
        title = " ".join(
            (
                entry.findtext("atom:title", default="", namespaces=_ATOM_NAMESPACES)
                or ""
            ).split()
        )
        published = (
            entry.findtext("atom:published", default="", namespaces=_ATOM_NAMESPACES)
            or ""
        ).strip()
        abstract = " ".join(
            (
                entry.findtext("atom:summary", default="", namespaces=_ATOM_NAMESPACES)
                or ""
            ).split()
        )
        items.append(
            {
                "id": entry_id or None,
                "title": title or None,
                "authors": ", ".join(authors) or None,
                "published_date": published or None,
                "abstract": abstract or None,
                "pdf_url": pdf_url,
                "doi": _arxiv_doi(entry_id),
                "url": entry_id or pdf_url,
                "source": "arxiv",
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
        "total_pages": math.ceil(total_results / results_per_page)
        if results_per_page
        else 0,
    }


def resolve_semantic_scholar_api_key() -> str | None:
    """Env var wins, then the ``[API] semantic_scholar_api_key`` config slot
    (house precedence: env -> config.toml)."""
    env_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if env_key:
        return env_key
    return get_cli_setting("API", "semantic_scholar_api_key", None) or None


def search_semantic_scholar(
    *,
    query: str,
    fields_of_study: list[str] | str | None = None,
    publication_types: list[str] | str | None = None,
    year_range: str | None = None,
    venue: list[str] | str | None = None,
    min_citations: int | None = None,
    page: int = 1,
    results_per_page: int = 10,
    api_key: str | None = None,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search Semantic Scholar over httpx with retry/backoff and an optional
    ``x-api-key`` (resolved from config when not supplied); returns the
    legacy runner shape with items additionally carrying ``doi``/``source``."""
    import json as _json

    offset = max(page - 1, 0) * results_per_page

    def _coerce_csv(value: list[str] | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return ",".join(str(item) for item in value)

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
        "fieldsOfStudy": _coerce_csv(fields_of_study),
        "publicationTypes": _coerce_csv(publication_types),
        "year": year_range,
        "venue": _coerce_csv(venue),
        "minCitationCount": min_citations,
    }
    params.update(
        {key: value for key, value in optional_params.items() if value is not None}
    )
    headers = {"x-api-key": api_key} if api_key else {}

    owned_client = client is None
    http = client or httpx.Client()
    try:
        response = _request_with_retries(
            http,
            "GET",
            SEMANTIC_SCHOLAR_API_ENDPOINT,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
            params=params,
        )
        payload = _json.loads(response.text)
    finally:
        if owned_client:
            http.close()

    items: list[dict[str, Any]] = []
    for raw_item in list(payload.get("data") or []):
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
        item["doi"] = doi
        item["url"] = item.get("url") or (
            f"https://doi.org/{doi}" if doi else None
        )
        item["source"] = "semantic_scholar"
        items.append(item)
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
        "total_pages": math.ceil(total_results / results_per_page)
        if results_per_page
        else 0,
    }


def papers_to_evidence(papers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map normalized paper records into the search-result shape the
    deep-search evidence pool consumes (``{title, url, content, metadata}``)."""
    evidence: list[dict[str, Any]] = []
    for paper in papers:
        if not isinstance(paper, Mapping):
            continue
        title = str(paper.get("title") or "Untitled")
        url = str(
            paper.get("url")
            or paper.get("pdf_url")
            or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else "")
            or ""
        )
        evidence.append(
            {
                "title": title,
                "url": url,
                "content": str(paper.get("abstract") or title),
                "metadata": {
                    "source": "academic",
                    "provider": paper.get("source"),
                    "doi": paper.get("doi"),
                    "authors": paper.get("authors"),
                    "published_date": paper.get("published_date"),
                },
            }
        )
    return evidence


def merge_evidence_pools(
    web_results: list[Mapping[str, Any]],
    academic_papers: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One evidence pool: web results first, then DOI-deduped papers. Papers
    sharing a DOI collapse to the FIRST occurrence (cross-provider duplicates
    are the point); papers without a DOI are kept -- URL-level dedup stays
    the caller's concern (the engine already does it for web results)."""
    merged: list[dict[str, Any]] = [dict(item) for item in web_results]
    seen_dois: set[str] = set()
    for paper in papers_to_evidence(list(academic_papers)):
        doi = paper.get("metadata", {}).get("doi")
        if doi:
            if doi in seen_dois:
                continue
            seen_dois.add(doi)
        merged.append(paper)
    return merged


async def search_papers(
    query: str,
    *,
    results_per_page: int = 5,
    semantic_scholar_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Query arXiv and Semantic Scholar together for one query string
    (task-16328) and return DOI-deduped normalized papers — the default
    ``paper_search_fn`` for the engine's academic lane.

    Per-provider degradation: one provider failing logs and contributes
    nothing while the other still returns; only total failure raises (the
    engine lane turns that into a warning, never a run failure).
    """
    from asyncio import to_thread

    failures: list[str] = []
    papers: list[dict[str, Any]] = []
    seen_dois: set[str] = set()

    def _collect(items: list[Any]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            doi = item.get("doi")
            if doi:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)
            papers.append(item)

    try:
        arxiv_out = await to_thread(
            search_arxiv, query=query, results_per_page=results_per_page
        )
        _collect(arxiv_out.get("items") or [])
    except AcademicProviderError as exc:
        failures.append(f"arxiv: {exc}")
        logger.warning(f"search_papers arxiv lane failed: {exc}")

    try:
        s2_out = await to_thread(
            search_semantic_scholar,
            query=query,
            results_per_page=results_per_page,
            api_key=semantic_scholar_api_key
            if semantic_scholar_api_key is not None
            else resolve_semantic_scholar_api_key(),
        )
        _collect(s2_out.get("items") or [])
    except AcademicProviderError as exc:
        failures.append(f"semantic_scholar: {exc}")
        logger.warning(f"search_papers semantic scholar lane failed: {exc}")

    if not papers and failures:
        raise AcademicProviderError(
            "all academic providers failed: " + "; ".join(failures)
        )
    return papers
