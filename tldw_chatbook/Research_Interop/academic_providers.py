"""Academic search providers (task-16326).

httpx-based arXiv and Semantic Scholar runners replacing the inline urllib
calls in ``local_research_search_service``: configurable timeouts, retry
with exponential backoff (+ jitter) on transport errors and 429/5xx,
constants-driven endpoints, a config-resolved Semantic Scholar API key, and
DOI normalization so papers can dedup against each other in one evidence
pool with web results (``merge_evidence_pools``).
"""

from __future__ import annotations

import datetime
import json
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

from .research_source_catalog import expand_source_selection

__all__ = [
    "ARXIV_API_ENDPOINT",
    "BIORXIV_API_BASE",
    "PUBMED_EUTILS_BASE",
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
BIORXIV_API_BASE = "https://api.biorxiv.org"
PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_API_BASE = "https://api.openalex.org"
CROSSREF_API_BASE = "https://api.crossref.org"
ZENODO_API_BASE = "https://zenodo.org/api/records"
FIGSHARE_API_BASE = "https://api.figshare.com/v2/articles/search"
OSF_API_BASE = "https://api.osf.io/v2/preprints"

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


def _json_or_error(response: Any, provider: str) -> Any:
    """Parse a provider response body as JSON; a malformed payload is a
    typed provider failure (task-16812: a raw JSONDecodeError escaped the
    lane's degradation catch and killed the OTHER providers' results too)."""
    try:
        return json.loads(response.text)
    except (ValueError, TypeError, AttributeError) as exc:
        raise AcademicProviderError(
            f"invalid JSON payload from {provider}: {exc}"
        ) from exc


def _request_with_retries_json(
    client: Any,
    method: str,
    url: str,
    *,
    timeout: Any,
    max_retries: int,
    body: dict[str, Any],
) -> Any:
    """JSON-body variant of the retry ladder (Figshare's POST search)."""
    last_failure: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.request(
                method, url, json=body, timeout=timeout
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
        # Phrase-quote multi-word queries: arXiv token matching ranks papers
        # sharing ANY tokens (a "retrieval augmented generation" token search
        # surfaces cognitive-augmentation and image-generation papers), while
        # a phrase match returns papers that actually use the term.
        if " " in query.strip():
            search_parts.append(f'all:"{query.strip()}"')
        else:
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
    """Resolve the Semantic Scholar API key (house precedence).

    Returns:
        The key from the ``SEMANTIC_SCHOLAR_API_KEY`` env var when set, else
        the ``[API] semantic_scholar_api_key`` config slot, else ``None``.
    """
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


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str | None) -> str | None:
    """Strip HTML/JATS tags and collapse whitespace (Crossref abstracts,
    Zenodo/Figshare descriptions)."""
    if not text:
        return None
    return " ".join(_TAG_RE.sub("", text).split()) or None


def _clean_doi(raw: str | None) -> str | None:
    """Normalize a DOI to its bare form (OpenAlex prefixes https://doi.org/)."""
    if not raw:
        return None
    return raw.removeprefix("https://doi.org/").removeprefix("doi:") or None


def _invert_abstract(index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct an abstract from OpenAlex's inverted index."""
    if not isinstance(index, dict):
        return None
    words: dict[int, str] = {}
    for word, positions in index.items():
        for position in positions or []:
            words[int(position)] = word
    return " ".join(words[i] for i in sorted(words)) or None


def search_openalex(
    query: str,
    *,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search OpenAlex works (task-16792; server catalog parity).

    Args:
        query: The search term.
        results_per_page: Page size (per-page).
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget.

    Returns:
        The runner dict shape (abstracts reconstructed from the inverted
        index; DOIs normalized; ``source`` set to ``"openalex"``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard error.
    """
    params = {
        "search": query,
        "per-page": max(1, results_per_page),
        "page": 1,
    }
    if client is not None:
        response = _request_with_retries(
            client, "GET", f"{OPENALEX_API_BASE}/works",
            timeout=timeout, max_retries=max_retries, params=params,
        )
        data = _json_or_error(response, "openalex")
    else:
        with httpx.Client() as http:
            response = _request_with_retries(
                http, "GET", f"{OPENALEX_API_BASE}/works",
                timeout=timeout, max_retries=max_retries, params=params,
            )
            data = _json_or_error(response, "openalex")

    items: list[dict[str, Any]] = []
    for raw in data.get("results") or []:
        if not isinstance(raw, dict):
            continue
        location = raw.get("primary_location") or {}
        doi = _clean_doi(raw.get("doi"))
        items.append(
            {
                "id": raw.get("id"),
                "doi": doi,
                "title": raw.get("title"),
                "authors": ", ".join(
                    str((a.get("author") or {}).get("display_name"))
                    for a in (raw.get("authorships") or [])
                    if isinstance(a, dict) and (a.get("author") or {}).get("display_name")
                ) or None,
                "published_date": str(raw.get("publication_year") or "") or None,
                "abstract": _invert_abstract(raw.get("abstract_inverted_index")),
                "url": location.get("landing_page_url") or (
                    f"https://doi.org/{doi}" if doi else None
                ),
                "pdf_url": location.get("pdf_url"),
                "source": "openalex",
            }
        )

    return {
        "query_echo": {"query": query},
        "items": items,
        "total_results": int((data.get("meta") or {}).get("count") or 0),
        "results_per_page": results_per_page,
    }


def search_crossref(
    query: str,
    *,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search Crossref works (task-16792; server catalog parity).

    Args:
        query: The search term.
        results_per_page: Page size (rows).
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget.

    Returns:
        The runner dict shape (JATS stripped from abstracts; ``source``
        set to ``"crossref"``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard error.
    """
    params = {"query": query, "rows": max(1, results_per_page), "offset": 0}
    if client is not None:
        response = _request_with_retries(
            client, "GET", f"{CROSSREF_API_BASE}/works",
            timeout=timeout, max_retries=max_retries, params=params,
        )
        data = _json_or_error(response, "crossref")
    else:
        with httpx.Client() as http:
            response = _request_with_retries(
                http, "GET", f"{CROSSREF_API_BASE}/works",
                timeout=timeout, max_retries=max_retries, params=params,
            )
            data = _json_or_error(response, "crossref")

    message = data.get("message") or {}
    items: list[dict[str, Any]] = []
    for raw in message.get("items") or []:
        if not isinstance(raw, dict):
            continue
        authors = ", ".join(
            f"{a.get('given', '').strip()} {a.get('family', '').strip()}".strip()
            for a in (raw.get("author") or [])
            if isinstance(a, dict)
        ).strip() or None
        issued = raw.get("issued") or {}
        date_parts = (issued.get("date-parts") or [[None]])[0]
        items.append(
            {
                "id": raw.get("DOI"),
                "doi": _clean_doi(raw.get("DOI")),
                "title": (raw.get("title") or [None])[0],
                "authors": authors,
                "published_date": str(date_parts[0]) if date_parts and date_parts[0] else None,
                "abstract": _strip_tags(raw.get("abstract")),
                "url": f"https://doi.org/{raw['DOI']}" if raw.get("DOI") else None,
                "pdf_url": None,
                "source": "crossref",
            }
        )

    return {
        "query_echo": {"query": query},
        "items": items,
        "total_results": int(message.get("total-results") or 0),
        "results_per_page": results_per_page,
    }


def search_zenodo(
    query: str,
    *,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search Zenodo records (task-16792; server catalog parity).

    Args:
        query: The search term.
        results_per_page: Page size.
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget.

    Returns:
        The runner dict shape (HTML stripped from descriptions;
        ``source`` set to ``"zenodo"``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard error.
    """
    params = {"q": query, "size": max(1, results_per_page), "page": 1}
    if client is not None:
        response = _request_with_retries(
            client, "GET", ZENODO_API_BASE,
            timeout=timeout, max_retries=max_retries, params=params,
        )
        data = _json_or_error(response, "zenodo")
    else:
        with httpx.Client() as http:
            response = _request_with_retries(
                http, "GET", ZENODO_API_BASE,
                timeout=timeout, max_retries=max_retries, params=params,
            )
            data = _json_or_error(response, "zenodo")

    hits = (data.get("hits") or {})
    items: list[dict[str, Any]] = []
    for raw in hits.get("hits") or []:
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata") or {}
        record_id = raw.get("id")
        items.append(
            {
                "id": str(record_id) if record_id is not None else None,
                "doi": _clean_doi(raw.get("doi")),
                "title": metadata.get("title"),
                "authors": ", ".join(
                    str(c.get("name"))
                    for c in (metadata.get("creators") or [])
                    if isinstance(c, dict) and c.get("name")
                ) or None,
                "published_date": metadata.get("publication_date"),
                "abstract": _strip_tags(metadata.get("description")),
                "url": f"https://zenodo.org/records/{record_id}" if record_id is not None else None,
                "pdf_url": None,
                "source": "zenodo",
            }
        )

    return {
        "query_echo": {"query": query},
        "items": items,
        "total_results": int(hits.get("total") or 0),
        "results_per_page": results_per_page,
    }


def search_figshare(
    query: str,
    *,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search Figshare articles (task-16792; server catalog parity --
    POST search, matching the server's adapter).

    Args:
        query: The search term.
        results_per_page: Page size.
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget.

    Returns:
        The runner dict shape (HTML stripped from descriptions;
        ``source`` set to ``"figshare"``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard error.
    """
    body = {"search_for": query, "page": 1, "page_size": max(1, results_per_page)}
    if client is not None:
        response = _request_with_retries_json(
            client, "POST", FIGSHARE_API_BASE, timeout=timeout,
            max_retries=max_retries, body=body,
        )
        data = _json_or_error(response, "figshare")
    else:
        with httpx.Client() as http:
            response = _request_with_retries_json(
                http, "POST", FIGSHARE_API_BASE, timeout=timeout,
                max_retries=max_retries, body=body,
            )
            data = _json_or_error(response, "figshare")

    items: list[dict[str, Any]] = []
    for raw in data if isinstance(data, list) else []:
        if not isinstance(raw, dict):
            continue
        article_id = raw.get("id")
        items.append(
            {
                "id": str(article_id) if article_id is not None else None,
                "doi": _clean_doi(raw.get("doi")),
                "title": raw.get("title"),
                "authors": ", ".join(
                    str(a.get("full_name"))
                    for a in (raw.get("authors") or [])
                    if isinstance(a, dict) and a.get("full_name")
                ) or None,
                "published_date": str(raw.get("published_date") or "") or None,
                "abstract": _strip_tags(raw.get("description")),
                "url": raw.get("url_publication") or raw.get("url") or (
                    f"https://figshare.com/articles/_{article_id}" if article_id is not None else None
                ),
                "pdf_url": raw.get("download_url"),
                "source": "figshare",
            }
        )

    return {
        "query_echo": {"query": query},
        "items": items,
        "total_results": len(items),
        "results_per_page": results_per_page,
    }


def search_osf(
    query: str,
    *,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search OSF preprints (task-16792; server catalog parity).

    Args:
        query: The search term.
        results_per_page: Page size (page[size], capped at the API's 100).
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget.

    Returns:
        The runner dict shape (``source`` set to ``"osf"``; dates truncated
        to ``YYYY-MM-DD``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard error.
    """
    params = {
        "q": query,
        "page[size]": max(1, min(results_per_page, 100)),
        "page[number]": 1,
    }
    headers = {"Accept": "application/json"}  # server-adapter parity: OSF returns HTML without it
    if client is not None:
        response = _request_with_retries(
            client, "GET", OSF_API_BASE,
            timeout=timeout, max_retries=max_retries, params=params,
            headers=headers,
        )
        data = _json_or_error(response, "osf")
    else:
        # OSF intermittently 301s (first-hit redirects); httpx does not
        # follow redirects by default, which yielded empty 301 bodies.
        with httpx.Client(follow_redirects=True) as http:
            response = _request_with_retries(
                http, "GET", OSF_API_BASE,
                timeout=timeout, max_retries=max_retries, params=params,
                headers=headers,
            )
            data = _json_or_error(response, "osf")

    items: list[dict[str, Any]] = []
    for raw in data.get("data") or []:
        if not isinstance(raw, dict):
            continue
        attributes = raw.get("attributes") or {}
        links = raw.get("links") or {}
        osf_id = raw.get("id")
        created = str(attributes.get("date_created") or "")
        items.append(
            {
                "id": osf_id,
                "doi": _clean_doi(attributes.get("doi")),
                "title": attributes.get("title"),
                "authors": None,
                "published_date": created[:10] if created else None,
                "abstract": _strip_tags(attributes.get("description")),
                "url": links.get("html") or (
                    f"https://osf.io/preprints/{osf_id}" if osf_id else None
                ),
                "pdf_url": None,
                "source": "osf",
            }
        )

    return {
        "query_echo": {"query": query},
        "items": items,
        "total_results": len(items),
        "results_per_page": results_per_page,
    }


def search_biorxiv(
    query: str,
    *,
    server: str = "biorxiv",
    from_date: str | None = None,
    to_date: str | None = None,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search BioRxiv/MedRxiv preprints (task-16790; port of the server's
    Third_Party/BioRxiv.py details flow, simplified to the lane's needs).

    The details API lists by date range (default: last 30 days, matching the
    server), so the query filters client-side over title/abstract.

    Args:
        query: Topic query; matched case-insensitively against titles and
            abstracts.
        server: ``"biorxiv"`` or ``"medrxiv"`` (shared API, per the server's
            MedRxiv aliasing).
        from_date: Optional ``YYYY-MM-DD`` range start.
        to_date: Optional ``YYYY-MM-DD`` range end.
        results_per_page: Maximum items returned after filtering.
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget for retryable failures.

    Returns:
        The runner dict shape (``items`` normalized to the shared paper
        shape with ``source`` set to the server name).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard HTTP
            error.
    """
    server_norm = server.lower().strip() if server else "biorxiv"
    if server_norm not in {"biorxiv", "medrxiv"}:
        server_norm = "biorxiv"

    def _validate_date(d: str | None) -> str | None:
        try:
            datetime.date.fromisoformat(str(d or ""))
            return str(d)
        except ValueError:
            return None

    f = _validate_date(from_date)
    t = _validate_date(to_date)
    if not (f and t):
        end = datetime.datetime.now(datetime.timezone.utc).date()
        start = end - datetime.timedelta(days=30)
        f, t = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    url = f"{BIORXIV_API_BASE}/details/{server_norm}/{f}/{t}/0"
    if client is not None:
        response = _request_with_retries(
            client, "GET", url, timeout=timeout, max_retries=max_retries
        )
        data = _json_or_error(response, "biorxiv")
    else:
        with httpx.Client() as http:
            response = _request_with_retries(
                http, "GET", url, timeout=timeout, max_retries=max_retries
            )
            data = _json_or_error(response, "biorxiv")

    total = 0
    messages = data.get("messages") or []
    if messages and isinstance(messages[0], dict):
        try:
            total = int(messages[0].get("total") or messages[0].get("count") or 0)
        except (TypeError, ValueError):
            total = 0

    needle = " ".join(query.casefold().split())
    items: list[dict[str, Any]] = []
    for raw in data.get("collection") or []:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "")
        abstract = str(raw.get("abstract") or "")
        if needle and needle not in title.casefold() and needle not in abstract.casefold():
            continue
        doi = str(raw.get("doi") or "")
        version = raw.get("version")
        v_suffix = f"v{version}" if str(version or "").isdigit() else ""
        base_host = "biorxiv.org" if server_norm == "biorxiv" else "medrxiv.org"
        content_url = f"https://www.{base_host}/content/{doi}{v_suffix}" if doi else None
        items.append(
            {
                "id": doi or None,
                "doi": doi or None,
                "title": title or None,
                "authors": raw.get("authors"),
                "published_date": raw.get("date"),
                "abstract": abstract or None,
                "url": content_url,
                "pdf_url": f"{content_url}.full.pdf" if content_url else None,
                "source": server_norm,
            }
        )
        if len(items) >= results_per_page:
            break

    return {
        "query_echo": {"query": query, "server": server_norm,
                       "from_date": f, "to_date": t},
        "items": items,
        "total_results": total,
        "results_per_page": results_per_page,
    }


def search_pubmed(
    query: str,
    *,
    from_year: int | None = None,
    to_year: int | None = None,
    results_per_page: int = 5,
    client: Any = None,
    timeout: Any = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Search PubMed via ESearch + ESummary (task-16790; port of the
    server's Third_Party/PubMed.py two-step).

    Args:
        query: The search term (relevance-sorted).
        from_year: Optional publication-year lower bound.
        to_year: Optional publication-year upper bound.
        results_per_page: Page size (retmax, clamped to the API's 200).
        client: Injectable httpx-like client for tests.
        timeout: Per-request timeout.
        max_retries: Retry budget per request.

    Returns:
        The runner dict shape (``items`` normalized to the shared paper
        shape with ``source`` set to ``"pubmed"``).

    Raises:
        AcademicProviderError: After exhausting retries or on a hard HTTP
            error.
    """
    if not query.strip():
        return {"query_echo": {"query": query}, "items": [], "total_results": 0,
                "results_per_page": results_per_page}

    esearch_params: dict[str, Any] = {
        "db": "pubmed",
        "term": query.strip(),
        "retmode": "json",
        "retstart": "0",
        "retmax": str(max(1, min(results_per_page, 200))),
        "sort": "relevance",
    }
    if from_year or to_year:
        fy = int(from_year or to_year)
        ty = int(to_year or from_year)
        esearch_params.update({"datetype": "pdat", "mindate": str(fy), "maxdate": str(ty)})

    def _run(http: Any) -> tuple[list[str], int, dict[str, Any]]:
        esearch = _request_with_retries(
            http, "GET", f"{PUBMED_EUTILS_BASE}/esearch.fcgi",
            timeout=timeout, max_retries=max_retries, params=esearch_params,
        )
        esr = json.loads(esearch.text).get("esearchresult") or {}
        ids = [str(i) for i in (esr.get("idlist") or [])]
        count = int(esr.get("count") or 0)
        if not ids:
            return ids, count, {}
        esummary = _request_with_retries(
            http, "GET", f"{PUBMED_EUTILS_BASE}/esummary.fcgi",
            timeout=timeout, max_retries=max_retries,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        )
        return ids, count, json.loads(esummary.text).get("result") or {}

    if client is not None:
        idlist, total, result = _run(client)
    else:
        with httpx.Client() as http:
            idlist, total, result = _run(http)
    if not idlist:
        return {"query_echo": {"query": query}, "items": [],
                "total_results": total, "results_per_page": results_per_page}

    items: list[dict[str, Any]] = []
    for uid in result.get("uids") or idlist:
        raw = result.get(str(uid))
        if not isinstance(raw, dict):
            continue
        doi = None
        pmcid = None
        for it in raw.get("articleids") or []:
            idtype = (it.get("idtype") or "").lower()
            value = it.get("value")
            if not value:
                continue
            if idtype == "doi":
                doi = value
            elif idtype == "pmc":
                pmcid = value.replace("PMC", "") if value.startswith("PMC") else value
        authors = ", ".join(
            str(a.get("name"))
            for a in (raw.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ) or None
        items.append(
            {
                "id": str(uid),
                "pmid": str(uid),
                "doi": doi,
                "title": raw.get("title"),
                "authors": authors,
                "journal": raw.get("fulljournalname") or raw.get("source"),
                "published_date": raw.get("pubdate") or raw.get("epubdate"),
                "abstract": raw.get("abstract"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                "pdf_url": f"https://pmc.ncbi.nlm.nih.gov/{pmcid}/pdf" if pmcid else None,
                "source": "pubmed",
            }
        )

    return {
        "query_echo": {"query": query, "from_year": from_year, "to_year": to_year},
        "items": items,
        "total_results": total,
        "results_per_page": results_per_page,
    }


_QUESTION_PREFIXES = (
    "what is",
    "what are",
    "what was",
    "what were",
    "how does",
    "how do",
    "how to",
    "why is",
    "why does",
    "explain",
    "describe",
    "tell me about",
)


def _paper_query(question: str) -> str:
    """Normalize a natural-language question into a topical search query:
    paper engines tokenize everything, so interrogative prefixes ("what
    is...") pollute arXiv/S2 ranking with off-topic matches."""
    query = str(question or "").strip().rstrip("?.!").strip()
    lowered = query.casefold()
    for prefix in _QUESTION_PREFIXES:
        if lowered.startswith(prefix + " "):
            query = query[len(prefix) + 1 :].strip()
            break
    # "the main differences between X and Y" -> "differences between X and Y"
    for filler in ("the main ", "the "):
        if query.casefold().startswith(filler):
            query = query[len(filler) :].strip()
            break
    return query


def _default_academic_providers() -> list[str]:
    """Config-driven provider set for the academic lane (task-16790):
    ``[SearchSettings] research_academic_providers`` as a comma list;
    default arXiv + Semantic Scholar. Unknown names are filtered at use
    time by the lane registry, so a typo costs nothing but a warning.

    Returns:
        The configured provider names, lowercased and deduplicated.
    """
    raw = get_cli_setting(
        "SearchSettings", "research_academic_providers",
        "arxiv,semantic_scholar",
    )
    names: list[str] = []
    for part in str(raw or "").split(","):
        name = part.strip().lower()
        if name and name not in names:
            names.append(name)
    return names or ["arxiv", "semantic_scholar"]


async def search_papers(
    query: str,
    *,
    providers: list[str] | None = None,
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
    import asyncio
    from asyncio import to_thread

    failures: list[str] = []
    papers: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    topic_query = _paper_query(query)

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

    def _arxiv_lane() -> Any:
        return search_arxiv(query=topic_query, results_per_page=results_per_page)

    def _s2_lane() -> Any:
        return search_semantic_scholar(
            query=topic_query,
            results_per_page=results_per_page,
            api_key=semantic_scholar_api_key
            if semantic_scholar_api_key is not None
            else resolve_semantic_scholar_api_key(),
        )

    def _biorxiv_lane() -> Any:
        return search_biorxiv(query=topic_query, results_per_page=results_per_page)

    def _medrxiv_lane() -> Any:
        return search_biorxiv(
            query=topic_query, server="medrxiv", results_per_page=results_per_page
        )

    def _pubmed_lane() -> Any:
        return search_pubmed(query=topic_query, results_per_page=results_per_page)

    def _openalex_lane() -> Any:
        return search_openalex(query=topic_query, results_per_page=results_per_page)

    def _crossref_lane() -> Any:
        return search_crossref(query=topic_query, results_per_page=results_per_page)

    def _zenodo_lane() -> Any:
        return search_zenodo(query=topic_query, results_per_page=results_per_page)

    def _figshare_lane() -> Any:
        return search_figshare(query=topic_query, results_per_page=results_per_page)

    def _osf_lane() -> Any:
        return search_osf(query=topic_query, results_per_page=results_per_page)

    lanes: dict[str, Any] = {
        "arxiv": _arxiv_lane,
        "semantic_scholar": _s2_lane,
        "biorxiv": _biorxiv_lane,
        "medrxiv": _medrxiv_lane,
        "pubmed": _pubmed_lane,
        "openalex": _openalex_lane,
        "crossref": _crossref_lane,
        "zenodo": _zenodo_lane,
        "figshare": _figshare_lane,
        "osf": _osf_lane,
    }
    # task-16792: tokens may be source ids OR category names ("biomedical",
    # "repositories", ...); unknown names raise instead of silently
    # narrowing the search.
    requested = expand_source_selection(
        list(providers)
        if providers is not None
        else _default_academic_providers()
    )
    selected: list[str] = []
    for name in requested:
        if name in lanes and name not in selected:  # order-preserving dedupe
            selected.append(name)
    dropped = [name for name in requested if name not in lanes]
    if dropped:
        # Defense in depth: expand_source_selection already rejects unknown
        # tokens, but a future lane rename must not silently narrow.
        logger.warning(f"search_papers ignoring unknown providers: {dropped}")

    async def _lane_runner(name: str, lane: Any) -> Any:
        try:
            return await to_thread(lane)
        except AcademicProviderError as exc:
            failures.append(f"{name}: {exc}")
            logger.warning(f"search_papers {name} lane failed: {exc}")
            return None

    # task-16789: all selected providers run CONCURRENTLY (serial execution
    # added avoidable latency); per-provider degradation lives in each lane.
    outcomes = await asyncio.gather(
        *(_lane_runner(name, lanes[name]) for name in selected)
    )
    for outcome in outcomes:
        if outcome is not None:
            _collect(outcome.get("items") or [])

    if not papers and failures:
        raise AcademicProviderError(
            "all academic providers failed: " + "; ".join(failures)
        )
    return papers
