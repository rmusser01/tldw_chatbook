"""[dreams] Candidate discovery: web-search pool, watchlist pool, rank.

The cycle's two candidate sources meet here (spec §discovery): queries run
through the injected ``perform_websearch`` seam become ``web`` candidates,
fresh unread watchlist items become ``watchlist`` candidates, and
``dedupe_and_rank`` collapses both pools against the seen ledger and the
media library before ranking by lexical topic overlap.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

#: Locale defaults for the injected ``perform_websearch`` seam; Dreams has
#: no per-user locale setting in Phase 1, so the seams get the app's
#: customary US/English baseline.
_DEFAULT_CONTENT_COUNTRY = "us"
_DEFAULT_SEARCH_LANG = "en"
_DEFAULT_OUTPUT_LANG = "en"

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Watchlist pool bounds: leading item content carried as the snippet, and
#: how many fresh items one cycle considers.
_WATCHLIST_SNIPPET_CHARS = 800
_WATCHLIST_POOL_LIMIT = 200


@dataclass(slots=True)
class Candidate:
    """One discovery candidate before story generation.

    Attributes:
        title: Result headline / item title.
        url: Source URL (original, un-normalized).
        snippet: Search-result snippet, or a watchlist item's leading content.
        source: Which pool produced it: ``'web'`` or ``'watchlist'``.
    """

    title: str
    url: str
    snippet: str
    source: str  # 'web' | 'watchlist'


#: Query parameters that only track the click, never identify the page.
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ref",
    "ref_src", "igshid", "si",
})


def normalize_url(url: str) -> str:
    """Dedupe identity: case-folded host, no fragment/tracking/trailing slash.

    ``https://A.x/rust-tui/?utm_source=feed`` and ``https://a.x/rust-tui``
    are the same page. The path keeps its case and non-tracking query
    parameters survive: ``youtube.com/watch?v=A`` and ``?v=B`` are different
    videos (dropping the whole query string collapsed every video into one
    key, so one library video hid all of them). Used for both in-pool
    dedupe and the cross-cycle seen ledger, so the two agree.

    Args:
        url: Candidate URL, verbatim.

    Returns:
        The comparison key; the candidate keeps its original URL.
    """
    parts = urlsplit(url.strip())
    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_PARAMS
    ])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), query, ""))


def _extract_results(payload: Any) -> list[dict]:
    """Pull the standardized result rows out of one ``perform_websearch`` return.

    Real payload shapes (``Web_Scraping/WebSearch_APIs.py``): success is
    ``process_web_search_results``'s dict whose ``results`` items carry
    ``title`` / ``url`` / ``content`` (the snippet; ``metadata`` is
    auxiliary); every failure path converges on
    ``_set_search_processing_error``'s ``{"results": [],
    "processing_error": ..., "error_kind": ...}`` (the request-error path
    returns ``{"processing_error", "error_kind"}`` without ``results``).
    A truthy ``error``/``processing_error`` marker therefore means "no
    results", never a partial read — both paths empty ``results`` before
    the marker survives.
    """
    if not isinstance(payload, dict):
        return []
    if payload.get("error") or payload.get("processing_error"):
        return []
    out: list[dict] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        out.append({
            "title": str(item.get("title") or ""),
            "url": url,
            "snippet": str(item.get("content") or ""),
        })
    return out


async def run_queries(
    perform: Callable[..., Any],
    *,
    engine: str,
    queries: list[str],
    result_count: int = 8,
    date_range: str | None = "m",
) -> tuple[list[Candidate], int]:
    """Run every query through the injected ``perform_websearch`` seam.

    Each search is blocking network I/O, so it runs under
    ``asyncio.to_thread`` (repo-global constraint) — never on the event
    loop. ``date_range="m"`` by default enforces the spec's recency
    promise for time-sensitive angles. A query whose search errors — error
    payload or exception — contributes no candidates but still counts as
    spent: the daily search budget was consumed either way.

    Args:
        perform: ``perform_websearch``-shaped callable (injected).
        engine: Search engine id (e.g. ``"duckduckgo"``).
        queries: Query strings, typically from ``synthesize_queries``.
        result_count: Results to request per query.
        date_range: Recency filter passed through (``'m'``, ``'w'``,
            ``'y'``, or ``None`` for no filter).

    Returns:
        ``(candidates, searches_used)`` — flattened non-error candidates
        in query order, and the number of searches attempted
        (``len(queries)``).
    """
    candidates: list[Candidate] = []
    for query in queries:
        try:
            payload = await asyncio.to_thread(
                perform, engine, query,
                content_country=_DEFAULT_CONTENT_COUNTRY,
                search_lang=_DEFAULT_SEARCH_LANG,
                output_lang=_DEFAULT_OUTPUT_LANG,
                result_count=result_count,
                date_range=date_range,
            )
        except Exception as exc:  # noqa: BLE001 - one dead engine is degradation
            # Type name only: the query is built from the user's interest
            # profile (notes keywords, Personal Context), never logged.
            logger.warning("Dreams search failed: {}", type(exc).__name__)
            continue
        for item in _extract_results(payload):
            candidates.append(Candidate(
                title=item["title"], url=item["url"], snippet=item["snippet"],
                source="web",
            ))
    return candidates, len(queries)


def fetch_watchlist_candidates(
    subs_db: SubscriptionsDB,
    *,
    freshness_hours: int,
    now_epoch: float,
) -> list[Candidate]:
    """Fresh, unread watchlist items as ``watchlist`` candidates.

    ``subscription_items`` rows with ``status = 'new'`` whose publish date
    falls within ``freshness_hours`` of ``now_epoch``. The window compares
    the table's ``effective_date`` generated column
    (``COALESCE(datetime(published_date), datetime(created_at))``) against
    ``datetime(?)`` — the same normalized floor
    ``SubscriptionsDB.get_unread_items_count_since`` uses, because stored
    ``published_date`` strings are mixed naive/aware and cannot be
    compared raw (the documented trap in ``Subscriptions/item_dates.py``).
    The window is bounded above by ``now`` too: a feed item stamped in the
    future is not "fresh" before it is published. The item's stored content
    (bounded) rides along as the snippet so the story step has source
    material, not just a headline.

    Args:
        subs_db: Subscriptions database to read the item pool from.
        freshness_hours: How many hours back an item still counts as fresh.
        now_epoch: Current time as seconds since the epoch (injected for
            determinism).

    Returns:
        Candidates freshest-first (at most ``_WATCHLIST_POOL_LIMIT``);
        ``snippet`` is the item's leading content, ``""`` when it has none.
    """
    cutoff = datetime.fromtimestamp(
        now_epoch - freshness_hours * 3600.0, tz=UTC
    ).isoformat()
    upper = datetime.fromtimestamp(now_epoch, tz=UTC).isoformat()
    with subs_db.transaction() as conn:
        rows = conn.execute(
            "SELECT i.url AS url, i.title AS title,"
            " substr(COALESCE(i.content, ''), 1, ?) AS snippet"
            " FROM subscription_items AS i"
            " JOIN subscriptions AS s ON s.id = i.subscription_id"
            " WHERE i.status = 'new'"
            " AND i.effective_date >= datetime(?)"
            " AND i.effective_date <= datetime(?)"
            " ORDER BY i.effective_date DESC, i.id ASC"
            " LIMIT ?",
            (_WATCHLIST_SNIPPET_CHARS, cutoff, upper, _WATCHLIST_POOL_LIMIT),
        ).fetchall()
    return [
        Candidate(title=str(row["title"] or ""), url=str(row["url"] or ""),
                  snippet=str(row["snippet"] or ""), source="watchlist")
        for row in rows
        if row["url"]
    ]


def _overlap_score(text: str, topics: list[dict]) -> float:
    """1.0 per topic that lexically overlaps ``text``, summed.

    A topic counts when its normalized text appears in the candidate text
    as a substring (multi-word topics like "rust tui" match whole) or
    shares any word token (single-word pivots like the "jazz" of "jazz
    guitar" still match). Case-insensitive throughout.

    Args:
        text: Candidate title + snippet, already lowercased by the caller.
        topics: Snapshot topic rows carrying ``text``.

    Returns:
        The summed score (1.0 per matching topic).
    """
    words = set(_WORD_RE.findall(text))
    score = 0.0
    for topic in topics:
        topic_text = str(topic.get("text", "")).strip().lower()
        if not topic_text:
            continue
        if topic_text in text or words & set(_WORD_RE.findall(topic_text)):
            score += 1.0
    return score


def dedupe_and_rank(
    candidates: list[Candidate],
    *,
    seen_urls: set[str],
    library_urls: set[str],
    topics: list[dict],
    limit: int,
) -> list[Candidate]:
    """Drop seen/library/duplicate URLs, rank by topic overlap, cap at ``limit``.

    Pure — inputs are never mutated. URL identity is ``normalize_url``
    (host case-folded, tracking parameters dropped, trailing slash stripped) for both the
    seen/library membership checks and exact-duplicate detection; the
    surviving candidate keeps its original URL verbatim and the FIRST
    occurrence wins. Ranking is lexical topic overlap (see
    ``_overlap_score``) over title + snippet; ties keep original order
    (stable sort), so zero-overlap pools fall back to arrival order.

    Args:
        candidates: Both pools' candidates, web and watchlist interleaved.
        seen_urls: URLs already surfaced by earlier cycles (seen ledger).
        library_urls: URLs already in the media library.
        topics: Snapshot topic rows carrying ``text``.
        limit: Maximum number of candidates to return.

    Returns:
        The top ``limit`` candidates, best-overlap first.
    """
    excluded = {normalize_url(u) for u in seen_urls}
    excluded |= {normalize_url(u) for u in library_urls}
    kept: list[Candidate] = []
    seen_keys: set[str] = set()
    for cand in candidates:
        key = normalize_url(cand.url)
        if key in excluded or key in seen_keys:
            continue
        seen_keys.add(key)
        kept.append(cand)
    scored = [
        (_overlap_score(f"{cand.title} {cand.snippet}".lower(), topics), index, cand)
        for index, cand in enumerate(kept)
    ]
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [cand for _, _, cand in scored[:limit]]
