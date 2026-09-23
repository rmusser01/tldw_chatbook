# Tests/Dreams/test_discovery.py
"""Discovery pools: web-search run, watchlist pool, dedupe and rank."""
from datetime import UTC, datetime

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Dreams.discovery import (
    Candidate,
    dedupe_and_rank,
    fetch_watchlist_candidates,
    run_queries,
)

CANDS = [
    Candidate("Rust TUI guide", "https://a.x/Rust-Tui/", "…", "web"),
    Candidate("Dup", "https://a.x/rust-tui?utm=1", "…", "web"),
    Candidate("Jazz near Seattle", "https://b.x/jazz", "…", "watchlist"),
    Candidate("Seen already", "https://c.x/old", "…", "web"),
    Candidate("In library", "https://d.x/lib", "…", "web"),
    Candidate("Off-topic", "https://e.x/taxes", "…", "web"),
]
TOPICS = [{"facet": "topic", "text": "rust tui", "weight": 0.9},
          {"facet": "topic", "text": "jazz guitar", "weight": 0.6}]


def test_dedupe_and_rank_drops_seen_library_and_duplicates_and_ranks_by_overlap():
    ranked = dedupe_and_rank(
        CANDS, seen_urls={"https://c.x/old"}, library_urls={"https://d.x/lib"},
        topics=TOPICS, limit=3,
    )
    urls = [c.url for c in ranked]
    assert urls == ["https://a.x/Rust-Tui/", "https://b.x/jazz", "https://e.x/taxes"]


def test_ranking_falls_back_to_original_order_on_no_overlap():
    ranked = dedupe_and_rank([CANDS[5]], seen_urls=set(), library_urls=set(),
                             topics=[], limit=5)
    assert ranked[0].url == "https://e.x/taxes"


@pytest.mark.asyncio
async def test_run_queries_counts_searches_and_skips_error_results():
    # Result payload keys are the REAL perform_websearch shapes: success is
    # process_web_search_results' standardized dict with per-result
    # title/url/content; every failure path converges on
    # _set_search_processing_error's {"results": [], "processing_error",
    # "error_kind"}.
    def perform(search_engine, search_query, **kwargs):
        if "bad" in search_query:
            return {
                "results": [],
                "processing_error": "Search provider returned an invalid response.",
                "error_kind": "response",
            }
        return {"results": [{"title": "t", "url": "https://ok.x/1", "content": "s"}]}

    cands, used = await run_queries(perform, engine="duckduckgo",
                                    queries=["good query", "bad query"])
    assert used == 2
    assert [c.url for c in cands] == ["https://ok.x/1"]
    assert cands[0].title == "t" and cands[0].snippet == "s"
    assert cands[0].source == "web"


def test_fetch_watchlist_candidates_returns_only_fresh_new_items(tmp_path):
    subs = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    try:
        # Source row via direct SQL, the verified Tests/Tools fixture idiom.
        with subs.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO subscriptions (
                    name, type, source, is_active, is_paused, created_at, updated_at,
                    last_checked, last_successful_check
                ) VALUES (?, 'rss', ?, ?, ?, ?, ?, ?, ?)
                """,
                ("Blog", "https://example.test/feed", 1, 0,
                 "2026-09-01 09:00:00", "2026-09-02 10:00:00",
                 "2026-09-22 11:00:00", "2026-09-22 10:55:00"),
            )
            sub_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO subscription_items"
                " (subscription_id, url, title, published_date, status,"
                "  created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    # Fresh, aware-ISO publish date.
                    (sub_id, "https://example.test/fresh", "Fresh",
                     "2026-09-22T10:00:00+00:00", "new",
                     "2026-09-22 10:05:00", "2026-09-22 10:05:00"),
                    # Fresh, naive space-separated publish date -- the mixed
                    # stored format item_dates.py documents.
                    (sub_id, "https://example.test/naive", "Naive format",
                     "2026-09-22 09:00:00", "new",
                     "2026-09-22 09:05:00", "2026-09-22 09:05:00"),
                    # Outside the 24h window.
                    (sub_id, "https://example.test/stale", "Stale",
                     "2026-09-20T10:00:00+00:00", "new",
                     "2026-09-20 10:05:00", "2026-09-20 10:05:00"),
                    # Inside the window but already read.
                    (sub_id, "https://example.test/reviewed", "Already read",
                     "2026-09-22T10:00:00+00:00", "reviewed",
                     "2026-09-22 10:05:00", "2026-09-22 10:05:00"),
                ],
            )
        now_epoch = datetime(2026, 9, 22, 12, 0, tzinfo=UTC).timestamp()
        out = fetch_watchlist_candidates(subs, freshness_hours=24, now_epoch=now_epoch)
        urls = [c.url for c in out]
        assert "https://example.test/fresh" in urls
        assert "https://example.test/naive" in urls
        assert "https://example.test/stale" not in urls
        assert "https://example.test/reviewed" not in urls
        assert all(c.source == "watchlist" for c in out)
        assert {c.url: c.title for c in out}["https://example.test/fresh"] == "Fresh"
    finally:
        subs.close_all_connections()
