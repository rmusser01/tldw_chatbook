"""Wiring tests: Article_Extractor_Lib routes fetches through the egress guard."""

from unittest.mock import patch

import pytest

from tldw_chatbook.Utils.egress import EgressBlockedError
from tldw_chatbook.Web_Scraping import Article_Extractor_Lib as AEL


def test_get_page_title_uses_guarded_fetch_and_contains_block():
    with patch.object(
        AEL, "guarded_fetch_requests", side_effect=EgressBlockedError("u", "private")
    ) as mocked:
        title = AEL.get_page_title("http://internal.example/x")
    assert mocked.called
    kwargs = mocked.call_args.kwargs
    assert kwargs["max_bytes"] == 10 * 1024 * 1024
    assert kwargs["timeout"] == 10
    assert title == "Untitled (Blocked URL)"


def test_scrape_from_sitemap_blocked_returns_empty():
    with patch.object(
        AEL, "guarded_fetch_requests", side_effect=EgressBlockedError("u", "private")
    ) as mocked:
        result = AEL.scrape_from_sitemap("http://sitemap.internal/map.xml")
    assert result == []
    assert mocked.call_args.kwargs["max_bytes"] == 50 * 1024 * 1024
    assert mocked.call_args.kwargs["trusted_origins"] == frozenset(
        {"sitemap.internal"}
    )


def test_scrape_article_signature_defaults_fail_closed():
    import inspect

    sig = inspect.signature(AEL.scrape_article)
    assert sig.parameters["trusted_origins"].default == frozenset()
    sig_async = inspect.signature(AEL.scrape_article_async)
    assert sig_async.parameters["trusted_origins"].default == frozenset()
