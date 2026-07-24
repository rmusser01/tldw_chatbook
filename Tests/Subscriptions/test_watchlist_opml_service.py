import pytest
from tldw_chatbook.Subscriptions.watchlist_opml_service import WatchlistOpmlService


def test_parse_opml():
    xml = '''<?xml version="1.0"?><opml version="2.0"><body><outline text="Tech" title="Tech"><outline text="AI" title="AI" type="rss" xmlUrl="http://example.com/ai"/></outline></body></opml>'''
    svc = WatchlistOpmlService()
    items = svc.parse(xml)
    assert len(items) == 1
    assert items[0]["url"] == "http://example.com/ai"
    assert items[0]["source_type"] == "rss"


def test_export_opml():
    svc = WatchlistOpmlService()
    xml = svc.export([
        {"name": "AI", "url": "http://example.com/ai", "source_type": "rss"}
    ])
    assert "http://example.com/ai" in xml
