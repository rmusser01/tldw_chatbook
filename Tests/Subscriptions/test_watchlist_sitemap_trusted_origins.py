"""TASK-32894: a sitemap `<loc>` must not seed its own SSRF trust.

`Utils/egress.py`'s module contract:

    Shared pipeline code must NEVER auto-trust its own input URL -- trust is
    seeded only at boundaries where user intent is known and threaded down.

`Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py` fixed four
caller-less copies of this and explicitly cleared the Watchlists path --
but it only checked the sitemap DOCUMENT's own fetch (`_urls_for_sitemap`,
which is provenance-correct). The per-`<loc>` fetches downstream went
through `URLMonitor._fetch_url_content`, whose `trusted_origins=
origin_set(url)` read `url` out of the config dict the sitemap loop had
just overwritten with the DISCOVERED value. A `<loc>` naming loopback or a
LAN address therefore authorized itself.

Nothing here touches the network: `guarded_fetch_httpx_async` is replaced
by a recorder.
"""

from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Subscriptions import monitoring_engine
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)

SITEMAP_SOURCE = "https://sitemap.example.com/sitemap.xml"
MALICIOUS_LOC = "http://127.0.0.1:8080/admin"


class _InertDB:
    """`_check_url_guarded` only needs `id(db)`; `check_url` unwinds early."""

    def __getattr__(self, name):  # pragma: no cover - defensive
        def _noop(*args, **kwargs):
            return None

        return _noop


@pytest.fixture
def captured_fetches(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    async def _recorder(url, *, client, max_bytes, trusted_origins, headers=None):
        seen.append({"url": url, "trusted_origins": trusted_origins})
        raise RuntimeError("fetch stopped after recording the egress trust")

    monkeypatch.setattr(monitoring_engine, "guarded_fetch_httpx_async", _recorder)
    return seen


def _run_sitemap_source(monkeypatch, captured) -> None:
    async def _fake_urls(cls_subscription):
        return [MALICIOUS_LOC]

    monkeypatch.setattr(
        LocalWatchlistsService,
        "_urls_for_sitemap",
        classmethod(lambda cls, subscription: _fake_urls(subscription)),
    )
    service = LocalWatchlistsService(db_factory=lambda: _InertDB())
    subscription = {
        "id": 1,
        "type": "sitemap",
        "source": SITEMAP_SOURCE,
        "name": "example sitemap",
    }
    asyncio.run(service._default_run_executor(subscription, _InertDB()))


def test_discovered_loc_is_not_its_own_trusted_origin(monkeypatch, captured_fetches):
    _run_sitemap_source(monkeypatch, captured_fetches)
    assert captured_fetches, "the per-<loc> fetch never happened; the test proved nothing"
    record = captured_fetches[-1]
    assert record["url"] == MALICIOUS_LOC
    assert "127.0.0.1" not in record["trusted_origins"], (
        "the discovered <loc> authorized itself: trusted_origins="
        f"{set(record['trusted_origins'])}"
    )
    assert record["trusted_origins"] == frozenset({"sitemap.example.com"}), (
        "trust must be seeded from the subscription's configured source"
    )


def test_a_configured_url_source_still_trusts_itself(monkeypatch, captured_fetches):
    """The user's OWN configured LAN/loopback source must keep working."""
    monitor = monitoring_engine.URLMonitor(_InertDB())
    config = {"id": 2, "type": "url", "source": "http://192.168.1.5/status"}
    with pytest.raises(Exception):
        asyncio.run(monitor._fetch_url_content(config))
    assert captured_fetches[-1]["trusted_origins"] == frozenset({"192.168.1.5"})
