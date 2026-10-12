"""Review of #2921, item 2: a ``url`` subscription's own source is a trusted
egress origin only when a USER configured it.

``URLMonitor._fetch_url_content`` fetches with the subscription's source as
a trusted origin, and the egress policy lets a trusted host through before
it looks at the address the host resolves to. That is right for a URL the
user typed (an intranet page is a legitimate thing to watch) and wrong for
one a Dreams story supplied: track a public URL whose host later resolves
to a private address, and every later check fetched that address.

What this is NOT: address pinning. The policy check and the connection
resolve the name separately (``Utils/egress.py`` documents that as a
non-goal), so a name that changes its answer between the two -- DNS
rebinding with a zero TTL -- is not caught by this.

A subscription whose URL nobody typed says so in ``processing_options``
(``url_provenance = "unknown"``), and the fetch grants it no trusted origin
-- on every fetch, through the real egress policy, which is what these
tests run. Only the HTTP transport and DNS are faked.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.monitoring_engine import (
    FetchBlockedError,
    URLMonitor,
)
from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import URL_PROVENANCE_OPTION, UrlProvenance

_URL = "https://later-private.example/pricing"
_STORY_DERIVED = {URL_PROVENANCE_OPTION: UrlProvenance.UNKNOWN.value}


@pytest.fixture()
def monitor(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "provenance-test")
    try:
        yield URLMonitor(db)
    finally:
        db.close()


@pytest.fixture()
def wire(monkeypatch):
    """Fake DNS and the HTTP transport; leave the egress policy real.

    Returns a handle whose ``resolves_to`` list the test sets and whose
    ``sent`` list records every request that actually left.
    """
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default=None: default)

    class Wire:
        resolves_to = ["93.184.216.34"]
        sent: list[str] = []

    handle = Wire()
    handle.sent = []

    async def resolve(_host):
        return list(handle.resolves_to)

    async def send(self, request, **kwargs):
        handle.sent.append(str(request.url))
        return httpx.Response(
            200, request=request, headers={"content-type": "text/html"},
            content=b"<html><body><p>price: 10</p></body></html>")

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return handle


def _subscription(processing_options=None) -> dict:
    row = {"id": 1, "source": _URL, "type": "url", "extraction_method": "auto"}
    if processing_options is not None:
        row["processing_options"] = processing_options
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored",
    [_STORY_DERIVED, json.dumps(_STORY_DERIVED)],
    ids=["parsed-config", "raw-row-json"],
)
async def test_story_derived_source_is_refused_once_its_host_resolves_private(
        monitor, wire, stored):
    wire.resolves_to = ["192.168.1.1"]

    with pytest.raises(FetchBlockedError):
        await monitor._fetch_url_content(_subscription(stored))

    assert wire.sent == [], "nothing was sent to the private address"


@pytest.mark.asyncio
async def test_story_derived_source_is_fetched_while_its_host_is_public(
        monitor, wire):
    result = await monitor._fetch_url_content(_subscription(_STORY_DERIVED))

    assert wire.sent == [_URL]
    assert "price: 10" in result["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored", [None, {}, {"max_items": 5}, "not json"],
    ids=["no-options", "empty", "other-options", "unreadable"],
)
async def test_user_configured_private_source_is_still_fetched(
        monitor, wire, stored):
    """The user's own subscription keeps its behaviour: an intranet page is
    a legitimate thing to watch, and its source vouches for itself."""
    wire.resolves_to = ["192.168.1.1"]

    await monitor._fetch_url_content(_subscription(stored))

    assert wire.sent == [_URL]


# --- Review round 3, T17: every fetch arm honours the mark ----------------------
#
# The mark was read only by the ``url`` arm. The feed, sitemap and API arms
# trusted their own source unconditionally, and a URL edit kept the mark.
# No UI reaches either today; they are closed anyway.


def _blocked_or_unsent(wire) -> None:
    assert wire.sent == [], "nothing was sent to the private address"


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [_STORY_DERIVED, json.dumps(_STORY_DERIVED)],
                         ids=["parsed-config", "raw-row-json"])
async def test_story_derived_feed_source_is_refused_once_it_resolves_private(
        tmp_path, wire, stored):
    from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor

    wire.resolves_to = ["192.168.1.1"]
    monitor = FeedMonitor()
    with pytest.raises(FetchBlockedError):
        await monitor._fetch_and_parse_feed(
            {"id": 1, "source": _URL, "type": "rss",
             "processing_options": stored})

    _blocked_or_unsent(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["sitemap", "api"])
async def test_story_derived_sitemap_and_api_sources_earn_no_trusted_origin(
        wire, arm):
    from tldw_chatbook.Subscriptions import LocalWatchlistsService
    from tldw_chatbook.Utils.egress import EgressBlockedError

    wire.resolves_to = ["192.168.1.1"]
    subscription = {"id": 1, "source": _URL, "processing_options": _STORY_DERIVED}
    fetch = (LocalWatchlistsService._urls_for_sitemap if arm == "sitemap"
             else LocalWatchlistsService._items_for_api_source)

    with pytest.raises(EgressBlockedError):
        await fetch(subscription)

    _blocked_or_unsent(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["feed", "sitemap", "api"])
async def test_user_configured_private_feed_sitemap_and_api_are_still_fetched(
        wire, arm):
    """The other arms keep their behaviour for a source the user typed."""
    from tldw_chatbook.Subscriptions import LocalWatchlistsService
    from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor

    wire.resolves_to = ["192.168.1.1"]
    subscription = {"id": 1, "source": _URL, "type": "rss"}
    try:
        if arm == "feed":
            await FeedMonitor()._fetch_and_parse_feed(subscription)
        elif arm == "sitemap":
            await LocalWatchlistsService._urls_for_sitemap(subscription)
        else:
            await LocalWatchlistsService._items_for_api_source(subscription)
    except Exception:  # noqa: BLE001 - the fake page is not a feed/sitemap/JSON
        pass

    assert wire.sent == [_URL], "the request went out: the source is trusted"


@pytest.mark.asyncio
async def test_a_user_editing_the_url_makes_the_source_user_entered(
        tmp_path, monkeypatch):
    """Owner ruling: when the USER changes the URL it is their entry; any
    other write -- Dreams' own included -- keeps the mark.

    Round 4: the edit RECORDS that (``url_provenance = "user_entered"``)
    instead of leaving no record. "No record" is also what a source from
    before the mark existed looks like, and Dreams stamps those -- so it
    stamped the user's edited source back at the next switch-off."""
    from tldw_chatbook.Subscriptions import LocalWatchlistsService

    # ``create_source`` freezes a config default through the guarded
    # loader, which fails closed under the per-test config redirect.
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default=None: default)
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "provenance-test")
    try:
        service = LocalWatchlistsService(db_factory=lambda: db)
        created = await service.create_source(
            {"name": "Dreams: page", "source_type": "url", "source": _URL,
             "processing_options": {**_STORY_DERIVED, "max_items": 5}})
        source_id = created["source_id"]

        def options() -> dict:
            return json.loads(
                db.get_subscription(source_id)["processing_options"] or "{}")

        await service.update_source(source_id, {"active": False})
        await service.update_source(source_id, {"name": "renamed"})
        await service.update_source(source_id, {"url": _URL})  # unchanged
        assert options() == {**_STORY_DERIVED, "max_items": 5}

        await service.update_source(
            source_id, {"url": "https://intranet.example/status"})

        assert options() == {
            URL_PROVENANCE_OPTION: UrlProvenance.USER_ENTERED.value,
            "max_items": 5,
        }, "recorded as the user's entry; the rest stays"
        from tldw_chatbook.Utils.egress import source_trusted_origins

        assert source_trusted_origins(
            "https://intranet.example/status", options()), (
            "and fetched as any address the user typed"
        )
    finally:
        db.close()
