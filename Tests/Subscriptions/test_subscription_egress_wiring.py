"""Subscription fetch paths route through the egress guard."""

import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import EgressBlockedError


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _ok(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _ok)
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda s, k=None, d=None: d
    )


def test_validate_feed_url_delegates_and_fails_closed_on_dns(monkeypatch):
    from tldw_chatbook.Subscriptions.security import SecurityValidator, SSRFError

    def _fail(host):
        raise OSError("nxdomain")

    monkeypatch.setattr(egress, "_resolve", _fail)
    with pytest.raises(SSRFError, match="dns_failure"):
        SecurityValidator.validate_feed_url("https://unresolvable.example/feed")


def test_validate_feed_url_same_origin_private_item_allowed(monkeypatch):
    from tldw_chatbook.Subscriptions.security import SecurityValidator

    monkeypatch.setattr(egress, "_resolve", lambda host: ["192.168.1.10"])
    out = SecurityValidator.validate_feed_url(
        "http://wiki.corp.example/item/1",
        trusted_origins=frozenset({"wiki.corp.example"}),
    )
    assert out.startswith("http://wiki.corp.example/")


def test_validate_feed_url_cross_origin_private_item_blocked(monkeypatch):
    from tldw_chatbook.Subscriptions.security import SecurityValidator, SSRFError

    monkeypatch.setattr(egress, "_resolve", lambda host: ["192.168.1.10"])
    with pytest.raises(SSRFError):
        SecurityValidator.validate_feed_url("http://other.internal/item")


def test_warn_insecure_ssl_once_per_host(monkeypatch):
    counted = []
    monkeypatch.setattr(egress, "log_counter", lambda name, **kw: counted.append(name))
    egress._INSECURE_SSL_WARNED.clear()
    egress.warn_insecure_ssl("selfsigned.example")
    egress.warn_insecure_ssl("selfsigned.example")
    assert counted.count("web_insecure_ssl_fetch") == 2


@pytest.mark.asyncio
async def test_feed_monitor_blocked_feed_contained(monkeypatch):
    """A blocked feed URL surfaces as the module's failure type, not a crash."""
    from tldw_chatbook.Subscriptions import monitoring_engine as me

    async def _blocked(*args, **kwargs):
        raise EgressBlockedError("http://internal.example/feed", "private")

    monkeypatch.setattr(me, "guarded_fetch_httpx_async", _blocked)
    monitor = me.FeedMonitor()
    subscription = {
        "id": 1,
        "source": "http://internal.example/feed",
        "type": "rss",
        "auth_config": None,
    }
    with pytest.raises(Exception) as exc:
        await monitor.check_feed(subscription)
    assert not isinstance(exc.value, EgressBlockedError)  # mapped, not leaked
