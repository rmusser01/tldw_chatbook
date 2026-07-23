"""Policy tests for tldw_chatbook.Utils.egress (no real DNS, no network)."""

import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    check_url_or_raise,
    evaluate_url_policy,
    evaluate_url_policy_async,
)


@pytest.fixture(autouse=True)
def _no_real_dns_or_config(monkeypatch):
    """Default: everything resolves public; config enabled with no allowlist."""
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])

    async def _fake_async(host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _resolve_to(monkeypatch, ips):
    monkeypatch.setattr(egress, "_resolve", lambda host: list(ips))

    async def _fake_async(host):
        return list(ips)

    monkeypatch.setattr(egress, "_resolve_async", _fake_async)


def test_public_url_allowed():
    d = evaluate_url_policy("https://example.com/page")
    assert d.allowed and d.reason == "ok"


def test_non_http_schemes_blocked():
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x", "data:text/html,hi"):
        d = evaluate_url_policy(url)
        assert not d.allowed and d.reason == "scheme", url


def test_missing_host_blocked():
    assert not evaluate_url_policy("https:///nohost").allowed


def test_loopback_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["127.0.0.1"])
    d = evaluate_url_policy("http://myhost.example/")
    assert not d.allowed and d.reason == "private"


def test_rfc1918_blocked(monkeypatch):
    for ip in ("10.0.0.5", "172.16.3.4", "192.168.1.1"):
        _resolve_to(monkeypatch, [ip])
        assert not evaluate_url_policy("http://h.example/").allowed


def test_ipv6_ula_and_mapped_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["fd12::1"])
    assert not evaluate_url_policy("http://h.example/").allowed
    _resolve_to(monkeypatch, ["::ffff:192.168.0.1"])
    assert evaluate_url_policy("http://h.example/").reason == "private"


def test_cgnat_blocked(monkeypatch):
    _resolve_to(monkeypatch, ["100.64.0.7"])
    assert not evaluate_url_policy("http://h.example/").allowed


def test_metadata_ip_blocked_even_when_trusted(monkeypatch):
    _resolve_to(monkeypatch, ["169.254.169.254"])
    d = evaluate_url_policy(
        "http://h.example/", trusted_origins=frozenset({"h.example"})
    )
    assert not d.allowed and d.reason == "metadata"


def test_metadata_hostname_blocked_pre_resolution(monkeypatch):
    def _boom(host):  # pragma: no cover - must not be called
        raise AssertionError("resolved a metadata hostname")

    monkeypatch.setattr(egress, "_resolve", _boom)
    d = evaluate_url_policy("http://metadata.google.internal/computeMetadata/")
    assert not d.allowed and d.reason == "metadata"


def test_ip_literal_hosts_classified_directly():
    d4 = evaluate_url_policy("http://169.254.169.254/latest/meta-data/")
    assert not d4.allowed and d4.reason == "metadata"
    d6 = evaluate_url_policy("http://[::1]:8080/")
    assert not d6.allowed and d6.reason == "private"


def test_any_bad_record_blocks(monkeypatch):
    _resolve_to(monkeypatch, ["93.184.216.34", "192.168.0.9"])
    assert not evaluate_url_policy("http://h.example/").allowed


def test_dns_failure_fail_closed(monkeypatch):
    def _fail(host):
        raise OSError("nxdomain")

    monkeypatch.setattr(egress, "_resolve", _fail)
    d = evaluate_url_policy("http://nope.invalid/")
    assert not d.allowed and d.reason == "dns_failure"


def test_trusted_origin_allows_private(monkeypatch):
    _resolve_to(monkeypatch, ["192.168.1.50"])
    d = evaluate_url_policy(
        "http://wiki.corp.example/page",
        trusted_origins=frozenset({"wiki.corp.example"}),
    )
    assert d.allowed


def test_trusted_match_is_hostname_only_case_insensitive(monkeypatch):
    _resolve_to(monkeypatch, ["10.1.2.3"])
    d = evaluate_url_policy(
        "https://Wiki.CORP.example:8443/x",
        trusted_origins=frozenset({"wiki.corp.example"}),
    )
    assert d.allowed


def test_allowlist_overrides_metadata(monkeypatch):
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda s, k=None, d=None: ["metadata.google.internal"]
        if k == "allowed_hosts"
        else d,
    )
    d = evaluate_url_policy("http://metadata.google.internal/")
    assert d.allowed


def test_kill_switch_short_circuits_before_dns(monkeypatch):
    def _boom(host):  # pragma: no cover
        raise AssertionError("resolved while disabled")

    monkeypatch.setattr(egress, "_resolve", _boom)
    monkeypatch.setattr(
        egress,
        "get_cli_setting",
        lambda s, k=None, d=None: False if k == "enabled" else d,
    )
    d = evaluate_url_policy("http://192.168.0.1/")
    assert d.allowed and d.reason == "disabled"


def test_check_url_or_raise_raises_with_remedy(monkeypatch):
    _resolve_to(monkeypatch, ["127.0.0.1"])
    with pytest.raises(EgressBlockedError) as exc:
        check_url_or_raise("http://internal.example/")
    assert "internal.example" in str(exc.value)
    assert "allowed_hosts" in str(exc.value)
    assert exc.value.reason == "private"


@pytest.mark.asyncio
async def test_async_variant_same_policy(monkeypatch):
    async def _fake(host):
        return ["169.254.169.254"]

    monkeypatch.setattr(egress, "_resolve_async", _fake)
    d = await evaluate_url_policy_async("http://h.example/")
    assert not d.allowed and d.reason == "metadata"
