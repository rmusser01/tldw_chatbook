# Web-Fetch Hardening Implementation Plan (TASK-328 + TASK-329)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** App-wide SSRF protection (policy module + per-hop-validated guarded fetch helpers) and response-size caps/timeouts across every web-content fetcher.

**Architecture:** New `tldw_chatbook/Utils/egress.py` owns the single policy rule (public-or-trusted-or-allowlisted; metadata blocked unless allowlisted) with sync+async DNS evaluation, plus guarded transport helpers for httpx/requests/aiohttp and a Playwright navigation-chain validator. ~15 fetch surfaces get wired to the helpers; `Subscriptions/security.py`'s dead validator delegates to the new module.

**Tech Stack:** Python ≥3.11 stdlib `ipaddress`/`socket`, httpx 0.28, requests, aiohttp 3.13, Playwright 1.58, pytest + httpx.MockTransport + hand-rolled fakes (no third-party HTTP-mock libs are installed).

**Spec:** `Docs/superpowers/specs/2026-07-23-web-fetch-hardening-design.md` (committed `33e3c9416`). Read it for policy rationale; THIS plan carries the exact code.

## Global Constraints

1. **Fail-closed trust-threading:** shared pipeline functions (`scrape_article*`, `Scraper._fetch_html`, `get_page_title`, the guarded helpers) take `trusted_origins: frozenset[str]` **defaulting to `frozenset()`** and NEVER auto-trust their own input URL's host. Trust is seeded only at boundaries where user intent is known (named per task).
2. **Credential stripping:** manual redirect loops strip `Authorization`/`Cookie`/`Proxy-Authorization` and suppress httpx `auth=` / `requests.Session.auth` on any hop whose host differs from the FIRST URL's host.
3. **Caps measure decompressed bytes** (httpx `iter_bytes`/requests `iter_content`/aiohttp `iter_chunked` — never switch to raw streams), aborting the instant the running total crosses the cap.
4. **`EgressBlockedError` containment:** every wired surface maps it (and `EgressFetchError`) into that surface's existing failure path — never an unhandled crash from a worker/gather loop.
5. **Async paths use `evaluate_url_policy_async` / `check_url_or_raise_async`** (event-loop `getaddrinfo`) — never the sync resolver on the event loop.
6. Metadata endpoints blocked even for `trusted_origins`; ONLY `[web_security] allowed_hosts` overrides. Kill switch `[web_security] enabled=false` short-circuits BEFORE DNS.
7. Helpers never call `raise_for_status()` — status semantics belong to callers. Redirect hop cap = 10 (`MAX_REDIRECT_HOPS`).
8. Byte caps: pages/feeds/API 10MB (`MAX_FETCH_BYTES_PAGE`), sitemaps 50MB (`MAX_FETCH_BYTES_SITEMAP`), GitHub file/tree 20MB (`MAX_FETCH_BYTES_GITHUB_FILE`), media/audio 500MB (`MAX_FETCH_BYTES_MEDIA`).
9. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-web-fetch-hardening` (branch `feat/web-fetch-hardening`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only files you changed — never `-A`.
10. Existing suites that must stay green after each wiring task: `Tests/Web_Scraping/`, `Tests/Subscriptions/`, `Tests/Local_Ingestion/`, `Tests/Utils/` (run the ones matching your task's files). Deliberate behavior changes to old tests are listed per task — nothing else may change.

**Known pre-existing baseline:** none in these suites at branch point `8a46af45e` (verify with a quick run before your first change; report any pre-existing failure rather than fixing it).

---

### Task 1: Egress policy core (`Utils/egress.py`) + `[web_security]` config

**Files:**
- Create: `tldw_chatbook/Utils/egress.py`
- Create: `Tests/Utils/test_egress.py`
- Modify: `tldw_chatbook/config.py` (CONFIG_TOML default template — add `[web_security]` section)

**Interfaces (Produces):**
- `EgressBlockedError(url, reason, detail="")` — `.url`, `.reason`, `.detail`; message names URL, reason, and remedy.
- `EgressDecision(allowed: bool, reason: str, host: str, resolved_ips: tuple[str, ...])` — frozen dataclass; reason ∈ `{"ok","scheme","metadata","private","dns_failure","disabled"}`.
- `evaluate_url_policy(url, *, trusted_origins=frozenset()) -> EgressDecision`
- `evaluate_url_policy_async(url, *, trusted_origins=frozenset()) -> EgressDecision`
- `check_url_or_raise(url, *, trusted_origins=frozenset())` / `check_url_or_raise_async(...)` — raise `EgressBlockedError` when not allowed.
- Constants: `MAX_REDIRECT_HOPS=10`, `MAX_FETCH_BYTES_PAGE=10*1024*1024`, `MAX_FETCH_BYTES_SITEMAP=50*1024*1024`, `MAX_FETCH_BYTES_GITHUB_FILE=20*1024*1024`, `MAX_FETCH_BYTES_MEDIA=500*1024*1024`.
- Test seams: module-level `_resolve(host) -> list[str]` (sync) and `_resolve_async(host) -> list[str]` — monkeypatch these in ALL tests (never real DNS).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Utils/test_egress.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`
Expected: collection error — `ModuleNotFoundError`/`ImportError` for `tldw_chatbook.Utils.egress`.

- [ ] **Step 3: Implement `tldw_chatbook/Utils/egress.py`**

```python
"""App-wide egress policy for outbound URL fetching (SSRF protection).

One rule: a URL is allowed iff every resolved IP is public and not a cloud
metadata endpoint, OR its hostname is in ``trusted_origins`` (a host the USER
explicitly typed/configured), OR its hostname is in the ``[web_security]
allowed_hosts`` config allowlist. Metadata endpoints are stricter: blocked
even for trusted origins; only the config allowlist overrides them.

Shared pipeline code must NEVER auto-trust its own input URL — trust is
seeded only at boundaries where user intent is known and threaded down
(see Docs/superpowers/specs/2026-07-23-web-fetch-hardening-design.md).

Non-goals (documented residual risk): DNS-rebinding IP pinning (we
resolve-and-check; the HTTP client re-resolves to connect), proxy-aware
policy (env-var proxies keep working; the target URL is what's validated),
and DNS caching (OS resolver caches suffice).
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Iterable, List
from urllib.parse import urlparse

from loguru import logger

from ..config import get_cli_setting
from ..Metrics.metrics_logger import log_counter

# Cloud metadata endpoints: blocked even for trusted origins.
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure IPv4
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IPv6
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud
    }
)
METADATA_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.azure.com"})

MAX_REDIRECT_HOPS = 10
MAX_FETCH_BYTES_PAGE = 10 * 1024 * 1024
MAX_FETCH_BYTES_SITEMAP = 50 * 1024 * 1024  # sitemap protocol allows 50MB uncompressed
MAX_FETCH_BYTES_GITHUB_FILE = 20 * 1024 * 1024
MAX_FETCH_BYTES_MEDIA = 500 * 1024 * 1024


class EgressBlockedError(Exception):
    """URL blocked by the egress policy (SSRF guard)."""

    def __init__(self, url: str, reason: str, detail: str = ""):
        self.url = url
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"Egress blocked ({reason}) for {url}"
            + (f": {detail}" if detail else "")
            + " [remedy: add the host to [web_security] allowed_hosts in"
            " config.toml, or set [web_security] enabled = false]"
        )


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str  # "ok" | "scheme" | "metadata" | "private" | "dns_failure" | "disabled"
    host: str
    resolved_ips: tuple = ()


def _resolve(host: str) -> List[str]:
    """Resolve every A/AAAA record for ``host`` (test seam — monkeypatched)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


async def _resolve_async(host: str) -> List[str]:
    """Async resolution via the event loop (test seam — monkeypatched)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def _config_enabled() -> bool:
    value = get_cli_setting("web_security", "enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def _config_allowed_hosts() -> frozenset:
    value = get_cli_setting("web_security", "allowed_hosts", [])
    if not isinstance(value, (list, tuple, set)):
        return frozenset()
    return frozenset(str(h).strip().lower() for h in value if str(h).strip())


def _classify_ip(ip_str: str) -> str:
    """Classify one resolved IP: "metadata" | "private" | "public"."""
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip in _METADATA_IPS:
        return "metadata"
    return "public" if ip.is_global else "private"


def _blocked(url: str, reason: str, host: str, detail: str = "") -> EgressDecision:
    logger.warning(f"Egress blocked ({reason}): {url} {detail}".rstrip())
    log_counter("egress_blocked", labels={"reason": reason})
    return EgressDecision(allowed=False, reason=reason, host=host)


def _pre_resolution(url: str, trusted_origins: frozenset):
    """Checks that need no DNS. Returns EgressDecision, or the host to resolve."""
    if not _config_enabled():
        logger.debug(f"Egress check disabled by [web_security] for {url}")
        log_counter("egress_check_skipped")
        return EgressDecision(allowed=True, reason="disabled", host="")
    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return _blocked(url, "scheme", "", "unparseable URL")
    if parsed.scheme not in ("http", "https") or not host:
        return _blocked(url, "scheme", host or "", "only http/https with a host")
    h = host.lower()
    allowed_hosts = _config_allowed_hosts()
    if h in allowed_hosts:
        return EgressDecision(allowed=True, reason="ok", host=h)
    if h in METADATA_HOSTNAMES:
        return _blocked(url, "metadata", h, "cloud metadata hostname")
    try:
        ipaddress.ip_address(h)
    except ValueError:
        return h  # hostname — caller resolves and calls _post_resolution
    # IP-literal host (incl. bracketed IPv6): classify directly, no DNS.
    return _post_resolution(url, h, (h,), trusted_origins)


def _post_resolution(
    url: str, host: str, ips: Iterable[str], trusted_origins: frozenset
) -> EgressDecision:
    ips = tuple(ips)
    classes = {_classify_ip(ip) for ip in ips}
    if "metadata" in classes:
        return _blocked(url, "metadata", host, f"resolves to metadata IP ({ips})")
    if host in trusted_origins:
        return EgressDecision(allowed=True, reason="ok", host=host, resolved_ips=ips)
    if "private" in classes:
        return _blocked(url, "private", host, f"resolves to private IP ({ips})")
    return EgressDecision(allowed=True, reason="ok", host=host, resolved_ips=ips)


def _normalize_trusted(trusted_origins) -> frozenset:
    return frozenset(str(h).strip().lower() for h in (trusted_origins or ()) if h)


def evaluate_url_policy(url: str, *, trusted_origins=frozenset()) -> EgressDecision:
    """Evaluate the egress policy for ``url`` (sync — blocking DNS).

    Never call from an asyncio event loop; use
    :func:`evaluate_url_policy_async` there.
    """
    trusted = _normalize_trusted(trusted_origins)
    pre = _pre_resolution(url, trusted)
    if isinstance(pre, EgressDecision):
        return pre
    try:
        ips = _resolve(pre)
    except (OSError, socket.gaierror) as exc:
        return _blocked(url, "dns_failure", pre, str(exc))
    return _post_resolution(url, pre, ips, trusted)


async def evaluate_url_policy_async(
    url: str, *, trusted_origins=frozenset()
) -> EgressDecision:
    """Async variant of :func:`evaluate_url_policy` (event-loop DNS)."""
    trusted = _normalize_trusted(trusted_origins)
    pre = _pre_resolution(url, trusted)
    if isinstance(pre, EgressDecision):
        return pre
    try:
        ips = await _resolve_async(pre)
    except (OSError, socket.gaierror) as exc:
        return _blocked(url, "dns_failure", pre, str(exc))
    return _post_resolution(url, pre, ips, trusted)


def check_url_or_raise(url: str, *, trusted_origins=frozenset()) -> None:
    decision = evaluate_url_policy(url, trusted_origins=trusted_origins)
    if not decision.allowed:
        raise EgressBlockedError(url, decision.reason)


async def check_url_or_raise_async(url: str, *, trusted_origins=frozenset()) -> None:
    decision = await evaluate_url_policy_async(url, trusted_origins=trusted_origins)
    if not decision.allowed:
        raise EgressBlockedError(url, decision.reason)
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`
Expected: all pass. If the `@pytest.mark.asyncio` test errors with "async def not natively supported", check how other async tests in `Tests/Subscriptions/test_local_watchlists_service.py` are marked and mirror that convention.

- [ ] **Step 5: Add `[web_security]` to the config template**

In `tldw_chatbook/config.py`, find the default-config TOML template string (search for the literal line `[image_generation]` — around line 2759). Insert BEFORE that line:

```toml
[web_security]
# Egress guard (SSRF protection) for web scraping / ingestion / subscription
# fetches. When enabled, content-derived URLs (redirects, sitemap/crawl
# discoveries, feed items) must resolve to public IPs; URLs you explicitly
# configure (feed sources, Confluence base_url, ingest URLs) may be private.
# Cloud metadata endpoints (169.254.169.254 etc.) are always blocked unless
# the exact host is listed in allowed_hosts.
enabled = true
allowed_hosts = []

```

No loader change is needed: `egress.py` reads via `get_cli_setting("web_security", key, default)` with safe defaults, which resolves the section from the nested TOML tree and falls back cleanly for configs without the section.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py tldw_chatbook/config.py
git commit -m "feat(security): egress policy module for SSRF protection (TASK-328)"
```

---

### Task 2: `GuardedResponse` + guarded httpx helpers (sync + async)

**Files:**
- Modify: `tldw_chatbook/Utils/egress.py` (append)
- Modify: `Tests/Utils/test_egress.py` (append)

**Interfaces:**
- Consumes: Task 1's policy functions/constants.
- Produces:
  - `class EgressFetchError(Exception)` — size cap / hop cap / missing-Location failures; `.url` attribute.
  - `@dataclass GuardedResponse`: `status_code: int`, `headers`, `content: bytes`, `final_url: str`, `_response` (underlying stack response or `None`); `.text` property (charset from content-type, `errors="replace"`), `.json()`, `.raise_for_status()` (delegates to `_response.raise_for_status()` when present so callers keep their stack's exception types; else raises `EgressFetchError` on status ≥ 400).
  - `guarded_fetch_httpx(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None) -> GuardedResponse` (sync)
  - `async guarded_fetch_httpx_async(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None) -> GuardedResponse`

- [ ] **Step 1: Write the failing tests** (append to `Tests/Utils/test_egress.py`)

```python
# ---------------------------------------------------------------------------
# Guarded httpx helpers
# ---------------------------------------------------------------------------
import httpx

from tldw_chatbook.Utils.egress import (
    EgressFetchError,
    GuardedResponse,
    guarded_fetch_httpx,
    guarded_fetch_httpx_async,
)


def _transport(routes, seen):
    """MockTransport: routes = {url_prefix: (status, headers, body)}."""

    def handler(request):
        seen.append(request)
        for prefix, (status, headers, body) in routes.items():
            if str(request.url).startswith(prefix):
                return httpx.Response(status, headers=headers, content=body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_httpx_basic_fetch_returns_guarded_response():
    seen = []
    routes = {"https://example.com/": (200, {"content-type": "text/html; charset=utf-8"}, b"<html>ok</html>")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://example.com/page", client=client, max_bytes=1024
        )
    assert isinstance(resp, GuardedResponse)
    assert resp.status_code == 200
    assert resp.content == b"<html>ok</html>"
    assert resp.text == "<html>ok</html>"
    assert resp.final_url == "https://example.com/page"


def test_httpx_redirect_to_internal_blocked(monkeypatch):
    def fake_resolve(host):
        return ["192.168.1.1"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    seen = []
    routes = {
        "https://example.com/": (302, {"location": "http://internal.example/"}, b""),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressBlockedError) as exc:
            guarded_fetch_httpx("https://example.com/r", client=client, max_bytes=1024)
    assert exc.value.reason == "private"
    # the internal host was never actually requested
    assert all("internal.example" not in str(r.url) for r in seen)


def test_httpx_same_origin_redirect_allowed_and_followed():
    seen = []
    routes = {
        "https://example.com/old": (301, {"location": "/new"}, b""),
        "https://example.com/new": (200, {"content-type": "text/html"}, b"moved"),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://example.com/old", client=client, max_bytes=1024
        )
    assert resp.status_code == 200 and resp.content == b"moved"
    assert resp.final_url == "https://example.com/new"


def test_httpx_cross_origin_hop_strips_credentials(monkeypatch):
    _resolve_to(monkeypatch, ["93.184.216.34"])
    seen = []
    routes = {
        "https://a.example/": (302, {"location": "https://b.example/x"}, b""),
        "https://b.example/": (200, {}, b"done"),
    }
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx(
            "https://a.example/start",
            client=client,
            max_bytes=1024,
            headers={"Authorization": "Bearer secret", "Cookie": "sid=1", "X-Keep": "y"},
        )
    assert resp.content == b"done"
    first, second = seen[0], seen[1]
    assert first.headers.get("authorization") == "Bearer secret"
    assert "authorization" not in second.headers
    assert "cookie" not in second.headers
    assert second.headers.get("x-keep") == "y"


def test_httpx_hop_cap():
    seen = []
    routes = {"https://example.com/": (302, {"location": "/loop"}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="redirect"):
            guarded_fetch_httpx("https://example.com/loop", client=client, max_bytes=64)


def test_httpx_redirect_without_location():
    seen = []
    routes = {"https://example.com/": (302, {}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="[Ll]ocation"):
            guarded_fetch_httpx("https://example.com/x", client=client, max_bytes=64)


def test_httpx_byte_cap_aborts():
    seen = []
    routes = {"https://example.com/": (200, {}, b"x" * 2048)}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        with pytest.raises(EgressFetchError, match="exceeds"):
            guarded_fetch_httpx("https://example.com/big", client=client, max_bytes=1024)


def test_httpx_304_passes_through_without_raise():
    seen = []
    routes = {"https://example.com/": (304, {"etag": "abc"}, b"")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx("https://example.com/feed", client=client, max_bytes=64)
    assert resp.status_code == 304


def test_guarded_response_raise_for_status_delegates_httpx():
    seen = []
    routes = {"https://example.com/": (404, {}, b"nope")}
    with httpx.Client(transport=_transport(routes, seen)) as client:
        resp = guarded_fetch_httpx("https://example.com/x", client=client, max_bytes=64)
    with pytest.raises(httpx.HTTPStatusError):
        resp.raise_for_status()


@pytest.mark.asyncio
async def test_httpx_async_variant_and_auth_suppression(monkeypatch):
    seen = []
    routes = {
        "https://a.example/": (302, {"location": "https://b.example/t"}, b""),
        "https://b.example/": (200, {}, b"ok"),
    }
    async with httpx.AsyncClient(transport=_transport(routes, seen)) as client:
        resp = await guarded_fetch_httpx_async(
            "https://a.example/s",
            client=client,
            max_bytes=64,
            auth=("user", "pw"),
        )
    assert resp.content == b"ok"
    assert "authorization" in seen[0].headers
    assert "authorization" not in seen[1].headers
```

- [ ] **Step 2: Run tests — verify the new ones fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`
Expected: ImportError for `GuardedResponse`/`guarded_fetch_httpx` (Task 1 tests still pass).

- [ ] **Step 3: Implement** (append to `tldw_chatbook/Utils/egress.py`)

```python
# ---------------------------------------------------------------------------
# Guarded transport helpers
# ---------------------------------------------------------------------------
import json as _json
from dataclasses import field
from urllib.parse import urljoin

_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization")


class EgressFetchError(Exception):
    """Guarded-fetch transport failure: size cap, hop cap, missing Location."""

    def __init__(self, message: str, url: str = ""):
        self.url = url
        super().__init__(f"{message}" + (f" [{url}]" if url else ""))


@dataclass
class GuardedResponse:
    """Stack-neutral capped-fetch result. Never raises on construction."""

    status_code: int
    headers: object
    content: bytes
    final_url: str
    _response: object = field(default=None, repr=False)

    @property
    def text(self) -> str:
        ctype = ""
        try:
            ctype = self.headers.get("content-type", "") or ""
        except Exception:
            pass
        charset = "utf-8"
        for part in ctype.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                charset = part.split("=", 1)[1].strip().strip('"') or "utf-8"
        try:
            return self.content.decode(charset, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self):
        if self._response is not None:
            return self._response.raise_for_status()
        if self.status_code >= 400:
            raise EgressFetchError(f"HTTP {self.status_code}", url=self.final_url)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _hop_headers(headers, same_origin: bool) -> dict:
    hop = {str(k): v for k, v in dict(headers or {}).items()}
    if not same_origin:
        for key in list(hop):
            if key.lower() in _STRIP_HEADERS:
                hop.pop(key)
    return hop


def guarded_fetch_httpx(
    url,
    *,
    client,
    max_bytes,
    trusted_origins=frozenset(),
    headers=None,
    params=None,
):
    """Capped GET via httpx.Client with per-hop egress re-validation."""
    first_host = _host_of(url)
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        check_url_or_raise(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, same_origin),
            params=params if hop == 0 else None,
        )
        response = client.send(request, stream=True, follow_redirects=False)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            for chunk in response.iter_bytes():
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
        finally:
            response.close()
    raise EgressFetchError("too many redirects", url=url)


async def guarded_fetch_httpx_async(
    url,
    *,
    client,
    max_bytes,
    trusted_origins=frozenset(),
    headers=None,
    params=None,
    auth=None,
):
    """Async capped GET via httpx.AsyncClient with per-hop re-validation.

    ``auth`` is applied on same-origin hops only (credential-stripping rule).
    """
    first_host = _host_of(url)
    current = url
    for hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        request = client.build_request(
            "GET",
            current,
            headers=_hop_headers(headers, same_origin),
            params=params if hop == 0 else None,
        )
        send_kwargs = {"stream": True, "follow_redirects": False}
        if auth is not None and same_origin:
            send_kwargs["auth"] = auth
        response = await client.send(request, **send_kwargs)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            async for chunk in response.aiter_bytes():
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status_code,
                headers=response.headers,
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
        finally:
            await response.aclose()
    raise EgressFetchError("too many redirects", url=url)
```

Note: httpx auto-sends `auth` on `client.send` only when passed; on cross-origin hops we omit it AND strip `Authorization` from explicit headers. Note also `response.close()`/`aclose()` before reading is prevented by reading inside the `try` — the `finally` close after a full read is a no-op, after an early raise it releases the connection.

- [ ] **Step 4: Run tests — verify all pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py
git commit -m "feat(security): guarded httpx fetch helpers with per-hop egress validation and byte caps"
```

---

### Task 3: Guarded requests helper (session-auth-safe, optional sink)

**Files:**
- Modify: `tldw_chatbook/Utils/egress.py` (append)
- Modify: `Tests/Utils/test_egress.py` (append)

**Interfaces:**
- Consumes: Task 1 policy + Task 2 `EgressFetchError`.
- Produces: `guarded_fetch_requests(url, *, session=None, max_bytes, trusted_origins=frozenset(), timeout=30.0, headers=None, sink=None) -> requests.Response` — returns the FINAL `requests.Response` with `._content` preloaded under the cap (callers keep `.json()`/`.text`/`.status_code`/`.headers`). When `sink` (a writable binary file object) is given, body bytes stream into it instead (`.content` stays `b""`); used by 500MB media/audio downloads so nothing big is buffered in RAM. Session auth (`session.auth`) applies on same-origin hops only. Propagates `requests` exception types (timeouts, connection errors) untouched.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Utils/test_egress.py`)

```python
# ---------------------------------------------------------------------------
# Guarded requests helper
# ---------------------------------------------------------------------------
import io

import requests as requests_lib
from requests.adapters import BaseAdapter
from requests.models import Response as RequestsResponse

from tldw_chatbook.Utils.egress import guarded_fetch_requests


class _FakeHTTPAdapter(BaseAdapter):
    """Serves canned responses; records every prepared request."""

    def __init__(self, routes):
        super().__init__()
        self.routes = routes  # {url_prefix: (status, headers, body)}
        self.seen = []

    def send(self, request, **kwargs):
        self.seen.append(request)
        for prefix, (status, headers, body) in self.routes.items():
            if request.url.startswith(prefix):
                resp = RequestsResponse()
                resp.status_code = status
                resp.headers.update(headers)
                resp.raw = io.BytesIO(body)
                resp.url = request.url
                resp.request = request
                return resp
        resp = RequestsResponse()
        resp.status_code = 404
        resp.raw = io.BytesIO(b"")
        resp.url = request.url
        return resp

    def close(self):
        pass


def _session_with(routes):
    sess = requests_lib.Session()
    adapter = _FakeHTTPAdapter(routes)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess, adapter


def test_requests_basic_fetch_preloads_content():
    sess, adapter = _session_with(
        {"https://example.com/": (200, {"content-type": "text/html"}, b"<p>hi</p>")}
    )
    resp = guarded_fetch_requests(
        "https://example.com/p", session=sess, max_bytes=1024
    )
    assert resp.status_code == 200
    assert resp.content == b"<p>hi</p>"
    assert resp.text == "<p>hi</p>"


def test_requests_redirect_to_internal_blocked(monkeypatch):
    def fake_resolve(host):
        return ["10.0.0.5"] if host == "internal.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    sess, adapter = _session_with(
        {"https://example.com/": (302, {"location": "http://internal.example/"}, b"")}
    )
    with pytest.raises(EgressBlockedError):
        guarded_fetch_requests("https://example.com/r", session=sess, max_bytes=64)
    assert all("internal.example" not in r.url for r in adapter.seen)


def test_requests_session_auth_suppressed_cross_origin():
    sess, adapter = _session_with(
        {
            "https://a.example/": (302, {"location": "https://b.example/n"}, b""),
            "https://b.example/": (200, {}, b"fin"),
        }
    )
    sess.auth = ("user", "pw")
    resp = guarded_fetch_requests("https://a.example/s", session=sess, max_bytes=64)
    assert resp.content == b"fin"
    assert "Authorization" in adapter.seen[0].headers
    assert "Authorization" not in adapter.seen[1].headers


def test_requests_byte_cap():
    sess, _ = _session_with({"https://example.com/": (200, {}, b"y" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests("https://example.com/big", session=sess, max_bytes=1024)


def test_requests_sink_streams_without_buffering():
    body = b"z" * 3000
    sess, _ = _session_with({"https://example.com/": (200, {}, body)})
    sink = io.BytesIO()
    resp = guarded_fetch_requests(
        "https://example.com/file", session=sess, max_bytes=4096, sink=sink
    )
    assert sink.getvalue() == body
    assert resp.content == b""
    assert resp.status_code == 200


def test_requests_sink_cap_still_enforced():
    sess, _ = _session_with({"https://example.com/": (200, {}, b"w" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests(
            "https://example.com/file", session=sess, max_bytes=1024, sink=io.BytesIO()
        )
```

- [ ] **Step 2: Run — verify new tests fail** (ImportError for `guarded_fetch_requests`).

- [ ] **Step 3: Implement** (append to `tldw_chatbook/Utils/egress.py`)

```python
def guarded_fetch_requests(
    url,
    *,
    session=None,
    max_bytes,
    trusted_origins=frozenset(),
    timeout=30.0,
    headers=None,
    sink=None,
):
    """Capped GET via requests with per-hop egress re-validation.

    Returns the final ``requests.Response`` with ``._content`` preloaded
    (unless ``sink`` is given, in which case bytes stream to ``sink`` and
    ``.content`` is empty). ``session.auth`` is suppressed on cross-origin
    hops (credential-stripping rule).
    """
    import requests

    sess = session or requests.Session()
    owns_session = session is None
    try:
        first_host = _host_of(url)
        current = url
        for _hop in range(MAX_REDIRECT_HOPS + 1):
            check_url_or_raise(current, trusted_origins=trusted_origins)
            same_origin = _host_of(current) == first_host
            prepared = sess.prepare_request(
                requests.Request(
                    "GET", current, headers=_hop_headers(headers, same_origin)
                )
            )
            if not same_origin:
                # prepare_request applies session.auth/cookies into headers;
                # a cross-origin hop must not carry them.
                for key in ("Authorization", "Cookie", "Proxy-Authorization"):
                    prepared.headers.pop(key, None)
            response = sess.send(
                prepared, stream=True, timeout=timeout, allow_redirects=False
            )
            if response.is_redirect:
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray() if sink is None else None
            received = 0
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise EgressFetchError(
                            f"response exceeds {max_bytes} bytes", url=current
                        )
                    if sink is None:
                        collected += chunk
                    else:
                        sink.write(chunk)
            finally:
                response.close()
            response._content = bytes(collected) if collected is not None else b""
            response._content_consumed = True
            return response
        raise EgressFetchError("too many redirects", url=url)
    finally:
        if owns_session:
            sess.close()
```

- [ ] **Step 4: Run — verify all pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py
git commit -m "feat(security): guarded requests fetch helper (session-auth-safe hops, sink streaming)"
```

---

### Task 4: Guarded aiohttp helper + Playwright navigation-chain validation

**Files:**
- Modify: `tldw_chatbook/Utils/egress.py` (append)
- Modify: `Tests/Utils/test_egress.py` (append)

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces:
  - `async guarded_fetch_aiohttp(url, *, session, max_bytes, trusted_origins=frozenset(), headers=None, timeout=None) -> GuardedResponse` (`_response` = the aiohttp response, so `raise_for_status()` raises `aiohttp.ClientResponseError` for existing callers).
  - `collect_navigation_chain(response) -> list[str]` — Playwright `Response` → every URL in the redirect chain (via `request.redirected_from` walk) plus the final URL, oldest first.
  - `async validate_navigation_chain_async(urls, *, trusted_origins=frozenset())` and sync `validate_navigation_chain(urls, *, trusted_origins=frozenset())` — raise `EgressBlockedError` on the first violating URL.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Utils/test_egress.py`)

```python
# ---------------------------------------------------------------------------
# Guarded aiohttp helper + Playwright chain validation
# ---------------------------------------------------------------------------
from tldw_chatbook.Utils.egress import (
    collect_navigation_chain,
    guarded_fetch_aiohttp,
    validate_navigation_chain,
    validate_navigation_chain_async,
)


class _FakeAiohttpContent:
    def __init__(self, body):
        self._body = body

    async def iter_chunked(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]


class _FakeAiohttpResponse:
    def __init__(self, status, headers, body, url):
        self.status = status
        self.headers = headers
        self.content = _FakeAiohttpContent(body)
        self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _FakeAiohttpSession:
    def __init__(self, routes):
        self.routes = routes
        self.seen = []

    def get(self, url, **kwargs):
        self.seen.append((url, kwargs))
        for prefix, (status, headers, body) in self.routes.items():
            if url.startswith(prefix):
                return _FakeAiohttpResponse(status, headers, body, url)
        return _FakeAiohttpResponse(404, {}, b"", url)


@pytest.mark.asyncio
async def test_aiohttp_fetch_and_redirect_revalidation(monkeypatch):
    def fake_resolve_ok(host):
        return ["10.9.9.9"] if host == "intra.example" else ["93.184.216.34"]

    async def fake_async(host):
        return fake_resolve_ok(host)

    monkeypatch.setattr(egress, "_resolve_async", fake_async)
    session = _FakeAiohttpSession(
        {"https://example.com/": (302, {"Location": "http://intra.example/"}, b"")}
    )
    with pytest.raises(EgressBlockedError):
        await guarded_fetch_aiohttp(
            "https://example.com/r", session=session, max_bytes=64
        )
    assert all("intra.example" not in u for u, _ in session.seen)


@pytest.mark.asyncio
async def test_aiohttp_basic_fetch_capped():
    session = _FakeAiohttpSession(
        {"https://example.com/": (200, {"Content-Type": "text/html"}, b"page")}
    )
    resp = await guarded_fetch_aiohttp(
        "https://example.com/x", session=session, max_bytes=64
    )
    assert resp.status_code == 200 and resp.content == b"page"
    # allow_redirects must be forced off (manual loop owns redirects)
    assert session.seen[0][1].get("allow_redirects") is False

    big = _FakeAiohttpSession({"https://example.com/": (200, {}, b"q" * 4096)})
    with pytest.raises(EgressFetchError, match="exceeds"):
        await guarded_fetch_aiohttp("https://example.com/b", session=big, max_bytes=1024)


class _FakePlaywrightRequest:
    def __init__(self, url, redirected_from=None):
        self.url = url
        self.redirected_from = redirected_from


class _FakePlaywrightResponse:
    def __init__(self, request):
        self.request = request


def test_collect_navigation_chain_walks_redirects():
    first = _FakePlaywrightRequest("https://a.example/start")
    second = _FakePlaywrightRequest("https://a.example/hop", redirected_from=first)
    final = _FakePlaywrightRequest("https://b.example/end", redirected_from=second)
    chain = collect_navigation_chain(_FakePlaywrightResponse(final))
    assert chain == [
        "https://a.example/start",
        "https://a.example/hop",
        "https://b.example/end",
    ]


def test_validate_navigation_chain_blocks_internal_hop(monkeypatch):
    def fake_resolve(host):
        return ["169.254.169.254"] if host == "meta.example" else ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve", fake_resolve)
    with pytest.raises(EgressBlockedError):
        validate_navigation_chain(
            ["https://ok.example/", "http://meta.example/x"]
        )
    validate_navigation_chain(["https://ok.example/"])  # no raise


@pytest.mark.asyncio
async def test_validate_navigation_chain_async(monkeypatch):
    async def fake_async(host):
        return ["192.168.7.7"]

    monkeypatch.setattr(egress, "_resolve_async", fake_async)
    with pytest.raises(EgressBlockedError):
        await validate_navigation_chain_async(["http://internal.example/"])
    # trusted origin passes
    await validate_navigation_chain_async(
        ["http://internal.example/"],
        trusted_origins=frozenset({"internal.example"}),
    )
```

- [ ] **Step 2: Run — verify new tests fail** (ImportError).

- [ ] **Step 3: Implement** (append to `tldw_chatbook/Utils/egress.py`)

```python
async def guarded_fetch_aiohttp(
    url,
    *,
    session,
    max_bytes,
    trusted_origins=frozenset(),
    headers=None,
    timeout=None,
):
    """Capped GET via aiohttp.ClientSession with per-hop re-validation."""
    first_host = _host_of(url)
    current = url
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(current, trusted_origins=trusted_origins)
        same_origin = _host_of(current) == first_host
        kwargs = {
            "allow_redirects": False,
            "headers": _hop_headers(headers, same_origin),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        async with session.get(current, **kwargs) as response:
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    raise EgressFetchError("redirect without Location", url=current)
                current = urljoin(current, location)
                continue
            collected = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                collected += chunk
                if len(collected) > max_bytes:
                    raise EgressFetchError(
                        f"response exceeds {max_bytes} bytes", url=current
                    )
            return GuardedResponse(
                status_code=response.status,
                headers=dict(response.headers),
                content=bytes(collected),
                final_url=str(response.url),
                _response=response,
            )
    raise EgressFetchError("too many redirects", url=url)


def collect_navigation_chain(response) -> list:
    """Playwright ``Response`` -> every URL in its redirect chain, oldest first.

    Playwright route handlers intercept only the INITIAL request of a
    navigation; server redirect hops are followed by the browser without
    re-invoking the route. Post-navigation validation of this chain is
    therefore the enforcement point (the pre-``goto`` check blocks bad
    initial targets outright).
    """
    urls = []
    request = getattr(response, "request", None) if response is not None else None
    while request is not None:
        urls.append(request.url)
        request = getattr(request, "redirected_from", None)
    return list(reversed(urls))


def validate_navigation_chain(urls, *, trusted_origins=frozenset()) -> None:
    for u in urls:
        check_url_or_raise(u, trusted_origins=trusted_origins)


async def validate_navigation_chain_async(urls, *, trusted_origins=frozenset()) -> None:
    for u in urls:
        await check_url_or_raise_async(u, trusted_origins=trusted_origins)
```

- [ ] **Step 4: Run — verify all pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_egress.py -q`

- [ ] **Step 5: Empirical Playwright check (report-only, no code change)**

Run this snippet with the venv python to record whether `page.route` fires on redirect hops in Playwright 1.58 (implementation is identical either way — post-nav validation stays the net):

```python
# scratch check — do NOT commit. Run:
#   /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/pw_check.py
import http.server, threading, functools
from playwright.sync_api import sync_playwright

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/start":
            self.send_response(302); self.send_header("Location", "/end"); self.end_headers()
        else:
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(b"<html>end</html>")
    def log_message(self, *a): pass

srv = http.server.HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
seen = []
with sync_playwright() as p:
    b = p.chromium.launch(); page = b.new_page()
    page.route("**/*", lambda route: (seen.append(route.request.url), route.continue_()))
    resp = page.goto(f"http://127.0.0.1:{port}/start")
    print("route saw:", seen)
    print("final url:", resp.url)
    req = resp.request; chain = []
    while req: chain.append(req.url); req = req.redirected_from
    print("chain (reversed):", list(reversed(chain)))
    b.close()
srv.shutdown()
```

If Chromium is not installed (`playwright install` never run), record "browser unavailable — check skipped" in your report and move on.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Utils/egress.py Tests/Utils/test_egress.py
git commit -m "feat(security): guarded aiohttp helper + Playwright navigation-chain validation"
```

---

### Task 5: Wire `Article_Extractor_Lib` (requests sites + Playwright paths)

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py`
- Create: `Tests/Web_Scraping/test_web_fetch_wiring.py`

**Interfaces:**
- Consumes: `guarded_fetch_requests`, `check_url_or_raise_async`, `collect_navigation_chain`, `validate_navigation_chain_async`, `EgressBlockedError`, `EgressFetchError`, `MAX_FETCH_BYTES_PAGE`, `MAX_FETCH_BYTES_SITEMAP` from `tldw_chatbook.Utils.egress`.
- Produces (later tasks rely on): `scrape_article(url, custom_cookies=None, trusted_origins=frozenset())` and `scrape_article_async(context, url, trusted_origins=frozenset())` — new trailing keyword-only param, default keeps every existing caller working (fail-closed content-derived posture).

**Boundary classification (verified callers on this branch):** `get_page_title`, `scrape_from_sitemap`, `collect_internal_links`, `scrape_entire_site`, `recursive_scrape` have NO callers outside this module — they ARE user boundaries and seed trust from their own input. `scrape_article` has external callers (`WebSearch_APIs` = content-derived → default; `website_monitor`/`generic_scraper` = user-configured → they pass origins in Tasks 8/9).

- [ ] **Step 1: Add the import** near the other `tldw_chatbook` imports at the top of `Article_Extractor_Lib.py`:

```python
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    EgressFetchError,
    MAX_FETCH_BYTES_PAGE,
    MAX_FETCH_BYTES_SITEMAP,
    check_url_or_raise_async,
    collect_navigation_chain,
    guarded_fetch_requests,
    validate_navigation_chain_async,
)
```

- [ ] **Step 2: `get_page_title`** — change signature to `def get_page_title(url: str, *, trusted_origins=frozenset()) -> str:` and replace

```python
        response = requests.get(url, timeout=10)  # Add timeout
```

with

```python
        response = guarded_fetch_requests(
            url,
            max_bytes=MAX_FETCH_BYTES_PAGE,
            trusted_origins=trusted_origins,
            timeout=10,
        )
```

and add, alongside the existing `except requests.Timeout:` handler (BEFORE the broad handlers):

```python
    except (EgressBlockedError, EgressFetchError) as e:
        logging.warning(f"Blocked/oversize page-title fetch for {url}: {e}")
        return "Untitled (Blocked URL)"
```

- [ ] **Step 3: `scrape_from_sitemap`** (user boundary — seeds trust) — replace the body's fetch portion:

```python
def scrape_from_sitemap(sitemap_url: str) -> list:
    """Scrape articles from a sitemap URL."""
    origins = frozenset({(urlparse(sitemap_url).hostname or "").lower()})
    try:
        response = guarded_fetch_requests(
            sitemap_url,
            max_bytes=MAX_FETCH_BYTES_SITEMAP,
            trusted_origins=origins,
            timeout=30,
        )
        response.raise_for_status()
        root = xET.fromstring(response.content)

        return [
            article
            for url in root.findall(
                ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
            )
            if (article := scrape_article(url.text, trusted_origins=origins))
        ]
    except (EgressBlockedError, EgressFetchError) as e:
        logging.error(f"Sitemap fetch blocked or too large: {e}")
        return []
    except requests.RequestException as e:
        logging.error(f"Error fetching sitemap: {e}")
        return []
```

(Sitemap-discovered article URLs inherit the sitemap host as trusted origin — same-origin items on an intranet sitemap keep working; cross-host items must be public. NOTE: `scrape_article` here is the module's async-wrapped sync entry — keep the existing call shape, only add the `trusted_origins=origins` argument.)

- [ ] **Step 4: `collect_internal_links`** (user boundary) — seed origins and guard the loop fetch:

At the top of the function add:

```python
    origins = frozenset({(urlparse(base_url).hostname or "").lower()})
```

Replace `response = requests.get(current_url)` with:

```python
            response = guarded_fetch_requests(
                current_url,
                max_bytes=MAX_FETCH_BYTES_PAGE,
                trusted_origins=origins,
                timeout=30,
            )
```

And extend the loop's `except requests.RequestException as e:` with a preceding:

```python
        except (EgressBlockedError, EgressFetchError) as e:
            logging.warning(f"Skipping blocked/oversize link {current_url}: {e}")
            continue
```

- [ ] **Step 5: Playwright paths.** Thread `trusted_origins` and validate navigations:

a) `scrape_article` / its nested `fetch_html`: add `trusted_origins=frozenset()` keyword-only param to `scrape_article` and pass it into `fetch_html`. In `fetch_html`, immediately BEFORE `await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)` insert:

```python
                    await check_url_or_raise_async(
                        url, trusted_origins=trusted_origins
                    )
```

Capture the goto response and validate the chain — change the goto line to:

```python
                    nav_response = await page.goto(
                        url, wait_until="domcontentloaded", timeout=timeout_ms
                    )
```

and AFTER the `content = await page.content()` line insert:

```python
                    await validate_navigation_chain_async(
                        collect_navigation_chain(nav_response),
                        trusted_origins=trusted_origins,
                    )
```

Wrap-level containment: `fetch_html`'s caller-visible failure mode is returning `""` — add to the function (at the same level as its existing exception handling; if it has none, wrap the browser block):

```python
        except EgressBlockedError as e:
            logging.warning(f"Navigation blocked by egress policy: {e}")
            return ""
```

b) `scrape_article_async(context, url)` → `async def scrape_article_async(context, url: str, trusted_origins=frozenset()) -> Dict[str, Any]:`; before `await page.goto(url)` insert `await check_url_or_raise_async(url, trusted_origins=trusted_origins)`; capture `nav_response = await page.goto(url)` and after `content = await page.content()` insert the same `validate_navigation_chain_async(...)` call. The existing `except Exception` there already returns `{"extraction_successful": False, ...}` — containment for free.

c) `recursive_scrape`: it computes `base_url`; near the top add `origins = frozenset({(urlparse(base_url).hostname or "").lower()})`; pass `trusted_origins=origins` to its `scrape_article_async(context, current_url)` call; for the inner link-discovery `await page.goto(current_url)` add the pre-check line before it and change to `nav_response = await page.goto(current_url)` + chain validation after `wait_for_load_state`. Its enclosing `except Exception` logs and continues — containment holds.

- [ ] **Step 6: Write wiring tests**

Create `Tests/Web_Scraping/test_web_fetch_wiring.py`:

```python
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
```

- [ ] **Step 7: Run tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_web_fetch_wiring.py Tests/Web_Scraping/test_article_extractor.py -q`
Expected: new tests pass; `test_article_extractor.py` stays green (it must not regress — if it fails, your edit broke an existing contract; fix the edit, not the test).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Web_Scraping/Article_Extractor_Lib.py Tests/Web_Scraping/test_web_fetch_wiring.py
git commit -m "feat(security): wire Article_Extractor_Lib through egress guard (TASK-328/329)"
```

---

### Task 6: Wire `Article_Scraper` (crawler + Scraper) and Confluence

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/Article_Scraper/config.py` (ScraperConfig)
- Modify: `tldw_chatbook/Web_Scraping/Article_Scraper/scraper.py` (`_fetch_html`)
- Modify: `tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py` (`crawl_site`, `get_urls_from_sitemap`)
- Modify: `tldw_chatbook/Web_Scraping/Confluence/confluence_auth.py` (`make_request`)
- Modify: `tldw_chatbook/Web_Scraping/Confluence/confluence_scraper.py` (`__init__`, `_extract_page_id_from_url`)
- Modify: `Tests/Web_Scraping/test_web_fetch_wiring.py` (append)

**Interfaces:**
- Consumes: Task 1-4 exports; `ScraperConfig` dataclass.
- Produces: `ScraperConfig.trusted_origins: frozenset = frozenset()` — the per-instance trust set `Scraper._fetch_html` enforces; `ConfluenceScraper.__init__` adds its `base_url` host to it.

- [ ] **Step 1: `ScraperConfig`** — add a field after the existing attributes (it's a `@dataclass`; `frozenset()` is immutable so a plain default is safe):

```python
    trusted_origins: frozenset = frozenset()
```

Document it in the class docstring attribute list: `trusted_origins: Hostnames (lowercase) allowed to resolve to private IPs — seed from user-configured origins only.`

- [ ] **Step 2: `scraper.py _fetch_html`** — import at top of file:

```python
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    check_url_or_raise_async,
    collect_navigation_chain,
    validate_navigation_chain_async,
)
```

Inside the retry loop, immediately BEFORE `await page.goto(...)`:

```python
                await check_url_or_raise_async(
                    url, trusted_origins=self.config.trusted_origins
                )
```

Change `await page.goto(` to `nav_response = await page.goto(` and AFTER `content = await page.content()` (before `await page.close()`):

```python
                await validate_navigation_chain_async(
                    collect_navigation_chain(nav_response),
                    trusted_origins=self.config.trusted_origins,
                )
```

Containment: wrap so a block aborts retries and returns the function's existing failure value `""` — add around/inside the attempt handling:

```python
            except EgressBlockedError as e:
                logging.warning(f"Scrape navigation blocked by egress policy: {e}")
                await page.close()
                return ""
```

(Place it BEFORE the broader `except` that triggers retries, so blocked URLs don't burn retry attempts.)

- [ ] **Step 3: `crawler.py`** — import at top:

```python
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    EgressFetchError,
    MAX_FETCH_BYTES_PAGE,
    MAX_FETCH_BYTES_SITEMAP,
    guarded_fetch_aiohttp,
)
```

a) `crawl_site`: near the top (where `base_url` is in scope) add `origins = frozenset({(urlparse(base_url).hostname or "").lower()})` (the module already imports `urlparse`; add it if not). Replace:

```python
                async with session.get(current_url, timeout=10) as response:
                    if response.status != 200:
```

with:

```python
                    response = await guarded_fetch_aiohttp(
                        current_url,
                        session=session,
                        max_bytes=MAX_FETCH_BYTES_PAGE,
                        trusted_origins=origins,
                        timeout=10,
                    )
                    if response.status_code != 200:
```

then inside that block: `response.headers.get("Content-Type", "")` stays; `html = await response.text()` becomes `html = response.text` (GuardedResponse property, not a coroutine); un-indent the block one level (no more `async with`). Add to the surrounding `try`'s handlers, before the generic one:

```python
            except (EgressBlockedError, EgressFetchError) as e:
                log_counter("crawler_url_filtered", labels={"reason": "egress_blocked"})
                logging.warning(f"Skipping blocked/oversize URL {current_url}: {e}")
                continue
```

b) `get_urls_from_sitemap`: replace the `async with session.get(sitemap_url, timeout=30) as response:` block:

```python
        async with aiohttp.ClientSession() as session:
            fetch_start = time.time()
            origins = frozenset({(urlparse(sitemap_url).hostname or "").lower()})
            response = await guarded_fetch_aiohttp(
                sitemap_url,
                session=session,
                max_bytes=MAX_FETCH_BYTES_SITEMAP,
                trusted_origins=origins,
                timeout=30,
            )
            response.raise_for_status()
            xml_content = response.text
```

(keep the existing log_histogram/log_counter lines, changing `response.status` → `response.status_code`). The function's existing broad `except` handles `EgressBlockedError`/`EgressFetchError` (they log + return `[]`) — verify that's true when editing; if the except clause is narrower than `Exception`, add the two error types to it.

- [ ] **Step 4: Confluence.** In `confluence_auth.py`, import `from tldw_chatbook.Utils.egress import MAX_FETCH_BYTES_PAGE, check_url_or_raise, guarded_fetch_requests` and rewrite the request portion of `make_request`:

```python
        url = f"{self.base_url}{endpoint}"

        # Set default headers
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].setdefault("Accept", "application/json")
        kwargs["headers"].setdefault("Content-Type", "application/json")
        kwargs.setdefault("timeout", 30)

        trusted = frozenset({(urlparse(self.base_url).hostname or "").lower()})
        if method.upper() == "GET" and set(kwargs) <= {"headers", "timeout"}:
            response = guarded_fetch_requests(
                url,
                session=self.session,
                max_bytes=MAX_FETCH_BYTES_PAGE,
                trusted_origins=trusted,
                timeout=kwargs["timeout"],
                headers=kwargs["headers"],
            )
        else:
            # Non-GET / param-carrying calls: pre-check + timeout (no manual
            # redirect loop; the Confluence API does not redirect these).
            check_url_or_raise(url, trusted_origins=trusted)
            response = self.session.request(method, url, **kwargs)
```

(add `from urllib.parse import urlparse` if the module lacks it). Callers keep receiving a real `requests.Response` either way. Note `make_request` calls that pass `params=` take the pre-checked branch — that is deliberate simplicity, not an oversight: the base URL host is user-configured (trusted), and same-host API calls don't traverse redirect chains.

In `confluence_scraper.py`:
- `__init__`: after `self.base_url = auth.base_url`, add (ScraperConfig is a mutable dataclass, direct assignment works):

```python
        base_host = (urlparse(self.base_url).hostname or "").lower()
        self.config.trusted_origins = frozenset(
            set(self.config.trusted_origins) | {base_host}
        )
```

(add the `urlparse` import if missing).
- `_extract_page_id_from_url`: replace `response = self.auth.session.get(url, allow_redirects=True)` with:

```python
                response = guarded_fetch_requests(
                    url,
                    session=self.auth.session,
                    max_bytes=MAX_FETCH_BYTES_PAGE,
                    trusted_origins=self.config.trusted_origins,
                    timeout=30,
                )
```

with import `from tldw_chatbook.Utils.egress import MAX_FETCH_BYTES_PAGE, guarded_fetch_requests`. The enclosing `except Exception` already contains failures.

- [ ] **Step 5: Append wiring tests** to `Tests/Web_Scraping/test_web_fetch_wiring.py`:

```python
def test_scraper_config_has_fail_closed_trusted_origins_default():
    from tldw_chatbook.Web_Scraping.Article_Scraper.config import ScraperConfig

    assert ScraperConfig().trusted_origins == frozenset()


def test_confluence_make_request_gets_timeout_and_guard(monkeypatch):
    from tldw_chatbook.Web_Scraping.Confluence import confluence_auth as ca

    calls = {}

    def fake_guarded(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)

        class R:
            status_code = 200

        return R()

    monkeypatch.setattr(ca, "guarded_fetch_requests", fake_guarded)
    auth = ca.ConfluenceAuth("https://wiki.corp.example")
    auth._auth_configured = True
    auth.make_request("GET", "/rest/api/content/123")
    assert calls["url"] == "https://wiki.corp.example/rest/api/content/123"
    assert calls["timeout"] == 30
    assert calls["trusted_origins"] == frozenset({"wiki.corp.example"})
```

(Adjust the `ConfluenceAuth` constructor call to its real signature — check the class `__init__`; it takes `base_url` and an auth-method enum with a default.)

- [ ] **Step 6: Run**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/ -q`
Expected: new tests pass; existing `Tests/Web_Scraping/Confluence/` and scraper tests stay green.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Web_Scraping/Article_Scraper/config.py tldw_chatbook/Web_Scraping/Article_Scraper/scraper.py tldw_chatbook/Web_Scraping/Article_Scraper/crawler.py tldw_chatbook/Web_Scraping/Confluence/confluence_auth.py tldw_chatbook/Web_Scraping/Confluence/confluence_scraper.py Tests/Web_Scraping/test_web_fetch_wiring.py
git commit -m "feat(security): wire Article_Scraper and Confluence through egress guard"
```

---

### Task 7: Wire `web_article_ingestion` (swap hand-rolled loop for the guarded helper)

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/web_article_ingestion.py`
- Modify: `Tests/Web_Scraping/test_web_article_ingestion.py` (existing suite — adapt fakes if they mock the raw httpx client; behavior contract itself is unchanged)

**Interfaces:**
- Consumes: `guarded_fetch_httpx`, `EgressBlockedError`, `EgressFetchError`, `MAX_FETCH_BYTES_PAGE`.
- Contract that MUST be preserved (existing docstring): retryable transport failures propagate as `httpx.HTTPError`; permanent failures raise `PermanentIngestError`; non-HTML content-type → permanent; oversize → permanent; returned `url` = canonical post-redirect URL.

- [ ] **Step 1: Read the current fetch block** (`extract_article_for_ingest`, the `with httpx.Client(...)` / `client.stream(...)` section) and replace it with:

```python
    import httpx

    from urllib.parse import urlparse as _urlparse

    from tldw_chatbook.Utils.egress import (
        EgressBlockedError,
        EgressFetchError,
        MAX_FETCH_BYTES_PAGE,
        guarded_fetch_httpx,
    )

    origins = frozenset({(_urlparse(url).hostname or "").lower()})
    try:
        with httpx.Client(
            timeout=30.0,
            headers={"User-Agent": _UA, "Accept": "text/html,*/*"},
        ) as client:
            resp = guarded_fetch_httpx(
                url,
                client=client,
                max_bytes=min(_MAX_BYTES, MAX_FETCH_BYTES_PAGE),
                trusted_origins=origins,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRYABLE_STATUS:
                    raise  # retryable
                raise PermanentIngestError(
                    f"URL fetch failed ({status}) for {url}"
                ) from exc
            ctype = (
                resp.headers.get("content-type", "").split(";")[0].strip().lower()
            )
            if ctype and "html" not in ctype and "xml" not in ctype:
                raise PermanentIngestError(
                    f"URL is not a web page (content-type {ctype!r}): {url}"
                )
            final_url = resp.final_url
            body = resp.text
    except EgressBlockedError as exc:
        raise PermanentIngestError(
            f"URL blocked by egress policy (SSRF guard): {exc}"
        ) from exc
    except EgressFetchError as exc:
        raise PermanentIngestError(
            f"URL response too large or redirect chain invalid: {exc}"
        ) from exc
    except httpx.InvalidURL as exc:
        raise PermanentIngestError(f"Invalid URL: {url}") from exc
    except httpx.HTTPError as exc:
```

…keeping the existing `except httpx.HTTPError` classification block and everything after (`trafilatura` extraction etc.) untouched, with `body`/`final_url` now sourced as above. Remove the now-dead `follow_redirects=True` client arg (the helper owns redirects), the manual `collected` loop, and the `encoding` juggling (`resp.text` handles charset). Note `guarded_fetch_httpx` raises `httpx`-native transport errors (timeouts, connect errors) from `client.send` — the retryable path is intact.

- [ ] **Step 2: Run the existing suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_web_article_ingestion.py -q`
Expected: if the suite mocks `httpx.Client`/transport, it keeps passing unchanged (the helper drives the real httpx request machinery, so `MockTransport`-style fakes still see requests). If a test asserts on `client.stream` specifically, update THAT mock to a `httpx.MockTransport` handler equivalent — behavior assertions (retryable/permanent/oversize/content-type) must not change. Add one new case:

```python
def test_blocked_url_is_permanent(monkeypatch):
    from tldw_chatbook.Local_Ingestion import web_article_ingestion as wai
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(egress, "_resolve", lambda host: ["169.254.169.254"])
    with pytest.raises(wai.PermanentIngestError, match="egress"):
        wai.extract_article_for_ingest("http://metadata-ish.example/x", {})
```

(match the module's existing test style/imports; note `trusted_origins={host}` does NOT save it — metadata beats trust.)

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/Local_Ingestion/web_article_ingestion.py Tests/Web_Scraping/test_web_article_ingestion.py
git commit -m "feat(security): web_article_ingestion via guarded httpx helper (per-hop egress checks)"
```

---

### Task 8: Wire Subscriptions monitors + `security.py` delegation + ssl_verify warning

**Files:**
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py` (`FeedMonitor._fetch_and_parse_feed`, `URLMonitor._fetch_url_content`)
- Modify: `tldw_chatbook/Subscriptions/website_monitor.py` (`_fetch_url_content` + the two `scrape_article(...)` calls)
- Modify: `tldw_chatbook/Subscriptions/security.py` (`SecurityValidator.validate_feed_url`, `SSRFProtector.is_ip_allowed`, `sanitize_item`)
- Modify: `tldw_chatbook/Utils/egress.py` (append `warn_insecure_ssl`)
- Create: `Tests/Subscriptions/test_subscription_egress_wiring.py`

**Interfaces:**
- Consumes: `guarded_fetch_httpx_async`, `evaluate_url_policy`, `_classify_ip`, `EgressBlockedError`, `EgressFetchError`, `MAX_FETCH_BYTES_PAGE`.
- Produces:
  - `egress.warn_insecure_ssl(host: str)` — loguru WARNING once per host per process (module-level `_INSECURE_SSL_WARNED: set`), `log_counter("web_insecure_ssl_fetch")` every call.
  - `SecurityValidator.validate_feed_url(url, trusted_origins=frozenset()) -> str` — same normalize-and-return contract; raises `SSRFError` on policy block. **Deliberate behavior change:** unresolvable hostnames now BLOCK (`dns_failure`, fail-closed) where the old code warned and passed — update any existing test pinning the old behavior, listing it in your report.
  - `SecurityValidator.sanitize_item(item, trusted_origins=frozenset())` — passes origins through to the URL validation.

- [ ] **Step 1: `egress.warn_insecure_ssl`** (append to `egress.py`):

```python
_INSECURE_SSL_WARNED: set = set()


def warn_insecure_ssl(host: str) -> None:
    """Record a TLS-verification-disabled fetch (warn once per host)."""
    log_counter("web_insecure_ssl_fetch")
    h = (host or "").lower()
    if h not in _INSECURE_SSL_WARNED:
        _INSECURE_SSL_WARNED.add(h)
        logger.warning(
            f"TLS certificate verification DISABLED for fetches to {h} "
            "(subscription ssl_verify=0) — traffic to this host can be "
            "intercepted; only use for trusted self-signed intranet services."
        )
```

- [ ] **Step 2: `FeedMonitor._fetch_and_parse_feed`.** Import at top of `monitoring_engine.py`:

```python
from tldw_chatbook.Utils.egress import (
    EgressBlockedError,
    EgressFetchError,
    MAX_FETCH_BYTES_PAGE,
    guarded_fetch_httpx_async,
    warn_insecure_ssl,
)
```

Replace the client block:

```python
        # Fetch feed
        feed_host = (urlparse(feed_url).hostname or "").lower()
        if subscription.get("ssl_verify", True) == 0:
            warn_insecure_ssl(feed_host)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            verify=subscription.get("ssl_verify", True) != 0,
        ) as client:
            response = await guarded_fetch_httpx_async(
                feed_url,
                client=client,
                max_bytes=MAX_FETCH_BYTES_PAGE,
                trusted_origins=frozenset({feed_host}),
                headers=headers,
                auth=auth,
            )
```

(`follow_redirects=True` is REMOVED from the client — the helper owns redirects; add `from urllib.parse import urlparse` if the module lacks it.) The status handling below (`304`/`401`/`429`/`raise_for_status`/`response.text`) works unchanged on `GuardedResponse` (`raise_for_status` delegates to the real httpx response, preserving `httpx.HTTPStatusError` for callers). Find `_fetch_and_parse_feed`'s caller (`check_feed`) and add to its error handling, mapped like network errors (same category the circuit breaker uses):

```python
        except (EgressBlockedError, EgressFetchError) as e:
            logger.warning(f"Feed fetch blocked by egress policy: {e}")
            raise MonitoringError(f"Feed URL blocked or oversize: {e}") from e
```

(Use the module's actual failure-exception type — read `check_feed`'s except clauses and mirror how `httpx.HTTPError` is wrapped; the requirement is containment into the existing failure path, not a new exception surface.)

- [ ] **Step 3: `URLMonitor._fetch_url_content`** — same transformation: compute `url_host`, `warn_insecure_ssl` when `ssl_verify==0`, client without `follow_redirects`, `guarded_fetch_httpx_async(url, client=client, max_bytes=MAX_FETCH_BYTES_PAGE, trusted_origins=frozenset({url_host}), headers=headers)`, then `response.raise_for_status()` and `response.text` as today. Containment mirrors Step 2 in ITS caller.

- [ ] **Step 4: `website_monitor.py`** — the httpx fallback in `_fetch_url_content`:

```python
            from tldw_chatbook.Utils.egress import (
                MAX_FETCH_BYTES_PAGE,
                guarded_fetch_httpx_async,
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await guarded_fetch_httpx_async(
                    url,
                    client=client,
                    max_bytes=MAX_FETCH_BYTES_PAGE,
                    trusted_origins=frozenset({(urlparse(url).hostname or "").lower()}),
                )
                response.raise_for_status()
                return response.text
```

The enclosing `except Exception` already returns `None` (containment). Also pass origins to the two `scrape_article` calls (`:292` and `:317` — the monitored URL is user-configured):

```python
                article_data = await scrape_article(
                    subscription["source"],
                    trusted_origins=frozenset(
                        {(urlparse(subscription["source"]).hostname or "").lower()}
                    ),
                )
```

(and equivalently for the `url` variant at the second site; add the `urlparse` import if missing).

- [ ] **Step 5: `security.py` delegation.** Replace the body of `validate_feed_url` from the metadata-endpoint check down to (but excluding) the "Normalize URL" section with:

```python
        from tldw_chatbook.Utils.egress import evaluate_url_policy

        decision = evaluate_url_policy(url, trusted_origins=trusted_origins)
        if not decision.allowed and decision.reason != "disabled":
            raise SSRFError(
                f"URL blocked by egress policy ({decision.reason}): {parsed.hostname}"
            )
```

New signature: `def validate_feed_url(cls, url: str, trusted_origins=frozenset()) -> str:`. Keep the empty/scheme/hostname `ValueError` pre-checks and the normalization return untouched. Delete `PRIVATE_IP_RANGES` and the now-unused `socket`/`ipaddress` imports IF nothing else in the file uses them (`SSRFProtector.is_ip_allowed` does — rewrite it):

```python
    def is_ip_allowed(self, ip: str) -> bool:
        try:
            from tldw_chatbook.Utils.egress import _classify_ip

            return _classify_ip(ip) == "public"
        except ValueError:
            return False
```

`sanitize_item` → `def sanitize_item(item: dict, trusted_origins=frozenset()) -> dict:` and its internal call becomes `SecurityValidator.validate_feed_url(sanitized["url"], trusted_origins=trusted_origins)`. Grep for `sanitize_item(` callers (expected: within `monitoring_engine.py` parse paths) and pass `trusted_origins=frozenset({feed_host})` where a feed host is in scope; leave callers without feed context on the default.

- [ ] **Step 6: New tests** — create `Tests/Subscriptions/test_subscription_egress_wiring.py`:

```python
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
```

(The last test's subscription dict must satisfy `check_feed`'s field access — read the function and fill required keys; the assertion that matters is *mapped, not leaked*.)

- [ ] **Step 7: Run**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Subscriptions/ Tests/Utils/test_egress.py -q`
Expected: new tests pass. Any existing test that pinned the old fail-OPEN dns behavior of `validate_feed_url` fails — update it to expect `SSRFError` and list the change in your report. Everything else stays green.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Subscriptions/monitoring_engine.py tldw_chatbook/Subscriptions/website_monitor.py tldw_chatbook/Subscriptions/security.py tldw_chatbook/Utils/egress.py Tests/Subscriptions/test_subscription_egress_wiring.py
git commit -m "feat(security): subscriptions monitors via egress guard; security.py delegates; ssl_verify=0 warning"
```

---

### Task 9: Wire watchlists + the six subscription scrapers (+ existing-test updates)

**Files:**
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py` (`_urls_for_sitemap`, `_items_for_api_source`)
- Modify: `tldw_chatbook/Subscriptions/scrapers/custom_scraper.py`, `generic_scraper.py`, `github_scraper.py`, `hackernews_scraper.py`, `reddit_scraper.py`, `youtube_scraper.py`
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py` (fakes → helper-level monkeypatch)
- Modify (if it mocks httpx): `Tests/Scheduling/test_watchlist_check_handler.py` — grep it first; update only if it fakes `httpx.AsyncClient`.

**Interfaces:** Consumes Task 2's `guarded_fetch_httpx_async` + `GuardedResponse` (`.text`, `.json()`, `.raise_for_status()`, `.status_code`).

**Uniform transformation** (apply at every `client.get(...)` site listed below): keep the `httpx.AsyncClient` construction but REMOVE any `follow_redirects=True` argument, then replace

```python
            response = await client.get(url, headers=headers, follow_redirects=True)
```

with

```python
            response = await guarded_fetch_httpx_async(
                url,
                client=client,
                max_bytes=MAX_FETCH_BYTES_PAGE,
                trusted_origins=trusted,
                headers=headers,
            )
```

where `trusted` is per the table below, and `params=...` is passed through to the helper where the old call had it. `response.raise_for_status()` / `.text` / `.json()` / `.status_code` all keep working on `GuardedResponse`. Import per file:

```python
from tldw_chatbook.Utils.egress import (
    MAX_FETCH_BYTES_PAGE,
    guarded_fetch_httpx_async,
)
```

(plus `MAX_FETCH_BYTES_SITEMAP` in `local_watchlists_service.py`, and `from urllib.parse import urlparse` where a trusted host is computed and the import is missing).

| Site | trusted | cap | notes |
|---|---|---|---|
| `local_watchlists_service._urls_for_sitemap` (`client.get(source)`) | `frozenset({(urlparse(source).hostname or "").lower()})` | `MAX_FETCH_BYTES_SITEMAP` | user-configured source |
| `local_watchlists_service._items_for_api_source` (`client.get(source, **request_kwargs)`) | same, from `source` | `MAX_FETCH_BYTES_PAGE` | pass `headers=headers` and `params=request_kwargs.get("params")` |
| `custom_scraper._fetch_static` L117 | `frozenset({(urlparse(url).hostname or "").lower()})` | PAGE | user-configured pipeline URL |
| `generic_scraper.fetch_content` L137 | same, from `url` | PAGE | ALSO: this file calls `scrape_article(url)` when the article path is available — pass `trusted_origins=frozenset({(urlparse(url).hostname or "").lower()})` there too |
| `github_scraper._fetch_atom_feed` L145, `_fetch_via_api` L192 | `frozenset()` | PAGE | fixed vendor hosts (github.com/api.github.com) resolve public; `_fetch_via_api` passes `params=params` |
| `hackernews_scraper` L168/L212/L231/L241/L273 | `frozenset()` | PAGE | L212 passes `params=params`; the per-item loop calls (L241/L273) use the same client — transform identically |
| `reddit_scraper` L154/L195 | `frozenset()` | PAGE | |
| `youtube_scraper` L183/L208 | `frozenset()` | PAGE | L208 (`_resolve_channel_id`) had NO `follow_redirects` (httpx default False) — the helper now follows up to 10 validated hops; that widens tolerance, which is fine |

- [ ] **Step 1: Apply the transformation to all sites** (table above; every edit is the uniform pattern — no variants beyond `trusted`/cap/params noted).

- [ ] **Step 2: Update `Tests/Subscriptions/test_local_watchlists_service.py`.** Two `FakeAsyncClient` fixtures (around L321 and L421) monkeypatch `httpx.AsyncClient`. Replace each with a helper-level monkeypatch. First site (sitemap test, records fetched URLs and returns sitemap XML):

```python
    async def fake_guarded(url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None):
        fetched_sitemaps.append(url)
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/xml"},
            text=SITEMAP_XML,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service.guarded_fetch_httpx_async",
        fake_guarded,
    )
```

(`SITEMAP_XML` = whatever the current `FakeResponse` returned — reuse its literal; `from types import SimpleNamespace`. Delete the now-unused `FakeAsyncClient`/`FakeResponse` classes and the `monkeypatch.setattr("httpx.AsyncClient", ...)` line for that test.) Second site (API-source test, records `{"url", **kwargs}`): same shape, but `fake_guarded` appends `{"url": url, "headers": headers, "params": params}` to `requests` and returns a namespace whose `json` is a `lambda: API_PAYLOAD` (reuse the old fake's payload literal). Keep each test's assertions; adjust key names only if the old kwargs dict recorded httpx-specific keys.

- [ ] **Step 3: Grep `Tests/Scheduling/test_watchlist_check_handler.py`** for `httpx`/`AsyncClient`/`client.get`. If it fakes the client for these code paths, apply the same helper-level monkeypatch conversion; if it stubs at a higher level (service/handler), leave it alone.

- [ ] **Step 4: Run**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Subscriptions/ Tests/Scheduling/ -q`
Expected: green (updated fakes included).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/scrapers/custom_scraper.py tldw_chatbook/Subscriptions/scrapers/generic_scraper.py tldw_chatbook/Subscriptions/scrapers/github_scraper.py tldw_chatbook/Subscriptions/scrapers/hackernews_scraper.py tldw_chatbook/Subscriptions/scrapers/reddit_scraper.py tldw_chatbook/Subscriptions/scrapers/youtube_scraper.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Scheduling/test_watchlist_check_handler.py
git commit -m "feat(security): watchlists + subscription scrapers via guarded httpx helper"
```

(drop the Scheduling test from `git add` if Step 3 required no change)

---

### Task 10: Media/audio real byte caps + GitHub API response caps

**Files:**
- Modify: `tldw_chatbook/Media/local_media_reading_service.py` (`_default_url_file_downloader`)
- Modify: `tldw_chatbook/Local_Ingestion/audio_processing.py` (`download_audio_file`)
- Modify: `tldw_chatbook/Utils/github_api_client.py` (`get_file_content`, `get_repository_tree`, `get_directory_contents`)
- Modify: `Tests/Utils/test_egress.py` — no; new file: `Tests/Utils/test_download_caps_wiring.py`

**Interfaces:** Consumes `guarded_fetch_requests` (with `sink=`), `guarded_fetch_httpx_async`, `EgressBlockedError`, `EgressFetchError`, `MAX_FETCH_BYTES_MEDIA`, `MAX_FETCH_BYTES_GITHUB_FILE`.

- [ ] **Step 1: `_default_url_file_downloader`** — replace the `requests.get(url, stream=True, timeout=timeout)` + manual chunk loop with a sink-mode guarded fetch. The new body of the download portion:

```python
        from tldw_chatbook.Utils.egress import (
            MAX_FETCH_BYTES_MEDIA,
            guarded_fetch_requests,
        )

        opts = dict(options or {})
        timeout = float(opts.get("timeout") or 30)
        trusted = frozenset({(urlparse(url).hostname or "").lower()})
        # HEAD-less probe: we need headers before choosing the suffix, so fetch
        # into a temp file via the guarded helper's sink mode (real streamed cap).
        fd, path = tempfile.mkstemp(prefix="tldw_url_ingest_", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                response = guarded_fetch_requests(
                    url,
                    max_bytes=MAX_FETCH_BYTES_MEDIA,
                    trusted_origins=trusted,
                    timeout=timeout,
                    sink=handle,
                )
            response.raise_for_status()
            suffix = cls._download_suffix_for_url(
                url,
                media_type=media_type,
                content_type=response.headers.get("content-type"),
            )
            final_path = path[: -len(".part")] + suffix if suffix else path
            if final_path != path:
                os.replace(path, final_path)
                path = final_path
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return {"path": path, "cleanup": True, "source_path": urlparse(url).path}
```

CAREFUL: the original called `raise_for_status()` BEFORE downloading; sink mode downloads first. To preserve fail-fast on error statuses, `raise_for_status` after the sink write is acceptable (the cap bounds wasted bytes), but you must delete the temp file on that raise — the `except Exception` cleanup above does. The original suffix came from headers pre-download; here we rename after. Keep the original `import` lines (`os`, `tempfile`, `urlparse`, `Path` are already imported in the function/file — verify and reuse).

- [ ] **Step 2: `download_audio_file`** — replace `requests.get(url, headers=headers, stream=True, timeout=120)` and the manual loop:

```python
            from tldw_chatbook.Utils.egress import (
                EgressBlockedError,
                EgressFetchError,
                guarded_fetch_requests,
            )

            trusted = frozenset({(urlparse(url).hostname or "").lower()})
            # Fast-fail on declared size, then enforce the REAL streamed size.
            save_path = None
            try:
                probe_headers = dict(headers)
                # single guarded fetch; filename needs response headers, so fetch
                # to a temp .part file then rename
                tmp_path = Path(target_dir) / (uuid.uuid4().hex + ".part")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    response = guarded_fetch_requests(
                        url,
                        max_bytes=self.max_file_size,
                        trusted_origins=trusted,
                        timeout=120,
                        headers=probe_headers,
                        sink=f,
                    )
                response.raise_for_status()
                declared = int(response.headers.get("content-length", 0))
                if declared > self.max_file_size:
                    raise AudioDownloadError(
                        f"File size ({declared / (1024 * 1024):.2f} MB) exceeds limit"
                    )
                filename = self._get_filename_from_response(response, url)
                save_path = Path(target_dir) / filename
                save_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.replace(save_path)
            except (EgressBlockedError, EgressFetchError) as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise AudioDownloadError(f"Download blocked or too large: {e}") from e
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

            logger.info(f"Downloaded audio file: {save_path}")
            return str(save_path)
```

(add `import uuid` at module top if missing; the existing `except requests.RequestException` → `AudioDownloadError` wrapper below stays — it still fires because the helper propagates requests exception types. NOTE the real enforcement is `max_bytes=self.max_file_size` on streamed bytes — the spoofed/missing Content-Length bypass is closed; the declared-size check remains only as a courtesy error message.)

- [ ] **Step 3: `github_api_client`** — in the three content-bearing methods (`get_file_content`, `get_repository_tree`, `get_directory_contents`), replace

```python
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
```

with

```python
            from tldw_chatbook.Utils.egress import (
                MAX_FETCH_BYTES_GITHUB_FILE,
                guarded_fetch_httpx_async,
            )

            response = await guarded_fetch_httpx_async(
                url,
                client=self.client,
                max_bytes=MAX_FETCH_BYTES_GITHUB_FILE,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
```

(adjust per-method for the exact local variable names; `get_repository_tree` has two `client.get` calls — transform both). `raise_for_status` delegation preserves the `except httpx.HTTPStatusError as e: ... e.response.status_code == 404` handling. No `trusted_origins` — fixed public vendor host; the guard just passes. `EgressFetchError` (oversize) will be caught by each method's trailing `except Exception as e: raise GitHubAPIError(...)` — containment holds; verify per method.

- [ ] **Step 4: Tests** — create `Tests/Utils/test_download_caps_wiring.py`:

```python
"""Sink-mode caps for media/audio downloads; GitHub client caps."""

import io

import pytest

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import EgressFetchError, guarded_fetch_requests


@pytest.fixture(autouse=True)
def _dns(monkeypatch):
    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda s, k=None, d=None: d
    )


def test_audio_download_real_cap_beats_spoofed_content_length(monkeypatch):
    """A body larger than max_bytes fails even when Content-Length lies."""
    import requests as requests_lib
    from requests.adapters import BaseAdapter
    from requests.models import Response as RequestsResponse

    class LyingAdapter(BaseAdapter):
        def send(self, request, **kwargs):
            resp = RequestsResponse()
            resp.status_code = 200
            resp.headers["Content-Length"] = "10"  # lie
            resp.raw = io.BytesIO(b"a" * 4096)
            resp.url = request.url
            resp.request = request
            return resp

        def close(self):
            pass

    sess = requests_lib.Session()
    sess.mount("http://", LyingAdapter())
    with pytest.raises(EgressFetchError, match="exceeds"):
        guarded_fetch_requests(
            "http://media.example/file.mp3",
            session=sess,
            max_bytes=1024,
            sink=io.BytesIO(),
        )
```

(The per-surface wiring of Steps 1-3 is covered by: this transport-level regression + the existing suites for those modules staying green + the containment checks each surface's own error handling provides. If `Tests/` has suites for `audio_processing`/`local_media_reading_service`/`github_api_client` (grep for their names), run them.)

- [ ] **Step 5: Run**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/test_download_caps_wiring.py Tests/Utils/test_egress.py -q` plus any suite found for the three modules.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Media/local_media_reading_service.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Utils/github_api_client.py Tests/Utils/test_download_caps_wiring.py
git commit -m "feat(security): real streamed byte caps for media/audio downloads; GitHub API response caps (TASK-329)"
```

---

### Task 11: Backlog bookkeeping + follow-up tasks

**Files:**
- Modify: `backlog/tasks/task-328 - Add-SSRF-protection-to-outbound-URL-fetching.md` (ACs → `- [x]`, status Done, Implementation Notes)
- Modify: `backlog/tasks/task-329 - Add-response-size-caps-and-timeouts-to-fetchers.md` (same)
- Create: two new follow-up backlog task files (IDs per the scan below)

- [ ] **Step 1: Assign follow-up IDs safely.** Backlog IDs have collided TEN+ times; scan BOTH namespaces against origin/dev AND the working tree:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-web-fetch-hardening
git fetch -q origin
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
import os, re, subprocess
ids = set()
for name in subprocess.run(
    ["git", "ls-tree", "-r", "--name-only", "origin/dev", "backlog/"],
    capture_output=True, text=True).stdout.splitlines():
    m = re.search(r"task-(\d+)", name)
    if m: ids.add(int(m.group(1)))
for root in ("backlog/tasks", "backlog/drafts"):
    if os.path.isdir(root):
        for name in os.listdir(root):
            m = re.search(r"task-(\d+)", name)
            if m: ids.add(int(m.group(1)))
print("max id:", max(ids), "-> next:", max(ids) + 1)
EOF
```

Use `next` and `next+1` for the two follow-ups below.

- [ ] **Step 2: Follow-up task files** (create with the backlog CLI if available — `backlog task create "<title>" -d "<desc>" -l <labels>` — else hand-write matching the existing file format):

1. **"Confluence: replace sync requests calls inside async methods"** — labels `subscriptions,performance`; description: `ConfluenceAuth.make_request` and `ConfluenceScraper._extract_page_id_from_url` make synchronous `requests` calls from inside `async def` methods, blocking the event loop for up to the full request timeout (30s after task-328's fix; previously unbounded). Port to httpx async or run via a thread executor. AC: no sync HTTP call remains on the event loop in `Web_Scraping/Confluence/`.
2. **"Image-gen adopts shared egress module (Utils/egress.py)"** — labels `image-generation,security,followup`; description: TASK-328 shipped `Utils/egress.py` (policy + guarded helpers with per-hop redirect re-validation). TASK-498's port should now REUSE it: `Image_Generation/http_client.py`'s light `_validate_egress_or_raise` and `fetch_json` manual-hop loop can delegate to `evaluate_url_policy`/`guarded_fetch_httpx`, with API-returned image URLs evaluated as content-derived (`trusted_origins=frozenset()`) and user-configured backend `base_url` hosts as trusted origins. AC: `fetch_image_bytes`/`fetch_json` validate via the shared module; local backends keep working. Also add one line to `backlog/tasks/task-498 - Port-image-generation-egress-SSRF-protections-from-tldw_server.md` under its Description pointing at `Utils/egress.py` as the now-existing shared module.

- [ ] **Step 3: Close out 328/329.** Check every AC box, set `status: Done`, add `## Implementation Notes` summarizing: the policy rule, the four helpers, the surfaces wired (list them), deliberate behavior changes (validate_feed_url dns fail-closed; sitemap-item same-origin trust), residuals (DNS-rebinding TOCTOU, Playwright mid-chain GET), and the files touched. Keep it concise — the spec and plan carry the detail.

- [ ] **Step 4: Run the full affected test set one more time**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Utils/ Tests/Web_Scraping/ Tests/Subscriptions/ Tests/Scheduling/ Tests/Local_Ingestion/ -q`
Expected: green (report any pre-existing failure unrelated to these files as baseline, do not fix).

- [ ] **Step 5: Commit**

```bash
git add "backlog/tasks/task-328 - Add-SSRF-protection-to-outbound-URL-fetching.md" "backlog/tasks/task-329 - Add-response-size-caps-and-timeouts-to-fetchers.md" backlog/tasks/task-<next>*.md "backlog/tasks/task-498 - Port-image-generation-egress-SSRF-protections-from-tldw_server.md"
git commit -m "docs(backlog): close TASK-328/329; file Confluence-async + egress-adoption follow-ups"
```

---

## Post-plan notes for the controller (not for task implementers)

- Suggested SDD models: Tasks 1-4 carry complete code → cheapest tier; Tasks 5-10 are integration edits on real files → mid tier; final whole-branch review → most capable.
- The Playwright empirical check (Task 4 Step 5) is report-only; capture its output in the task report.
- Live checks before PR (controller, not CI): one guarded fetch of a real public page; `http://127.0.0.1:1/` and `http://169.254.169.254/` rejected with the remedy message.
- `.superpowers/` is TRACKED on dev — SDD ledger/briefs go in the scratchpad, implementers `git add` only the files their task lists.
