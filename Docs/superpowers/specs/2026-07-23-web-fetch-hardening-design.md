# Web-Fetch Hardening: SSRF Protection + Size Caps/Timeouts (TASK-328 + TASK-329)

**Date:** 2026-07-23
**Backlog:** TASK-328 (SSRF protection for all outbound URL fetching, HIGH) + TASK-329 (response-size caps and timeouts on remaining fetchers, MEDIUM; depends on 328). One cohesive sub-project, one PR.
**Branch:** `feat/web-fetch-hardening` (worktree off `origin/dev` @ `8a46af45e`).

## Problem

There is no SSRF protection anywhere in the codebase. `validate_url`
(`Utils/input_validation.py`) is syntactic-only and intentionally permits
localhost/metadata/internal hosts. Every fetcher follows redirects with no
post-redirect re-validation (`302 -> http://169.254.169.254/` is followed).
`Scraper.page.goto()` has no URL validation. Separately, many fetchers read
unbounded response bodies and several have no timeout at all.

A full inventory (origin/dev @ `8a46af45e`) found ~30 fetch sites across
`Web_Scraping/`, `Local_Ingestion/`, `Media/`, `Subscriptions/`, and
`Utils/github_api_client.py`, plus **dead SSRF wiring**:
`Subscriptions/security.py` defines a real `SecurityValidator.validate_feed_url`
(metadata + private-range blocking via DNS) and `SSRFProtector`, constructed and
passed into `FeedMonitor` — but never called in any fetch path.

## Threat model / posture (user decision)

This is a local-first TUI. Users legitimately ingest from intranet Confluence,
localhost services, and internal wikis. The SSRF vector is **content-derived**
URLs: redirect targets, sitemap/crawl-discovered links, feed-item links.

**Chosen posture — trust user-configured origins.** One uniform rule:

> A URL is allowed iff **(every resolved IP is public AND not a metadata
> endpoint)** OR **its host is in `trusted_origins`** OR **its host is in the
> config allowlist**. Metadata endpoints are stricter: blocked even for
> `trusted_origins`; ONLY the exact-host config allowlist overrides them.

"User-supplied" is not a flag with special cases — the host of a URL the user
explicitly typed/configured (Confluence `base_url`, a subscription source, a
URL pasted into ingest) simply **seeds `trusted_origins`** for that fetch and
its redirect chain. Content-derived URLs get an empty (or inherited) set.
Consequences:

- Intranet Confluence, localhost feeds, and private sitemap hosts keep working
  with zero config.
- Same-origin redirects (`http→https`, `/login`, port changes) inherit trust —
  `trusted_origins` matches on lowercase **hostname only** (not scheme/port).
- Cross-origin redirect targets must resolve public → blocks
  `302 → http://169.254.169.254/` and `302 → http://192.168.1.1/`.
- A user who explicitly types a URL resolving to a private IP is allowed (their
  choice, accepted residual); one resolving to a **metadata** endpoint is still
  blocked unless they allowlist that exact host in config.

## Architecture

### Unit 1: Policy module — `tldw_chatbook/Utils/egress.py` (new)

The single source of SSRF truth app-wide (also the module TASK-498/image-gen
adopts later; its API is a superset of image-gen's light guard).

**Classification** (via stdlib `ipaddress`, superseding the dead validator's
hand-rolled range list):

- `metadata` — IPs `169.254.169.254`, `fd00:ec2::254` (AWS IPv6),
  `100.100.100.200` (Alibaba); hostnames `metadata.google.internal`,
  `metadata.azure.com` (union of `SecurityValidator.METADATA_ENDPOINTS` + the
  cloud-IPs it missed). Hostname matches are checked pre-resolution.
- `private` — anything where `ip.is_private or ip.is_loopback or
  ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified`,
  plus CGNAT `100.64.0.0/10`; IPv6 incl. ULA `fc00::/7` and IPv4-mapped
  addresses classified by their embedded IPv4.
- `public` — everything else.

**API:**

```python
class EgressBlockedError(Exception):
    """URL blocked by egress policy. Carries url, reason, resolved detail."""

@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str        # "ok" | "scheme" | "metadata" | "private" | "dns_failure" | "disabled"
    host: str
    resolved_ips: tuple[str, ...]

def evaluate_url_policy(url, *, trusted_origins=frozenset()) -> EgressDecision
async def evaluate_url_policy_async(url, *, trusted_origins=frozenset()) -> EgressDecision
def check_url_or_raise(url, *, trusted_origins=frozenset()) -> None
async def check_url_or_raise_async(url, *, trusted_origins=frozenset()) -> None
```

- Non-`http(s)` schemes rejected first (kills `file:`, `ftp:`, `gopher:`,
  `javascript:`, `data:` — everything not http/https, an allowlist not a
  blocklist).
- DNS: sync path uses `socket.getaddrinfo`; **async path uses
  `loop.getaddrinfo`** — the sync resolver must never run on the event loop
  (subscriptions/watchlists/crawler are all async; sync-under-async is a
  repeated repo bug class). IP-literal hosts (incl. bracketed IPv6) classify
  directly without lookup.
- **Every** resolved IP (both families, deduplicated) must pass — any bad IP
  blocks (fail-closed). DNS resolution failure → blocked with
  `reason="dns_failure"` (fail-closed; a URL that cannot resolve cannot be
  fetched anyway).
- Config (`[web_security]`, read via the canonical `config.py` accessors, with
  defaults registered in `config.py`'s defaults so the section is surfaced —
  the `[image_generation]`/`load_settings()` lesson):
  - `enabled = true` — kill switch. When `false`, every decision returns
    `allowed=True, reason="disabled"` but still logs the would-block at DEBUG
    and increments the counter — protection can be turned off, visibility
    cannot.
  - `allowed_hosts = []` — exact lowercase hostnames/IP literals exempt from
    all blocking (incl. metadata). The escape hatch for exotic setups.
- Observability: `log_counter("egress_blocked", labels={"reason": ...})` on
  every block (and would-block when disabled). Block errors name the URL, the
  reason, and the remedy (`[web_security] allowed_hosts` / `enabled=false`).

**Explicit non-goals (documented in the module docstring):** DNS-rebinding IP
pinning (we resolve-and-check; the client re-resolves to connect — TOCTOU
residual accepted, same "where feasible" line TASK-498 draws), proxy-aware
policy (env-var proxies keep working; we validate the target URL), a DNS cache
(OS caches suffice).

### Unit 2: Guarded transport helpers (same module)

All helpers: pre-check via policy → issue request with redirects **disabled**
→ manual hop loop (max 10 hops) re-validating each `Location` (resolved via
`urljoin`) against the **same** `trusted_origins` → stream body with a byte
cap. Shared rules:

- **Credential stripping (Global Constraint):** the manual loop strips
  `Authorization`, `Cookie`, and `Proxy-Authorization` headers whenever a hop's
  host differs from the FIRST URL's host — replicating what
  `follow_redirects=True` did automatically. This includes **session-mounted
  auth**: `requests.Session.auth` re-signs every request the manual loop
  issues, so `guarded_fetch_requests` must suppress session auth on
  cross-origin hops (not merely strip headers). Without this, Confluence basic
  auth or a GitHub token would be forwarded to any host a redirect names —
  a credential-exfiltration vector this project must not introduce. Covered by
  a dedicated test per helper (incl. a session-auth case for requests).
- **Byte caps measure decompressed bytes** — httpx `iter_bytes`, requests
  `iter_content`, and aiohttp `iter_chunked` all yield decompressed data by
  default, so caps are decompression-bomb-safe. Helpers must not switch to raw
  streams.
- Oversize → error the instant the running total crosses the cap (never buffer
  first); redirect-loop/hop-cap, redirect-without-`Location`, and timeout
  failures produce explicit messages (TASK-329 AC#3; image-gen `fetch_json`
  precedent).
- Hop targets do NOT get added to `trusted_origins`; only the original set
  carries through the chain.

Helpers (thin — policy logic lives only in Unit 1):

- `guarded_fetch_httpx(url, *, client, max_bytes, trusted_origins, ...)` —
  sync + async variants; returns `(bytes, final_url, response_headers)`-shaped
  result. Used by subscriptions, watchlists, `web_article_ingestion`.
- `guarded_fetch_requests(url, *, session=None, max_bytes, trusted_origins,
  timeout, ...)` — accepts an existing `requests.Session` (Confluence auth
  lives on the session); raises/propagates `requests` exception types so
  existing callers' `except requests...` blocks keep working.
- `guarded_fetch_aiohttp(url, *, session, max_bytes, trusted_origins, ...)` —
  for `Article_Scraper/crawler.py` (`allow_redirects=False` + manual loop +
  `iter_chunked` cap).
- Playwright navigation guard:
  - `check` before `page.goto` (async or sync variant per call site) — a bad
    initial target never launches.
  - **Post-navigation validation**: after `goto` returns, walk
    `response.request.redirected_from` collecting every URL in the chain plus
    the final URL; validate each. On violation: discard content, raise the
    surface's error. This is required because Playwright route handlers
    intercept only the initial request of a navigation — server redirect hops
    are followed by the browser without re-invoking the route. The plan
    includes an empirical check of route/redirect behavior on our Playwright
    version; if hops ARE interceptable, a `page.route` guard on
    `document`-type requests aborts them too — post-hoc validation stays as
    the net either way. **Documented residual:** a mid-chain GET to an
    internal host may fire before we reject; the response is discarded, so
    nothing enters the app. Subresource requests (images/scripts) are not
    policed (rendering breakage; accepted scope).

### Unit 3: Wiring (the fan-out)

| Surface | trusted_origins seed | Cap | Timeout | Notes |
|---|---|---|---|---|
| `Article_Extractor_Lib.get_page_title` | target host (user URL) | 10MB | keep 10s | → `guarded_fetch_requests` |
| `scrape_from_sitemap` (L999) | sitemap host | **50MB** (sitemap protocol allows up to 50MB uncompressed) | **add 30s** (has none) | sitemap URL user-supplied; URLs *from* it are content-derived |
| `collect_internal_links` (L1055) | seed host | 10MB | **add 30s** (has none) | crawl-discovered links content-derived |
| `Article_Scraper/crawler.py` `crawl_site`/`get_urls_from_sitemap` | seed host | 10MB / 50MB | keep 10s/30s | → `guarded_fetch_aiohttp` |
| `Scraper._fetch_html` + Playwright paths (`fetch_html`, `recursive_scrape`, `scrape_article_async`) | initial URL host; Confluence subclass adds its `base_url` host | n/a (rendered page) | keep config-driven | pre-goto check + post-nav chain validation |
| Confluence `make_request`, `_extract_page_id_from_url` | `base_url` host | 10MB | **add 30s** (has none) | via `guarded_fetch_requests` w/ the auth session; **sync-in-async refactor filed as a follow-up backlog task, not fixed here** |
| `web_article_ingestion.extract_article_for_ingest` | target host | keep 10MB | keep 30s | swap hand-rolled stream loop → `guarded_fetch_httpx`; content-type allowlist, retryable-vs-permanent classification, and post-redirect canonical `url` semantics preserved at the call site |
| Subscriptions `FeedMonitor._fetch_and_parse_feed`, `URLMonitor._fetch_url_content`, `website_monitor._fetch_url_content` | subscription source host | 10MB | keep 30s | → async `guarded_fetch_httpx`; `URLMonitor` gains guarding it never had |
| `local_watchlists_service._urls_for_sitemap` / `_items_for_api_source` | source host | 50MB / 10MB | keep 30s | async `guarded_fetch_httpx` |
| `Subscriptions/scrapers/*` (custom, generic, github, hackernews, reddit, youtube) | source/vendor host | 10MB | keep 30s | uniform wiring even where hosts are fixed vendors (simplicity > case-splitting) |
| Media `_default_url_file_downloader` | target host (user URL) | **500MB streamed-bytes** (was: none) | keep 30s | real cap on received bytes |
| `audio_processing.download_audio_file` | target host (user URL) | `self.max_file_size` enforced on **actual streamed bytes** (header check kept as a fast-fail) | keep 120s | closes the spoofed/missing `Content-Length` bypass |
| `Utils/github_api_client` | n/a — fixed vendor host, **no SSRF wiring** | 20MB on file/tree responses | keep 30s | caps only |
| `distribution_manager._distribute_webhook` | — | — | — | **out of scope** (data egress to a user-configured webhook, not URL-content ingestion) |

Additional wiring:

- **`Subscriptions/security.py` delegation:** `SecurityValidator.validate_feed_url`
  and `SSRFProtector.is_safe_url` keep their public APIs but delegate all
  host/IP policy to `evaluate_url_policy`; the duplicate `PRIVATE_IP_RANGES` /
  `METADATA_ENDPOINTS` logic is deleted. Existing `Tests/Subscriptions`
  security tests must stay green (API-compatible delegation).
- **`ssl_verify=0`** (monitoring_engine L370/L778): feature kept (self-signed
  intranet feeds are legitimate) but each fetch with TLS verification disabled
  logs a loguru WARNING (once per host per process, module-level seen-set) and
  increments `log_counter("web_insecure_ssl_fetch")` every time.
- **`validate_url` deliberately unchanged** (satisfies TASK-328 AC#3 via its
  "or a new validate_public_url" arm — `evaluate_url_policy` is that
  function). Rationale: `validate_url` is called from ~20 UI/config validation
  sites (incl. LLM endpoint settings) where DNS resolution would add latency
  and false failures offline. Syntactic validation at the UI boundary,
  resolution-based policy at the network boundary.

### Error handling

`EgressBlockedError` **containment is a requirement**: every wired surface maps
it into that surface's existing failure path — `PermanentIngestError` in
ingestion, a failed-check/skip result in monitors and scrapers, an error return
in Media/audio download paths, a scrape-failure result in Article_Extractor
paths. A blocked URL must never escape as an unhandled exception from a worker
or `gather` loop (this repo's `run_worker(exit_on_error)` kills the app on
uncaught worker exceptions).

### Testing

- **Policy unit tests** (`Tests/Utils/test_egress.py`), DNS monkeypatched
  (`socket.getaddrinfo` / event-loop `getaddrinfo`): metadata IP + hostname,
  loopback, RFC1918, IPv6 ULA, IPv4-mapped IPv6, CGNAT, `file://`/`ftp://`/
  `data:`, IP-literal hosts (v4 + bracketed v6), multi-record any-bad-blocks,
  dns-failure fail-closed, `trusted_origins` allows private but never
  metadata, config `allowed_hosts` overrides metadata, kill switch
  logs-but-allows. (TASK-328 AC#4.)
- **Helper tests**: httpx via `MockTransport` (precedent:
  `Tests/Image_Generation/test_http_client.py`): redirect-to-internal blocked,
  same-origin redirect allowed, hop-cap, byte-cap aborts mid-stream,
  **credential stripping on cross-origin hop** (Authorization/Cookie present
  same-origin, absent cross-origin). requests/aiohttp equivalents with
  mocked transports (libraries already in the dev env, else hand-rolled
  fakes). Playwright guard: unit-test the chain-validation function on faked
  request/redirect objects (no browser in CI).
- **Wiring tests per surface**: a blocked URL produces the surface's error
  type (containment), oversize produces the explicit message.
- **Test-mock fallout inventoried up front:** existing Tests/Subscriptions
  (and any others) that mock `client.get(..., follow_redirects=True)` are
  enumerated during planning and updated to the manual-loop shape —
  not discovered as failures mid-implementation.
- **Live checks (documented, not CI):** fetch a real public page through a
  guarded path; confirm `http://127.0.0.1:1/` and a `169.254.169.254` URL are
  rejected with the remedy-bearing message.

### Follow-ups filed at implementation end (backlog CLI, IDs assigned against
origin/dev with the two-namespace dup scan)

1. Confluence sync-`requests`-inside-async refactor (event-loop blocking).
2. TASK-498 note: image-gen adopts `Utils/egress.py` (its light guard's
   `fetch_json` already has the identical manual-hop shape).

## Out of scope / residual risks (explicit)

- DNS-rebinding TOCTOU between policy resolution and client connection
  (IP-pinning across requests/httpx/aiohttp/Playwright is not feasible here).
- Playwright mid-chain GET fires before post-hoc rejection (response
  discarded); subresource requests unpoliced.
- User-typed URLs resolving to private (non-metadata) IPs are allowed by
  chosen posture.
- `WebSearch_APIs.py`, LLM/embeddings/TTS/image-gen provider calls,
  `distribution_manager` webhooks: unchanged.
- No per-cap config keys; caps are constants (callers pass overrides).
