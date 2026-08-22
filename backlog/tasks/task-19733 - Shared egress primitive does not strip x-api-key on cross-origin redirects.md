---
id: TASK-19733
title: >-
  Shared egress primitive does not strip x-api-key on cross-origin redirects
status: Done
assignee:
  - '@claude'
created_date: '2026-08-21 23:40'
labels:
  - security
  - credentials
  - http
  - egress
priority: high
dependencies: []
---

## Description

Source: surfaced while filing/reviewing **TASK-19557** (API-key headers survive
cross-origin redirects in two clients). TASK-19557 fixes two *direct callers*
(`tldw_api/client.py` and the Anthropic call sites) and deliberately does not
touch the shared primitive. This task is that primitive. Re-verified at this
branch base (`839819e1a`).

`tldw_chatbook/Utils/egress.py:348`:

```python
_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization", "x-goog-api-key")
```

`x-api-key` is **not** in the tuple. `_hop_headers` (`egress.py:491-497`) is the
single place every guarded fetch decides what a cross-origin redirect hop may
carry; it lowercases and drops only what that tuple names. The same tuple is
re-applied belt-and-braces at `egress.py:526-527`, `:588-589`, `:648` and
`:710`.

This matters more than the two callers TASK-19557 covers, because
`Utils/egress.py` is the shared guarded-fetch primitive the review praised as
well-built — roughly 25 call sites route through
`guarded_fetch_httpx` / `guarded_fetch_httpx_async`, and everything that adopts
the primitive inherits the gap.

**Reachable today, with a user-supplied credential.** Feed/watchlist auth is
user-configurable and lands in exactly this header:

* `Subscriptions/monitoring_engine.py:1006-1009` —
  `key_header = auth_config.get("header", "X-API-Key"); headers[key_header] =
  auth_config.get("key", "")` — and those `headers` are handed straight to
  `guarded_fetch_httpx_async` at `monitoring_engine.py:1024-1030` (same shape
  again at `:1709-1714` for the single-URL fetch path).
* `Subscriptions/site_config_manager.py:161-167` —
  `key_name = self.auth_credentials.get("key_name", "X-API-Key")` — consumed by
  every scraper's `self.get_headers()` (`generic_scraper.py:137`,
  `custom_scraper.py:117`, `github_scraper.py:143,191`,
  `reddit_scraper.py:152,199`, `hackernews_scraper.py:166,222,246,262,300`,
  `youtube_scraper.py:181,218`), each of which passes them to
  `guarded_fetch_httpx_async`.

A feed that starts 302-ing to another host (compromised, sold, or hostile from
the start) receives the user's feed API key verbatim.

**The header name is user-supplied, which is the load-bearing design point.**
Both producers above take the header NAME from user config and only *default*
to `X-API-Key`. Adding one more literal to `_STRIP_HEADERS` closes the default
and leaves every custom-named credential header open. Per the owner's standing
ruling (durable/pragmatic over clever), the fix should be the one that cannot
drift: on a cross-origin hop, drop **all caller-supplied headers except an
explicit non-credential allowlist** (User-Agent / Accept / Accept-Language /
Accept-Encoding and similar), rather than growing a denylist one incident at a
time. `_STRIP_HEADERS` would then remain only as the client-default-header
sweep it also performs at `:526`, `:588`, `:648`, `:710`.

**Named siblings, still unfixed, all outside the primitive** (verified at this
base; every one uses `requests` with default `allow_redirects=True` — grep for
`allow_redirects` in `Web_Scraping/WebSearch_APIs.py` returns nothing):

| site | header | call |
| --- | --- | --- |
| `LLM_Calls/LLM_API_Calls_Local.py:1011` (Kobold) | `X-Api-Key` | `session.post(...)` at `:1060` |
| `Web_Scraping/WebSearch_APIs.py:2711` (Bing) | `Ocp-Apim-Subscription-Key` | `session.get(...)` |
| `Web_Scraping/WebSearch_APIs.py:2964` (Brave) | `X-Subscription-Token` | `requests.get(...)` |
| `Web_Scraping/WebSearch_APIs.py:3985` (Serper) | `X-API-KEY` | `requests.post(...)` |
| `Web_Scraping/WebSearch_APIs.py:4055` (Exa) | `x-api-key` | `requests.post(...)` |

(Kagi at `:3672` and Yandex at `:4228` use `Authorization`, which `requests`
already strips cross-origin — record them as clean, not as fixes.)

**Knock-on the fix must not miss:** `Model_Artifacts/fetch.py:26` keeps a
hand-mirrored copy of the tuple, and
`Tests/Model_Artifacts/test_stream_fetch.py:276` pins
`set(fetch._STRIP_HEADERS) == set(egress._STRIP_HEADERS)`. Changing one without
the other turns that guard red.

## Acceptance Criteria

- [x] A credential header carrying a user-configured API key (default
      `X-API-Key` **and** a custom header name the user chose) is absent from
      the request when a guarded fetch follows a redirect to a different origin
- [x] The cross-origin rule is expressed so that a newly-introduced credential
      header name is safe by default — a header the caller supplies is not
      forwarded cross-origin unless it is on an explicit non-credential
      allowlist — rather than by adding another literal to a denylist
- [x] Same-origin redirects still carry the header, so authenticated feeds that
      redirect within their own origin keep working
- [x] Born-red tests drive an actual cross-origin redirect through both
      `guarded_fetch_httpx` and `guarded_fetch_httpx_async` and assert the
      credential header is absent on the second hop, including one case with a
      non-default header name; each is mutation-checked (restoring the old
      behaviour makes it red)
- [x] At least one test exercises the real producer path
      (`monitoring_engine`'s `api_key` auth config or a `SiteConfig` scraper),
      not only `_hop_headers` in isolation
- [x] `Model_Artifacts/fetch.py`'s mirrored tuple and
      `Tests/Model_Artifacts/test_stream_fetch.py`'s drift guard are updated in
      the same change and remain green
- [x] The five named non-egress siblings (Kobold, Bing, Brave, Serper, Exa) are
      either fixed the same way or recorded — with the reason — as a separate
      tracked item; the task is not closed with them silently unaddressed

## Notes

Deliberately scoped apart from TASK-19557: that task fixes two clients that do
not use the primitive. Fixing the primitive does not fix them, and fixing them
does not fix the ~25 call sites that route through `Utils/egress.py`. Both are
needed; neither subsumes the other.

## Implementation Plan

1. Write the born-red tests first, against unmodified `origin/dev`: a
   custom-named credential header (`X-Feed-Token` — on no denylist anywhere)
   and the default `X-API-Key`, each driven through a real cross-origin
   redirect in `guarded_fetch_httpx`, `guarded_fetch_httpx_async` and
   `guarded_fetch_requests`, plus the two real producer paths.
2. Invert `_hop_headers`'s cross-origin branch from a name denylist to an
   explicit allowlist, and apply the same allowlist to the BUILT request so
   client-default headers are covered too.
3. Delete `Model_Artifacts/fetch.py`'s hand-mirrored tuple; import the one
   policy object instead, and re-point the drift guard at identity.
4. Run egress / Model_Artifacts / subscriptions / web-scraping suites and a
   repo-wide `--collect-only`; baseline anything red against `origin/dev`.
5. Record the five non-egress siblings for separate filing.

## Implementation Notes

Inverted the cross-origin redirect rule in `Utils/egress.py` from a denylist
of credential header names to an allowlist of headers that may cross an
origin boundary. `x-api-key` was never added — adding it would have been the
wrong fix, because both producers take the header NAME from user config
(`monitoring_engine._fetch_and_parse_feed` and `SiteConfig.get_headers` only
*default* to `X-API-Key`), so any literal list closes one name and forwards
every other one verbatim.

**The rule.** New `CROSS_ORIGIN_SAFE_HEADERS` in `Utils/egress.py`: on a hop
that leaves the original origin, a header survives only if it is on that
frozenset. Membership test — forwarding it off-origin must be both *needed by
a real caller* and *incapable of carrying a secret*. That admits content
negotiation (`accept`, `accept-charset`, `accept-encoding`,
`accept-language`), cache validators (`cache-control`, `pragma`, `if-match`,
`if-none-match`, `if-modified-since`, `if-unmodified-since`), partial content
(`range`, `if-range`), and `user-agent`. A second, separate frozenset
`_TRANSPORT_HEADERS` (`host`, `connection`, `content-length`, `content-type`,
`transfer-encoding`, …) is exempt: those are framing owned by the HTTP client,
and stripping `host` off a built request would break the very request the
guard protects. `_STRIP_HEADERS` is retained but demoted — it is no longer the
rule, it is the never-cross FLOOR: both exemption sets are constructed by
subtracting it (`frozenset({...}) - frozenset(_STRIP_HEADERS)`), so a careless
future edit that adds `authorization` to either list cannot take effect.
`test_allowlist_never_admits_a_credential_shaped_name` asserts that wiring is
live, and additionally rejects any allowlist entry whose name reads like a
secret (`key`/`token`/`secret`/`auth`/`cookie`/`password`/`sig`).

Applied at both layers, because they leak independently:
- `_hop_headers()` filters what the CALLER passed (`filter_cross_origin_headers`).
- `strip_cross_origin_request_headers()` filters the BUILT request in
  `guarded_fetch_httpx`, `guarded_fetch_httpx_async` and
  `guarded_fetch_requests` — that is the only layer that sees the client
  object's DEFAULT headers (`httpx.Client(headers=...)`, a `requests.Session`'s
  `headers`/`auth`/cookies), which `_hop_headers` never sees. The old code
  popped a fixed four names here; it now applies the same allowlist, so a
  user-named credential set as a client default is covered too (proved by two
  of the born-red tests). Range/conditional headers are unaffected because they
  are on the allowlist.

`guarded_fetch_aiohttp` gets the allowlist through `_hop_headers` only; it has
no built-request object to post-filter, so a credential set as an
`aiohttp.ClientSession(headers=...)` default remains a documented residual —
unchanged by this task, and no live caller does that.

**What this might break — stated plainly.** A *non-credential* custom header a
user configured for a feed (`custom_headers` on a subscription, `custom_headers`
on a `SiteConfig`) also stops being forwarded once that feed redirects
off-origin. Nothing at this layer can tell `X-Feed-Token` from
`X-Client-Version` by name, and that is precisely why the denylist could not
work; being wrong in this direction costs a redirected request some optional
metadata, being wrong in the other direction hands a user's API key to whoever
bought the domain. Concretely reviewed against every live caller: the
subscriptions/watchlists header build (`User-Agent`, `Accept`,
`Accept-Encoding`, `If-None-Match`, `If-Modified-Since`) is entirely
allowlisted; `github_api_client` (`Accept: application/vnd.github.v3+json`,
`User-Agent`) is allowlisted, which matters because release-asset downloads
redirect to `objects.githubusercontent.com`; `Model_Artifacts` resume
(`Range`, `If-Range`) is allowlisted, which matters because catalog→CDN is the
normal artifact download; `Confluence`'s `Content-Type: application/json` is in
`_TRANSPORT_HEADERS`. Same-origin redirects are untouched — the same-origin
branch returns the caller's headers verbatim, so authenticated feeds that
redirect within their own origin keep authenticating (pinned by two tests).

**The hand-mirrored constant: removed, not re-synced.** `Model_Artifacts/fetch.py`
kept its own copy of `_STRIP_HEADERS` with `test_stream_fetch.py` pinning
`set(...) == set(...)`. Re-synchronising the copy would have preserved the
defect shape — a duplicated security constant whose correctness depends on
someone remembering a test. `fetch.py` now imports `CROSS_ORIGIN_SAFE_HEADERS`,
`filter_cross_origin_headers` and `strip_cross_origin_request_headers` from
`egress` and has no local copy; the guard became
`test_cross_origin_header_policy_is_shared_not_mirrored`, asserting object
IDENTITY (`fetch.CROSS_ORIGIN_SAFE_HEADERS is egress.CROSS_ORIGIN_SAFE_HEADERS`)
plus `not hasattr(fetch, "_STRIP_HEADERS")` so re-introducing a mirror fails.
Divergence is now not expressible rather than merely detected late.

**Born-red evidence.** `Tests/Utils/test_egress_cross_origin_header_allowlist.py`
was written and run FIRST against unmodified `origin/dev` (`3193816e7`):
**9 failed, 4 passed**. The failures show the sentinel arriving at the second
origin, e.g.

```
assert 'x-feed-token' not in Headers({'host': 'evil.example', ...,
    'user-agent': 'tldw-chatbook/1.0 (+https://github.com/tld...',
    'x-feed-token': 'sentinel-not-a-real-key-19733'})
```

— that particular one is the **producer** test, driving
`FeedMonitor._fetch_and_parse_feed` with a real `auth_config`
`{"type": "api_key", "header": "X-Feed-Token", ...}` (the feed's own
`User-Agent` in the dump is the tell that this is the production header
build, not a hand-made dict). The default-name case failed identically with
`'x-api-key': 'sentinel-...'`; the `requests` path failed with
`'X-Feed-Token'` present on the second prepared request. The 4 that passed at
base are the deliberate non-regression pins — same-origin redirect keeps the
header, and `User-Agent`/`Accept`/`Accept-Encoding`/`Accept-Language`/
`If-None-Match`/`If-Modified-Since`/`Range` still forward cross-origin — so the
fix could not be "drop everything".

Mutation re-check after implementation (patch saved, `git apply -R`, re-run,
`git apply`, checksums verified byte-identical): **13 failed, 17 passed**,
which additionally covers the two tests written after the fix
(`test_cross_origin_hop_keeps_range_but_drops_custom_credential`, the
identity drift guard). All in-process transports (`httpx.MockTransport`, a
`requests` `BaseAdapter` double); no sockets; every credential value is the
synthetic sentinel `sentinel-not-a-real-key-19733`.

**Two pre-existing tests changed on purpose.**
`test_httpx_cross_origin_hop_strips_credentials` and
`test_hop_headers_strips_x_goog_api_key_cross_origin` both asserted that an
arbitrary custom header (`X-Keep`) SURVIVES a cross-origin hop — they encoded
the old denylist rule as intended behaviour. Each now asserts `X-Keep` is
dropped and an allowlisted header (`Accept`) survives, with a comment naming
this task. These were the only two reds in the whole verified set.

**The five named non-egress siblings: recorded, not fixed** (re-verified at
this base; all still `requests` with default `allow_redirects=True`, and
`requests` strips only `Authorization`/`Cookie` cross-origin):

| site | header | call | why not fixed here |
| --- | --- | --- | --- |
| `LLM_Calls/LLM_API_Calls_Local.py:1010` (Kobold) | `X-Api-Key` | `session.post` `:1060` | POST; `guarded_fetch_*` is GET-only. **Highest risk of the five — `current_api_base_url` is user-configured.** |
| `Web_Scraping/WebSearch_APIs.py:2711` (Bing) | `Ocp-Apim-Subscription-Key` | `session.get` | GET, but re-routing it changes its `Retry`/`HTTPAdapter` session behaviour; user-configurable `bing_search_api_url`. |
| `Web_Scraping/WebSearch_APIs.py:2961` (Brave) | `X-Subscription-Token` | `requests.get` | hard-coded vendor endpoint. |
| `Web_Scraping/WebSearch_APIs.py:3985` (Serper) | `X-API-KEY` | `requests.post` | POST; hard-coded `google.serper.dev`. |
| `Web_Scraping/WebSearch_APIs.py:4055` (Exa) | `x-api-key` | `requests.post` | POST; hard-coded `api.exa.ai`. |

None of them touch `Utils/egress.py`, so fixing the primitive cannot reach
them and folding them in here would mix a shared-primitive change with five
per-call-site rewrites in one diff. Their minimal fix is `allow_redirects=False`
(none of these APIs legitimately redirects), which is a different, testable
change. They were already listed as residue in TASK-19557's notes (items 2 and
3) and remain unfiled — **they need a task of their own; this one is closed
having named them, not having fixed them.**

Kagi (`:3672`) and Yandex (`:4228`) use `Authorization` and are clean, as the
filing said.

**Modified files.** `tldw_chatbook/Utils/egress.py`,
`tldw_chatbook/Model_Artifacts/fetch.py`,
`Tests/Utils/test_egress_cross_origin_header_allowlist.py` (new),
`Tests/Utils/test_egress.py`, `Tests/Model_Artifacts/test_stream_fetch.py`.
