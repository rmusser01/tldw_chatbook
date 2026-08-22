---
id: TASK-19733
title: >-
  Shared egress primitive does not strip x-api-key on cross-origin redirects
status: To Do
assignee: []
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

- [ ] A credential header carrying a user-configured API key (default
      `X-API-Key` **and** a custom header name the user chose) is absent from
      the request when a guarded fetch follows a redirect to a different origin
- [ ] The cross-origin rule is expressed so that a newly-introduced credential
      header name is safe by default — a header the caller supplies is not
      forwarded cross-origin unless it is on an explicit non-credential
      allowlist — rather than by adding another literal to a denylist
- [ ] Same-origin redirects still carry the header, so authenticated feeds that
      redirect within their own origin keep working
- [ ] Born-red tests drive an actual cross-origin redirect through both
      `guarded_fetch_httpx` and `guarded_fetch_httpx_async` and assert the
      credential header is absent on the second hop, including one case with a
      non-default header name; each is mutation-checked (restoring the old
      behaviour makes it red)
- [ ] At least one test exercises the real producer path
      (`monitoring_engine`'s `api_key` auth config or a `SiteConfig` scraper),
      not only `_hop_headers` in isolation
- [ ] `Model_Artifacts/fetch.py`'s mirrored tuple and
      `Tests/Model_Artifacts/test_stream_fetch.py`'s drift guard are updated in
      the same change and remain green
- [ ] The five named non-egress siblings (Kobold, Bing, Brave, Serper, Exa) are
      either fixed the same way or recorded — with the reason — as a separate
      tracked item; the task is not closed with them silently unaddressed

## Notes

Deliberately scoped apart from TASK-19557: that task fixes two clients that do
not use the primitive. Fixing the primitive does not fix them, and fixing them
does not fix the ~25 call sites that route through `Utils/egress.py`. Both are
needed; neither subsumes the other.
