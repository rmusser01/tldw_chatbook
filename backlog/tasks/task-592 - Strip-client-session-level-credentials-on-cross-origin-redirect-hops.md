---
id: TASK-592
title: Strip client/session-level credentials on cross-origin redirect hops
status: Done
assignee: [rmusser01]
created_date: '2026-07-23 12:00'
labels: [web, security, followup]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The guarded fetch helpers in `Utils/egress.py` strip `Authorization`/`Cookie`/`Proxy-Authorization` on cross-origin redirect hops, but only for credentials passed *through* the helper's `headers=`/`auth=` params (`_hop_headers`, and the `requests` helper's prepared-request auth suppression). Credentials attached at the transport-object level are invisible to this guard: an `httpx.Client`/`AsyncClient` default header or client-level `auth=`, and an `aiohttp.ClientSession(auth=...)`, are re-applied by the library on every hop — including a cross-origin one.

**UPDATE (PR #822):** the httpx **client-default header** case is now FIXED — `guarded_fetch_httpx`/`guarded_fetch_httpx_async` pop `Authorization`/`Cookie`/`Proxy-Authorization` off the *built* request on cross-origin hops, which strips client-default headers as well as per-call ones (regression-tested). REMAINING residual for this task: (1) `aiohttp.ClientSession(auth=...)` session-level BasicAuth is re-applied per hop and not suppressed (no live caller attaches auth to the aiohttp session — crawler uses a bare `ClientSession()`); (2) an httpx client-level `auth=` CALLABLE flow (not a default header) is not suppressed on cross-origin hops (no live caller uses it). `Utils/github_api_client.py` set `Authorization` as a client-default header, which is now covered by the PR #822 fix; `Subscriptions/scrapers/github_scraper.py` already passes the token via the helper's `headers=` param.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Either (a) the guarded httpx/aiohttp helpers strip client/session-default `Authorization`/`Cookie`/`Proxy-Authorization` and suppress client/session `auth` on cross-origin hops, OR (b) `github_api_client` passes its token via the helper's `headers=` param and a docstring contract on the helpers requires credentials to be attached only via the helper's `headers=`/`auth=` params
- [x] A test proves a client/session-level credential does NOT follow a cross-origin redirect through the guarded helper
- [x] No live functionality regresses
<!-- AC:END -->

## Implementation Plan

1. Re-audit the residuals against current code: (i) httpx client-level `auth=` — task-19733 already passes an explicit `auth=None` on cross-origin hops in both httpx helpers (tuple form is regression-tested; the CALLABLE `httpx.Auth` flow named in this task's update is not); (ii) aiohttp `ClientSession(auth=...)`/session-default headers — verified against aiohttp 3.14: applied inside `session.get()`, no per-request override exists (aiohttp even raises on combining an Authorization header with auth), so stripping is impossible and no live caller attaches session credentials (crawler builds a bare session).
2. httpx: add pin tests that a client-level `httpx.Auth` CALLABLE does not authenticate a cross-origin hop (sync + async), plus one test pinning the actual `github_api_client._build_client` header shape (client-default `Authorization: token ...`) through a cross-origin redirect. Expect green (mechanism already landed); the tests close the documentation gap this task filed.
3. aiohttp: fail closed. On a cross-origin hop, if the session carries a session-level credential — `session.auth` or any non-forwardable session-default header (public properties in aiohttp >=3.9; read defensively so duck-typed test fakes keep working) — raise `EgressFetchError` instead of forwarding the credential. Red tests first: session `auth=BasicAuth` and a session-default `X-Feed-Token` must each abort with the second host never contacted; a bare session cross-origin fetch must keep working (crawler path).
4. Docstring contract on all four guarded helpers: credentials are supplied via `headers=` (or the helper's `auth=` param); client/session-object-level credentials are stripped on cross-origin hops (httpx/requests) or make the helper refuse the hop (aiohttp).
5. Targeted runs: `Tests/Utils/test_egress_cross_origin_header_allowlist.py` and `Tests/Utils/test_egress.py` (aiohttp sections); `Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py` for the aiohttp caller.

ADR required: no
Reason: No new interface, storage, or policy surface — tightening existing helpers in place, and the fail-closed aiohttp refusal implements the same existing credential-forwarding rule (fail closed where stripping is impossible).

## Implementation Notes

- **Approach** — code reality at start differed from the filing's update in one respect and matched it in another:
  - httpx residual (client-level `auth=` CALLABLE): already suppressed — task-19733 passes an explicit `auth=None` to `send()` on cross-origin hops in BOTH httpx helpers, which covers tuple and `httpx.Auth`-flow auth alike. The tuple form was regression-tested; this task added the callable-flow pin (sync + async). Both green with no code change.
  - aiohttp residual (`ClientSession(auth=...)` + session-default headers): confirmed live. Verified against aiohttp 3.14: both are applied inside `session.get()`, there is no built-request object to post-filter, no per-request suppression exists, and aiohttp raises `ValueError` if an explicit `Authorization` header is combined with `auth=` — so strip/inject workarounds are closed off by the library. Suppression is therefore impossible without mutating caller-owned session state mid-flight (unsafe for concurrent use of one session). The helper now FAILS CLOSED: a cross-origin hop on a session carrying `session.auth` or any non-forwardable session-default header (classified with the same `_may_cross_origin` rule) raises `EgressFetchError` before the request is sent. Reads use the public `session.auth`/`session.headers` properties, defensively (`getattr`) so duck-typed test fakes without them keep working.
  - Delivered outcome per AC#1: (a) holds for the httpx/requests helpers (strip + suppress); for aiohttp the equivalent guarantee is refusal plus the (b)-style documented contract (`_CREDENTIAL_CONTRACT`, now in all four guarded helpers' docstrings: credentials only via `headers=`/`auth=` parameters). `github_api_client` passes its token as a client-DEFAULT header (not `headers=`); that shape is exactly what PR #822's built-request strip covers, and this task pinned it end to end with the `_build_client` header shape through a cross-origin redirect.
- **Evidence (red -> green)**:
  - Red: `.venv/bin/python -m pytest Tests/Utils/test_egress_cross_origin_header_allowlist.py -k aiohttp` before the fix -> `2 failed` (`test_aiohttp_session_level_auth_does_not_follow_cross_origin_redirect`, `test_aiohttp_session_default_credential_header_does_not_follow_cross_origin_redirect` — the fake session mimics aiohttp 3.14's apply-inside-`get()` semantics, so the credential genuinely reached the second origin).
  - Green after the refusal: the same file + all of `Tests/Utils/test_egress.py` -> `141 passed`; the four caller suites + census + skills file -> `261 passed, 1 failed` (the 1 is `test_sitemap_crawl_trusted_origins.py::test_crawl_site_does_not_trust_discovered_links`, verified red at baseline HEAD with this branch's egress.py reverted — a pre-existing crawler/link-discovery failure unrelated to credentials, surfaced only once the admission reds cleared; belongs to the crawler owner).
  - No live regressions: the crawler's bare-`ClientSession()` cross-origin path is pinned green (`test_aiohttp_bare_session_cross_origin_fetch_still_works`), and same-origin fetches keep session credentials (`test_aiohttp_session_credentials_still_work_same_origin`).
- **Tests added** (all in `Tests/Utils/test_egress_cross_origin_header_allowlist.py`): client-level callable-auth pins (sync+async); the `github_api_client` production header-shape pin; aiohttp session-auth refusal; aiohttp session-default-header refusal; aiohttp same-origin-keeps-credentials; aiohttp bare-session cross-origin still works.
- **Test-infra restoration (pre-existing reds, not this task's logic)**: all SSRF/egress suites had been red since TASK-32628's config-admission landed — every `guarded_fetch_*` call reads `get_cli_setting` (`check_url_or_raise` -> `_config_enabled`) and trips `RecoveryRequired("raw_source_selection_changed")` under the per-test sandbox; none of the files is in the PR fast lane, so it went unnoticed (20 red in the allowlist file, 15 in `test_skill_remote_fetch.py`, 8 in `test_egress.py`, plus 4 caller suites; verified identical with this branch's changes reverted and with a fresh HOME — structural, not machine state). Enrolled the seven affected files in `keep_bootstrap_profile` in `Tests/conftest.py`, exactly the pattern of commit 982265b3f8 / TASK-32873 / ADR-179 Task 11 (verified none of them re-selects a config).
- **Files changed**: `tldw_chatbook/Utils/egress.py` (aiohttp fail-closed refusal + `_aiohttp_session_level_credential` + `_CREDENTIAL_CONTRACT` docstring contract on all four helpers), `Tests/Utils/test_egress_cross_origin_header_allowlist.py` (6 new tests), `Tests/conftest.py` (admission enrollment for 7 egress/caller suites), this task file.
- **ADR required**: no (see plan).
