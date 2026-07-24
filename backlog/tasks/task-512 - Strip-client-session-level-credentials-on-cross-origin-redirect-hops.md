---
id: TASK-512
title: Strip client/session-level credentials on cross-origin redirect hops
status: To Do
assignee: []
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
- [ ] Either (a) the guarded httpx/aiohttp helpers strip client/session-default `Authorization`/`Cookie`/`Proxy-Authorization` and suppress client/session `auth` on cross-origin hops, OR (b) `github_api_client` passes its token via the helper's `headers=` param and a docstring contract on the helpers requires credentials to be attached only via the helper's `headers=`/`auth=` params
- [ ] A test proves a client/session-level credential does NOT follow a cross-origin redirect through the guarded helper
- [ ] No live functionality regresses
<!-- AC:END -->
