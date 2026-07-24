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

Only live instance today: `Utils/github_api_client.py` sets `Authorization: token <PAT>` as an httpx **client-default header** and calls the guarded helper with no per-call `headers`. Not exploitable currently (fixed vendor host `api.github.com`, JSON endpoints, no attacker-controllable cross-origin redirect), but it is a footgun for TASK-506 (image-gen adopting this module) and generalizes the known aiohttp-session-auth caveat to the httpx client level. `Subscriptions/scrapers/github_scraper.py` already does it correctly (token via the helper's `headers=` param).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Either (a) the guarded httpx/aiohttp helpers strip client/session-default `Authorization`/`Cookie`/`Proxy-Authorization` and suppress client/session `auth` on cross-origin hops, OR (b) `github_api_client` passes its token via the helper's `headers=` param and a docstring contract on the helpers requires credentials to be attached only via the helper's `headers=`/`auth=` params
- [ ] A test proves a client/session-level credential does NOT follow a cross-origin redirect through the guarded helper
- [ ] No live functionality regresses
<!-- AC:END -->
