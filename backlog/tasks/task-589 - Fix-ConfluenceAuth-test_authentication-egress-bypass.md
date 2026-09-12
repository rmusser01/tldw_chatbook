---
id: TASK-589
title: Fix ConfluenceAuth.test_authentication egress bypass
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-23 12:00'
labels: [web, security]
dependencies: [task-328]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`confluence_auth.ConfluenceAuth.test_authentication()` calls `self.session.get(base_url+/rest/api/user/current)` directly, bypassing `make_request` and thus the egress guard (including metadata hard-block). Route it through guarded_fetch_requests to apply SSRF protection consistently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] `test_authentication` uses guarded_fetch_requests or similar to validate the base_url against SSRF policy
- [x] Metadata IPs and private ranges are blocked even in test_authentication calls
<!-- AC:END -->

## Implementation Plan

1. Re-point the two existing `test_authentication` tests at the `guarded_fetch_requests` module seam (same mock pattern as `test_make_request`), preserving their 200/401 semantics.
2. Add regression tests: probe routes through the guard (raw `Session.get` never called; session/URL/timeout threaded through) and an egress rejection returns False.
3. Route `test_authentication` through `guarded_fetch_requests` with `trusted_origins=origin_set(base_url)`, `MAX_FETCH_BYTES_PAGE`, timeout 10, mirroring `make_request`'s GET path.
4. Run `Tests/Web_Scraping/Confluence/`; ruff-check touched files against the pre-change baseline.

ADR required: no
ADR path: N/A
Reason: Closes an SSRF-protection bypass at one call site using the module's established guard pattern; no new boundary or policy.

## Implementation Notes

`ConfluenceAuth.test_authentication` now fetches `/rest/api/user/current` via `guarded_fetch_requests` (session, `MAX_FETCH_BYTES_PAGE`, `trusted_origins=origin_set(self.base_url)`, timeout 10, JSON Accept header) instead of a raw `self.session.get`, mirroring `make_request`'s guarded GET path. The dropped `_session_lock` hold matches `make_request`'s GET path, which also calls the guard without the lock. Egress-policy raises hit the existing `except Exception` and return False (fail-closed).

TDD evidence: on unmodified dev the re-pointed tests failed AND the repo's autouse no-network guard (task-15111) flagged teardown errors showing the old path attempting a real socket connect to `example.atlassian.net:443` -- the bypass made concrete. After the fix: `Tests/Web_Scraping/Confluence/` 38 passed. Four `test_authentication` tests now pin the contract: success, 401, guard-routed (raw `Session.get` asserted not called), and egress-blocked returns False. Ruff: zero delta on `confluence_auth.py` (7 pre-existing fixables at HEAD, unchanged); test file clean after autofix.

Modified: `tldw_chatbook/Web_Scraping/Confluence/confluence_auth.py`, `Tests/Web_Scraping/Confluence/test_confluence_auth.py`.