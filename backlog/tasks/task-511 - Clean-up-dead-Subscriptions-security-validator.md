---
id: TASK-511
title: Clean up or wire dead Subscriptions/security.py validator
status: To Do
assignee: []
created_date: '2026-07-23 12:00'
labels: [subscriptions, cleanup]
dependencies: [task-328]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Subscriptions/security.py` `SecurityValidator.validate_feed_url`/`sanitize_item` and `SSRFProtector` are now policy-correct (delegate to egress) but have ZERO live callers. Either wire `sanitize_item` into the item-ingestion path (post-fetch item-URL validation) or DELETE the dead code. Live fetch-layer SSRF protection is already in place via `guarded_fetch_httpx_async` from TASK-328.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Either wire `sanitize_item` into the active item-ingestion path with live tests OR delete `SecurityValidator` and `SSRFProtector` as dead code
- [ ] No live functionality regresses
<!-- AC:END -->
