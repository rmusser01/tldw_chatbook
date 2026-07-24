---
id: TASK-509
title: Fix ConfluenceAuth.test_authentication egress bypass
status: To Do
assignee: []
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
- [ ] `test_authentication` uses guarded_fetch_requests or similar to validate the base_url against SSRF policy
- [ ] Metadata IPs and private ranges are blocked even in test_authentication calls
<!-- AC:END -->
