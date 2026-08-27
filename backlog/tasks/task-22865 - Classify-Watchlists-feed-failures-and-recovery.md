---
id: TASK-22865
title: Classify Watchlists feed failures and recovery
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
labels:
  - watchlists
  - feeds
  - network
  - ux
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-feed-and-interface-uat-remediation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Classify common transport and feed failures into stable, actionable, redacted outcomes while retaining the already-shipped product User-Agent as a no-regression contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Feed/source-check receipts classify access denied, authentication required, rate limited, invalid feed, connection failure, temporary server error, and policy-blocked outcomes with stable machine categories.
- [ ] #2 Each category carries a bounded safe next action and relevant status/retry metadata without response bodies, auth/custom headers, signed queries, database paths, or unsanitized exception text.
- [ ] #3 Existing redirect, SSRF, authentication, custom-header, and cross-origin credential-stripping policies remain unchanged.
- [ ] #4 The already-shipped product User-Agent remains present for feed, URL-family, and API requests unless a validated safe source override applies.
- [ ] #5 A local regression fixture reproduces an endpoint that rejects absent/default client identity and proves the product User-Agent path succeeds without any site-specific exception.
- [ ] #6 Monitoring, local service, operation-receipt, redaction, and user-facing recovery tests cover all categories.
<!-- AC:END -->
