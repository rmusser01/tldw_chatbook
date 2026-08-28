---
id: TASK-22865
title: Classify Watchlists feed failures and recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-28 13:57'
labels:
  - watchlists
  - feeds
  - network
  - ux
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-watchlists-feed-and-interface-uat-remediation.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add table-driven RED tests for the exact stable failure vocabulary, bounded retry metadata, fixed safe messages/actions, unknown fallback, and absolute exclusion of raw exception, response, URL, header, credential, and path content.
2. Implement one pure Watchlists failure classifier and route scheduled checks, Check Now, durable run `stats_json`, source `last_error`, normalizers, status tools, and user-facing recovery through that shared safe outcome without changing receipt storage.
3. Add local `httpx.MockTransport` regressions for the existing product User-Agent across feed, URL-family, and API builders, including redirects and validated safe overrides while preserving SSRF, auth, custom-header, and cross-origin credential stripping.
4. Add actionable recovery tests for retryable versus non-retryable categories across local service, operations receipts, Runs/Inspector/Console, and guide copy; legacy rows without machine fields retain the generic failure state.
5. Run task-targeted classifier/monitoring/service/transport/UI/tool tests, Ruff, diff checks, self-review, and independent review.

ADR required: no
ADR path: N/A
Reason: This task uses the existing durable run receipt, monitoring ownership, HTTP egress policy, and redaction boundaries. It adds a stable classification projection and regression coverage without changing storage, security authority, provider boundaries, or runtime ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a pure seven-category Watchlists failure classifier plus one validated
public recovery projection shared by durable run normalization, Console operation
receipts, Check Now toasts, and the Runs pane. Failed writes now retain only bounded
machine metadata and validated accounting, use fixed safe copy in run/source rows,
and keep legacy or tampered rows generic and non-retryable. URL-list aggregation is
order-independent: mixed categories are generic, while a uniform category carries
HTTP status and Retry-After only when every actual failure agrees; skipped-in-flight
items remain separate counters.

Monitoring now classifies malformed/non-feed payloads, bounds Retry-After before
conversion, preserves wrapped policy failures, and keeps the exact shipped Chatbook
User-Agent across feed, page, sitemap, and API builders. Existing guarded-fetch,
redirect, SSRF, authentication, custom-header, and cross-origin credential authority
remain owned by `Utils/egress.py` and were not changed. The Watchlists guide documents
fixed recovery actions and when unchanged-input re-run is available. ADR required:
no; the existing run receipt, service ownership, and egress/redaction boundaries are
unchanged.

Targeted evidence: 310 backend/monitoring/service/tool/Check Now tests passed; 47 fresh
Runs/toast/egress-preservation tests passed; a broader mounted Runs/Collections pass
recorded 123 passed with three independently reproduced unrelated nodes excluded.
The anti-vacuous mutation check failed after deliberately leaking `str(error)` and
passed after restoration. Ruff, the Impeccable detector, and `git diff --check` pass.
The task intentionally remains In Progress with all ACs unchecked for independent
review and UAT.

Review round 1 remediation removed raw exception persistence/logging from the
scheduled outer and fallback paths, separated true egress policy blocks from generic
guarded-fetch container failures, and replaced exception-name matching with concrete
owned types. Operation recovery now follows terminal failed status, nested JSON Feed
shapes fail as `invalid_feed`, and feed-specific 401/429 exceptions retain only their
validated structured status (plus bounded Retry-After for 429). Mutation-resistant
MockTransport, direct/wrapped classifier, SQLite receipt, and scheduler canary probes
cover each distinction. A final review additionally pinned and fixed terminal
scheduler-write traceback leakage plus forged structured-failure copy, then approved
the resulting category-only boundaries. Updated targeted evidence is 384 task-path
tests and 47 Runs/toast/egress-preservation tests passing; Ruff and diff checks remain
clean.
<!-- SECTION:NOTES:END -->
