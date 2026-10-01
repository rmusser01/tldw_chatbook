---
id: TASK-32680
title: Schedule bounded Stop continuations and teardown
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:25'
updated_date: '2026-10-01 23:07'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32679
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow useful automatic follow-up while preserving user priority, cancellation and finite scheduler ownership.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stop proposals combine into at most one scheduler turn keyed by parent turn and event, with three-turn and 120-second chain caps and inherited restrictions.
- [x] #2 Foreground user work, vetoes, update drain, revocation, closure and uncertain dispatch prevent stale continuation or automatic replay; continuations do not fire UserPromptSubmit.
- [x] #3 Interrupt and SessionEnd are bounded observations after admission sealing, cannot prompt/connect or extend cleanup, and revoked plugin handlers are suppressed.
- [x] #4 Mounted and viewless tests exercise continuation admission, deduplication, concurrent settlement and teardown with legacy hook behavior preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
Reason: implements accepted scheduler continuation admission, durable deduplication receipts and teardown custody through existing queue/runtime owners.
1. Port reviewed H5 fixtures with current profile/worker/resource ownership and establish real scheduler RED after accepted ordinary Send.
2. Integrate reviewed H5 increment preserving maintenance gates, dispatch recovery, exact hook consent, transformed input carriers and current schema contracts.
3. Qualify bounded chains, inherited budgets, human priority, concurrency/deduplication and uncertain dispatch, actual mounted Send/Stop at supported sizes, viewless teardown and migration/reopen.
4. Run targeted legacy and affected queue/dispatch neighbors, token governance, lint/format and shared diagnostic comparison; self-review ACs, record evidence/limits, complete via CLI and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded Stop scheduling through existing queue/runtime owners, atomic v74 deduplication receipts, inherited budgets, untrusted machine input and fixed Interrupt/SessionEnd observation windows. Preserved actual cancellation and maintenance custody; reused finite worker connection retirement. Final targeted H5 qualification: 105 passed in 453.52s, no warnings/skips; 133 queue/dispatch/maintenance controls passed and 1,000 real turns passed in 465.05s. Additional archive/fleet/log controls: 85 passed, two failures reproduced on frozen pre-H5 production; pre-existing dispatch/Stop fixture failures also recorded without weakening guards. New hooks/tests Ruff+format, parse, whitespace and shared diagnostic comparison pass; exact v74 wheel resource verified. ADR-162/163/063 apply. Evidence and platform limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.

PR #2946 derived-artifact repair registers the existing continuation table in the SQL identifier allowlist and pins the actual populated conversation DELETE cascade query plan without ANALYZE. Migration/runtime ownership is unchanged; the focused repaired-runtime plus migration group passes 7 cases. Existing ADR-163 applies; exact evidence is in the integration report.
<!-- SECTION:NOTES:END -->
