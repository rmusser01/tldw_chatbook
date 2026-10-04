---
id: TASK-34356
title: Bound native rule helper request lifetime and usage
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
updated_date: '2026-10-04 07:17'
labels: []
dependencies:
  - TASK-34355
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 3. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Four app-wide and one per-Chat helper slots remain held through physical provider settlement.
- [x] #2 Native helpers obey deadlines and zero transport retries without changing existing auxiliary behavior.
- [x] #3 Late cancelled output is rejected and original-owner usage is charged once.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: implements accepted runtime and provider boundary. Follow Task 3: watched failing physical-lease and provider tests, app-owned capacity and usage-once lifecycle, native-only gateway timeout/output/no-retry changes, targeted ordinary auxiliary regressions and static verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-219 helper custody through physical synchronous-worker finally or genuine async transport cleanup, retaining four app/one Chat reservations after cancelled awaiters. Native calls clamp output to 4096/current model cap, transport to remaining at most 30 seconds, and retries to zero; ordinary auxiliary defaults remain intact. Captured late usage is charged once to the original source; unknown usage remains unknown. App shutdown seals admission while reporting pending physical cleanup. Held-thread, retained-task cancellation, late async usage and slow usage persistence tests pass; Black/Ruff/mypy pass. Broad targeted gateway/runtime runs pass after excluding only 22 cases reproduced against unchanged owners; baseline limitations are recorded in the execution ledger.
<!-- SECTION:NOTES:END -->
