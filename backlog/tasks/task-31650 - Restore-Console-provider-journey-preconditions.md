---
id: TASK-31650
title: Restore Console provider journey preconditions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:46'
updated_date: '2026-09-05 16:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore durable seed and post-adoption fault injection preconditions in provider journey tests so their existing ownership assertions exercise the intended production paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing chat action routes exercise persisted seeded messages and retain chat-owned provider settings.
- [x] #2 Both vLLM rollback projection cases inject failure after endpoint adoption and verify restoration and replay.
- [x] #3 Affected provider journey tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the two baseline failures and trace durable enrollment and fault injection timing.
2. Correct stale test preconditions without weakening behavior assertions.
3. Run the complete provider Apply/defaults journey file, scoped Ruff/format checks, and review.
ADR required: no
ADR path: N/A
Reason: test-only repair of preconditions for existing persistence and handoff contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored durable seed enrollment for the Retry/Continue/Regenerate/Edit journey, asserting durable user and refreshed assistant records. Armed vLLM sync failures only after the real endpoint adoption, because pre-adoption summary capture legitimately synchronizes core state; retained restoration and replay assertions and allowed additional nested rollback sync calls. Both failures reproduced before the repair. All 34 provider Apply/defaults journeys passed in67.75s. Scoped Ruff, changed-range formatting, diff checks, and independent review passed. No production changes or new ADR required.
<!-- SECTION:NOTES:END -->
