---
id: TASK-31702
title: Wait for real scheduler dispatch before lifecycle shutdown
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:48'
updated_date: '2026-09-05 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove a startup wall-clock race from scheduler lifecycle coverage while preserving real dispatch and bounded shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The lifecycle regression observes an awaited due-reminder handler before requesting stop, even when startup takes longer than ten milliseconds.
- [x] #2 Shutdown remains bounded and is attempted on assertion failure; the complete scheduler loop file passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve isolated baseline and deterministic RED with a 50ms delayed real queue reload. 2. Replace arbitrary 10ms sleep with a bounded asyncio event set by the real AsyncMock handler; always stop and join in finally. 3. Repeat the controlled delay and run the full scheduler loop file, scoped checks, review, and commit. ADR required: no. ADR path: N/A. Reason: test-only lifecycle readiness fix, no scheduler behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the arbitrary ten-millisecond lifecycle sleep with a bounded event set when the due-reminder AsyncMock handler is actually awaited. Shutdown is always requested and joined in finally. The original test failed deterministically with a fifty-millisecond delay around the real queue reload; the corrected test passed the same probe. All 44 scheduler-loop cases passed within the complete 125-test fixture gate in 10.32s (/private/tmp/tldw-31702-31703-31705-first.xml). Scoped Ruff/format and diff checks passed. No production changes or new ADR required.

Parent reviewed the final scoped diff with no actionable findings.
<!-- SECTION:NOTES:END -->
