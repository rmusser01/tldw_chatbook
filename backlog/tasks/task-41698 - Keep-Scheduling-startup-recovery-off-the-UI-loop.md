---
id: TASK-41698
title: Keep Scheduling startup recovery off the UI loop
status: In Progress
created_date: 2026-11-18 16:06
assignee:
- '@codex'
labels:
- performance
- scheduling
- sqlite
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 16:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first Console frame waits behind synchronous storage recovery. Move the measured finite Scheduling work off the UI loop while preserving the existing recovery order, execution admission and native database ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stale automation runs settle before definition recovery, followed by local reminder discovery. First dispatch waits for that local phase, and an in-flight server listing can finish independently.
- [x] #2 Exact file-backed startup operations use fresh guarded admission on their worker. A held callback retires before cancellation finishes, and custom and in-memory stores retain their existing invocation behavior.
- [x] #3 Three alternating pairs in the existing sealed Terminal fixture retain first input, cold and warm Library, and ordinary retirement evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the current on-mount and post-mount Scheduler entry points, local transfer recovery, worker admission, and quit drain.
2. Keep the DB schema check with construction. Move the existing stale-run transaction, definition recovery and initial reminder read to finite workers for the stock file-backed store. Keep server selection and matching on the current owner.
3. Keep the mount-time recovery task app-owned. Release the Scheduler after the ordered local pass and the existing UI-ready gate. Keep the remote listing independent and recheck execution admission where work actually runs.
4. Exercise real native order, cancellation and admission. Check Scheduler and boot guards, then compare the same three alternating 120×50 PTY pairs with full observation windows.
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md and backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: The existing boot and storage policies apply. The change moves finite local calls and preserves Scheduler, transfer authority, and database boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Moved stale automation-run reconciliation, definition recovery and initial reminder discovery onto finite workers for the exact file-backed Scheduling store. Each worker reacquires and holds the original SQLite execution admission. The app drains recovery on shutdown and gates the scheduler on local completion after UI readiness. Remote reminder matching can finish independently.

Thirteen Scheduling recovery tests, five native checks, the real slow-mount scheduler test, the boot census and three isolated pre-ready exit controls passed. Six paired 120×50 Terminal runs confirmed order and ordinary quit with zero scoped SQLite descriptors. Median UI-thread Scheduling preparation fell from 258 to 66 ms. The QA record retains every measured window, loop-gap misses, the reproduced baseline fixture failures and the private receipt hash.

ADRs 097 and 126 cover the existing boot budget and guarded storage boundaries. The ordered startup path is documented in Docs/Architecture/scheduling-workflows.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
