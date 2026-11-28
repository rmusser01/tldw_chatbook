---
id: TASK-34978
title: Research workspace deletes a note on the UI loop
status: Done
created_date: 2026-10-12 04:58
labels:
- research-workspace
- performance
priority: medium
assignee:
- '@codex'
updated_date: 2026-11-24 16:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deleting a note from the Research workspace runs the database write on the app's event loop instead of a worker thread. Since PR #3090 (TASK-34789) the delete asks for the write lock before reading, so when another writer holds that lock the whole UI waits, in the worst case for the 15 second busy timeout, where it used to fail at once. The Library runs the same delete on a worker thread and is not affected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deleting a note from the Research workspace does not block the UI while another connection holds the database write lock
- [x] #2 The delete still completes, or reports its failure, once the lock is released or the wait times out
- [x] #3 A test holds the write lock on another connection during the delete and shows the event loop keeps running
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
Reason: Keep the existing canonical Notes delete and durable membership receipt. Match the local worker boundary already used by Library.

1. Reproduce the event-loop stall in the Local Research SQLite round trip by holding a second connection at BEGIN IMMEDIATE.
2. Isolate the existing canonical delete in a finite worker for both direct deletion and receipt reconciliation. Preserve the optimistic version and cleanup order.
3. Verify event-loop progress, SQLite tombstone and membership cleanup, and existing owner conflict and receipt cases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The Local Research adapter dispatches the canonical Notes delete through a finite worker for both direct deletes and receipt recovery. Version checks, error propagation, and membership cleanup stay in the existing owners. RED held a real SQLite writer lock until the event-loop-side delete timed out. GREEN observed the loop while SQLite waited, then verified the tombstone and removal of both workspace memberships. All 119 targeted Research Quick Notes and Notes cases passed. The aggregate descriptor warning also reproduced with the original synchronous delete and the new regression deselected. ADR-322 applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from task-34810 to TASK-34978 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
