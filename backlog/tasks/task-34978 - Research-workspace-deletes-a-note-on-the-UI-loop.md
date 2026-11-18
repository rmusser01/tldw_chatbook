---
id: TASK-34978
title: Research workspace deletes a note on the UI loop
status: To Do
created_date: 2026-10-12 04:58
labels:
- research-workspace
- performance
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deleting a note from the Research workspace runs the database write on the app's event loop instead of a worker thread. Since PR #3090 (TASK-34789) the delete asks for the write lock before reading, so when another writer holds that lock the whole UI waits, in the worst case for the 15 second busy timeout, where it used to fail at once. The Library runs the same delete on a worker thread and is not affected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deleting a note from the Research workspace does not block the UI while another connection holds the database write lock
- [ ] #2 The delete still completes, or reports its failure, once the lock is released or the wait times out
- [ ] #3 A test holds the write lock on another connection during the delete and shows the event loop keeps running
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
