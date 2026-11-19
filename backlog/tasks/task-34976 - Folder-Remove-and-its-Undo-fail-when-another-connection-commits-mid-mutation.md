---
id: TASK-34976
title: Folder Remove and its Undo fail when another connection commits mid-mutation
status: To Do
created_date: 2026-10-12 04:58
labels:
- notes
- data-integrity
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Removing a Notes folder and undoing the removal each read the folder and then write it inside one transaction that starts without the write lock. If any other connection commits to the same database between that read and the write, SQLite refuses the lock upgrade at once, without waiting, and the Library reports that the folder changed during the mutation. A retry works. The same shape was fixed for note delete and its Undo in PR #3090 (TASK-34789), where it was seen three times in local test loops; the folder paths were found by that PR's review from the code and were not reproduced. Affected: the folder repository's remove and restore, and the synchronized-profile wrapper that runs every folder mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Removing a folder succeeds when another connection commits between the mutation's read and its write
- [ ] #2 Undoing a folder removal succeeds under the same interleaving
- [ ] #3 A folder mutation on a synchronized profile succeeds under the same interleaving
- [ ] #4 A test drives the interleaving directly (no timing) for each of the three and fails on the code before the fix
- [ ] #5 A mutation that is nested in a caller's transaction behaves as it does today
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

Renumbered from task-34808 to TASK-34976 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
