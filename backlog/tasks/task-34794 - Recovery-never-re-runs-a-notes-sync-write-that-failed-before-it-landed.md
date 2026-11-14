---
id: TASK-34794
title: Recovery never re-runs a notes sync write that failed before it landed
status: To Do
created_date: 2026-10-11 17:52
labels:
- notes-sync
- data-integrity
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an automatic notes sync write fails after it was admitted but before its bytes reach the file, the operation is left for Recovery. After Recovery settles it, the folder reads up to date and offers Sync now, but the file never receives that note version and stays stale until the note is edited again. The operation is identified by the state that was observed when it was planned, the settle keeps the reviewed baseline, and the pass that follows plans the same operation again, finds it recorded as completed, and returns without writing. Found while fixing TASK-34789: reproduced by failing one guarded replace before its write and then resolving the cleanup; the result was no open entry, an up-to-date folder, and the old bytes in the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Recovery settles an operation whose write never landed, the file receives the note's current version without another edit to the note
- [ ] #2 A folder is never reported up to date while a bound file holds an older note version than the database
- [ ] #3 An operation whose write did land is not written a second time by the settle
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
