---
id: TASK-34809
title: Note save and other read-then-write Notes transactions fail at once when another
  connection commits
status: To Do
created_date: 2026-10-12 04:58
labels:
- notes
- data-integrity
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Many transactions in the main database read a row and then update it inside a transaction that starts without the write lock (saving a note is the most used one). When another connection commits between the read and the write, SQLite refuses the lock upgrade immediately, with no wait, and the caller reports a failure for an edit that would have succeeded a moment later. PR #3090 (TASK-34789) fixed note delete and its Undo. 153 other transactions in that module start the same way; how many of them read before they write has not been counted. That PR changed none of them because each one needs its callers checked first: taking the write lock up front makes the call wait for other writers, which matters for any caller on the UI thread.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving a note succeeds when another connection commits between its version read and its write, proven by a test that drives the interleaving directly and fails before the fix
- [ ] #2 Every read-then-write transaction on the Notes paths the Library and Console can reach is listed with a decision: takes the write lock up front, or stays as it is with the reason
- [ ] #3 No call that now waits for the write lock runs on the UI thread
- [ ] #4 Callers that relied on failing fast, if any, are named and keep working
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
