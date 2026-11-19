---
id: TASK-38607
title: Harden scheduler pop_due against naive timestamps
status: Done
created_date: 2026-11-02 14:29
labels:
- bug
- scheduling
priority: low
updated_date: 2026-11-03 02:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A naive run_at string raises TypeError inside pop_due which terminates the scheduler worker on every restart; sibling code already normalizes naive stamps. Only reachable with hand-edited data today. See bug-sweep-2026-11-02-dev.md services long tail findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Naive run_at strings are normalized to UTC or skipped like load filters malformed rows
- [x] #2 The scheduler worker survives a hand-edited poison row
- [x] #3 Test covers a naive timestamp
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
_coerce_run_at normalizes naive stamps to UTC and skips junk rows with a warning, mirroring the sibling db parser; poison-row test drains the queue. Commit ec0f348363.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
