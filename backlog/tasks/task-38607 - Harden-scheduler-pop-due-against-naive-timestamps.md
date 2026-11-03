---
id: TASK-38607
title: Harden scheduler pop_due against naive timestamps
status: In Progress
created_date: 2026-11-02 14:29
labels:
- bug
- scheduling
priority: low
updated_date: 2026-11-02 14:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A naive run_at string raises TypeError inside pop_due which terminates the scheduler worker on every restart; sibling code already normalizes naive stamps. Only reachable with hand-edited data today. See bug-sweep-2026-11-02-dev.md services long tail findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Naive run_at strings are normalized to UTC or skipped like load filters malformed rows
- [ ] #2 The scheduler worker survives a hand-edited poison row
- [ ] #3 Test covers a naive timestamp
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
