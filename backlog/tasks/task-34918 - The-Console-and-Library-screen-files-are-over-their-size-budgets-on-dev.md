---
id: TASK-34918
title: The Console and Library screen files are over their size budgets on dev
status: To Do
created_date: 2026-10-12 22:38
labels:
- architecture
- tests
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The screen size ratchet pins a line budget per screen file that may only go down. Since the Dreams and Guardian merge on 2026-10-12 (PR #2921, 22:07Z) it fails on dev for two files: chat_screen.py at 25,209 lines against 25,204, and library_screen.py at 36,969 against 36,968. The ratchet is not in the required check, so the merge was not blocked, and every later change to either file inherits the red. PR #3091 (TASK-34789) adds 13 lines to chat_screen.py for a product fix and lands on top of it. The owner's rule is that a budget is never raised to fit growth; the files must shrink.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The screen size ratchet passes on dev for chat_screen.py and library_screen.py with their budgets unchanged or lowered
- [ ] #2 The lines removed move into owner modules that tests already cover, with no behaviour change
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
