---
id: TASK-34987
title: The Watchlists screen file is over its module-size budget on dev
status: To Do
created_date: 2026-10-14 16:21
labels:
- architecture
- tests
- watchlists
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Architecture/test_module_size_ratchet.py pins a line budget per large module that may only go down. tldw_chatbook/UI/Screens/watchlists_collections_screen.py was 14,378 lines on dev on 2026-10-14 against a budget of 14,324, so the ratchet fails on dev and every change to the file inherits the red (PR #3107, which fixes a Watchlists race for TASK-34951, brings it to 14,374, still 50 over). The ratchet is not in the required check, which is why it went unnoticed. The owner's rule is that a budget is never raised to fit growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The module size ratchet passes on dev for watchlists_collections_screen.py with its budget unchanged or lowered
- [ ] #2 The lines removed move into Watchlists modules that tests already cover, with no behaviour change
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
