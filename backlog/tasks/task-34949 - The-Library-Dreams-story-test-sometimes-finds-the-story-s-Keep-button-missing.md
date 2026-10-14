---
id: TASK-34949
title: The Library Dreams story test sometimes finds the story's Keep button missing
status: To Do
created_date: 2026-10-14 13:47
labels:
- tests
- flaky
- dreams
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_dreams_reader.py::test_production_library_dreams_story_actions_goals_and_literal_notice failed three times in the required check with 'DreamsStoryModal did not mount #dsm-keep-button': on feat/dreams-phase-4 (PR #2921, 2026-10-12), on PR #3097 (2026-10-13) and on the branch of PR #3020 (2026-10-13). It passed every local run tried (4 of 4 on dev, 4 of 4 on PR #3097's branch, the whole file 3 of 3). Commit aea8948fdb ('test(library): wait for Dreams action hit targets'), which reached dev with PR #3020 on 2026-10-13 22:58Z, rewrote this test to wait for the actions button to be a real hit target and to assert that the click lands, which is exactly the failure above; all three failures predate it and none has been seen since. A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 20 consecutive CI runs after aea8948fdb, or, if it fails again, the cause of the missing button is identified in the test or the product
- [ ] #2 If the product was at fault, a test that fails deterministically without the fix pins it
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
