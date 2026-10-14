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
Tests/UI/test_library_dreams_reader.py::test_production_library_dreams_story_actions_goals_and_literal_notice failed twice in the required check, on two unrelated pull requests (PR #3097 on 2026-10-13, and a branch changing the default web font size on 2026-10-13), each time with 'DreamsStoryModal did not mount #dsm-keep-button', and passed on the re-run. It passed every local run tried (4 of 4 on dev, 4 of 4 on PR #3097's branch, the whole file 3 of 3). A flaky test in the required check can evict a good pull request from the merge queue. Found while re-measuring the required check for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the missing button is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive runs in CI, or fails only for a real defect
- [ ] #3 If the product was at fault, a test that fails deterministically without the fix pins it
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
