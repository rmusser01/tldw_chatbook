---
id: TASK-33374
title: 'Keystroke census scale test is flaky: context_estimate_max_rows reads 0 or
  1'
status: To Do
created_date: 2026-09-28 20:12
dependencies:
- TASK-33260
labels:
- testing
- performance
- flaky
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Performance/test_console_keystroke_work_census.py::test_keystroke_work_does_not_scale_with_transcript_length failed 4 of 4 standalone runs at 9cd9aad65f (context_estimate_max_rows 0 vs 1), yet passed in two full-suite runs. The 2026-09-27 audit's Console probe saw the same thing and noted that the test's strict equality is tighter than its own <=1 bound. The PERF-01 agent suspected a debounce timer firing mid-burst but did not prove it. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root cause is identified with evidence
- [ ] #2 The test is deterministic: 20 of 20 standalone runs and full-suite runs agree
- [ ] #3 Any budget change keeps the property the test exists to pin (per-key work does not scale with transcript length)
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
