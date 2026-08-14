---
id: TASK-16000
title: 'Fix pre-existing red test: test_schedules_ux_fixes fails on dev'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - tests
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_schedules_ux_fixes` is red on dev — surfaced while baselining test failures during the TASK-15450 review (it is NOT attributable to the consolidation; it failed identically at the pre-consolidation base) and is absent from the known-red batch task-15766 filed 2026-08-13. Diagnose whether the test or the production surface drifted, fix accordingly, and if the investigation shows a class of similar drift, note it. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified (test drift vs production regression) with the introducing commit named
- [ ] #2 The test passes, or is corrected to pin current intended behavior with the change justified
- [ ] #3 If a production regression: the fix ships with born-red evidence
<!-- AC:END -->
