---
id: TASK-35648
title: The Library panels sheet is over the 2,000 active-line CSS ceiling on dev
status: To Do
created_date: 2026-10-20 04:30
dependencies: []
labels:
- follow-up
- css
- tests
priority: medium
references:
- tldw_chatbook/css/features/_library_panels.tcss
- Tests/UI/test_component_pattern_governance.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of Phase 9's rebase onto dev 88fe3108e5 (2026-10-20). `Tests/UI/test_component_pattern_governance.py::test_sheet_size_ceiling` fails on plain dev: `css/features/_library_panels.tcss` has 2,004 comment-stripped active lines against the 2,000 ceiling (`SHEET_ACTIVE_LINE_CEILING`, ADR-161 spec 3.5, TASK-24451). The file has 2,982 lines in all. Its recent growth is TASK-34000.25's Library Reader work (47f5bbea32, 8b9041834f, bbc7c2c6d2). The test runs in no PR lane (not in scripts/ui_pr_gate_census.txt or any workflow), so dev's CI never reported it. The ceiling exists to stop the next 12,000-line grab-bag sheet; the test says a sheet over it splits by vocabulary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_sheet_size_ceiling passes on dev with the ceiling unchanged
- [ ] #2 The Library panels rules are split by vocabulary into sheets that each stay under the ceiling, and the CSS bundle still reproduces (preflight)
- [ ] #3 The Library screens render the same at 211x44 and 235x52 before and after the split
- [ ] #4 The ceiling test runs in a PR lane, so the next sheet that crosses it fails a PR
<!-- AC:END -->
