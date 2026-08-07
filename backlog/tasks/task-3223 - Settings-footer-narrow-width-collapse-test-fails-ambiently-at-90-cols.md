---
id: TASK-3223
title: Settings footer narrow-width collapse test fails ambiently at 90 cols
status: To Do
assignee: []
created_date: '2026-08-07 19:10'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_narrow_footer_collapses_but_f1_help_stays_truthful (Tests/UI/test_settings_footer_hints.py) fails on unmodified dev: at 90 cols the Settings footer's Storage-category hints (s save category | r revert category | t check storage) do not collapse to an ellipsis+globals form the way the test expects -- the full text plus GLOBAL_HINTS_COMPACT still fits under the current width-ladder math. Confirmed ambient (A/B'd against clean HEAD @ c9dfc48cb, unrelated to task-2860's footer dedup fix -- same failure reproduces with AppFooterStatus.py fully reverted to HEAD).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev, or its width assumption is corrected to match the real ladder math
- [ ] #2 No regression to the other test_settings_footer_hints.py cases (F1 help truthfulness, per-category hint sets)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Discovered while running the footer regression sweep for task-2860 (Library footer F6 hint fix). Isolated repro: pytest Tests/UI/test_settings_footer_hints.py::test_narrow_footer_collapses_but_f1_help_stays_truthful -q. Not caused by task-2860 -- confirmed by reverting tldw_chatbook/Widgets/AppFooterStatus.py to HEAD and re-running (still fails identically).
<!-- SECTION:NOTES:END -->
