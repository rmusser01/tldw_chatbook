---
id: TASK-33120
title: Theme switches skip the stylesheet re-parse when returning to a theme
status: To Do
assignee: []
created_date: '2026-09-28 00:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-33075 profiling (2026-09-27, 211x44, TLDW_TEST_CSS_CACHE=0): a theme switch costs ~0.9-1.2 s, ~430 ms of it Textual re-parsing the ~830 KB stylesheet bundle, identical across screens. A parse cache keyed on the theme variables on the app's TieAwareStylesheet would make returning to a theme (Try then Revert, cycling rows) skip that cost; a first use would still pay it. Pytest's Tests/UI/css_cache.py hides this cost — probes need TLDW_TEST_CSS_CACHE=0.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Returning to a previously used theme does not re-parse the stylesheet
- [ ] #2 The first use of a theme still renders correctly
- [ ] #3 A measurement shows the saved time
<!-- AC:END -->
