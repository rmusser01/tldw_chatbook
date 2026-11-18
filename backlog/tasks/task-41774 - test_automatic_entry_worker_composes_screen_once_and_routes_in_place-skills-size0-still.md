---
id: TASK-41774
title: "test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size0] still expects the empty Skills work pane at 60x20, which TASK-38981's list-first rule replaced"
status: To Do
created_date: 2026-11-18 17:00
labels:
- follow-up
- architecture-review-2026-10-25
- library
- tests
priority: low
references:
- Tests/UI/test_library_entry_compose_once.py
- tldw_chatbook/UI/Library_Modules/screen_constants.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_entry_compose_once.py::test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size0]` fails at dev 908e9840aa with "skills terminal copy was not painted" (`:1061`).

**What fails.** At 60x20 the Skills case expects the empty Work pane's copy, "Select a skill to inspect it here." (`:1040-1043`, commented "paints the Work pane until the user opens Items explicitly"). The test's first wait (`_entry_worker_terminal`, `:834-835`) already requires `#library-skill-row-code-review` to be mounted, and it passes. In every captured failure the painted screen shows the Skills list with "code-review" in it and no Work-pane copy. So the list and its row are painted; the empty Work-pane copy never is.

**Likely cause, to be confirmed.** cf65608380 (TASK-38981, 2026-11-05) gave `LIBRARY_SKILLS_READER_PROFILE` `list_first_when_empty=True` (`screen_constants.py:91-95` at dev; TASK-34851's rule). Since then, an empty narrow Skills stage shows the list rather than the Work pane, and the test's 60x20 expectation has not caught up. That points to test drift.

**Age.** The case predates the TASK-36952.3 series:
- it failed with the same message under the private-profile marker at TASK-36952.3.3's base, a4d43518e0, and at its head (2026-11-15), and in the 3.4, 3.5 and rb4 marker probes after it;
- cf65608380 is an ancestor of 08b3c9d337, the series' original base;
- 13cb59ee12 (TASK-36952.3.6, 2026-11-15), the file's last change, retargeted two media tests (`[media]` and `test_stale_media_detail_cannot_project_after_route_switch`) receiver-only and is not in a4d43518e0, so it did not cause this.

**Re-checked on 2026-11-18** at dev 908e9840aa under a temporary private-profile marker at `-n 3`: `[skills-size0]` failed in 2 of 2 rounds (load 16 and 49), while the test's other 17 cases passed in both, including `[skills-size1]` at 170x48. Locally the file stops at `RecoveryRequired` without the marker, and no PR lane runs it; the whole test is among TASK-41772's Recovery-blocked guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes confirm the cause (TASK-38981's list-first rule or another) and say whether the 60x20 expectation or the product is wrong
- [ ] #2 The case passes 3 of 3, under the private-profile marker if the file still stops at RecoveryRequired locally
- [ ] #3 The test's other cases keep their assertions; the 60x20 Skills expectation changes only if the notes show it is the stale side
<!-- AC:END -->
