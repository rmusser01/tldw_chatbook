---
id: TASK-42422
title: "Switching from Notes to a resident Media list by press still lands Library focus on the pane grip and fails UI Fast Lane 33"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- ci
- library
- focus
- flaky
priority: high
references:
- Tests/UI/test_library_rail_switch_keeps_reader_and_note.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/screen_constants.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** The required "UI Fast Lane (33)" fails on `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_each_library_switch_lands_on_its_first_row_never_a_grip[notes-press-resident-media]` with `library-media-row-0: focus on LibraryAdaptiveReaderPaneGrip(id='library-browse-library-grip', name='Collapse Library pane', ...) (in the focus chain: True)`, after the test's full wait (call 21.9 s). In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), it failed 5 of the lane's 6 failed jobs, from 2026-11-21 02:26Z to 2026-11-22 15:09Z: runs 44758099535, 44944358256 (listed in TASK-33620.15.61's notes), 44954347282, 44974765316 and 45031398878. It passes locally at the base.

**Defect class.** Focus landing on a pane grip instead of the first row is TASK-41504's defect class; its description recorded exactly this switch ("Notes then resident Media `press()` lands on a grip about 1 in 3"). TASK-41504 is Done, so this residual is untracked. It is not TASK-32301's ceiling: when a list arrives after the fixed entry window, focus lands nowhere, while here it lands on a focusable grip, which is what a user on a slow machine would be left on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The path that leaves focus on `#library-browse-library-grip` after a Notes-to-resident-Media press is identified and recorded in the notes
- [ ] #2 After that switch, focus lands on the Media list's first row and never on a pane grip, including when the switch's work is slowed (shown with a temporary delay, reverted, that fails before the fix)
- [ ] #3 The parametrized test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
