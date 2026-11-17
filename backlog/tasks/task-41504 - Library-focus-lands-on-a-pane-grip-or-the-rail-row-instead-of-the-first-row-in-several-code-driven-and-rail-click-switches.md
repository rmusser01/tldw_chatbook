---
id: TASK-41504
title: "Library focus lands on a pane grip or the rail row instead of the first row in several code-driven and rail-click switches"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- focus
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41128
- TASK-41129
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the libfocus stream and its review (TASK-41128 to TASK-41130 area, round 15); all measured as present on base too. Observed landings:
- in 1 of 4 wide probe runs a rail click left focus on the Notes rail row instead of the first note row;
- code-driven switches landing on a pane grip: landing then empty Notes `press()` at 170x48 lands on `#library-browse-library-grip`; Media then Collections `press()` lands on `#library-collections-library-grip`; Notes then resident Media `press()` lands on a grip about 1 in 3;
- a rail click Media then already-open Notes lands on the rail row (3 of 3 on base, 1 of 3 on head);
- at 170x48 rail entry into an empty Notes list sometimes leaves focus on the Library grip (libtests fix round);
- "View imported notes" on the Import once receipt still lands in the filter (`library_notes_controller.py:6546`), inconsistent with the TASK-41128 landing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed switch lands on its first row (or the documented empty-state control) 10 of 10 runs, and never on a pane grip
- [ ] #2 "View imported notes" lands where TASK-41128 decided Escape lands, or the difference is documented as a decision
- [ ] #3 Tests drive each switch and assert the focused widget id
<!-- AC:END -->
