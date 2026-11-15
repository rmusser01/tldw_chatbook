---
id: TASK-41128
title: "Owner decision needed: Escape from Library Notes Create or Add from files lands on the Notes filter, not on the control that opened it"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- notes
- owner-decision
- ux
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A keyboard user who opens Create (or Add from files) and presses Escape expects to return to where they started; instead focus lands in the filter field, where the next letter types into the filter. TASK-40294 found the behaviour is deliberate and re-pinned its test to it, so whether it is right is a product question this task asks; it does not decide it.

**Evidence.** `library_notes_controller.py:5689-5772` (`_try_switch_retained_library_notes_route`) restores the navigator with `semantic_role="create-template:blank" if entering_create else "filter"` (`:5767`). TASK-40294's notes: both Escape steps in `test_library_note_footer_covers_navigator_create_sync_and_exit` "restore the navigator's `filter` role on purpose", and a step walk showed focus in `#library-notes-filter` at both, with the footer reading `esc rail | typing in field`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on where Escape from Create and from Add from files lands is recorded verbatim in the notes
- [ ] #2 The behaviour and the footer follow the decision, the Library Notes guide page says where focus lands, and the test pins it
<!-- AC:END -->
