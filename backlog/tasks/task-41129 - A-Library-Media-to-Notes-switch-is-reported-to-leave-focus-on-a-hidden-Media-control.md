---
id: TASK-41129
title: "A Library Media-to-Notes switch is reported to leave focus on a hidden Media control"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- focus
- investigation
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Reported by the round-14 Library stream as a bug present on its base: after switching from Media to Notes, focus can stay on a Media control that is no longer displayed, so keys go nowhere visible. It was not reproduced at closeout and no node pins it.

**Evidence to start from.** The Notes route switch restores focus through `_restore_library_notes_focus_identity` after `_try_switch_retained_library_notes_route` (`library_notes_controller.py:5689-5772`); if that restore finds nothing focusable it does not move focus off the previous canvas's widget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test reproduces the switch at the sizes the Library supports and records where focus lands, or the notes record the bug cannot be reproduced and the task is closed
- [ ] #2 If reproduced, focus lands on a visible Notes control after the switch, with the test red before the fix
<!-- AC:END -->
