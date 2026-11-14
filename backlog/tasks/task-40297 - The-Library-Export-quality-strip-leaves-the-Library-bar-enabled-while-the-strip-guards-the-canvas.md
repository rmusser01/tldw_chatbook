---
id: TASK-40297
title: "The Library Export quality strip leaves the ‹ Library bar enabled while the strip guards the canvas"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- export
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_export_controller.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified the node as a product defect and left it red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** While the Export quality strip is open and guarding the canvas, the emergency "‹ Library" bar stays enabled. It offers a way out that the strip's own guard should be deciding.

**Cause, from TASK-39152's notes.** Since 43849abe13 (TASK-21243) the strip opens canvas-scoped, and nothing re-syncs the bar's guard presentation after the open. A temporary `_sync_library_emergency_guard_presentation()` call after the open (`UI/Screens/library_screen.py:9118`) made the node pass.

**Node.** `Tests/UI/test_library_shell.py::test_library_emergency_real_guards_keep_specific_action_authoritative[cancel]`: `LibraryEmergencyReturn(id='library-emergency-return').disabled` is False.

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While the Export quality strip guards the canvas, the ‹ Library bar is disabled, and it is enabled again once the strip closes
- [ ] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
