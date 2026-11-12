---
id: TASK-40299
title: "The compact Library Notes Preview loses its scroll offset and responsive memory across a compact-wide-compact round trip under load"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- notes
- flaky
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down. Review r13l-rev1 of TASK-39152 classified both nodes as one product defect: the compact Preview's scroll offset and its responsive memory are lost across a compact -> wide -> compact round trip. The offset stays (0, 0) where (0, 2) was kept, and the deliberate scroll is overwritten.

**How often.** Load-dependent:
- TASK-39152 measured 3 of 3 whole-file runs red at load 41-51, and 6 of 7 alone in review.
- Checked at the round-13 stack top 8b633d0101, 2026-11-12, at load 13-15: both passed in a batch, and 3 of 3 alone each.

So the loss shows only when the machine is busy. It is still to be decided whether the product restores the scroll before layout settles (a real race a busy user would hit) or the test reads too early.

**Nodes.**
- `Tests/UI/test_library_shell.py::test_library_note_breakpoint_round_trips_restore_every_region_focus_role`: `preview-body (0, 0) == (0, 2)`.
- `::test_library_note_deliberate_scroll_override_replaces_responsive_memory`: "Compact Preview did not restore the deliberate scroll offset."

Both are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say whether the loss is a product race or a test that reads too early, with the evidence (for example a probe of when the offset is restored against when layout settles)
- [ ] #2 A product race is fixed with a test that is red under load (or under a temporary delay) before the fix; a test race waits for the restored offset instead
- [ ] #3 Both nodes pass 5 of 5 alone at load 30 or more, and their lines leave `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
