---
id: TASK-32253
title: >-
  Library Notes editor: Shift+Tab into the Title selects the whole title, so
  one keystroke destroys it and autosave commits the loss
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduced twice: `Ideas for study decks` became `!` on a single keypress after Shift+Tab moved focus into the Title, and autosave committed the loss (`R/caps/08`, `09`).

This is Textual's `Input.select_on_focus` default and it matches browser behaviour, which is exactly why it is filed at P2 rather than higher. What makes it a defect and not a default is that the pairing is backwards: the one field on the screen whose content must not be destroyed selects on focus, while the path fields that would genuinely benefit from select-on-focus do not (task-32251). Filed together, they are one decision about the same widget default, not two.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Moving focus into the note Title by keyboard places the caret without selecting the existing title
- [ ] #2 A title replaced in one keystroke is recoverable: either undo inside the field restores it, or the change is not autosaved until the field is left
- [ ] #3 Covered by a test: Shift+Tab into the Title followed by one character leaves the title intact apart from that character
<!-- AC:END -->
