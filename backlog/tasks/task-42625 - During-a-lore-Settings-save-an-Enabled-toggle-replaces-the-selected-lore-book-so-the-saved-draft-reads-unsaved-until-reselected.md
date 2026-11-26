---
id: TASK-42625
title: 'During a lore Settings save an Enabled toggle replaces the selected lore book, so the saved draft reads unsaved until reselected'
status: To Do
assignee: []
created_date: '2026-11-26 16:00'
labels:
  - follow-up
  - roleplay
dependencies: []
references:
  - tldw_chatbook/UI/Persona_Modules/persona_save_completion.py
  - tldw_chatbook/UI/Screens/personas_screen.py
  - tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py
  - backlog/tasks/task-33910.48 - Roleplay-lore-settings-draft-is-outside-the-unsaved-changes-aggregate.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The interim lore-Settings check that decides whether an import may auto-select (TASK-33910.48, shipped in PR #3518) loses its baseline in one interleaving. This defect exists on dev since `52351c4e0b` (#3518's merge commit); no open PR fixes it (`gh pr list --state open --search "lore Enabled toggle settings save"` returns nothing). B9a (TASK-33910.14) is expected to replace the interim check with a lore-settings domain in the unsaved-changes aggregate, which would remove it.

What the code does at `52351c4e0b`, read, not run:
- A Settings save's `reconcile` (`persona_save_completion.py`, the lore save at about lines 175-250) calls `detail.mark_settings_saved(submitted_snapshot)` only when its `is_current()` holds, and `is_current()` requires `screen._selected_lore_book is selected_record`.
- The Enabled toggle handler (`personas_screen.py` about lines 6985-7010) reloads the book and assigns a new record to `self._selected_lore_book`. If that lands while the save is in flight, the save's `is_current()` fails, so it takes the "Saved." branch and skips `mark_settings_saved`.
- `PersonasLoreDetailWidget._loaded_settings` (`personas_lore_detail.py`, `has_unsaved_settings`, `mark_settings_saved`) then keeps the pre-save baseline, so when the saved edit changed any non-Enabled control the detail reads unsaved until the book is reselected (`load_book` resets it).
- `_lore_settings_unsaved_for_landing` (`personas_screen.py` about line 14378) feeds `_land_import`'s clean test, so an import finishing after that does not select: it toasts where the item landed.

It fails safe (a toast instead of a selection, never a lost draft) and needs a save and a toggle to overlap, so the priority is low.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a lore Settings save completes, the draft reads saved even if an Enabled toggle replaced the selected book while the save was in flight.
- [ ] #2 A mounted test pins it: hold a Settings save, toggle Enabled, release, then land an import and see it select.
- [ ] #3 If B9a (TASK-33910.14) lands first and removes the interim check, this task is closed as superseded, with the evidence.
<!-- AC:END -->
