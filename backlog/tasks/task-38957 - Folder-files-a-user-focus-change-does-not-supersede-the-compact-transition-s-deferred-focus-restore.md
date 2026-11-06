---
id: TASK-38957
title: "Folder files: a user focus change does not supersede the compact transition's deferred focus restore"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- library
- focus
- file-notes
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38695 (Done) first raised this as a "Proposed follow-up (product, not fixed here)".

**Code path.**
- `_transition_library_notes_presentation` (`tldw_chatbook/UI/Screens/library_screen.py:8180-8257`) takes a snapshot of `focus_intent_generation` for its guard (`:8220-8226`).
- It then queues `_restore_library_notes_focus_identity` (`:8248`), plus a settled restore two turns later (`tldw_chatbook/UI/Library_Modules/library_notes_controller.py:2764-2774`).
- A user's focus change advances the generation only when `relevant` holds (`library_screen.py:11240-11255`). `relevant` requires the Database Notes workflow (`_library_notes_workflow_active`, `:5128-5133`), the rail, Media or Conversations.
- In Folder files mode none of those applies, so the generation stays put and the guard (`:5717-5734`) still passes. The deferred restore then moves focus back to where it was before the user's newer choice.

**Observed.** `Tests/UI/test_library_file_notes_workspace.py:772` (`test_folder_files_automatic_items_priority_does_not_steal_focus_or_persist`) failed 2 of 72 runs from exactly this sequence. TASK-38695 fixed only the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In Folder files mode, a user focus change made after a compact/wide transition but before its settled restore keeps the user's focus; a test focuses the editor between the two restores without waiting for the settled one and is red on dev
- [ ] #2 Programmatic focus and the Database Notes, rail, Media and Conversations focus behaviour are unchanged, and the existing Notes focus and canvas-sync pins stay green
- [ ] #3 The new test passes 20 of 20 runs under -n 4, with the load recorded
<!-- AC:END -->
