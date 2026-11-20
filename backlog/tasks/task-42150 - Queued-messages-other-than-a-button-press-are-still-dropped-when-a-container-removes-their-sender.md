---
id: TASK-42150
title: "Queued messages other than a button press are still dropped when a container removes their sender"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- ui
- textual
priority: low
references:
- tldw_chatbook/__init__.py
- Tests/Library/test_library_notes_canvas_pending_lookup.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the library stream's TASK-42014 (round 18). TASK-42014 (round-18 library branch, not yet on dev) makes every recompose hand a queued `Button.Pressed`, whose sender is being removed, to the nearest ancestor that stays (`keep_queued_presses` in `tldw_chatbook/__init__.py`). Two cases remain:
- other user messages queued the same way are still dropped: `Input.Submitted`, `Checkbox.Changed` and `Switch.Changed`, `Select.Changed`, `ListView.Selected`;
- a press is still dropped when a container swaps its children without recomposing (`remove_children()` then `mount()`); the keeper runs on the recompose route and in the two overrides TASK-42014 patched.

Each loses a user action that arrived just before the rebuild. This builds on TASK-42014's keeper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes list which message types and which child-swap routes (recompose, `remove_children()` then `mount()`) can drop a queued user message, and where each occurs in the app
- [ ] #2 Each listed case either keeps its queued message (delivered to the nearest staying ancestor, as TASK-42014 does for presses) or is recorded as safe with the reason
- [ ] #3 For each case that keeps its message, a pilot fails without the change
<!-- AC:END -->
