---
id: TASK-31804
title: Roleplay Inspector shows the previous character's avatar while reporting 'Selected: none'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-05 19:15'
updated_date: '2026-09-06 14:47'
labels:
  - bug
  - ui
  - roleplay
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). After deselecting (or when selection clears), the Inspector's status line says 'Selected: none' but the previously selected character's avatar stays rendered - contradictory state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When selection is none, the Inspector shows no stale avatar.
<!-- AC:END -->

## Implementation Plan

1. Reproduce with a mounted inspector test: show_selection(character) + set_avatar_thumbnail, then clear_selection, assert the avatar box has no children (fails on dev).
2. Fix at the widget seam: blank the avatar in `PersonasInspectorPane.clear_selection()` (`set_avatar_thumbnail(None)`), matching the non-character `show_selection` paths in `personas_screen.py` that already blank the portrait "so the rail never shows a face that belongs to a different selection".
3. Verify: new test passes; full inspector-pane suite green.

## Implementation Notes

Reproduced on dev tip 5894f4755e. `PersonasInspectorPane.clear_selection()` updated the summary line to "Selected: none" and cleared conversations/validation but never cleared the portrait box (`#personas-inspector-avatar-thumb`), so a previous character's avatar lingered — the exact contradiction reported. The character-selection path renders the avatar via a worker (`_render_inspector_avatar`), and the persona/dictionary/lore `show_selection` paths already call `set_avatar_thumbnail(None)` to drop a stale face; only the deselect path was missing it.

Fix: added a single `self.set_avatar_thumbnail(None)` call inside `clear_selection()`. Centralizing the invariant at the widget means every screen-side `clear_selection()` caller gets it for free. The avatar worker's own staleness guard (keyed on `selected_entity_id`) already prevents a late repaint after deselection, so no worker-cancellation change was needed.

Modified files:
- `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- `Tests/UI/test_personas_inspector_pane.py` (new `test_clear_selection_drops_stale_avatar`)
