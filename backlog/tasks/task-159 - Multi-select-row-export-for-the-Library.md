---
id: TASK-159
title: Multi-select row export for the Library
status: Done
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - export
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F4 shipped current-scope export (everything, or a section's current filter). Add per-row multi-select so users can export an arbitrary subset. Requires a selection mode on the Library list widgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can check individual Library rows and export exactly those
- [x] #2 Export form accepts an explicit id set as its scope
<!-- AC:END -->

## Implementation Notes

- `ExportScope` gained an `ids: tuple[str, ...]` field: when non-empty it
  overrides the normal current-scope-filter resolve/count/label branches
  with an explicit-subset export ("Selected media/conversations/notes ·
  N items").
- `RowSelection` (pure, Textual-free) is a per-source checked-id
  accumulator: `toggle`/`select_all`/`clear`/`reconcile` on the id set,
  plus `export_scope()` to turn it into an `ExportScope`. The screen owns
  one instance per browsable source (media/conversations/notes).
- Each of the three Library row-state builders (`build_library_media_state`,
  `build_library_conversations_state`, `build_library_notes_list_state`)
  gained `checked`/`select_mode`/`selected_count` so the canvases can
  render the ☑/☐ glyph and selection count without any screen-side
  bookkeeping duplicated in the widgets.
- Select mode is per-source: entering it on one canvas doesn't affect the
  others, and switching a source's sort/filter/type-filter clears that
  source's selection and exits select mode (reconciled against the
  currently rendered rows so a stale selection can never silently export
  more/less than what's on screen -- WYSIWYG).
- All select-mode transitions recompose via the screen's
  `self.refresh(recompose=True)` (the only reliable update path already
  used everywhere else in this canvas family) rather than a targeted
  per-widget update.
- Notes is the one source where entering/toggling select mode is `async`:
  it awaits `_flush_library_note_save()` first (mirroring the existing
  note-row-press flush) so entering select mode, or toggling a row while
  select mode is active, never strands a dirty note edit mid-save. Notes
  rows also had no marker at all before this task, so the ☑/☐ glyph only
  ever appears once select mode is active; normal mode keeps its
  markerless label.
- Modified/added files: `tldw_chatbook/Library/library_export_scope.py`,
  `tldw_chatbook/Library/row_selection.py`,
  `tldw_chatbook/Library/library_media_state.py`,
  `tldw_chatbook/Library/library_conversations_state.py`,
  `tldw_chatbook/Library/library_notes_state.py`,
  `tldw_chatbook/Widgets/Library/library_media_canvas.py`,
  `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`,
  `tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
  `tldw_chatbook/UI/Screens/library_screen.py`,
  `Tests/Library/*`, `Tests/UI/test_library_multiselect_media.py`,
  `Tests/UI/test_library_multiselect_conversations.py`,
  `Tests/UI/test_library_multiselect_notes.py`.
