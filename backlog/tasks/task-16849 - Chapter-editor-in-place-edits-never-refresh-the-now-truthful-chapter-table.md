---
id: TASK-16849
title: Chapter editor in-place edits never refresh the now-truthful chapter table
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16 18:20'
labels:
  - bug
  - ui
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15773 (PR #1710) fixed the recompose race that left the audiobook chapter table
empty for every population, so the table finally shows real rows. Its review's residual 4
(pre-existing, disclosed, not that task's regression) is now user-visible and still holds
at dev `ee741cf10`: the editor's own edit buttons mutate the `chapters` reactive **in
place** and never notify anything, so the table the fix made truthful goes stale on the
first edit.

`Widgets/TTS/chapter_editor_widget.py`: `_add_chapter` (`:369`,
`self.chapters.insert(...)`), `_split_chapter` (`:390`), `_merge_chapters` (`:424`),
`_delete_chapter` (`:446`) all mutate `self.chapters` directly and post a
`ChapterEditEvent`; none of them calls `_refresh_chapter_table` (`:263`) or
`mutate_reactive`. The review probed it post-fix: after `_add_chapter`, `chapters=6` but
`rows=5`; after `_delete_chapter`, `chapters=5`, `rows=5` — counts, titles, and the
selected-row mapping all drift from reality, and the preview/selection indices then point
at the wrong chapters. Only `set_chapters` (the detection path) refreshes.

Fix direction: route the four mutations through `mutate_reactive(...)` or an explicit
refresh (table + preview + selection clamp) after each edit — mirroring what
`watch_chapters`/`on_mount` already do for the set-path. Note `chapters` must stay
`reactive(list)` (callable default, task-15771's merged guard pins it).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After each of add/split/merge/delete, the table's rows, numbering, and the selected row match the chapters list (born-red tests per operation)
- [x] #2 The preview pane reflects the post-edit selection (no stale index into the old list)
- [x] #3 The 15773 population-race and detection suites stay green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read `Widgets/TTS/chapter_editor_widget.py` end to end; re-locate the four mutation methods and confirm `watch_chapters`/`watch_selected_chapter_index`/`on_mount`'s exact refresh shape post-15773.
2. Check the sibling `CharacterVoiceWidget` (`Widgets/TTS/character_voice_widget.py`, task-15479) for this codebase's established idiom for the exact same in-place-mutation-vs-reactive-watcher problem, and match it for consistency.
3. Add two small helpers (`_sync_table_cursor`, `_sync_after_edit`) and route `_add_chapter`/`_split_chapter`/`_merge_chapters`/`_delete_chapter` through `mutate_reactive(ChapterEditorWidget.chapters)` (table refresh, once) + `_sync_after_edit()` (cursor + preview refresh, once), deciding a selection policy per operation.
4. Write four born-red tests (one per operation) asserting `table.row_count == len(chapters)` and that the edited chapter's content is what the table/preview actually show.
5. Verify born red at HEAD by toggling the widget file back to its pre-fix content (Edit/Write-based, not git), confirm all 4 fail, then restore the fix and confirm all 4 pass.
6. Run the 15773 population-race suite (3), the 15772 tuple-order test (1), `test_reactive_default_aliasing.py`'s chapter-editor test, and the 8-test detection suite (`test_speech_audiobook_chapter_detection.py`) for regressions.
7. `ruff check` the touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Root cause confirmed**: `_add_chapter`/`_split_chapter`/`_merge_chapters`/`_delete_chapter` mutate `self.chapters` via `list.insert`/`list.pop` (same list object), so Textual's reactive equality check never fires `watch_chapters` -> `_refresh_chapter_table`. Only `set_chapters` (reassignment) got a free refresh. Additionally, even where `selected_chapter_index` WAS reassigned (delete), a numerically-unchanged clamp (`min(index, len-1)`) can silently name a *different* chapter object after the list shrinks -- so relying on that reactive's own change-detection for the preview is not sufficient either.

**Fix shape** (mirrors `CharacterVoiceWidget`'s task-15479 fix for the identical bug class, and the file's own `on_mount` set-path):
- Right after each list mutation (+ `_renumber_chapters()`), call `self.mutate_reactive(ChapterEditorWidget.chapters)` -- this forces `watch_chapters` to run exactly once, refreshing the table.
- `selected_chapter_index` updates (add: `insert_pos`; delete: `min(index, len-1)`, unchanged from the original clamp logic) go through `self.set_reactive(...)` -- silent, no watcher -- specifically so the table-cursor + preview refresh happens through exactly ONE path (`_sync_after_edit`), not potentially twice (once via a would-be watcher firing, once via the explicit call).
- New helper `_sync_table_cursor()` moves the DataTable cursor onto `selected_chapter_index` after each refresh (the clear+repopulate in `_refresh_chapter_table` resets the cursor, so this keeps the highlighted row and the reactive in sync).
- New helper `_sync_after_edit()` = `_sync_table_cursor()` + `_update_preview()`, called exactly once per operation, after `chapters`/`selected_chapter_index` are both finalized.

**Selection policy chosen per operation** (AC #2):
- **Add**: selects the newly-created chapter (`insert_pos`) -- the user's next action is almost always to edit what they just added.
- **Split**: selection stays on the original, now-truncated first-half chapter (unchanged index) -- the split was made from that chapter's own preview cursor position.
- **Merge**: selection stays on the (now-merged) current chapter (unchanged index).
- **Delete**: unchanged from the pre-fix clamp (`min(selected_chapter_index, len(chapters)-1)`) -- selects the neighbor that shifted into the deleted slot, or the new last chapter if the last one was deleted. Only the refresh was missing, not the selection logic itself.

**Tests added**: `Tests/Widgets/test_chapter_editor_widget_inplace_edit_refresh.py`, one test per operation, each driving the real `on_button_pressed` dispatch via a mounted `Button` and asserting `table.row_count == len(editor.chapters)` plus that the table/preview show the actually-edited content (not just a row-count coincidence). All 4 confirmed born red against the pre-fix widget (toggled back in via Write, then restored -- not via git) with the exact `table has N rows for M chapters` mismatch the task predicted; all 4 pass against the fix.

**Regression suites** (all green): the 15773 population-race suite (3 tests), the 15772/16849-adjacent tuple-order test (1), `test_reactive_default_aliasing.py`'s `test_chapter_editor_widgets_do_not_share_chapters` (+ 4 sibling tests in that file), and the 8-test detection suite `test_speech_audiobook_chapter_detection.py`. 21/21 passed together. `ruff check` clean on both touched files.

**Files changed**:
- `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py` -- the four mutation methods + two new private helpers.
- `Tests/Widgets/test_chapter_editor_widget_inplace_edit_refresh.py` -- new, four born-red pins.
<!-- SECTION:NOTES:END -->
