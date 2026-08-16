---
id: TASK-16849
title: 'Chapter editor in-place edits never refresh the now-truthful chapter table'
status: To Do
assignee: []
created_date: '2026-08-16'
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
- [ ] #1 After each of add/split/merge/delete, the table's rows, numbering, and the selected row match the chapters list (born-red tests per operation)
- [ ] #2 The preview pane reflects the post-edit selection (no stale index into the old list)
- [ ] #3 The 15773 population-race and detection suites stay green
<!-- AC:END -->
