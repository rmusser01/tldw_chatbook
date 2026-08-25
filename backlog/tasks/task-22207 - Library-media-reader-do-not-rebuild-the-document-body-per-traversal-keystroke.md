---
id: TASK-22207
title: >-
  Library media reader: do not rebuild the document body per traversal keystroke
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - library
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22207).

New with PR #2064. `UI/Screens/library_screen.py:7407-7419`: `on_descendant_focus` now
opens the reader on focus, so arrow-keying through the Items list drives it per keystroke.
`_select_library_media_reader_row` runs `_sync_library_media_viewer_or_recompose()`
synchronously (`:12424`) — the 120 ms settle timer debounces only the DB fetch, not the
recompose — and the `unchanged` test at `:33096-33100` compares `viewer.loading` against a
freshly-flipped pending flag, so it always falls through to `viewer.refresh(recompose=True)`
(`:33141`). Compose constructs a brand-new full-document body
(`Widgets/Library/library_media_viewer.py:296-311`): in rendered mode a fresh
`Markdown(content)` whose mount parses the whole document on the loop. The settle path
then recomposes a second time. Net: 2 full-document rebuilds per settled selection, 1 per
keystroke during fast traversal — and the wasted one re-parses the document being LEFT,
purely to paint a "Loading..." placeholder. There is no windowing of the reader body.
While Find is open, each rebuild adds 3 more O(document) match scans
(`library_media_viewer.py:296`, `library_media_content.py:34`, `:155`).

## Acceptance Criteria

- [ ] Traversing N rows performs zero document-body rebuilds for pass-through rows; only the settled row renders, once (probe counts `LibraryMediaContentBody` constructions per 10-row traversal)
- [ ] Showing the loading placeholder does not require recomposing the document body
- [ ] A 1 MB document can be traversed past with no per-keystroke parse (measured)
- [ ] Reader behavior (settle, modes, search) unchanged; existing reader tests green
