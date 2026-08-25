---
id: TASK-22208
title: >-
  Media reader no-change syncs: stop rebuilding PIL previews and copying the document
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - library
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22208).

New with PR #2064. `_sync_library_media_viewer_state` computes the image-preview
projection BEFORE its `unchanged` test (`library_screen.py:33079-33085` vs `:33110`), so
every interaction — traversal step, mode switch, More toggle, Escape — synchronously runs
`build_media_image_widget` (`Widgets/Library/library_media_image_preview.py:127-175`): a
PIL LANCZOS resize plus a per-cell Python loop over the mosaic grid
(`Utils/mosaic_render.py:216-240`), on the event loop, then discards the widget when
nothing changed. Separately, every interaction rebuilds the full viewer display state:
`build_library_media_viewer_state` copies the whole content string (`str(...).strip()` at
`Library/library_media_viewer_state.py:305` — a trailing newline forces a full copy) and
the `unchanged` compare memcmps it (`:33087`).

## Acceptance Criteria

- [ ] A no-change sync builds no preview widget and performs no O(document) string copies (probe)
- [ ] Preview construction is memoized by source (and/or moved off the loop) with the memo's invalidation stated
- [ ] The change test compares cheap identity (ids/revisions), falling back to content compare only when identity is inconclusive
- [ ] Per-interaction cost measured before/after on an image-typed item
