---
id: TASK-2853
title: Media Select mode ships a real selection toolbar
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - media
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-05, observed at dev `6ffa56516`). Owner ruling 2026-08-07: ship a
selection toolbar (not remove Select).

Media (3) → "Select" enters a mode offering only checkboxes, an "N selected" count, and "Done".
No action consumes the selection; pressing Done discards it; the bottom preview pane meanwhile
keeps showing a previously selected different item. The control advertises bulk capability — the
power user's #1 need (bulk export/delete) — and delivers a no-op, poisoning trust in every other
control's promise.

Scope per ruling: a selection toolbar with real actions. Export selection first (Export canvas +
context-scoped export already exist — wire selection in as a scope). Delete-selected with confirm.
"Add to collection" only when collection item adapters exist (do not block on them).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While items are selected, a toolbar offers at least "Export selection" and "Delete selected" with the selection count
- [ ] #2 Export selection produces a bundle containing exactly the selected items (verified against the zip)
- [ ] #3 Delete selected asks for confirmation naming the count, then soft-deletes and updates list + rail counts
- [ ] #4 Leaving Select mode without acting discards the selection explicitly (copy states it) and the preview pane never shows an item outside the current selection context
- [ ] #5 Live TUI verification of both actions end-to-end
<!-- AC:END -->
