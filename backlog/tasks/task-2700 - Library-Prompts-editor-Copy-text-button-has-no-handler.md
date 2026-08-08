---
id: TASK-2700
title: 'Library Prompts editor: "Copy text" button has no handler (dead button)'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-08 14:37'
labels:
  - library
  - bug
  - ui
dependencies: []
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implemented in the TASK-202 PR; add the missing handler through LibraryScreen and reuse the canonical Prompt Markdown renderer. ADR required: no; ADR path: N/A; reason: UI-only defect repair under ADR-011/ADR-040.
<!-- SECTION:PLAN:END -->

## Description (the why)

The prompt editor's action row includes a **"Copy text"** button
(`#library-prompt-copy`, `library_prompts_canvas.py:372`), but no
`Button.Pressed` handler for that id exists anywhere in the codebase (no
`@on` decorator, no generic dispatcher on the screen or app) — pressing it
does nothing: no clipboard write, no toast. Found during the G2 user-guide
verification session (dev @ bd05a692a, 2026-07-31; grep-verified — the only
references are the canvas compose and a label-only test). The notes editor's
"Copy" (which toasts "Note copied to clipboard as markdown!") shows the
intended pattern.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Copy Markdown copies the live unsaved working copy.
- [ ] Clipboard success is reported only after a successful copy.
- [ ] Unavailable or failed clipboard support is reported honestly.
<!-- AC:END -->
