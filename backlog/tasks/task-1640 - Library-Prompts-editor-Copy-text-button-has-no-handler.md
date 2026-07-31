---
id: task-1640
title: 'Library Prompts editor: "Copy text" button has no handler (dead button)'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [library, bug, ui]
dependencies: []
---

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

## Acceptance Criteria (the what)

- [ ] Pressing "Copy text" copies the prompt's user-prompt text (or the
      sensible chosen payload) to the clipboard and confirms with a toast.
- [ ] A test covers the handler (button press → clipboard/toast), so the
      button cannot silently detach again.
- [ ] The User Guide quirk in `Docs/User_Guide/library/prompts.md` is
      updated/removed to match.
