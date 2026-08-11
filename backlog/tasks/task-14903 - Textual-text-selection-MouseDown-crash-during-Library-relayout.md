---
id: TASK-14903
title: Text-selection MouseDown crash during Library relayout kills the app
status: To Do
assignee: []
created_date: '2026-08-10 17:20'
labels:
  - library
  - stability
dependencies: []
priority: high
---

## Description

Observed once during task-4023's live verification (2026-08-10, 170x24 terminal,
isolated profile sdd_hat3): a mouse click landing on the Search/RAG canvas's
`#library-rag-query-quiet-line` Static ~1s after a 50→24-row terminal resize
raised an unhandled exception inside Textual's text-selection machinery and
terminated the whole app:

```
File .../textual/screen.py:1914, in <text-selection begin>
    event.screen_offset - container.region.offset,
AttributeError: 'NoneType' object has no attribute 'region'
locals: container = None
        content_widget = Static(id='library-rag-query-quiet-line', classes='library-rag-quiet-line')
        event = MouseDown(x=42, y=11, button=1)
```

Not deterministically reproducible (three replay attempts, including at the
task-2 base c3bfddc0b, did not trigger it) — the window appears to be a click
dispatched while the clicked Static's ancestor chain is being replaced by a
recompose/relayout, so the selection container resolves to None. Nothing in the
Library screen's own code is on the stack; this is a Textual 8.x race the app
still owns the consequences of (an app-killing click). Worth a guarded
reproduction attempt (synthetic MouseDown into a screen mid-recompose), then
either an upstream report + pinned workaround (e.g. a defensive guard via an
App-level exception handler for this signature) or a Textual version bump that
fixes it.

## Acceptance Criteria

- [ ] The crash signature is reproduced in a test or conclusively attributed with a written analysis
- [ ] A click during Library recompose can no longer terminate the application (guard, upstream fix, or pinned Textual bump — whichever lands first)
