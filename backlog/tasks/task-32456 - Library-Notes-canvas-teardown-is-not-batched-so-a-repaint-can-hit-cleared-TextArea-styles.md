---
id: TASK-32456
title: >-
  Library Notes: canvas teardown is not batched, so a repaint can hit cleared
  TextArea styles
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
  - flake
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The durable half of task-32298. `KeyError: "No 'text-area--gutter' key in
COMPONENT_CLASSES"` is raised when Textual repaints a `TextArea` whose
component styles teardown has already cleared. TASK-32114 proved that this
reaches a REAL user path (Linux CI, Escape closing the MCP Test Tool panel)
and fixed it there by removing the panel inside `app.batch_update()`.

Library does not do that: `grep -rn batch_update tldw_chatbook/UI/Screens/library_screen.py
tldw_chatbook/UI/Library_Modules/` returns nothing, so every route switch
that unmounts the Notes editor (`#library-note-body` is a
`NoteEditorTextArea`) leaves the same window open. task-32298 mitigated the
test symptom with a per-test loop drain in
`Tests/UI/test_library_notes_reader.py`; the product guard is still missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving a Library route that has a mounted `TextArea` removes it inside `app.batch_update()` (or an equivalent guard), so no repaint can land between the styles being cleared and the widget leaving the compositor
- [ ] #2 A deterministic regression test drives that teardown with a pending repaint and reproduces the `text-area--gutter` KeyError before the fix, as TASK-32114's does for the MCP panel
- [ ] #3 With the product guard in place, the `_drain_pending_repaints` fixture task-32298 added is removed or justified in its docstring
<!-- AC:END -->
