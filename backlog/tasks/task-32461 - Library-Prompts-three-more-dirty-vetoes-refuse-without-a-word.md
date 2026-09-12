---
id: TASK-32461
title: 'Library Prompts: three more dirty vetoes refuse without a word'
status: To Do
assignee: []
created_date: '2026-09-12 00:10'
labels:
  - library
  - prompts
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32393 fixed the Escape/Back seam: a dirty Prompts editor now says "Unsaved Prompt changes — Save or Discard changes first." instead of refusing in silence. Three sibling vetoes read the same flag and still say nothing, so the same click-does-nothing-says-nothing defect survives on three other surfaces.

All three refuse on `_flush_library_prompt_save()` returning False (which is exactly `not _prompts_state.dirty`) and return without notifying: the prompt-row switch (`library_screen.py` ~25675, pressing another prompt row while the open one is dirty), select-mode entry (`library_prompts_controller.py` ~1433, pressing "Select" while dirty), and the entry-reconcile path (~33622, a deep link into a prompt arriving while the editor is dirty). The rail-row switch and the app-level navigation guard already notify (`library_screen.py:21405`, `:10485`), so the copy and the pattern exist — these three were simply never wired to them.

The background reconcile path is the one that needs a judgement rather than a copy-paste: a toast raised by something the user did not just press may be noise, so decide whether it explains, defers, or stays silent by design and record which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing another prompt row while the open prompt is dirty states why the switch was refused, on the same line as the next step
- [ ] #2 Pressing Select while the open prompt is dirty states why it was refused
- [ ] #3 The entry-reconcile veto's behaviour is decided and recorded in the task (explain, defer, or deliberately silent), and matches what ships
- [ ] #4 Each wired refusal is covered by a test that fails if it becomes a silent no-op again
<!-- AC:END -->
