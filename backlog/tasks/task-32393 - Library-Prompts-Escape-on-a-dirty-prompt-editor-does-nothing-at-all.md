---
id: TASK-32393
title: 'Library Prompts: Escape on a dirty prompt editor does nothing at all'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - prompts
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an unsaved edit in the Prompts editor, Escape neither leaves nor explains. Verified live at 235x52 on a seeded profile during the task-32366 reconciliation: with the editor showing "Modified 4h - v1 - Unsaved changes" and the footer advertising `esc back to list`, four Escape presses (from an Input and after tabbing out of one) produced no exit, no notice and no visible change. A dirty veto exists in code (`_notify_prompt_dirty_veto`, `LIBRARY_PROMPT_DIRTY_VETO_COPY`) but no toast reached the screen. A key the footer names must either do what it says or say why it will not; a silent no-op reads as the app having hung. The veto itself is correct and deliberate -- `_exit_library_prompt_editor_guarded` (`library_prompts_controller.py:3456-3459`) returns False on a dirty flush -- so the defect is only the missing message. task-2702 (Done) shipped `LIBRARY_PROMPT_DIRTY_VETO_COPY` for the nav-bar veto and task-32133 is the nearest precedent for wiring a refusal to its reason; the Skills twin (`library_screen.py:24802-24810`) already calls its notifier from the same seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Escape on a dirty prompt editor either leaves the editor or states why it cannot, on the same line as the next step
- [ ] #2 The footer chip and what Escape does agree in every prompt-editor state
- [ ] #3 The dirty-Escape path is covered by a test that fails if the key becomes a silent no-op again
<!-- AC:END -->
