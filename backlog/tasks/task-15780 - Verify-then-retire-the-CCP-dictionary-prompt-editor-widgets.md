---
id: TASK-15780
title: Verify-then-retire the CCP dictionary/prompt editor widgets
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - cleanup
priority: low
---

## Description

Verify-then-retire candidate surfaced while reviewing task-15476 (input-latency
burn-down's picker-debounce task, which touched
`ccp_dictionary_editor_widget.py:730/:848` and `ccp_prompt_editor_widget.py:876`
for consistency without checking whether the widgets are actually reachable
in production). Confirmed by a repo-wide grep: `CCPDictionaryEditorWidget`
and `CCPPromptEditorWidget` are referenced only inside their own module and
`Widgets/CCP_Widgets/__init__.py`'s re-export — zero other production
importers anywhere in `tldw_chatbook/`. The package's own `__init__.py`
docstring says as much: "Surviving prompt/dictionary editor widgets. The
legacy CCP screen chrome (sidebar, character card/editor, conversation view,
persona card/editor) was retired in favor of the Personas workbench
(tldw_chatbook/Widgets/Persona_Widgets/)."

This is exactly the same shape as task-15481's dead-scheduler/dead-DB-module
sweep in the same programme: code that looks alive (and gets reflexively
touched by unrelated fixes, as task-15476 did) but is unreachable from any
production screen. Per the same standing preference task-15481 applied:
delete (with git-log provenance) or explicitly quarantine — do not leave a
loaded gun a future contributor might wire up without noticing it was
already retired.

## Acceptance Criteria

- [ ] Re-verify at implementation time (not trusting this task's grep without
      re-checking) that `CCPDictionaryEditorWidget` and
      `CCPPromptEditorWidget` have zero production callers/importers outside
      their own module and `__init__.py`
- [ ] If still dead: delete both widget modules (with git-log provenance
      recorded in the notes) and their now-orphaned test-only importers, or
      trim tests that also cover live code the same way task-15481 did for
      `Research_DB`/`Sync_Client`
- [ ] If a live caller is found (contradicting the grep above): the task
      closes as "not dead," with the caller documented, and no deletion
      happens
- [ ] `pytest --collect-only` over the whole tree has zero errors after the
      change; a final grep sweep for both class names returns no production
      hits
