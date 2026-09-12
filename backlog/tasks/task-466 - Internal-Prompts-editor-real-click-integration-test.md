---
id: TASK-466
title: 'Internal Prompts editor: real-click integration test for Save/Reset/Cancel'
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - test-coverage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The InternalPromptEditorModal's Save/Reset/Cancel are exercised via the `_save_from_test` seam and the panel's `_apply_editor_result` entry point, but no test drives the real Button.Pressed -> push_screen -> callback -> run_worker chain. A wrong `@on` selector (e.g. on Reset) would go undetected. Add a pilot-driven integration test that clicks the actual buttons.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pilot test clicks the modal's Save button and asserts the override persists + the row badge updates
- [x] #2 A pilot test clicks Reset and asserts the override is cleared
- [x] #3 A pilot test clicks Cancel (and presses Escape) and asserts no change
<!-- AC:END -->

## Implementation Notes

Three real-click integration tests in ``Tests/UI/test_internal_prompts_panel_editing.py``, completing the panel's earlier unit-level coverage (which drove ``_apply_editor_result`` directly) with the full chain: pressing a prompt ROW mounts ``InternalPromptEditorModal``; the modal's ACTUAL buttons drive the outcome; both the persisted override and the panel row badge reflect it.

- Save: ``#internal-prompt-editor-save`` press persists "CLICK-SAVED TEXT" (``override_state().active_text``) and the row gains ``row-customized`` (AC#1).
- Reset: ``#internal-prompt-editor-reset`` press clears a seeded override and drops the badge (AC#2).
- Cancel/Escape: parametrized over the Cancel button and the ``escape`` key -- neither persists edited text nor badges the row (AC#3).

All 7 tests in the file pass together (3 unit + 4 new). Format clean.

ADR required: no
ADR path: N/A
Reason: Test-only integration coverage.

Modified: ``Tests/UI/test_internal_prompts_panel_editing.py``.
