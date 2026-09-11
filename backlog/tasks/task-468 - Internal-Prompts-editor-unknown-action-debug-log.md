---
id: TASK-468
title: 'Internal Prompts editor: debug-log the modal''s unknown-action branch'
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - polish
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InternalPromptsPanel._apply_editor_result has a silent `else: return` for a result action other than save/reset/None. It is currently unreachable (the modal's dismiss contract is closed), but a future modal change could hit it silently. Add a one-line debug log for defense-in-depth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unknown-action branch logs at debug level with the unexpected action value
- [x] #2 No behavior change for the save/reset/None paths
<!-- AC:END -->

## Implementation Plan

1. RED: add a test driving _apply_editor_result with an unknown action, capturing loguru records at DEBUG level and asserting the action value is logged and nothing persists.
2. Add the debug log to the else branch (plus the loguru import).

ADR required: no
ADR path: N/A
Reason: One-line defense-in-depth log; no behavior or contract change.

## Implementation Notes

``_apply_editor_result``'s unknown-action branch now logs at debug level with the unexpected action value and prompt id before returning. TDD evidence: ``test_unknown_editor_action_is_debug_logged_and_ignored`` (Tests/UI/test_internal_prompts_panel_editing.py) failed on unmodified dev (no record captured) and passes with the log; it also asserts the override is not persisted. Save/reset paths unchanged (their existing tests still pass; 12 passed across the four internal-prompts files). Ruff: one pre-existing BLE001 remains in the file at HEAD; my lines are clean.
