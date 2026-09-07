---
id: TASK-31381
title: Console Ctrl+C clears an unselected focused composer draft
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 00:00'
updated_date: '2026-09-04 19:24'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user discard the complete prompt currently being composed with Ctrl+C while the Console composer owns the cursor, without requiring repeated Backspace presses or losing the existing selected-draft copy behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Ctrl+C clears the complete unselected Console draft when the composer owns focus and resets the caret to the start.
- [x] #2 The Ctrl+C clear is undoable with Ctrl+Z.
- [x] #3 Ctrl+C continues to copy a fully selected draft instead of clearing it.
- [x] #4 Ctrl+C does not clear a draft when the composer does not own focus.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add mounted Console key-path regressions for focused clear, undo, selected copy, and the unfocused boundary; confirm the new clear behavior fails before production changes.
2. Route focused, unselected Ctrl+C through the composer's existing undoable full-clear operation, preserving the later selected-copy path.
3. Update the Console user guide to describe the contextual Ctrl+C behavior.
4. Run the targeted composer keymap/draft-change tests, focused static checks, and diff hygiene checks.
5. Add direct handler-level unit coverage for key consumption, history, clearing, and the `DraftChanged` notification in response to PR review.

ADR required: no
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
Reason: This is a focused text-editing behavior implemented through the existing composer key router, not a new screen/widget `BINDINGS` app action; ADR-031 remains the governing keybinding decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added one contextual Ctrl+C branch to the existing Console composer key router. When the composer itself owns focus and no full-draft selection is active, the branch reuses the established history-recording clear operation and posts the normal draft-change notification; the screen's later selected-copy path remains unchanged. Added mounted keypress coverage for clear/undo and the unfocused boundary, retained the existing selected-copy regression, and documented the contextual shortcut in the Console guide.

PR review follow-up added a direct `@pytest.mark.unit` handler test covering key consumption, the undo snapshot, the cleared draft and caret, and the non-insertion `DraftChanged` message while retaining the mounted integration cases.

ADR required: no. Existing ADR-031 remains applicable because this is composer text editing routed outside `BINDINGS`, not a new application action.

Verification: the new focused-clear test failed RED against the prior implementation while the other 14 composer-keymap cases passed. After implementation and the review follow-up, the complete composer-keymap file plus the selected-copy regression passed (17 tests). Ruff check, Ruff format check, `py_compile`, and scoped `git diff --check` passed. Pytest reported pre-existing dependency and temporary-directory cleanup warnings after the passing cases; no task-specific test failed.
<!-- SECTION:NOTES:END -->
