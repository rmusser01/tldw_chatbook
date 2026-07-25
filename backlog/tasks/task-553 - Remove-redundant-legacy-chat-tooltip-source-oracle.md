---
id: TASK-553
title: Remove redundant legacy chat tooltip source oracle
status: Done
assignee: []
created_date: '2026-07-25 17:02'
updated_date: '2026-07-25 17:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stop treating source-code whitespace in the legacy chat window as tooltip behavior when mounted widget tests already verify the same controls and dynamic send/stop state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tooltip behavior remains covered by mounted Textual tests
- [x] #2 The formatting-sensitive source oracle is removed without production changes
- [x] #3 Adjacent legacy chat tooltip and send-stop tests pass
- [x] #4 Task notes record RED evidence ADR decision and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact source-string failure and inventory overlapping behavioral coverage.
2. Remove only the redundant source-text oracle.
3. Run the mounted tooltip and send-stop tests plus the full UI fail-fast slice.
4. Review and document the change.

ADR required: no
ADR path: N/A
Reason: This removes a redundant formatting-sensitive test and changes no runtime behavior or application boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Removed the redundant legacy chat tooltip source-text test while retaining the mounted Textual tests that verify the same buttons and dynamic send/stop behavior.

RED evidence:
- The source oracle required a single-line `tooltip="Send message" if ...` expression, while Ruff-formatted production source correctly split the expression across lines.
- `test_chat_window_tooltips.py`, `test_chat_window_tooltips_fixed.py`, and `test_send_stop_button.py` already mount the widget and verify the actual tooltip values and state transitions.

Verification:
- The three adjacent behavioral modules: 14 passed.
- Full UI fail-fast rerun passed the removed oracle and advanced through 530 tests before stopping on an unrelated collapsed-composer interaction failure.
- Ruff check for the retained behavioral modules: passed.
- `git diff --check`: passed.
- Diff review confirmed no production file changed and the broader legacy chat decomposition remains deferred.

ADR required: no
ADR path: N/A
Reason: This removes redundant formatting-sensitive test coverage and changes no runtime behavior or application boundary.

Files modified:
- `Tests/UI/test_chat_window_tooltips_simple.py` (removed)
- `backlog/tasks/task-553 - Remove-redundant-legacy-chat-tooltip-source-oracle.md`
<!-- SECTION:NOTES:END -->
