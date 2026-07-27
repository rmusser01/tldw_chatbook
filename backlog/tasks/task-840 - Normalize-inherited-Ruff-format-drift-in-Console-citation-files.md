---
id: TASK-840
title: Normalize inherited Ruff format drift in Console citation files
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 04:02'
updated_date: '2026-07-27 19:54'
labels:
  - maintenance
  - formatting
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair pre-existing whole-file Ruff formatting drift in the four Console files surfaced by TASK-553.15 without mixing broad mechanical churn into the citation-repair feature implementation:

- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Widgets/Console/console_transcript.py`

The same four files fail the formatter check on `origin/dev`, so this work is tracked independently of TASK-553.15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The exact TASK-553.15 eleven-file Ruff format check exits zero
- [x] #2 Formatting changes are mechanical and behavior-preserving
- [x] #3 Scoped Console citation and UI tests pass after formatting
- [x] #4 `git diff --check` passes and formatter-only scope is independently reviewed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact TASK-553.15 eleven-file Ruff format check and confirm only the four recorded Console files fail.
2. Run Ruff format on exactly those four files; do not hand-edit or change behavior.
3. Review the complete diff as formatter-only and run the exact eleven-file formatter gate, Ruff check on the four touched files, focused Console citation/UI tests, and git diff --check.
4. Record verification and self-review, complete all acceptance criteria, and mark TASK-840 Done.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter normalization only; no behavior, interface, ownership, storage, security, or architectural decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ran Ruff format on exactly the four inherited-drift files recorded by
TASK-553.15. No production logic or tests were hand-edited. A complete
zero-context diff review plus parsed-AST and comment-token comparisons against
`dev` confirmed that executable structure and comment text are unchanged.

ADR required: no. ADR path: N/A. This is mechanical formatter normalization
with no architectural or behavioral change.

Verification: the exact eleven-file Ruff format check reports all eleven files
formatted; Ruff check passes on the four touched production files; 95 focused
controller/agent citation tests and 20 transcript/native-flow citation tests
pass with offline model flags; and `git diff --check` passes. The UI group
emits one existing RequestsDependencyWarning.
<!-- SECTION:NOTES:END -->
