---
id: TASK-24194
title: Move rewind-position mutation behind the Console message controller
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 05:53'
updated_date: '2026-08-29 05:57'
labels:
  - console
  - architecture-gate
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the one-way ChatScreen size ratchet after durable rewind-before-first handling added screen-owned state mutation that belongs to the existing Console message controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ChatScreen line-count and method-count ratchets pass without increasing either budget.
- [x] #2 Rewind to the first and later prompts preserves active-path behavior and the restart-persistence warning.
- [x] #3 Focused rewind and architecture regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a routine ownership repair within the approved Console decomposition and durable rewind design; it changes no storage policy or public contract.

1. Use the existing red screen-size ratchet and rewind tests as characterization.
2. Move rewind-position mutation and persistence-warning ownership into ConsoleMessageController while keeping ChatScreen orchestration thin.
3. Run focused rewind and architecture tests, Ruff, and diff checks; document evidence and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the durable rewind cursor mutation and persistence-failure warning from
`ChatScreen` into the existing `ConsoleMessageController`. The screen remains the
modal/orchestration owner while the message controller now owns active-path mutation;
user-visible behavior and storage policy are unchanged. `chat_screen.py` fell from
16,974 to 16,965 physical lines against the unchanged 16,966-line budget. Verification:
23 rewind UI tests and four Console architecture/size ratchets passed; scoped Ruff,
`py_compile`, and `git diff --check` passed. ADR required: no; this is a routine repair
inside the existing Console decomposition boundary.
<!-- SECTION:NOTES:END -->
