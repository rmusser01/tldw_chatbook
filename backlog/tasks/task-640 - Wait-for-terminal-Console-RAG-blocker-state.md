---
id: TASK-640
title: Wait for terminal Console RAG blocker state
status: Done
assignee: []
created_date: '2026-07-25 21:30'
updated_date: '2026-07-25 21:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the missing-service Console RAG integration test synchronize on the terminal blocked outcome instead of the status card selector that is already mounted during the intermediate searching state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test waits for the RAG run button to become enabled before pressing it
- [x] #2 The test waits for visible Status: blocked text rather than selector existence
- [x] #3 The recoverable blocker assertions remain unchanged
- [x] #4 Focused and surrounding Console RAG tests pass repeatedly
- [x] #5 Production Console and RAG code remains unchanged
- [x] #6 Task notes record full-suite RED evidence ADR decision verification and self-review
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the 59% full-suite race and trace the worker/card transition.
2. Replace fixed timing and non-terminal selector waits with existing semantic wait helpers.
3. Run the exact test repeatedly, the neighboring Console RAG tests, and the full Console internals file.
4. Run Ruff/format checks and git diff --check; self-review the test-only delta.

ADR required: no
ADR path: N/A
Reason: This corrects test synchronization around an existing asynchronous UI contract and changes no production behavior or interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Removed a load-sensitive race from the missing-service Console RAG integration test by synchronizing on existing semantic UI states.

RED evidence and root cause:
- The permitted repository-wide fail-fast run reached 7,521 passed and 198 skipped before this test failed at 59%; the status selector existed but visible text had not yet reached Status: blocked.
- The run action synchronously stages an intermediate Status: searching card, then a Textual worker resolves the absent service and replaces the same card with Status: blocked. Waiting for #console-live-work-status therefore admitted the non-terminal state.
- Replaced the fixed 0.1-second query delay with the existing run-button-state helper and replaced selector existence with the existing visible-text helper for Status: blocked.
- All recoverable blocker assertions remain unchanged. No production Console, Library RAG, worker, or UI code changed.

Verification:
- Exact regression in five independent pytest processes: 5/5 passed.
- Five neighboring Console RAG workflow tests: 5 passed.
- Full Tests/UI/test_console_internals_decomposition.py: 123 passed.
- Ruff format check: file already formatted.
- Ruff check: all checks passed.
- py_compile: passed.
- git diff --check: passed.
- Self-review: both replacements remove timing assumptions and wait for public mounted behavior; assertion scope and production behavior are unchanged.

ADR required: no
ADR path: N/A
Reason: This corrects test synchronization around an existing asynchronous UI contract and makes no architectural decision.

Files modified:
- Tests/UI/test_console_internals_decomposition.py
- backlog/tasks/task-640 - Wait-for-terminal-Console-RAG-blocker-state.md
<!-- SECTION:NOTES:END -->
