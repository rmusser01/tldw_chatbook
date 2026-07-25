---
id: TASK-640
title: Wait for terminal Console RAG blocker state
status: Done
assignee: []
created_date: '2026-07-25 21:30'
updated_date: '2026-07-25 22:59'
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
- [x] #7 The missing-service scenario explicitly sets app.library_rag_search_service to None
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
Summary: Made the missing-service Console RAG test independent of both asynchronous card timing and process-wide RAG singleton state.

RED evidence and complete root cause:
- The first permitted full-suite run reached 7,521 passed and 198 skipped, then the test read before its worker reached the terminal card. The run action mounts Status: searching before the worker updates the same selector.
- Replacing the selector wait with a semantic Status: blocked wait exposed a second failure in a fresh full-suite run at the same 7,521/198 point: terminal Status: failed with retrieval-failed recovery.
- `_build_test_app()` always wires `LibraryLocalRagSearchService`; the test title and assertions claimed a missing service but never removed it. Isolated runs happened to see the lazy shared runtime unavailable, while full-suite order could leave the process-wide runtime initialized, allowing the wired service to run and fail.
- The final test fixture explicitly sets `app.library_rag_search_service = None`, then waits for the RAG button to become enabled and for visible Status: blocked. This directly constructs the scenario under test and removes both timing and singleton-order dependence.
- Recoverable blocker assertions are unchanged. No production Console, Library RAG, worker, singleton, or UI code changed.

Verification:
- Exact regression with a deliberately pre-populated exploding shared RAG runtime: 1 passed, proving the global runtime is irrelevant.
- Exact regression in five fresh independent pytest processes after the final fix: 5/5 passed.
- Five neighboring Console RAG workflow tests: 5 passed.
- Full Tests/UI/test_console_internals_decomposition.py after the final fix: 123 passed.
- Ruff format check: file already formatted.
- Ruff check: all checks passed.
- py_compile: passed.
- git diff --check: passed.
- Self-review: the test now explicitly owns its missing-service precondition and waits on public terminal UI behavior; assertions and production behavior are unchanged.

ADR required: no
ADR path: N/A
Reason: This corrects test setup and synchronization around existing service and asynchronous UI contracts; it makes no architectural decision.

Files modified:
- Tests/UI/test_console_internals_decomposition.py
- backlog/tasks/task-640 - Wait-for-terminal-Console-RAG-blocker-state.md
<!-- SECTION:NOTES:END -->
