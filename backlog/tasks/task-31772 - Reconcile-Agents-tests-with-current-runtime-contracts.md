---
id: TASK-31772
title: Reconcile Agents tests with current runtime contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 09:08'
updated_date: '2026-09-05 09:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the latest dev Agents suite after intentional runtime-surface changes and make filesystem isolation tests hermetic in linked worktrees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime-tool and Library inventory assertions derive from the current canonical sets.
- [x] #2 Filesystem and worktree integration tests exercise current checkout code in linked worktrees.
- [x] #3 Raw-shell and trace tests supply current context requirements.
- [x] #4 The complete `Tests/Agents` suite passes.
- [x] #5 Static checks and diff validation pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update stale exact-inventory assertions to derive from canonical runtime contracts.
2. Replace linked-worktree-incompatible subprocess test executors with an in-process protocol executor at the test seam.
3. Supply the current profile and model context required by raw-shell and trace tests.
4. Run focused regressions followed by the complete Agents suite.
5. Run static checks and document verification.

ADR required: no

ADR path: N/A

Reason: test-harness and expectation repairs preserve existing production architecture and ADR-defined boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced stale fixed runtime/Library counts with assertions derived from the canonical sets, supplied the current raw-shell profile and large-context trace model, and routed stale-write plus dynamic agent-worktree tests through the current checkout's in-process workspace protocol harness. This preserves subprocess containment coverage elsewhere while making linked-worktree behavior tests hermetic. Recorded the editable-install checkout trap in the testing lessons. Verification: the eight-file focused set passed 545 tests, the final `Tests/Agents` run passed 2,581 tests, Ruff passed on all touched Python files, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
