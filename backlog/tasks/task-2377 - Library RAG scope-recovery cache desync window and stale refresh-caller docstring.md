---
id: TASK-2377
title: Library RAG scope-recovery cache desync window and stale refresh-caller docstring
status: To Do
assignee: []
created_date: '2026-08-04 20:07'
labels:
  - library
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

PR-T1 Task 6 (which closed task-2075) introduced a cached scope-recovery visibility flag (`_library_rag_scope_recovery_visible`) so the recovery banner would stay in sync with real source counts without redundant remove/mount work. Task 6's review noted two small residuals left behind:

1. The cache is not updated by `compose()` or by non-Search-row recompose paths, leaving a narrow desync window (it self-heals on the next real change to the recovery state, so this is low severity, not a repeat of the task-2075 defect).
2. `_sync_library_rag_scope_toggle_and_run_gate_widgets`'s docstring claims there are "four other refresh callers" when there are actually five — `_apply_library_rag_answer` is missing from the enumerated list. This is a documentation gap only; the lock that makes the callers safe lives in the callee, so correctness is unaffected.

## Acceptance Criteria

- [ ] The scope-recovery cache stays consistent across `compose()` and non-Search-row recompose paths, or the residual desync window is explicitly documented as accepted (with the self-healing behavior noted)
- [ ] The docstring enumerates all five refresh callers accurately
