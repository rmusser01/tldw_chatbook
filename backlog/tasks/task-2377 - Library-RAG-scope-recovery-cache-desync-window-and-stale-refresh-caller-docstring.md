---
id: TASK-2377
title: >-
  Library RAG scope-recovery cache desync window and stale refresh-caller
  docstring
status: Done
assignee:
  - '@codex'
created_date: '2026-08-04 20:07'
updated_date: '2026-09-17 05:06'
labels:
  - library
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR-T1 Task 6 (which closed task-2075) introduced a cached scope-recovery visibility flag (`_library_rag_scope_recovery_visible`) so the recovery banner would stay in sync with real source counts without redundant remove/mount work. Task 6's review noted two small residuals left behind:

1. The cache is not updated by `compose()` or by non-Search-row recompose paths, leaving a narrow desync window (it self-heals on the next real change to the recovery state, so this is low severity, not a repeat of the task-2075 defect).
2. `_sync_library_rag_scope_toggle_and_run_gate_widgets`'s docstring claims there are "four other refresh callers" when there are actually five — `_apply_library_rag_answer` is missing from the enumerated list. This is a documentation gap only; the lock that makes the callers safe lives in the callee, so correctness is unaffected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The scope-recovery cache stays consistent across `compose()` and non-Search-row recompose paths, or the residual desync window is explicitly documented as accepted (with the self-healing behavior noted)
- [x] #2 The docstring enumerates all five refresh callers accurately
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A; existing backlog/decisions/003-settings-library-rag-defaults.md and backlog/decisions/150-design-token-system-and-design-language.md apply.
Reason: repair an existing recovery-display cache lifecycle; no new ownership, service contract, storage, or visual language.
1. Reproduce stale scope recovery through mounted Search panel rebuilds and source snapshots, preserving the real state/render/controller path.
2. Bind recovery change-gating to the current rendered scope so panel replacement/recompose cannot inherit an obsolete cache; retain no-churn snapshots and the shared refresh lock. Correct the stale caller documentation to describe current entry points.
3. Run targeted recovery/query-gate/state checks, verify native dark/light wide/compact recovery transitions, independently review, update evidence/task/audit and commit locally. No full suite, push or merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recovery visibility now follows the mounted scope container. The previous boolean cache could suppress a required update after Search navigation or recomposition; a weak container reference plus visibility invalidates replaced widgets while retaining the existing refresh lock and same-container no-op behavior. The controller docstring now names all five full-refresh callers.

Changed the controller, RAG state and Library screen cache writes; added a six-case mounted regression matrix for both source-count directions through navigation, screen recompose and panel recompose. All six reproduced the original failure before repair. The targeted gate passed 268 tests; a final six-case rerun passed after strengthening no-churn assertions with a wrapped real mirror hook (overlapping tests, not an additional total). New files/state pass Ruff lint and formatting; existing controller/screen diagnostics are unchanged, and changed ranges pass formatting. Independent review has no remaining findings.

Native verification covered dark/light at 170x48 and 80x24, eight recovery transitions, four real local keyword searches and four keyboard Import media actions. All ten private databases pass read-only quick_check, the seeded source is unchanged, default profile fingerprints match, and shutdown/exit/PID checks pass. Source counts were injected to control timing; this does not qualify deletion, semantic search or provider calls.

Evidence and limits: Docs/superpowers/qa/2026-09-17-rag-scope-recovery/README.md. Updated the Library audit and the testing lesson about cache identity and false no-churn probes. ADR required: no; this directly repairs existing ADR-003 and ADR-150 behavior without changing storage, ownership, service contracts or visual language. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
