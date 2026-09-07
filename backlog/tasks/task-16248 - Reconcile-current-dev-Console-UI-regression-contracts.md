---
id: TASK-16248
title: Reconcile current-dev Console UI regression contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 12:24'
updated_date: '2026-08-14 12:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console character swapping, staged evidence copy, compact geometry, and keyboard-focus regression tests aligned with the current supported UI contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character-swap coverage uses the typed prompt seed and current controller/store boundary.
- [x] #2 Staged-evidence assertions match the concise current copy.
- [x] #3 Compact geometry distinguishes off-screen clipped regions without assuming a hit-test target.
- [x] #4 The focus tour accounts for the actionable provider recovery control and still reaches status chips within ten stops.
- [x] #5 All four affected modules and checkpoint chunk 51 pass.
- [x] #6 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is test reconciliation for already-shipped Console contracts with no ownership or behavior change.

1. Preserve the 13 current-dev failures as RED evidence and trace each to its source contract.
2. Update only the four stale test modules with current typed, copy, geometry, and focus behavior.
3. Run focused nodes, full affected modules, and exact checkpoint chunk 51.
4. Run scoped static checks and compare any legacy failures with HEAD.
5. Self-review, record evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the obsolete positional character-swap fake with the typed prompt seed and real in-memory Console store, preserving the empty-chat-only greeting rule through the current roleplay persistence boundary.
- Updated staged-source and blocked-send assertions to the concise current copy; no production copy changed.
- Classified the three off-screen narrow-layout regions as clipped and made the geometry oracle accept Textual's explicit no-widget result without weakening mounted/display/positive-width checks.
- Allowed the focus tour to traverse the intentional provider-recovery CTA before reaching status chips while retaining the ten-stop and no-navigation detour bounds.
- RED evidence was 13 failures across four test modules. The four affected modules passed 77 tests, and exact checkpoint chunk 51 passed 275 tests in 4m13s.
- `git diff --check`, artifact scope, and scoped format checks passed for the two clean files. The other two files retain only their exact HEAD formatter drift; scoped Ruff improves the chip-actions file from two baseline findings to one unchanged unused import.
- ADR check: no ADR was required because this task updates tests for existing typed, copy, layout, and focus contracts without changing production architecture or behavior.
<!-- SECTION:NOTES:END -->
