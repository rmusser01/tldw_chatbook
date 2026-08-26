---
id: TASK-16247
title: Reconcile Console rail and popover interaction contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 12:02'
updated_date: '2026-08-14 12:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console rail geometry, switch guidance, and the quick-model popover tests aligned with the current visible interaction contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The workspace label contract preserves the 13-cell gutter.
- [x] #2 Disabled workspace switching exposes guidance only through its tooltip.
- [x] #3 The quick-popover streaming control is scrolled into view before pointer activation.
- [x] #4 The affected modules and exact checkpoint chunk pass.
- [x] #5 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: These are localized Console layout/copy and integration-test corrections within existing UI ownership.

1. Preserve the four chunk failures as RED evidence and trace their compositor state.
2. Remove the duplicate workspace recovery Static while retaining the disabled-button tooltip.
3. Update geometry and popover interaction tests to exercise the current visible controls.
4. Run focused tests, affected modules, the exact checkpoint chunk, and scoped static checks.
5. Self-review, record evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Removed the obsolete always-visible workspace-switch recovery Static; the existing disabled-button tooltip is now the single intent-scoped guidance surface described by the widget's contract.
- Updated the width test to the intentional 13-cell label column (12-character `Conversation` plus a one-cell gutter) and scrolled the popover's body control above its pinned footer before pointer activation.
- TDD evidence: the four chunk failures were reproduced before the edits and passed afterward. Full affected rail/workspace coverage passed 121 tests; exact checkpoint chunk 50 passed 480 tests, including TASK-16246's realtime regressions.
- `git diff --check` passed. Scoped Ruff lint/format reproduce only exact HEAD baseline drift in the legacy rail-section test and workspace-context widget; the changed width-budget test is formatted and no new lint finding was introduced.
- ADR check: no ADR was required because these changes preserve existing UI ownership and interaction policy.
<!-- SECTION:NOTES:END -->
