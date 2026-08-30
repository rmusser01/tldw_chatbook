---
id: TASK-24530
title: Preserve the initial Console row click target
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 00:59'
labels:
  - console
  - textual
  - interaction
dependencies:
  - TASK-24529
references:
  - >-
    Docs/superpowers/specs/2026-08-29-console-selection-click-target-stability-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-console-selection-stability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure a plain Console row click activates the message under the initial press even when dismissing text-selection UI changes transcript geometry before release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An empty left-button gesture toggles the message under its initial press exactly once whether or not Textual emits a later `Click`
- [ ] #2 Dismissing an existing selection menu and highlight cannot retarget same-row, different-row, Markdown, or diff row activation
- [ ] #3 Escape during an armed press clears the menu, highlight, selection manager, origin row, and mouse capture without replaying a stale target
- [ ] #4 Right-button, protected-control, menu-interior, negative-space, keyboard-selection, and genuine-drag behavior remain unchanged
- [ ] #5 Raw MouseDown/MouseUp and ordinary pilot interaction regressions prove missing-Click, optional-Click exact-once, repeated-gesture, and row-removal behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement and verify TASK-24529 first.
2. Add raw MouseDown/MouseUp and optional-Click exact-once regressions.
3. Preserve initial row identity through empty MouseUp and complete Escape cleanup.
4. Verify branch, Markdown, drag, keyboard, and dismissal controls.
5. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-console-selection-stability.md
ADR required: no
ADR path: N/A
Reason: existing Console pointer-event ordering only.
<!-- SECTION:PLAN:END -->
