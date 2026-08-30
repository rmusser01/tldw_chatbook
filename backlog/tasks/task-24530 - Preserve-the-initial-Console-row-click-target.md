---
id: TASK-24530
title: Preserve the initial Console row click target
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 02:03'
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
- [x] #1 An empty left-button gesture toggles the message under its initial press exactly once whether or not Textual emits a later `Click`
- [x] #2 Dismissing an existing selection menu and highlight cannot retarget same-row, different-row, Markdown, or diff row activation
- [x] #3 Escape during an armed press clears the menu, highlight, selection manager, origin row, and mouse capture without replaying a stale target
- [x] #4 Right-button, protected-control, menu-interior, negative-space, keyboard-selection, and genuine-drag behavior remain unchanged
- [x] #5 Raw MouseDown/MouseUp and ordinary pilot interaction regressions prove missing-Click, optional-Click exact-once, repeated-gesture, and row-removal behavior
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented press-time row latching before layout-changing cleanup, MouseUp-owned exact-once toggle with existing optional-Click suppression, and ordinary-Escape capture release. Added raw no-Click/replay, pilot exact-once, Escape, right-button, menu-descendant, same-row, and Markdown retargeting regressions; existing focused coverage retains diff-row, removed-row, protected-control, negative-space, drag, and keyboard behavior. Modified Tests/UI/test_console_selection_transcript.py and tldw_chatbook/Widgets/Console/console_transcript.py. TDD RED evidence reproduced the m1-versus-m2 pilot failure, None-versus-m2 raw failure, and retained Escape capture; task-specific GREEN verification passed 25 transcript tests. Complete six-file Console slice produced 193 passed and one inherited keyboard-anchor baseline failure at Tests/UI/test_console_keyboard_selection.py::test_menu_anchor_derives_from_row_region_and_stays_in_transcript; the identical y=4 versus y=5 failure was independently reproduced on untouched commit 1b4b0c86a7 and is unrelated to this mouse-event change. Ruff check, compileall, and git diff --check passed; Ruff format retained exactly the inherited three-file baseline. Independent specification and code-quality reviews approved the final implementation after stale lifecycle comments were corrected. Full repository suite not run under targeted-test policy. ADR required: no. Lessons learned: none added; no new general rule beyond the approved event-ordering design.
<!-- SECTION:NOTES:END -->
