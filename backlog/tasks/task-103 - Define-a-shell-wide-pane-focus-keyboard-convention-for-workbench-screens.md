---
id: TASK-103
title: Define a shell-wide pane-focus keyboard convention for workbench screens
status: Done
assignee: []
created_date: 2026-06-11 20:43
labels:
- ux
- keyboard
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Personas workbench (UX-E2) wanted pane-jump keys to cycle focus between the library list, work area, and inspector, but no house pattern exists: chat_screen.py and notes_screen.py define no pane-focus bindings at all, and ctrl+left/ctrl+right collide with Input/TextArea word-navigation (the keys are consumed inside text fields, making screen-level cycling inconsistent). Rather than invent a one-off convention on one screen, define a single shell-wide convention (key choice, wrap order, focus-steal rules) and apply it to Console, Notes, and Personas together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] A documented pane-focus key convention exists for workbench screens, including keys, cycle order, and text-input interaction rules.
- [x] Console, Notes, and Personas implement the convention consistently.
- [x] The chosen keys do not conflict with Input/TextArea default bindings or app-level bindings.
- [x] Focus cycling skips unavailable or collapsed panes and wraps predictably.
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: This standardizes keyboard focus behavior inside existing screen boundaries without changing storage, service contracts, data ownership, or long-lived architecture.

1. Add failing mounted UI regressions for F6 and Shift+F6 pane cycling in Console, Notes, and Personas.
2. Add a small shared workbench focus helper that resolves visible pane targets and focuses each pane's preferred child.
3. Add F6/Shift+F6 bindings and per-screen target maps without changing existing shortcuts.
4. Document the convention in project UX docs and link it from this task.
5. Run focused UI regressions and formatting/diff checks before marking done.

## Implementation Notes

- Added a shared `WorkbenchPaneTarget`/`focus_relative_workbench_pane` helper that resolves visible pane targets, skips hidden/collapsed descendants, and wraps through available panes.
- Added priority `F6` and `Shift+F6` bindings to Console, Notes, and Personas so explicit pane jumps work from focused text inputs without using `Ctrl+Left`/`Ctrl+Right`.
- Documented the convention in `Docs/superpowers/specs/2026-06-25-workbench-pane-focus-keyboard-convention-design.md`.
- Added mounted regressions in `Tests/UI/test_workbench_pane_focus.py` covering Console, Notes, Personas, hidden-target fallback, wrapping, and binding collision checks.
- Verification: `python -m pytest -q Tests/UI/test_workbench_pane_focus.py ... --tb=short` passed for the new module plus focused Notes/Personas keyboard regressions; `git diff --check` passed.
