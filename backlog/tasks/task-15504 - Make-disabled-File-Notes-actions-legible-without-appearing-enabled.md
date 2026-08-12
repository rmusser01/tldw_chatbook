---
id: TASK-15504
title: Make disabled File Notes actions legible without appearing enabled
status: Done
assignee: []
created_date: '2026-08-11 20:56'
updated_date: '2026-08-11 22:22'
labels:
  - notes
  - filesystem
  - accessibility
  - theming
  - ux
dependencies: []
references:
  - >-
    backlog/tasks/task-1801 -
    Disabled-control-labels-are-unreadable-at-1-1-contrast.md
  - DESIGN.md
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rendered File Notes actions inherit the global disabled stack of text-disabled, 50 percent color, and Textual dimming without the app-tier override already used on other surfaces. This can push labels below the DESIGN.md 3:1 minimum, hiding both the action and its reason in a trust-sensitive local-file workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled action labels in the File Notes workspace and Session Git panel render at least 3:1 against their own background in every shipped theme, measured in a running terminal rather than inferred from token values.
- [x] #2 Disabled controls remain visibly distinct from enabled controls through a stable non-color cue and never appear actionable.
- [x] #3 Whenever a disabled action has a reason, that reason remains readable without hover and identifies the recovery path when one exists.
- [x] #4 The fix uses the app stylesheet tier required to override Textual disabled styling and does not duplicate theme-specific literal colors inside File Notes widgets.
- [x] #5 Focused rendered-color regressions cover representative dark, light, and high-contrast themes plus 40x20 and normal layouts; targeted lint, CSS bundle generation checks, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add CSS-true rendered-color regressions for disabled File Notes workspace and Session Git actions across representative dark, light, high-contrast, normal, and 40x20 layouts.
2. Add a scoped app-tier disabled treatment using semantic theme tokens and preserve visible recovery reasons without duplicating literal colors.
3. Regenerate the CSS bundle and run focused behavior, rendered-color, layout, lint, compile, and diff checks.
4. Complete the task acceptance criteria and implementation notes after self-review.

ADR required: no
ADR path: N/A
Reason: This is a routine accessibility and visual-state correction within ADR-011 and DESIGN.md's established Legible Disabled contract; no storage, ownership, service, security, or long-lived application boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the Library's visible disabled-action marker to File Notes editor and Session Git buttons, including dynamic commit and push states, while preserving plain labels when actions become available again.
- Added explicit on-screen wait and recovery guidance for transient file and Git operations so disabled reasons do not depend on hover.
- Added an app-tier semantic-token override that neutralizes the compounded disabled opacity and keeps both action labels and recovery copy readable without theme-specific literals. Regenerated `tldw_cli_modular.tcss` from the source modules.
- Added CSS-true compositor regressions that measure actual painted contrast across both built-in themes and every shipped custom theme at normal size, plus dark, light, and high-contrast coverage at 40x20. The focused surrounding File Notes, commit, push, and layout matrix passed 28 tests.
- Verification: rendered-state behavior 1 passed; every-shipped-theme compositor sweep 1 passed; 40x20 compositor matrix 1 passed; focused surrounding UI matrix 28 passed; Ruff, compileall, CSS generation, and `git diff --check` passed.
- ADR required: no. ADR-011 and DESIGN.md's Legible Disabled rule already define the applicable UI and accessibility contract.
<!-- SECTION:NOTES:END -->
