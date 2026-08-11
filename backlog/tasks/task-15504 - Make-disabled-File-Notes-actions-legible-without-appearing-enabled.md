---
id: TASK-15504
title: Make disabled File Notes actions legible without appearing enabled
status: To Do
assignee: []
created_date: '2026-08-11 20:56'
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
- [ ] #1 Disabled action labels in the File Notes workspace and Session Git panel render at least 3:1 against their own background in every shipped theme, measured in a running terminal rather than inferred from token values.
- [ ] #2 Disabled controls remain visibly distinct from enabled controls through a stable non-color cue and never appear actionable.
- [ ] #3 Whenever a disabled action has a reason, that reason remains readable without hover and identifies the recovery path when one exists.
- [ ] #4 The fix uses the app stylesheet tier required to override Textual disabled styling and does not duplicate theme-specific literal colors inside File Notes widgets.
- [ ] #5 Focused rendered-color regressions cover representative dark, light, and high-contrast themes plus 40x20 and normal layouts; targeted lint, CSS bundle generation checks, and diff checks pass.
<!-- AC:END -->
