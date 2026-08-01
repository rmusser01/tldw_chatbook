---
id: TASK-1801
title: 'Disabled control labels are unreadable at ~1.1:1 contrast'
status: To Do
assignee: []
created_date: '2026-08-01 13:20'
labels:
  - console
  - ux
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of the temporary-conversations work (2026-08-01) measured a disabled control's label at foreground `rgb(31,31,31)` on background `rgb(3,3,3)` — roughly 1.1:1 contrast, effectively unreadable without hovering for the tooltip.

This is a general Console convention, not one feature's bug, but it undercuts that feature specifically. Temporary conversations communicate **every** restriction through a disabled control that states its reason: Generate Image, Save Chatbook, and six save-as sinks all render disabled-with-a-reason rather than being hidden, on the explicit principle that a user who cannot find an action assumes the app is broken, while one who sees it greyed out with a reason learns the rule.

That principle fails if the greyed-out label cannot be read. The user gets the worst of both: the action is visibly present, apparently broken, and its explanation is invisible.

A related instance was already fixed in the composer's ☰ menu (a disabled row's reason was tooltip-only and now renders on screen in `$warning`), but the underlying disabled-label styling is unchanged everywhere else.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disabled control labels meet a stated minimum contrast ratio against their background, and the chosen threshold is recorded in DESIGN.md
- [ ] #2 A disabled control's reason is discoverable without hovering, wherever a reason exists
- [ ] #3 Disabled still reads as visually distinct from enabled — fixing contrast must not make disabled controls look actionable
- [ ] #4 Verified by measuring real rendered colours in a terminal, not by reading token values, since the defect was found by measurement and token names did not reveal it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured during the 2026-08-01 live pass on the Console composer's Generate Image row in a temporary chat. The reason text itself was correct and correctly sourced from `EPHEMERAL_BLOCKED_ACTIONS`; only its legibility failed.

Check `.console-action-disabled` and the `$ds-*` disabled tokens in `tldw_chatbook/css/components/_agentic_terminal.tcss`. Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate it with `python tldw_chatbook/css/build_css.py`.
<!-- SECTION:NOTES:END -->
