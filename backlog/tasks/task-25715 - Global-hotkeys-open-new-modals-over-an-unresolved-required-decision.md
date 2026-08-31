---
id: TASK-25715
title: Global hotkeys open new modals over an unresolved required decision
status: Done
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 13:57'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a blocking interrupt card is mounted and asking the user to choose one action, Ctrl+K and F1 each open another modal on top of it, producing three stacked layers with the original decision still pending. Panels also render without a scrim, so a modal reads as an inline card and clicks outside it silently do nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Global hotkeys that open panels are suppressed while a blocking decision card is mounted
- [ ] #2 Modal surfaces render a scrim that visually separates them from live content
- [ ] #3 Clicks outside a modal produce a visible response rather than silently doing nothing
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the stacking half. The recovery callouts ask 'Choose one action' and hold the turn, but they are mounted INSIDE the workbench rather than pushed as screens, so nothing stopped Ctrl+K or F1 opening a panel over the top -- I reached three layers deep with the original decision unresolved. Added _console_decision_blocking(), mirroring the existing _console_setup_modal_blocking() that these same actions already honour for the first-run modal, and wired it into the Ctrl+K switcher and F1 help.

NOT addressed (needs a decision, not a patch): the missing scrim. Textual dims pushed SCREENS, but these callouts are in-workbench widgets by design -- that is what lets the transcript stay visible behind them (see TASK-25728). Adding a scrim means either promoting them to screens, which costs that visibility, or hand-dimming the region. Both are design changes rather than fixes, so I left the visual layer alone and closed the input-routing hole instead.

Baseline confirmed unchanged: 2 pre-existing failures in test_console_workbench_contract.py.
<!-- SECTION:NOTES:END -->
