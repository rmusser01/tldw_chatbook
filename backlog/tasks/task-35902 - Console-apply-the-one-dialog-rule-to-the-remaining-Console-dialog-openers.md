---
id: TASK-35902
title: 'Console: apply the one-dialog rule to the remaining Console dialog openers'
status: To Do
assignee: []
created_date: '2026-10-21 01:25'
labels:
  - console
  - crash
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-34720 put a 'the Console must be the top screen' check immediately before the push in the Console's status-line and Library access dialog openers, so a burst of activations opens one dialog instead of a stack (a typed sentence had stacked enough translucent modals to crash the app with RecursionError). Other Console openers still push without that check; the TASK-34720 review named the composer menu itself, the prompts modal, the '/system <args>' prompt-picker fallback, the style picker, the session and workspace switchers, and session settings. They are reached from Buttons (Enter only) or key bindings, so typed text does not repeat them the way it repeated the status-line segments, but nothing stops a burst of their triggers from stacking. Each new opener also has to remember the rule on its own; one shared helper would make it the default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Console push site is inventoried and classified as user-triggered dialog, nested sub-dialog or system prompt, with the classification recorded in the task notes
- [ ] #2 However many activations of each user-triggered Console dialog arrive together, at most one of its dialogs opens, and it opens again once that dialog closes
- [ ] #3 The Console's user-triggered dialog openers, including the ones TASK-34720 guarded, check through one shared helper
- [ ] #4 A parametrized Pilot test on the real ChatScreen covers each newly guarded opener and fails on the code before the change
<!-- AC:END -->
