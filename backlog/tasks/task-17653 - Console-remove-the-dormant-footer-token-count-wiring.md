---
id: TASK-17653
title: 'Console: remove the dormant footer token-count wiring'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - console
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AppFooterStatus` mounts `#footer-token-count` on Console with `show_token_count=True`, but the widget is dormant: its only writer (`update_token_count_display` -> `update_chat_token_counter`) gates the footer branch on `not app._use_screen_navigation`, which is never true in screen-navigation mode. It is one flag flip away from rendering a full token readout one row below the cost chip's compact "2.7k tok" — a silent future duplicate.

Owner decision (2026-08-17): the cost chip is the single token/cost surface on Console; delete the never-taken path rather than leaving the latent duplicate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The dormant footer token-count path for Console is removed (dead writer branch and/or the Console mount flag), with no visible change to the current default footer
- [ ] #2 The cost chip remains the only token/cost readout on Console, and a test pins that the footer token counter cannot appear there
- [ ] #3 Non-Console footer features (word count, DB size indicator, key hints, responsive reflow) are unaffected
<!-- AC:END -->
