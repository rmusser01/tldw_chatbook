---
id: TASK-25719
title: First-run wizard's final step removes every exit affordance
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Steps one through five advertise Esc to leave setup and show it in the footer hint. The final step silently withdraws it: Esc stops working, the footer shows only Back, and no control is labelled Finish or Done. The only ways out are three body buttons whose names describe destinations rather than completion, so users cannot tell that setup is over.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Esc behaves consistently on every wizard step or its removal is explained on screen
- [ ] #2 The final step offers a clearly labelled completion action
- [ ] #3 The footer hint line matches the keys that actually work on the current step
<!-- AC:END -->
