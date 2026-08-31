---
id: TASK-25733
title: >-
  Composer collapse hides the only input while the footer still advertises
  sending
status: To Do
assignee: []
created_date: '2026-08-31 05:10'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The control that collapses the composer sits immediately beside the input it hides and is labelled only with the word composer. After collapsing, the footer continues to advertise Enter to send and queue although no input exists. Console also reports no active conversation in the rail while the tab bar and title both name one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The collapse control names the outcome it produces
- [ ] #2 Footer hints reflect only the keys that work in the current state
- [ ] #3 Rail, tab bar and title agree on whether a conversation is active
<!-- AC:END -->
