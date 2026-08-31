---
id: TASK-25728
title: Interrupt card consumes the entire transcript at eighty by twenty-four
status: To Do
assignee: []
created_date: '2026-08-31 05:10'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At the default terminal size the Console chrome compacts correctly, but a mounted interrupt card fills the whole content area and leaves no transcript visible. The user is asked to choose between sending, retrying and cancelling without being able to see the message the decision applies to.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An interrupt card leaves the relevant transcript context visible at the narrowest supported size
- [ ] #2 The card scrolls within its own region rather than displacing all content
- [ ] #3 The message under decision is identifiable from the card itself
<!-- AC:END -->
