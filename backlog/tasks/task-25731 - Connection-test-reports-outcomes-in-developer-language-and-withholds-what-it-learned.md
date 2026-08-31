---
id: TASK-25731
title: >-
  Connection test reports outcomes in developer language and withholds what it
  learned
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
A failed provider test reports that the model listing request had a connection error, naming neither the address tried, the reason, nor a next step. A partial success reports that the listing was reached but the selected model was not confirmed, without naming the models the server just returned. In both cases the product knows more than it says, and the copy is written for the implementer rather than the user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed test names the address tried, the reason, and one concrete next step
- [ ] #2 A successful listing reports the models it discovered
- [ ] #3 Test result copy matches the plain-language standard used elsewhere in setup
<!-- AC:END -->
