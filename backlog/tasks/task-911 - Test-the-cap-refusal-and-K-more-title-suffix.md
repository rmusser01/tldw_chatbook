---
id: TASK-911
title: 'Test the cap-refusal and-K-more title suffix'
status: To Do
assignee: []
created_date: '2026-07-27 03:55'
labels: [console, tests]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
send_refusal_copy's cap message truncates busy-session titles to 3 plus an "and K more" suffix. The suffix branch (more than 3 busy sessions) has no test; it is user-facing spec copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A unit test drives 4+ busy sessions and asserts the exact "and K more" refusal copy.
<!-- AC:END -->
