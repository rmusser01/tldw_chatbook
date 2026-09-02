---
id: TASK-27019
title: Recompose census needs an anti-slack guard like the size ratchet
status: To Do
assignee: []
created_date: '2026-09-02 15:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final review of the library decomposition foundation: Tests/UI/test_library_recompose_ratchet.py pins a ceiling only; headroom drift happened twice before (107->80, 74->63). Mirror test_budget_is_not_left_slack_after_a_wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Census pin has a slack guard with a documented tolerance
- [ ] #2 Guard is mutation-tested (headroom injected -> fails)
<!-- AC:END -->
