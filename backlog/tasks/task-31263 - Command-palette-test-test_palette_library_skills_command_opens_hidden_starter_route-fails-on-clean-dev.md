---
id: TASK-31263
title: >-
  Command palette test
  test_palette_library_skills_command_opens_hidden_starter_route fails on clean
  dev
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-04 13:47'
updated_date: '2026-09-04 20:30'
labels:
  - tests
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_command_palette_providers.py::TestTabNavigationProvider::test_palette_library_skills_command_opens_hidden_starter_route fails identically on clean origin/dev (verified at 2516735cfd during PR #2374 work, run in its own process). It is in the same hidden-starter-route family as the six Library failures tracked by task-31249 but lives in a different file and is not covered by that task's list. Every PR touching command-palette or theme code has to hand-verify this failure is baseline, which is how real breaks hide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev, or is rewritten/removed with the reason recorded in this task (no bare skip markers)
- [ ] #2 Root cause identified and recorded (production code vs test contract)
<!-- AC:END -->
