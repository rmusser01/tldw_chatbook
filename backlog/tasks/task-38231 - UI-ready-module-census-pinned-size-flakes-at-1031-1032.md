---
id: TASK-38231
title: 'UI-ready module census pinned size flakes at 1031/1032'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - tests
dependencies: []
references:
  - Tests/Performance/test_ui_ready_module_census.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_ui_ready_module_census fails on both the base and head arms with a pinned size of 1032 against 1031. It exists on dev today; it is not B2b's and B2b does not fix it. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census test passes repeatedly on dev with a pinned size that matches the measured one, or the pin is made robust to the 1031/1032 variation
<!-- AC:END -->
