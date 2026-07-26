---
id: TASK-687
title: Library shell reads CLI config when app_config already answered
status: To Do
assignee: []
created_date: '2026-07-26 05:36'
labels:
  - library
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Library shell tests assert that get_cli_setting is not consulted once app_config already carries the search-history and rail-preference values, and both fail: something reads the CLI config anyway, so a value set in app_config can be overridden by a stale one on disk. Pre-existing on dev, found while regression-testing the 684.2 registry work; both fail identically at 05ebe2ab7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Search history and rail preferences come from app_config when it has them
- [ ] #2 get_cli_setting is consulted only as a fallback
- [ ] #3 Both existing precedence tests pass
<!-- AC:END -->
